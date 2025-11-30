import torch
import os, time
import ast
from torch.utils.cpp_extension import load

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout

import cuequivariance as cue
import cuequivariance_torch as cueq
import e3nn.o3 as o3

import math
import time 
from torch.autograd import Function
import numpy as np
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.jit.*"
)
torch.manual_seed(42)

cwtp_bwd = load(
    name="cwtp_bwd",
    sources=["cwtp_bwd.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90"
    ]
)

@tilelang.jit
def backward(B, U, dim, threads=128, dtype="float64", accum_dtype="float64"):
    
    block_B = 32
    total = U * dim

    @T.prim_func
    def main(
        grad_out: T.Tensor((B, dim, U), dtype), # type: ignore
        X: T.Tensor((B, U), dtype), # type: ignore
        Y: T.Tensor((B, dim), dtype), # type: ignore
        W: T.Tensor((B, U), dtype), # type: ignore
        B_buf: T.Tensor((B, U), dtype), # type: ignore
        grad_X: T.Tensor((B, U), dtype), # type: ignore
        grad_Y: T.Tensor((B, dim), dtype), # type: ignore
        grad_W: T.Tensor((B, U), dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(total, threads), threads=threads) as (b_idx, t_idx):
            t = T.thread_binding(0, threads, thread="threadIdx.x")
            tid = t_idx * threads + t

            grad_W_shared = T.alloc_shared((block_B, U), accum_dtype)

            for i in T.serial(T.ceildiv(U, threads)):
                idx = i * threads + t
                if idx < U:
                    grad_W_shared[b_idx * block_B, idx] = 0

            T.sync_threads()

            for bi in T.serial(block_B):
                b = b_idx * block_B + bi

                if tid < total:

                    tmp = tid
                    u = tmp // dim
                    i = tmp % dim

                    xv = T.cast(X[b, u], dtype)
                    yv = T.cast(Y[b, i], dtype)
                    wv = T.cast(W[B, u], dtype)
                    g = T.cast(grad_out[b, u, i], dtype)

                    T.atomic_add(grad_X[b, u], g * yv * wv)
                    T.atomic_add(grad_Y[b, i], g * xv * wv)
                    T.atomic_add(grad_W_shared[b, u], g * xv * yv)

            T.sync_threads()
            
            for i in T.serial(T.ceildiv(U, threads)):
                idx = i * threads + t
                if idx < U:
                    if grad_W_shared[b_idx, idx] != 0:
                        T.atomic_add(grad_W[b_idx * block_B, idx], grad_W_shared[b_idx, idx])
    return main

B = 368
U = 96
V = 1

retry = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype=torch.float64

#### cuequivarance ####
irreps_in1 = cue.Irreps("O3", "96x0e")
irreps_in2 = cue.Irreps("O3", "1x0e+1x1o+1x2e+1x3o")
irreps_out = cue.Irreps("O3", "96x0e+96x1o+96x2e+96x3o")

tp_cueq = cueq.ChannelWiseTensorProduct(
    irreps_in1, irreps_in2, 
    filter_irreps_out = irreps_out, 
    layout=cue.ir_mul,
    shared_weights=False,     
    internal_weights=False,
    device=device     
)

X = torch.randn(B, irreps_in1.dim, dtype=dtype, device=device, requires_grad=True)
Y = torch.randn(B, irreps_in2.dim, dtype=dtype, device=device, requires_grad=True)
W = torch.randn(B, tp_cueq.weight_numel, dtype=dtype, device=device, requires_grad=True)
Z = torch.zeros(B, U, device=device, dtype=dtype)

# print(X.shape)
# print(Y.shape)

print(f"irreps_in1: {irreps_in1.dim}, irreps_in2: {irreps_in2.dim}, irreps_out: {irreps_out.dim}, weight: {tp_cueq.weight_numel}")

instr = [
    (i_1, i_2, i_out, 'uvu', True, 1.0)
    for i_1, (mul1, ir_1) in enumerate(irreps_in1)     # i:编号
    for i_2, (mul2, ir_2) in enumerate(irreps_in2)     # mul:通道数
    for i_out, (mul3, ir_out) in enumerate(irreps_out) # ir.l:角量
    if ir_out in ir_1 * ir_2
]

dim_list = [
    (2 * ir_2.l + 1)
    for i_1, (_, ir_1) in enumerate(irreps_in1)
    for i_2, (_, ir_2) in enumerate(irreps_in2)
    for i_out, (_, ir_out) in enumerate(irreps_out)
    if ir_out in ir_1 * ir_2
]

cg_list = [
    (ir_1.l, ir_2.l, ir_out.l)
    for i_1, (_, ir_1) in enumerate(irreps_in1)
    for i_2, (_, ir_2) in enumerate(irreps_in2)
    for i_out, (_, ir_out) in enumerate(irreps_out)
    if ir_out in ir_1 * ir_2
]
path_num = len(instr)

### cueq ###
out_cueq = tp_cueq(X, Y, W)
grad_output = torch.ones_like(out_cueq)
print(grad_output.shape)

for retry_id in range(0, retry):
    if retry_id != 0:
        X.grad.zero_(), Y.grad.zero_(), W.grad.zero_()

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    out_cueq.backward(grad_output, retain_graph=True)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"cueq time: {(end - start):.4f} ms")

grad_X_cueq = X.grad.clone()
grad_Y_cueq = Y.grad.clone()
grad_W_cueq = W.grad.clone()

### cuda ###
W_reshaped = W.reshape(B, -1, U)
B_buf = X.unsqueeze(1) * W_reshaped
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    grad_X_cuda, grad_Y_cuda, grad_W_cuda = cwtp_bwd.backward(grad_output, X, Y, W, B_buf)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"cuda time: {(end - start):.4f} ms")

print(grad_W_cueq.shape)

cuda_err = max((grad_X_cueq - grad_X_cuda).abs().max(), (grad_Y_cueq - grad_Y_cuda).abs().max(), (grad_W_cueq - grad_W_cuda).abs().max())
print(f"cuda_err: {cuda_err.item()}")

### torch.einsum ###
for retry_id in range(0, retry):
    dim_offset = 0
    grad_X_einsum = torch.zeros_like(X)
    grad_Y = []
    grad_W = []

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    for dim_idx in range(0, path_num):
        dim = dim_list[dim_idx]

        x = X                                   # (B, U)
        y = Y[:, dim_offset:dim_offset + dim]   # (B, 1+3+5+7)
        w = W_reshaped[:, dim_idx]              # (B, 4, U)
        grad_out = grad_output[:, dim_offset * U: (dim_offset + dim) * U].reshape(B, -1, U)
        b_buf = B_buf[:, dim_idx]

        grad_x = torch.einsum("biu,bi,bu->bu", grad_out, y, w)
        grad_X_einsum += grad_x

        grad_y = torch.einsum("biu,bu->bi", grad_out, b_buf)
        grad_Y.append(grad_y)

        grad_w = torch.einsum("biu,bu,bi->bu", grad_out, x, y)
        grad_W.append(grad_w)

        dim_offset += dim

    grad_Y_einsum = torch.cat(grad_Y, dim=1)
    grad_W_einsum = torch.cat(grad_W, dim=1)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"torch time: {(end - start):.4f} ms")

torch_err = max((grad_X_cueq - grad_X_einsum).abs().max(), (grad_Y_cueq - grad_Y_einsum).abs().max(), (grad_W_cueq - grad_W_cuda).abs().max()) 
print(f"torch_err: {torch_err.item()}")

### tilelang ###
dim_offset = 0
grad_X_tilelang = torch.zeros_like(X)
grad_Y = []
grad_W = []
for dim_idx in range(0, path_num):
    dim = dim_list[dim_idx]
    kernel = backward(B, U, dim)

    x = X                                   # (B, U)
    y = Y[:, dim_offset:dim_offset + dim]   # (B, 1+3+5+7)
    w = W_reshaped[:, dim_idx]              # (B, 4, U)
    grad_out = grad_out = grad_output[:, dim_offset * U: (dim_offset + dim) * U].reshape(B, -1, U).contiguous()
    b_buf = B_buf[:, dim_idx, :].contiguous()

    grad_x = torch.zeros(B, U, dtype=torch.float64, device=device)
    grad_y = torch.zeros(B, dim, dtype=torch.float64, device=device)
    grad_w = torch.zeros(B, U, dtype=torch.float64, device=device)

    # kernel(grad_out, x, y, w, b_buf, grad_x, grad_y, grad_w)
    # grad_X_tilelang += grad_x
    # grad_Y.appned(grad_y)
    # grad_W.append(grad_w)

    dim_offset += dim

# grad_Y_tilelang = torch.cat(grad_Y, dim=1)
# grad_W_tilelang = torch.cat(grad_W, dim=1)
