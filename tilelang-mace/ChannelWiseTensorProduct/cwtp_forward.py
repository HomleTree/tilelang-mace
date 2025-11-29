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

cwtp_fwd = load(
    name="cwtp_fwd",
    sources=["cwtp_fwd.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90"
    ]
)

B = 5152
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

# print(cg_list)
path_num = len(instr)

for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    out_cueq = tp_cueq(X, Y, W)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"cueq time: {(end - start):.4f} ms")


W_reshaped = W.reshape(B, -1, U)
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    
    out_cuda, _ = cwtp_fwd.forward(X, Y, W)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"cuda time: {(end - start):.4f} ms")

cuda_err = (out_cueq - out_cuda).abs()
print(f"cuda_err: {cuda_err.max().item()}")

# print(out_cueq)
# print(out_cuda)

b_buf = X.unsqueeze(1) * W_reshaped
print(b_buf.shape)

dim_offset = 0
outs = []
for dim_idx in range(0, path_num):
    dim = dim_list[dim_idx]

    x = X                                   # (B, U)
    y = Y[:, dim_offset:dim_offset + dim]   # (B, 1+3+5+7)
    w = W_reshaped[:, dim_idx]              # (B, 4, W)
    
    # method 1
    # out1 = torch.einsum("bu,bu->bu", x, w)
    # out = torch.einsum("bu,bi->biu", out1, y)

    # method 2
    out = torch.einsum("bu,bi,bu->biu", x, y, w)

    # method cuda
    # out = b_buf[:, dim_idx, :].unsqueeze(1) * y.unsqueeze(2)

    out = out.reshape(B, -1)
    outs.append(out)

    dim_offset += dim

out_einsum = torch.cat(outs, dim=1)

torch_err = (out_cueq - out_einsum).abs()
print(f"torch_err: {torch_err.max().item()}")
