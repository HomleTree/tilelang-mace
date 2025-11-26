import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout

import cuequivariance as cue
import cuequivariance_torch as cueq
import e3nn.o3 as o3

import math
import time 
import torch
from torch.autograd import Function
import numpy as np
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.jit.*"
)
torch.manual_seed(42)

@tilelang.jit
def forward(B, U, V, dim,
            dtype="float16", 
            accum_dtype="float32",
            threads=128):
    
    # tile
    block_B = 64
    block_U = 64
    block_V = 64

    @T.prim_func
    def main(
            X: T.Tensor((B*dim, U), dtype), # type: ignore
            W: T.Tensor((U, V), dtype), # type: ignore
            Z: T.Tensor((B*dim, V), dtype), # type: ignore
    ):
        T.func_attr({"tir.noalias": True})
        T.func_attr({"tl.copy_vectorize": 4}) 
        T.func_attr({"tl.tensor_mem_size": 48*1024})

        with T.Kernel(T.ceildiv(B*dim, block_B), T.ceildiv(V, block_V), threads=threads) as (b_idx, v_idx):
            
            X_shared = T.alloc_shared((block_B, block_U), dtype)
            W_shared = T.alloc_shared((block_U, block_V), dtype)

            Z_local = T.alloc_fragment((block_B, block_V), accum_dtype)
            T.clear(Z_local)

            # data_shared
            for uo in T.Pipelined(T.ceildiv(U, block_U), num_stages=4):
                T.copy(X[b_idx * block_B, uo * block_U], X_shared)
                T.copy(W[uo * block_U, v_idx * block_V], W_shared)

                T.gemm(X_shared, W_shared, Z_local)

            T.copy(Z_local, Z[b_idx * block_B, v_idx * block_V])

    return main

@tilelang.jit
def backward(B, U, V, dim,
            dtype="float16", 
            accum_dtype="float32",
            threads=128):
    
    # tile
    block_B = 64
    block_U = 64
    block_V = 64

    @T.prim_func
    def main(
            grad_out: T.Tensor((B*dim, V), dtype), # type: ignore
            W: T.Tensor((V, U), dtype), # type: ignore
            grad_X: T.Tensor((B*dim, U), dtype), # type: ignore
    ):
        T.func_attr({"tir.noalias": True})
        T.func_attr({"tl.copy_vectorize": 4}) 
        T.func_attr({"tl.tensor_mem_size": 48*1024})

        with T.Kernel(T.ceildiv(B*dim, block_B), T.ceildiv(U, block_U), threads=threads) as (b_idx, u_idx):
            
            grad_out_shared = T.alloc_shared((block_B, block_V), dtype)
            W_shared = T.alloc_shared((block_V, block_U), dtype)

            X_local = T.alloc_fragment((block_B, block_U), accum_dtype)
            T.clear(X_local)

            # data_shared
            for vo in T.Pipelined(T.ceildiv(V, block_V), num_stages=4):
                T.copy(grad_out[b_idx * block_B, vo * block_V], grad_out_shared)
                T.copy(W[vo * block_V, u_idx * block_U], W_shared)

                T.gemm(grad_out_shared, W_shared, X_local)

            T.copy(X_local, grad_X[b_idx * block_B, u_idx * block_U])

    return main

### example ###
class MultiPathMatMul(Function):
    @staticmethod
    def forward(ctx, x, W, i_list, cg):
        """
        x: (B, I_total, u)
        W: (num_paths, u, v)
        i_list: python list of ints, lengths per path
        cg: scalar tensor or float (broadcastable)
        returns: out (B, I_total * v) where each path produces (B, i*v) concatenated along dim=1
        """
        device = x.device
        dtype = x.dtype

        num_paths = len(i_list)
        u = W.size(1)
        v = W.size(2)
        B = x.size(0)

        # compute forward as in your loop
        out_chunks = []
        offset = 0
        for p, i in enumerate(i_list):
            Xi = x[:, offset:offset + i, :].contiguous()   # (B, i, u)
            # reshape to (B*i, u) and multiply by W[p] (u,v) -> (B*i, v)
            Yi = Xi.view(B * i, u) @ W[p].view(u, v)       # (B*i, v)
            Yi = (cg * Yi)                                 # scale
            Yi = Yi.view(B, i, v)                         # (B, i*v)
            out_chunks.append(Yi)
            offset += i

        out = torch.cat(out_chunks, dim=1)  # (B, I_total * v)

        # save for backward
        # we need x and W and i_list and cg and shapes to compute grads
        ctx.save_for_backward(x, W, torch.tensor(i_list, dtype=torch.int64, device=device), cg)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (B, I_total * v)
        returns: gradients for (x, W, None(for i_list), None(for cg))
        """
        x, W, i_list_tensor, cg = ctx.saved_tensors
        i_list = i_list_tensor.cpu().tolist()
        device = x.device
        dtype = x.dtype

        B = x.size(0)
        num_paths = len(i_list)
        u = W.size(1)
        v = W.size(2)
        I_total = sum(i_list)

        # prepare gradients
        dX = torch.zeros_like(x)

        offset_out = 0
        out_chunks = []

        grad_out = grad_output.reshape(B, I_total, v)
        for p, i in enumerate(i_list):
            # slice grad_output for this path
            dY = grad_out[:, offset_out:offset_out + i, :].contiguous()

            # dXi = cg * dY @ W[p].T  -> (B, i, u)
            Wp = W[p].view(u, v)                                     # (u, v)
            dXi = torch.matmul(dY, Wp.t()) * cg                      # (B, i, u)
            out_chunks.append(dXi)

            offset_out += i

        dX = torch.cat(out_chunks, dim=1)
        # return gradients in same order as forward args: (x, W, i_list, cg)
        # i_list and cg are not tensors requiring grad (or we don't compute their grads) -> None
        return dX, None, None, None 

def multipath_matmul(x, W, i_list, cg):
    return MultiPathMatMul.apply(x, W, i_list, cg)

# -------- 基本参数 ----------
B = 5152
U = V = 96
dim_list = [1, 3, 5, 7]               # 4 条路径的 i 维
num_paths = len(dim_list)
dim_sum = sum(dim_list)               # 1+3+5+7 = 16

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

# -------- 构造输入 ----------
# x: [B, sum(i)*u] = [736, 1536]
X = torch.randn(B, dim_sum, U, dtype=dtype, device=device, requires_grad=True)

# weight: [1, 36864] = [1, num_paths * u * v] -> [num_paths, u, v]
W = torch.randn(1, num_paths * U * V, dtype=dtype, device=device)
W_reshaped = W.view(num_paths, U, V).contiguous() 
W_reshaped.requires_grad_(False)  

retry = 100

cg_val = 0.10206207261596577
cg_tensor = torch.tensor(cg_val, dtype=dtype, device=device)

### hand ###
# forward #
dim_offset = 0
outs = []

for i, dim in enumerate(dim_list):
    x = X[:, dim_offset : dim_offset + dim, :].contiguous() 
    w = W_reshaped[i]

    out = x.view(B * dim, U) @ w * cg_tensor
    out = out.view(B, dim, V)

    outs.append(out)
    dim_offset += dim

out_hand = torch.cat(outs, dim=1)

# backward #
grad_output = torch.rand(out_hand.numel(), dtype=dtype, device=device).reshape(out_hand.shape)
loss_hand = (out_hand * grad_output).sum()
for retry_id in range(0, retry):

    if retry_id != 0:
        X.grad.zero_()

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    loss_hand.backward(retain_graph=True)

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    t = end - start
    if retry_id == retry - 1:
        print(f"hand_backward_cost: {t:.4f} ms")

grad_X_hand = X.grad.clone()
# print(f"out_hand shape: {out_hand.shape}")

### tilelang ###
# # forward #
# dim_offset = 0
# outs = []

# for i, dim in enumerate(dim_list):
#     kernel1 = forward(B, U, V, dim)

#     x = X[:, dim_offset : dim_offset + dim, :].reshape(B*dim, U).contiguous()
#     w = W_reshaped[i]
#     z = torch.zeros(B*dim, V, dtype=dtype, device=device)

#     kernel1(x, w, z)

#     z = z.reshape(B, dim, V)

#     outs.append(z)
#     dim_offset += dim

# out_tilelang = torch.cat(outs, dim=1) * cg_val

# backward #
for retry_id in range(0, retry):
    dim_offset = 0
    grad_X = []

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000
    for i, dim in enumerate(dim_list):
        kernel = backward(B, U, V, dim)

        grad_out = grad_output[:, dim_offset : dim_offset + dim, :].reshape(B*dim, -1).contiguous() 
        w = W_reshaped[i].t().contiguous() 
        grad_x = torch.zeros(B*dim, U, dtype=dtype, device=device)

        kernel(grad_out, w, grad_x)

        grad_x = grad_x.reshape(B, dim, U)

        grad_X.append(grad_x)
        dim_offset += dim

    grad_X_tilelang = torch.cat(grad_X, dim=1) * cg_val

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    t = end - start
    if retry_id == retry - 1:
        print(f"tilelang_backward_cost: {t:.4f} ms")

### torch.einsum ###
# forward #
dim_offset = 0
outs = []

torch.cuda.synchronize()
start = time.perf_counter() * 1000
for i, dim in enumerate(dim_list):
    x = X[:, dim_offset : dim_offset + dim, :].contiguous()
    w = W_reshaped[i].view(U, V)

    out = torch.einsum("biu,uv->biv", x, w) * cg_tensor
    # out = out.view(B, -1)

    outs.append(out)
    dim_offset += dim

out_einsum = torch.cat(outs, dim=1)

# backward #
loss_einsum = (out_einsum * grad_output).sum()
for retry_id in range(0, retry):

    if retry_id != 0:
        X.grad.zero_()

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    loss_einsum.backward(retain_graph=True)

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    t = end - start
    if retry_id == retry - 1:
        print(f"einsum_backward_cost: {t:.4f} ms")

grad_X_einsum = X.grad.clone()

### multi_path ###
# forward #
x_test = X.clone().detach().requires_grad_(True)
W_test = W_reshaped.clone().detach().requires_grad_(False)
Y_test = multipath_matmul(x_test, W_test, dim_list, cg_tensor)

# backward #
loss_test = (Y_test * grad_output).sum()
for retry_id in range(0, retry):

    if retry_id != 0:
        x_test.grad.zero_()

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    loss_test.backward(retain_graph=True)

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    t = end - start
    if retry_id == retry - 1:
        print(f"multi_backward_cost: {t:.4f} ms")
# loss_test.backward()
grad_X_test = x_test.grad

# print(grad_X_tilelang)
# print(grad_X_test)

# print(f"forward_error: {(out_hand - out_tilelang).abs().max().item()}")
print(f"backward error: {(grad_X_test - grad_X_tilelang).abs().max().item()}")

    