import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout

import cuequivariance as cue
import cuequivariance_torch as cueq
import e3nn.o3 as o3

import math
import time 
import torch
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
def forward4(B, U, V, dim_sum,
            dtype="float16", 
            accum_dtype="float32",
            threads=256):
    
    # tile
    block_B = 64
    block_U = 64
    block_V = 64

    @T.prim_func
    def main(
            X: T.Tensor((B*dim_sum, U), dtype), # type: ignore
            W: T.Tensor((4, U, V), dtype), # type: ignore
            Z: T.Tensor((B*dim_sum, V), dtype), # type: ignore
    ):
        T.func_attr({"tir.noalias": True})
        T.func_attr({"tl.copy_vectorize": 4}) 

        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(V, block_V), threads=threads) as (b_idx, v_idx):
            
            # X_shared = T.alloc_shared((block_B, block_U), dtype)
            # W_shared = T.alloc_shared((block_U, block_V), dtype)

            # Z_local = T.alloc_fragment((block_B, block_V), accum_dtype)
            # T.clear(Z_local)

            for i in T.serial(4):
                X_shared = T.alloc_shared((block_B*(2*i-1), block_U), dtype)
                W_shared = T.alloc_shared((block_U, block_V), dtype)

                for uo in T.Pipelined(T.ceildiv(U, block_U), num_stages=4):
                    T.copy(X[b_idx * block_B, uo * block_U], X_shared)

            # # data_shared
            # for uo in T.Pipelined(T.ceildiv(U, block_U), num_stages=4):
            #     T.copy(X[b_idx * block_B, uo * block_U], X_shared)
            #     T.copy(W[uo * block_U, v_idx * block_V], W_shared)

            #     for i, j, k in T.Parallel(block_B, block_U, block_V):
            #         Z_local[i, k] += X_shared[i, j] * W_shared[j, k]

            # T.copy(Z_local, Z[b_idx * block_B, v_idx * block_V])

    return main

# -------- 基本参数 ----------
B = 368
U = V = 96
dim_list = [1, 3, 5, 7]               # 4 条路径的 i 维
num_paths = len(dim_list)
dim_sum = sum(dim_list)               # 1+3+5+7 = 16

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

# -------- 构造输入 ----------
# x: [B, sum(i)*u] = [736, 1536]
X = torch.randn(B, dim_sum * U, dtype=dtype, device=device)

# weight: [1, 36864] = [1, num_paths * u * v] -> [num_paths, u, v]
W = torch.randn(1, num_paths * U * V, dtype=dtype, device=device)
W_reshaped = W.view(num_paths, U, V).contiguous()   

retry = 100

cg_val = 0.10206207261596577
cg_tensor = torch.tensor(cg_val, dtype=dtype, device=device)

### hand ###
for retry_id in range(0, retry):
    dim_offset = 0
    outs = []

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000
    for i, dim in enumerate(dim_list):
        x = X[:, dim_offset * U: (dim_offset + dim) * U].view(B, dim, U).contiguous() 
        w = W_reshaped[i].view(U, V)

        out = x.view(B * dim, U) @ w * cg_tensor
        out = out.view(B, dim, V)

        outs.append(out)
        dim_offset += dim

    out_hand = torch.cat(outs, dim=1)

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"hand_forward_cost: {(end - start):.4f} ms")

# out_hand = torch.cat(outs, dim=1)
# print(output.shape)

### torch.einsum ###
for retry_id in range(0, retry):
    dim_offset = 0
    outs = []

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000
    for i, dim in enumerate(dim_list):
        x = X[:, dim_offset * U: (dim_offset + dim) * U].view(B, dim, U).contiguous()
        w = W_reshaped[i].view(U, V)

        out = torch.einsum("biu,uv->biv", x, w) * cg_tensor
        out = out.view(B, dim, V)

        outs.append(out)
        dim_offset += dim

    out_einsum = torch.cat(outs, dim=1)

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"einsum_forward_cost: {(end - start):.4f} ms")

err = (out_hand - out_einsum).abs()
print(f"error: {err.max().item()}")

### tilelang ###
for retry_id in range(0, retry):
    dim_offset = 0
    outs = []

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000
    for i, dim in enumerate(dim_list):
        kernel = forward(B, U, V, dim)

        x = X[:, dim_offset * U: (dim_offset + dim) * U].reshape(B*dim, U).contiguous()
        w = W_reshaped[i]
        z = torch.zeros(B*dim, V, dtype=dtype, device=device)

        kernel(x, w, z)

        z = z.reshape(B, dim, V)

        outs.append(z)
        dim_offset += dim

    out_tilelang = torch.cat(outs, dim=1) * cg_val

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"tilelang_forward_cost: {(end - start):.4f} ms")

err = (out_hand - out_tilelang).abs()
print(f"error: {err.max().item()}")

# dim_offset = 0
# outs = []
# for i, dim in enumerate(dim_list):
#     kernel = forward(B, U, V, dim)

#     x = X[:, dim_offset * U: (dim_offset + dim) * U].view(B, dim, U).reshape(B*dim, U).contiguous()
#     w = W_reshaped[i].view(U, V)
#     z = torch.zeros(B*dim, V, dtype=dtype, device=device)

#     # torch.cuda.cudart().cudaProfilerStart()
#     # torch.cuda.synchronize(device=device)

#     kernel(x, w, z)

#     # torch.cuda.synchronize(device=device)
#     # torch.cuda.cudart().cudaProfilerStop()

#     z = z.reshape(B, dim, V)

#     outs.append(z)
#     dim_offset += dim

# out_tilelang = torch.cat(outs, dim=1) * cg_val
    