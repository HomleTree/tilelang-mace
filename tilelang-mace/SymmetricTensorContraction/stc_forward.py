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

@tilelang.jit
def forward(B, U, V, W, dim,
            dtype="float16", 
            accum_dtype="float32",
            threads=128):
    
    # tile
    block_B = 64
    block_W = 64
    block_UV = 64
    block_U = 64

    @T.prim_func
    def main(
            X: T.Tensor((B, U, dim), dtype), # type: ignore
            Y: T.Tensor((B, V), dtype), # type: ignore
            W_tensor: T.Tensor((U*V, W), dtype), # type: ignore
            Z: T.Tensor((B*dim, W), dtype), # type: ignore
    ):
        T.func_attr({"tir.noalias": True})
        T.func_attr({"tl.copy_vectorize": 4}) 

        with T.Kernel(T.ceildiv(B*dim, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):
            
            Y_shared = T.alloc_shared((block_B, V), dtype)

            A_shared = T.alloc_shared((block_B, block_UV), dtype)
            B_shared = T.alloc_shared((block_UV, block_W), dtype)

            C_local = T.alloc_fragment((block_B, block_W), accum_dtype)
            T.clear(C_local)

            base_i = b_idx * block_B
            for i, v in T.Parallel(block_B, V):
                b = (base_i + i) // dim
                Y_shared[i, v] = Y[b, v]

            # data_shared
            for ko in T.Pipelined(T.ceildiv(U*V, block_UV), num_stages=4):
                T.copy(W_tensor[ko * block_UV, w_idx * block_W], B_shared)

                base_k = ko * block_UV
                for i, j in T.Parallel(block_B, block_UV):
                    b = (base_i + i) // dim
                    d = (base_i + i) % dim
                    u = (base_k + j) // V
                    v = (base_k + j) % V

                    A_shared[i, j] = X[b, u, d] * Y_shared[i, v]

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, Z[b_idx * block_B, w_idx * block_W])

    return main

kernel = forward(5152, 96, 10, 96, 3)

@tilelang.jit
def forward(B, U, V, W, dim,
            dtype="float16", 
            accum_dtype="float32",
            threads=128):
    
    # tile
    block_B = 64
    block_W = 64
    block_UV = 64
    block_U = 64

    @T.prim_func
    def main(
            X: T.Tensor((B, U, dim), dtype), # type: ignore
            Y: T.Tensor((B, V), dtype), # type: ignore
            W_tensor: T.Tensor((U*V, W), dtype), # type: ignore
            Z: T.Tensor((B*dim, W), dtype), # type: ignore
    ):
        T.func_attr({"tir.noalias": True})
        T.func_attr({"tl.copy_vectorize": 4}) 

        with T.Kernel(T.ceildiv(B*dim, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):
            
            Y_shared = T.alloc_shared((block_B, V), dtype)

            A_shared = T.alloc_shared((block_B, block_UV), dtype)
            B_shared = T.alloc_shared((block_UV, block_W), dtype)

            C_local = T.alloc_fragment((block_B, block_W), accum_dtype)
            T.clear(C_local)

            base_i = b_idx * block_B
            for i, v in T.Parallel(block_B, V):
                b = (base_i + i) // dim
                Y_shared[i, v] = Y[b, v]

            # data_shared
            for ko in T.Pipelined(T.ceildiv(U*V, block_UV), num_stages=4):
                T.copy(W_tensor[ko * block_UV, w_idx * block_W], B_shared)

                base_k = ko * block_UV
                for i, j in T.Parallel(block_B, block_UV):
                    b = (base_i + i) // dim
                    d = (base_i + i) % dim
                    u = (base_k + j) // V
                    v = (base_k + j) % V

                    A_shared[i, j] = X[b, u, d] * Y_shared[i, v]

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, Z[b_idx * block_B, w_idx * block_W])

    return main

# 参数设置
B, u = 368, 96
num_a = num_b = num_c = 16
num_i = 13
num_paths = 94

retry = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

x1 = torch.randn(B, num_a, u, dtype=dtype, device=device)
x0_g = torch.randn(B, num_i, u, dtype=dtype, device=device)

with open("cg_coeff_paths.txt", "r") as f:
    data = f.read()
host_paths = ast.literal_eval(data)
path_lens_tensor = torch.tensor([len(p) for p in host_paths], dtype=torch.int, device=device)
max_len = max(len(p) for p in host_paths)
padded_paths = [p + [0] * (max_len - len(p)) for p in host_paths]

paths_tensor = torch.tensor(padded_paths, dtype=torch.int, device=device)
coeffs = torch.randn(len(path_lens_tensor), dtype=dtype, device=device)

print(host_paths)

print(len(path_lens_tensor))

# baseline
def baseline():
    baseline_out = torch.zeros(B, u, dtype=dtype, device=device)
    for path_idx in range(0, len(paths_tensor)):
        path = paths_tensor[path_idx]
        real_path_len = path_lens_tensor[path_idx]
        if real_path_len == 3:
            a = path[0]
            i = path[1]
            baseline_out += x1[:, a] * x0_g[:, i] * coeffs[path_idx]
        if real_path_len == 4:
            a = path[0]
            b = path[1]
            i = path[2]
            baseline_out += x1[:, a] * x1[:, b] * x0_g[:, i] * coeffs[path_idx]
        if real_path_len == 5:
            a = path[0]
            b = path[1]
            c = path[2]
            i = path[3]
            baseline_out += x1[:, a] * x1[:, b] * x1[:, c] * x0_g[:, i] * coeffs[path_idx]
    return baseline_out

def baseline_opt():
    baseline_out = torch.zeros(B, u, dtype=dtype, device=device)

    # 按长度分 mask
    len3_mask = path_lens_tensor == 3
    len4_mask = path_lens_tensor == 4
    len5_mask = path_lens_tensor == 5

    print(len3_mask)

    # ---------- len=3 ----------
    if len3_mask.any():
        idx = paths_tensor[len3_mask]                  # [M3, 2]
        print(idx.shape)            
        c3  = coeffs[len3_mask]                        # [M3]
        x1_a = x1[:, idx[:, 0]]                      # [B, M3, u]
        x0_leg = x0_g[:, idx[:, 1]]                    # [B, M3, u]
        baseline_out += torch.einsum('bmu,bmu,m->bu', x1_a, x0_leg, c3)

    # ---------- len=4 ----------
    if len4_mask.any():
        idx = paths_tensor[len4_mask]                  # [M4, 3]
        print(idx.shape)  
        c4  = coeffs[len4_mask]
        x1_ab = x1[:, idx[:, 0]] * x1[:, idx[:, 1]]   # [B, M4, u]
        x0_leg = x0_g[:, idx[:, 2]]                   # [B, M4, u]
        baseline_out += torch.einsum('bmu,bmu,m->bu', x1_ab, x0_leg, c4)

    # ---------- len=5 ----------
    if len5_mask.any():
        idx = paths_tensor[len5_mask]                  # [M5, 4]
        print(idx.shape)  
        c5  = coeffs[len5_mask]
        x1_abc = x1[:, idx[:, 0]] * x1[:, idx[:, 1]] * x1[:, idx[:, 2]]  # [B, M5, u]
        x0_leg = x0_g[:, idx[:, 3]]                    # [B, M5, u]
        baseline_out += torch.einsum('bmu,bmu,m->bu', x1_abc, x0_leg, c5)

    return baseline_out



# 计时器
def benchmark(fn, label, retry=1):
    # torch.cuda.synchronize()
    # start = time.perf_counter() * 1000
    for retry_id in range(0, retry):

        torch.cuda.synchronize()
        start = time.perf_counter() * 1000
        out = fn()
        torch.cuda.synchronize()
        end = time.perf_counter() *1000
        if retry_id == retry - 1:
            t = end - start
    return label, out, t

### baseline ###
out_baseline = torch.zeros(B, u, dtype=dtype, device=device)

for retry_id in range(0, retry):

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000
    for path_idx in range(0, len(paths_tensor)):
        path = paths_tensor[path_idx]
        real_path_len = path_lens_tensor[path_idx]
        if real_path_len == 3:
            a = path[0]
            i = path[1]
            out_baseline += x1[:, a] * x0_g[:, i] * coeffs[path_idx]
        if real_path_len == 4:
            a = path[0]
            b = path[1]
            i = path[2]
            out_baseline += x1[:, a] * x1[:, b] * x0_g[:, i] * coeffs[path_idx]
        if real_path_len == 5:
            a = path[0]
            b = path[1]
            c = path[2]
            i = path[3]
            out_baseline += x1[:, a] * x1[:, b] * x1[:, c] * x0_g[:, i] * coeffs[path_idx]
        
    torch.cuda.synchronize()
    end = time.perf_counter() *1000
    t1 = end - start
    if retry_id == retry - 1:
        print(f"baseline time: {t1} ms")


### test ###
out_test = torch.zeros(B, u, dtype=dtype, device=device)

# 按长度分 mask
# len3_mask = path_lens_tensor == 3
# len4_mask = path_lens_tensor == 4
# len5_mask = path_lens_tensor == 5

# print(len3_mask)
K = 8
retry = 100
for retry_id in range(0, retry):

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    len3_mask = path_lens_tensor == 3
    len4_mask = path_lens_tensor == 4
    len5_mask = path_lens_tensor == 5
    # ---------- len=3 ----------
    if len3_mask.any():
        idx = paths_tensor[len3_mask]                  # [M3, 2]
        # if retry_id == retry - 1:
        #     print(idx.shape)            
        c3  = coeffs[len3_mask]                        # [M3]
        # if retry_id == retry - 1:
        #     print(c3)
        x1_leg = x1[:, idx[:, 0]]                      # [B, M3, u]
        x0_leg = x0_g[:, idx[:, 1]]                    # [B, M3, u]
        # out_test += torch.einsum('bmu,bmu,m->bu', x1_leg, x0_leg, c3)
        contrib = (x1_leg * x0_leg)            # [B,M3,u]
        out_test += contrib.mul(c3.view(1, -1, 1)).sum(dim=1)  # [B,u]

    # ---------- len=4 ----------
    if len4_mask.any():
        idx = paths_tensor[len4_mask]                  # [M4, 3]
        # if retry_id == retry - 1:
        #     print(idx.shape)  
        c4  = coeffs[len4_mask]
        # if retry_id == retry - 1:
        #     print(c4)
        x1_ab = x1[:, idx[:, 0]] * x1[:, idx[:, 1]]   # [B, M4, u]
        x0_leg = x0_g[:, idx[:, 2]]                   # [B, M4, u]
        # out_test += torch.einsum('bmu,bmu,m->bu', x1_ab, x0_leg, c4)
        contrib = (x1_ab * x0_leg)            # [B,M3,u]
        out_test += contrib.mul(c4.view(1, -1, 1)).sum(dim=1)  # [B,u]

    # ---------- len=5 ----------
    if len5_mask.any():
        idx = paths_tensor[len5_mask]                  # [M5, 4]
        # if retry_id == retry - 1:
        #     print(idx.shape)  
        c5  = coeffs[len5_mask]
        x1_abc = x1[:, idx[:, 0]] * x1[:, idx[:, 1]] * x1[:, idx[:, 2]]  # [B, M5, u]
        x0_leg = x0_g[:, idx[:, 3]]                    # [B, M5, u]
        # out_test += torch.einsum('bmu,bmu,m->bu', x1_abc, x0_leg, c5)
        contrib = (x1_abc * x0_leg)            # [B,M3,u]
        out_test += contrib.mul(c5.view(1, -1, 1)).sum(dim=1)  # [B,u]

    torch.cuda.synchronize()
    end = time.perf_counter() *1000
    t2 = end - start
    if retry_id == retry - 1:
        print(f"compute time: {t2} ms")

err = (out_baseline - out_test).abs()
print(f"error: {err.max().item()}")
print(f"speedup: {(t1/t2):.4f}x")
# print(out_baseline)
# print(out_test)
