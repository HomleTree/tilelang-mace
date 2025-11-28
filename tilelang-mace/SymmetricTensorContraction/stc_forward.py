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

# @tilelang.jit
# def forward(B, m, U,
#             dtype="float16", 
#             accum_dtype="float32",
#             threads=128):
    
#     # tile
#     block_B = 32
#     block_U = 32
#     block_V = 64
#     block_M = 16

#     @T.prim_func
#     def main(
#             X1: T.Tensor((B, m, U), dtype), # type: ignore
#             X0: T.Tensor((B, m, U), dtype), # type: ignore
#             cg: T.Tensor((m,), dtype), # type: ignore
#             out: T.Tensor((B, U), dtype), # type: ignore
#     ):
#         # T.func_attr({"tir.noalias": True})
#         # T.func_attr({"tl.copy_vectorize": 4}) 
#         # T.func_attr({"tl.tensor_mem_size": 48*1024})

#         with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(U, block_U), threads=threads) as (b_idx, u_idx):
            
#             X1_shared = T.alloc_shared((block_B, block_M, block_U), dtype)
#             X0_shared = T.alloc_shared((block_B, block_M, block_U), dtype)
#             cg_shared = T.alloc_shared((block_M,), dtype)

#             Z_local = T.alloc_fragment((block_B, block_U), accum_dtype)
#             T.clear(Z_local)

#             # data_shared
#             for mo in T.Pipelined(T.ceildiv(m, block_M), num_stages=3):
#                 # X1_shared = T.alloc_shared((block_B, m, block_U), dtype)
#                 # X0_shared = T.alloc_shared((block_B, m, block_U), dtype)
#                 # cg_shared = T.alloc_shared((m,), dtype)

#                 T.copy(X1[b_idx * block_B, mo*block_M, u_idx * block_U], X1_shared)
#                 T.copy(X0[b_idx * block_B, mo*block_M, u_idx * block_U], X0_shared)
#                 T.copy(cg[mo*block_M], cg_shared)

#                 # T.gemm(X_shared, W_shared, Z_local)
#                 for i, j, k in T.Parallel(block_B, block_U, block_M):
#                     Z_local[i, j] += X1_shared[i, j, k] * X1_shared[i, j, k] * cg_shared[k]

#             T.copy(Z_local, out[b_idx * block_B, u_idx * block_U])

#     return main

# @tilelang.jit
# def forward(B, num_a, num_i, u, path_num, 
#             dtype="float64", 
#             accum_dtype="float64",
#             threads=96):
    
#     # tile
#     block_B = 32
#     block_U = threads
#     block_V = 64
#     block_M = 16

#     @T.prim_func
#     def main(
#             X1: T.Tensor((B, num_a, u), dtype), # type: ignore
#             X0: T.Tensor((B, num_i, u), dtype), # type: ignore
#             cg: T.Tensor((path_num,), dtype), # type: ignore
#             paths: T.Tensor((path_num, 5), "int32"), # type: ignore
#             path_lens: T.Tensor((path_num,), "int32"), # type: ignore
#             out: T.Tensor((B, u), dtype), # type: ignore
#     ):
#         T.func_attr({"tir.noalias": True, "tl.copy_vectorize": 4})
        
#         with T.Kernel(B, T.ceildiv(u, block_U), threads=threads) as (b_idx, u_blk):
            
#             tx = T.thread_binding(0, block_U, "threadIdx.x")
#             u_base = u_blk * block_U
#             u_idx  = u_base + tx

#             X1_shared = T.alloc_shared((num_a, u), dtype)
#             X0_shared = T.alloc_shared((num_i, u), dtype)
#             # cg_shared = T.alloc_shared((block_M,), dtype)

#             # Z_local = T.alloc_fragment((u,), accum_dtype)
#             # T.clear(Z_local)

#             acc = T.alloc_fragment((1,), accum_dtype)
#             acc[0] = 0.0

#             T.copy(X1[b_idx, :, :], X1_shared)
#             T.copy(X0[b_idx, :, :], X0_shared)

#             T.sync_threads()

#             for p in T.serial(path_num):
#                 coeff = cg[p]
#                 len_p = path_lens[p]
#                 a = paths[p, 0]
#                 i = paths[p, 1] if len_p == 3 else paths[p, len_p - 2]

#                 val = T.alloc_fragment((1,), dtype)          
#                 val[0] = X1_shared[a, u_idx] 
#                 if len_p >= 4:
#                     b = paths[p, 1]
#                     val[0] *= X1_shared[b, u_idx]         # x1[b, u]
#                 if len_p == 5:
#                     c = paths[p, 2]
#                     val[0] *= X1_shared[c, u_idx]         # x1[c, u]
#                 val[0] *= X0_shared[i, u_idx]             # x0[i, u]
#                 val[0] *= coeff
#                 acc[0] += val[0]

#             T.atomic_add(out[b_idx, u_idx], acc[0])

#     return main

@tilelang.jit
def forward(B, num_a, num_i, u, path_num, 
            dtype="float16", 
            accum_dtype="float32",
            threads=96):
    
    # tile
    block_B = 16
    block_U = 96

    total = B * u

    @T.prim_func
    def main(
            X1: T.Tensor((B, num_a, u), dtype), # type: ignore
            X0: T.Tensor((B, num_i, u), dtype), # type: ignore
            cg: T.Tensor((path_num,), dtype), # type: ignore
            paths: T.Tensor((path_num, 5), "int32"), # type: ignore
            path_lens: T.Tensor((path_num,), "int32"), # type: ignore
            out: T.Tensor((B, u), dtype), # type: ignore
    ):
        T.func_attr({"tir.noalias": True, "tl.copy_vectorize": 4})
        
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(u, threads), threads=threads) as (b_blk, t_blk):
            
            tx = T.thread_binding(0, threads, "threadIdx.x")
            b_base = b_blk * block_B
            # u_base = u_blk * block_U
            u_idx  = t_blk * threads + tx

            X1_shared = T.alloc_shared((block_B, num_a, u), dtype)
            X0_shared = T.alloc_shared((block_B, num_i, u), dtype)

            acc = T.alloc_fragment((block_B,), accum_dtype)
            T.clear(acc)

            T.copy(X1[b_base, 0, 0], X1_shared)
            T.copy(X0[b_base, 0, 0], X0_shared)

            T.sync_threads()

            for p in T.serial(path_num):
                coeff = cg[p]
                len_p = path_lens[p]
                a = paths[p, 0]
                i = paths[p, 1] if len_p == 3 else paths[p, len_p - 2]

                for bi in T.serial(block_B):
                    if u_idx < u:
                        val = T.alloc_fragment((1,), dtype)          
                        val[0] = X1_shared[bi, a, u_idx] 
                        if len_p >= 4:
                            b = paths[p, 1]
                            val[0] *= X1_shared[bi, b, u_idx]         # x1[b, u]
                        if len_p == 5:
                            c = paths[p, 2]
                            val[0] *= X1_shared[bi, c, u_idx]         # x1[c, u]
                        val[0] *= X0_shared[bi, i, u_idx] * coeff            # x0[i, u]
                        acc[bi] += val[0]

            for bi in T.serial(block_B):
                if u_idx < u:
                    T.atomic_add(out[b_base + bi, u_idx], acc[bi])

    return main

# 设置 CUDA 调试同步
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

stp_kernel = load(
    name="segmented_tensor_product_kernel",
    sources=["baseline_cueq_stp.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90"
    ]
)

stc_fwd = load(
    name="stc_fwd",
    sources=["stc_fwd.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90"
    ]
)

# 参数设置
B, u = 368, 96
num_a = num_b = num_c = 16
num_i = 13
num_paths = 94

retry = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

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

# print(host_paths)
# print(path_lens_tensor)

### baseline ###
for retry_id in range(0, retry):
    out_baseline = torch.zeros(B, u, dtype=dtype, device=device)

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

# ### baseline_cuda ###
# for retry_id in range(0, retry):
#     out_baseline_cuda = torch.zeros(B, u, dtype=dtype, device=device) 

#     torch.cuda.synchronize()
#     start = time.perf_counter() * 1000

#     stp_kernel.segmented_tensor_product(
#         x1, x0_g, coeffs, paths_tensor, path_lens_tensor, out_baseline_cuda
#     )

#     torch.cuda.synchronize()
#     end = time.perf_counter() *1000
#     t3 = end - start
#     if retry_id == retry - 1:
#         print(f"baseline_cuda time: {t3} ms")

### test ###
# kernel = forward(B, num_a, num_i, u, 77)
for retry_id in range(0, retry):
    out_test = torch.zeros(B, u, dtype=dtype, device=device)

    len3_mask = path_lens_tensor == 3
    len4_mask = path_lens_tensor == 4
    len5_mask = path_lens_tensor == 5

    # torch.cuda.synchronize()
    # start = time.perf_counter() * 1000
    # ---------- len=3 ----------
    if len3_mask.any():
        idx = paths_tensor[len3_mask]                  # [M3, 2]        
        c3  = coeffs[len3_mask]                        # [M3]
        x1_leg = x1[:, idx[:, 0]]                      # [B, M3, u]
        x0_leg = x0_g[:, idx[:, 1]]                    # [B, M3, u]
        # out_test += torch.einsum('bmu,bmu,m->bu', x1_leg, x0_leg, c3)
        contrib = (x1_leg * x0_leg)            # [B,M3,u]
        out_test += contrib.mul(c3.view(1, -1, 1)).sum(dim=1)  # [B,u]

    # ---------- len=4 ----------
    if len4_mask.any():
        idx = paths_tensor[len4_mask]                  # [M4, 3]
        c4  = coeffs[len4_mask]
        x1_ab = x1[:, idx[:, 0]] * x1[:, idx[:, 1]]   # [B, M4, u]
        x0_leg = x0_g[:, idx[:, 2]]                   # [B, M4, u]
        # out_test += torch.einsum('bmu,bmu,m->bu', x1_ab, x0_leg, c4)
        contrib = (x1_ab * x0_leg)            # [B,M3,u]
        out_test += contrib.mul(c4.view(1, -1, 1)).sum(dim=1)  # [B,u]

    # ---------- len=5 ----------
    if len5_mask.any():

        idx = paths_tensor[len5_mask]                  # [M5, 4]
        c5  = coeffs[len5_mask]
        d5 = path_lens_tensor[len5_mask]
        
        if retry_id == retry - 1:
            print(c5)
            # print(idx)
            print(idx)
            print(idx[:, 0])

        x1_abc = x1[:, idx[:, 0]] * x1[:, idx[:, 1]] * x1[:, idx[:, 2]]  # [B, M5, u]
        x0_leg = x0_g[:, idx[:, 3]]               # [B, M5, u]
        # out_test += torch.einsum('bmu,bmu,m->bu', x1_abc, x0_leg, c5)
        contrib = (x1_abc * x0_leg)            
        out_test += contrib.mul(c5.view(1, -1, 1)).sum(dim=1)
        # out_test += out_tilelang1

    # torch.cuda.synchronize()
    # end = time.perf_counter() *1000
    # t2 = end - start
    # if retry_id == retry - 1:
    #     print(f"baseline_opt time: {t2} ms")

# ### opt_cuda ###
# for retry_id in range(0, retry):

#     torch.cuda.synchronize()
#     start = time.perf_counter() * 1000

#     out = stc_fwd.forward(
#         x1, x0_g, coeffs, paths_tensor, path_lens_tensor
#     )

#     torch.cuda.synchronize()
#     end = time.perf_counter() *1000
#     t4 = end - start
#     if retry_id == retry - 1:
#         print(f"opt_cuda time: {t4} ms")

### tilelang ###
kernel = forward(B, num_a, num_i, u, 94)
for retry_id in range(0, retry):
    out_tilelang = torch.zeros(B, u, dtype=dtype, device=device)

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    kernel(x1, x0_g, coeffs, paths_tensor, path_lens_tensor, out_tilelang)

    torch.cuda.synchronize()
    end = time.perf_counter() *1000
    t5 = end - start
    if retry_id == retry - 1:
        print(f"tilelang time: {t5} ms")

err = (out_baseline - out_test).abs()
print(f"test_error: {err.max().item()}")
# print(f"speedup: {(t1/t2):.4f}x")

# print(out_baseline)
# print(out_tilelang)

err = (out_baseline - out_tilelang).abs()
print(f"tilelang_error: {err.max().item()}")
