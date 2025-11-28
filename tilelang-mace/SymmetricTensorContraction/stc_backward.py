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
def backward(B, num_a, num_i, u, path_num, 
            dtype="float16", 
            accum_dtype="float16",
            threads=96):
    
    # tile
    block_B = 16
    block_U = 96

    @T.prim_func
    def main(
            grad_out: T.Tensor((B, u), dtype), # type: ignore
            X1: T.Tensor((B, num_a, u), dtype), # type: ignore
            X0: T.Tensor((B, num_i, u), dtype), # type: ignore
            cg: T.Tensor((path_num,), dtype), # type: ignore
            paths: T.Tensor((path_num, 5), "int32"), # type: ignore
            path_lens: T.Tensor((path_num,), "int32"), # type: ignore
            grad_x: T.Tensor((B, num_a, u), dtype), # type: ignore
    ):
        T.func_attr({"tir.noalias": True, "tl.copy_vectorize": 4})
        
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(u, block_U), threads=threads) as (b_blk, u_blk):
            
            tx = T.thread_binding(0, threads, "threadIdx.x")
            b_base = b_blk * block_B
            u_base = u_blk * block_U
            u_idx  = u_base + tx

            X1_shared = T.alloc_shared((block_B, num_a, u), dtype)
            X0_shared = T.alloc_shared((block_B, num_i, u), dtype)
            grad_out_shared = T.alloc_shared((block_B, u), dtype)

            T.copy(X1[b_base, 0, 0], X1_shared)
            T.copy(X0[b_base, 0, 0], X0_shared)
            T.copy(grad_out[b_base, 0], grad_out_shared)

            T.sync_threads()

            for p in T.serial(path_num):
                coeff = cg[p]
                len_p = path_lens[p]
                a = paths[p, 0]
                i = paths[p, 1] if len_p == 3 else paths[p, len_p - 2]

                for bi in T.serial(block_B):
                    base = grad_out_shared[bi, u_idx] * coeff * X0_shared[bi, i, u_idx]

                    if len_p == 3:
                        T.atomic_add(grad_x[b_base + bi, a, u_idx], base)
                    if len_p == 4:
                        b = paths[p, 1]
                        base_1 = base * X1_shared[bi, b, u_idx]
                        base_2 = base * X1_shared[bi, a, u_idx]
                        T.atomic_add(grad_x[b_base + bi, a, u_idx], base_1) 
                        T.atomic_add(grad_x[b_base + bi, b, u_idx], base_2)    
                    if len_p == 5:
                        b = paths[p, 1]
                        c = paths[p, 2]
                        base_3 = base * X1_shared[bi, b, u_idx] * X1_shared[bi, c, u_idx]
                        base_4 = base * X1_shared[bi, a, u_idx] * X1_shared[bi, c, u_idx]
                        base_5 = base * X1_shared[bi, a, u_idx] * X1_shared[bi, b, u_idx]

                        T.atomic_add(grad_x[b_base + bi, a, u_idx], base_3) 
                        T.atomic_add(grad_x[b_base + bi, b, u_idx], base_4)
                        T.atomic_add(grad_x[b_base + bi, c, u_idx], base_5)
    return main

# 设置 CUDA 调试同步
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

stc_bwd = load(
    name="stc_bwd",
    sources=["stc_bwd.cu"],
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

print(paths_tensor.shape)

grad_out = torch.randn(B, u, dtype=dtype, device=device)

# for retry_id in range(0, retry):
#     torch.cuda.synchronize()
#     start = time.perf_counter() * 1000

#     grad_x1_cuda = stc_bwd.backward(
#         grad_out, x1, x0_g, coeffs, paths_tensor, path_lens_tensor
#     )

#     torch.cuda.synchronize()
#     end = time.perf_counter() *1000
#     t1 = end - start
#     if retry_id == retry - 1:
#         print(f"opt_cuda time: {t1} ms")

for retry_id in range(0, retry):

    grad_x1_baseline = torch.zeros_like(x1)

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000
    for path_idx in range(0, len(paths_tensor)):
        path = paths_tensor[path_idx]
        real_path_len = path_lens_tensor[path_idx]
        
        if real_path_len == 3:
            a = path[0]
            i = path[1]
            grad_x1_baseline[:, a] += grad_out * x0_g[:, i] * coeffs[path_idx]
        elif real_path_len == 4:
            a = path[0]
            b = path[1]
            i = path[2]
            grad_x1_baseline[:, a] += grad_out * x1[:, b] * x0_g[:, i] * coeffs[path_idx]
            grad_x1_baseline[:, b] += grad_out * x1[:, a] * x0_g[:, i] * coeffs[path_idx] 
        elif real_path_len == 5:
            a = path[0]
            b = path[1]
            c = path[2]
            i = path[3]
            grad_x1_baseline[:, a] += grad_out * x1[:, b] * x1[:, c] * x0_g[:, i] * coeffs[path_idx]
            grad_x1_baseline[:, b] += grad_out * x1[:, a] * x1[:, c] * x0_g[:, i] * coeffs[path_idx]
            grad_x1_baseline[:, c] += grad_out * x1[:, a] * x1[:, b] * x0_g[:, i] * coeffs[path_idx]

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000
    t2 = end - start
    if retry_id == retry - 1:
        print(f"baseline time: {t2} ms")    

# err = (grad_x1 - grad_x1_baseline).abs()
# print(f"baseline error: {err.max().item()}")

kernel = backward(B, num_a, num_i, u, num_paths)
for retry_id in range(0, retry):
    grad_x1_tilelang = torch.zeros_like(x1)

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    kernel(grad_out, x1, x0_g, coeffs, paths_tensor, path_lens_tensor, grad_x1_tilelang)

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000
    t3 = end - start
    if retry_id == retry - 1:
        print(f"tilelang time: {t3} ms")

err = (grad_x1_baseline - grad_x1_tilelang).abs()
print(f"error: {err.max().item()}")
# print(grad_x1_cuda)
# print(grad_x1_tilelang)
# print(grad_x1_baseline)
