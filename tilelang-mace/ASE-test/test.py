import itertools
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
# def shared_reduce_mv(B, dtype="float64", accum_dtype="float64", threads=128):
#     warp_size = 32
#     warps_per_block = threads // warp_size
#     block_M = warp_size
#     block_N = warp_size

#     @T.prim_func
#     def main(
#         X: T.Tensor((B, B), dtype),
#         y: T.Tensor((B,),   dtype),
#         Z: T.Tensor((B,),   dtype),
#     ):
#         T.func_attr({"tir.noalias": True})
#         T.func_attr({"tl.copy_vectorize": 4})

#         with T.Kernel(T.ceildiv(B, block_M)) as m_blk:

#             X_sh = T.alloc_shared((block_M, block_N), dtype)
#             y_sh = T.alloc_shared((block_N,),        dtype)
#             # 每线程一个 partial 和
#             partial = T.alloc_shared((warps_per_block, warp_size), accum_dtype)

#             thread_x = T.thread_binding(0, threads, thread="threadIdx.x")
#             warp_id = thread_x // warp_size
#             lane_id = thread_x %  warp_size

#             acc = T.alloc_fragment((1,), accum_dtype)
#             acc[0] = 0.0

#             for n_blk in T.Pipelined(T.ceildiv(B, block_N), num_stages=2):
#                 T.copy(X[m_blk*block_M + warp_id, n_blk*block_N], X_sh[warp_id, :])
#                 T.copy(y[n_blk*block_N],                          y_sh)
#                 T.sync_threads()

#                 # ---- 每线程负责一列 ----
#                 val = X_sh[warp_id, lane_id] * y_sh[lane_id]
#                 acc[0] += val          # 线程内先累加当前 tile
#                 T.sync_threads()

#             # ---- 把私有 acc 写回 shared ----
#             partial[warp_id, lane_id] = acc[0]
#             T.sync_threads()

#             # ---- lane-0 把同 warp 32 个数累加 ----
#             if lane_id == 0:
#                 warp_sum = T.alloc_fragment((1,), accum_dtype)
#                 warp_sum[0] = 0.0
#                 for k in T.serial(warp_size):
#                     warp_sum[0] += partial[warp_id, k]
#                 # 写回 global
#                 Z[m_blk*block_M + warp_id] = warp_sum[0]

#     return main

# @tilelang.jit
# def naive_gemv(
#     N, K,
#     BLOCK_N,
#     reduce_threads,
#     dtype="float64",
#     accum_dtype="float64",
# ):

#     MAX_TRANSACTION_SIZE_IN_BITS = 128
#     TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // 32
#     BLOCK_K = reduce_threads * TILE_K

#     @T.prim_func
#     def main(
#             A: T.Tensor((K,), dtype), # type: ignore
#             B: T.Tensor((N, K), dtype), # type: ignore
#             C: T.Tensor((N,), dtype), # type: ignore
#     ):
#         with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
#             tn = T.get_thread_binding(0)
#             tk = T.get_thread_binding(1)
#             A_local = T.alloc_local((TILE_K,), dtype)
#             B_local = T.alloc_local((TILE_K,), dtype)
#             C_accum = T.alloc_local((1,), accum_dtype)

#             T.clear(C_accum)
#             # for bk in T.serial(T.ceildiv(K, BLOCK_K)):
#             #     for k in T.vectorized(TILE_K):
#             #         A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
#             #         B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
#             #     for k in T.serial(TILE_K):
#             #         C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)

#             for bk, k in T.Parallel(T.ceildiv(K, BLOCK_K), TILE_K):
#                 # for k in T.vectorized(TILE_K):
#                 A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
#                 B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
#                 # for k in T.serial(TILE_K):
#                 C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
#             C_reduced = T.alloc_local((1,), accum_dtype)
#             with T.attr(
#                     T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
#                     "reduce_scope",
#                     T.reinterpret(T.uint64(0), dtype="handle"),
#             ):
#                 T.evaluate(
#                     T.tvm_thread_allreduce(
#                         T.uint32(1),
#                         C_accum[0],
#                         True,
#                         C_reduced[0],
#                         tk,
#                         dtype="handle",
#                     ))

#             C[bn * BLOCK_N + tn] = C_reduced[0]

#     return main

B = 147
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype=torch.float64

retry = 100

### gemv ###
print(f"\n*** gemv ***")
X = torch.randn(B, B, dtype=dtype, device=device)
Y = torch.randn(B, dtype=dtype, device=device)

## baseline ##
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    gemv_baseline = X @ Y

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"gemv_baseline time: {(end - start):.4f} ms")

## tilelang_1 ##
@tilelang.jit
def gemv_1(B,
         dtype="float64", 
         accum_dtype="float64",
         threads=1):
    
    # tile
    block_B = 32

    @T.prim_func
    def main(
            X: T.Tensor((B, B), dtype), # type: ignore
            Y: T.Tensor((B,), dtype), # type: ignore
            Z: T.Tensor((B), dtype), # type: ignore
    ):
        T.func_attr({"tir.noalias": True})
        T.func_attr({"tl.copy_vectorize": 4}) 

        with T.Kernel(T.ceildiv(B, block_B), threads=threads) as (b_idx):

            X_shared = T.alloc_shared((block_B, block_B), dtype)
            Y_shared = T.alloc_shared((block_B,), dtype)

            Z_local = T.alloc_fragment((block_B,), accum_dtype)
            T.clear(Z_local)

            for bo in T.Pipelined(T.ceildiv(B, block_B), num_stages=3):
                T.copy(X[b_idx * block_B, bo * block_B], X_shared)
                T.copy(Y[bo * block_B], Y_shared)

                for i, j in T.Parallel(block_B, block_B):
                    Z_local[i] += X_shared[i, j] * Y_shared[j]

            T.copy(Z_local, Z[b_idx * block_B])

    return main

kernel = gemv_1(B)
for retry_id in range(0, retry):
    gemv_tilelang_1 = torch.zeros_like(gemv_baseline)

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    
    kernel(X, Y, gemv_tilelang_1)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"gemv_tilelang_1 time: {(end - start):.4f} ms")

gemv_tilelang_1_err = (gemv_baseline - gemv_tilelang_1).abs()
print(f"gemv_tilelang_1 error: {gemv_tilelang_1_err.max().item()}")

## tilelang_2 ##
@tilelang.jit
def gemv_2(
    N, K,
    BLOCK_N,
    reduce_threads,
    dtype="float64",
    accum_dtype="float64",
):

    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // 32
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
            A: T.Tensor((K,), dtype), # type: ignore
            B: T.Tensor((N, K), dtype), # type: ignore
            C: T.Tensor((N,), dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)

            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)

            T.clear(C_accum)
            # for bk in T.serial(T.ceildiv(K, BLOCK_K)):
            #     for k in T.vectorized(TILE_K):
            #         A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
            #         B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
            #     for k in T.serial(TILE_K):
            #         C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)

            for bk, k in T.Parallel(T.ceildiv(K, BLOCK_K), TILE_K):
                # for k in T.vectorized(TILE_K):
                A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                # for k in T.serial(TILE_K):
                C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            C_reduced = T.alloc_local((1,), accum_dtype)
            with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        C_accum[0],
                        True,
                        C_reduced[0],
                        tk,
                        dtype="handle",
                    ))

            C[bn * BLOCK_N + tn] = C_reduced[0]

    return main

kernel = gemv_2(B, B, 1, 4)
for retry_id in range(0, retry):
    gemv_tilelang_2 = torch.zeros_like(gemv_baseline)

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    
    kernel(Y, X, gemv_tilelang_2)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"gemv_tilelang_2 time: {(end - start):.4f} ms")

gemv_2_err = (gemv_baseline - gemv_tilelang_2).abs()
print(f"gemv_tilelang_2 error: {gemv_2_err.max().item()}")

### dot ###
print(f"\n*** dot ***")
from torch_scatter import scatter

B = 3675
X = torch.randn(B, dtype=dtype, device=device)
Y = torch.randn(B, dtype=dtype, device=device)

batch_indices = torch.arange(25, device=device).repeat(49)
# index = batch_indices.repeat_interleave(3)  # shape:[1225 * 3]

## baseline ##
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    
    dot_baseline = scatter(X * Y, batch_indices.repeat_interleave(3), reduce="sum") 

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"dot_baseline time: {(end - start):.4f} ms")

## torch ##
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    index = batch_indices.repeat_interleave(3)
    dim_size = int(index.max()) + 1

    dot_torch = torch.zeros(dim_size, dtype=dot_baseline.dtype, device=dot_baseline.device)
    
    out = X * Y
    dot_torch.scatter_add_(0, index, out)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"dot_torch time: {(end - start):.4f} ms")

dot_torch_err = (dot_baseline - dot_torch).abs()
print(f"dot_torch error: {dot_torch_err.max().item()}")

## tilelang ##
@tilelang.jit
def dot(B, dim_size, 
        dtype="float64", 
        accum_dtype="float64"):
    
    # tile
    block_B = 32

    @T.prim_func
    def main(
            X: T.Tensor((B,), "float64"), # type: ignore
            Y: T.Tensor((B,), "float64"), # type: ignore
            index: T.Tensor((B), "int64"), # type: ignore
            Z: T.Tensor((dim_size), "float64"), # type: ignore
    ):
        T.func_attr({"tir.noalias": True})
        T.func_attr({"tl.copy_vectorize": 4}) 

        with T.Kernel(T.ceildiv(B, block_B), threads=block_B) as (b_idx):
            t = T.thread_binding(0, block_B, "threadIdx.x")
            tid = b_idx * block_B + t

            if tid < B:
                i = index[tid]
                T.atomic_add(Z[i], X[tid] * Y[tid] )
                # Z[i] += X[tid] * Y[tid]   

    return main

dim_size = 25
kernel = dot(B, dim_size)
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    dot_tilelang = torch.zeros_like(dot_baseline)
    kernel(X, Y, index, dot_tilelang)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"dot_tilelang time: {(end - start):.4f} ms")

print(dot_baseline)
print(dot_tilelang)

