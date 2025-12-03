import itertools
import torch
import os, time
import ast
from torch.utils.cpp_extension import load

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout
from tvm import DataType

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
        print(f"gemv_baseline_time: {(end - start):.4f} ms")

## tilelang ##
@tilelang.jit(out_idx=[-1])
def gemv(
    N, K,
    BLOCK_N,
    reduce_threads,
    dtype="float64",
    accum_dtype="float64",
):

    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
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
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)

            # for bk, k in T.Parallel(T.ceildiv(K, BLOCK_K), TILE_K):
            #     # for k in T.vectorized(TILE_K):
            #     A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
            #     B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
            #     # for k in T.serial(TILE_K):
            #     C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
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

kernel = gemv(B, B, 2, 32)
for retry_id in range(0, retry):
    gemv_tilelang = torch.zeros_like(gemv_baseline)

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    
    gemv_tilelang = kernel(Y, X)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"gemv_tilelang_time: {(end - start):.4f} ms")

gemv_tilelang_err = (gemv_baseline - gemv_tilelang).abs()
print(f"gemv_tilelang_error: {gemv_tilelang_err.max().item()}")


### dot ###
print(f"\n*** dot ***")
from torch_scatter import scatter

B = 3675
X = torch.randn(B, dtype=dtype, device=device)
Y = torch.randn(B, dtype=dtype, device=device)

batch_size = 25
batch_indices = torch.arange(batch_size, dtype=torch.int64, device=device).repeat_interleave(49)
index = batch_indices.repeat_interleave(3)  # shape:[1225 * 3]
dim_size = batch_size

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
    dot_torch = torch.zeros(dim_size, dtype=dot_baseline.dtype, device=dot_baseline.device)
    
    out = X * Y
    dot_torch.scatter_add_(0, index, out)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"dot_torch_time: {(end - start):.4f} ms")

dot_torch_err = (dot_baseline - dot_torch).abs()
print(f"dot_torch_error: {dot_torch_err.max().item()}")

## tilelang ##
@tilelang.jit#(out_idx=[-1])
def dot(B, dim_size, 
        dtype="float64", 
        accum_dtype="float64"):
    
    # tile
    block_B = 64

    @T.prim_func
    def main(
            X: T.Tensor((B,), "float64"), # type: ignore
            Y: T.Tensor((B,), "float64"), # type: ignore
            # index: T.Tensor((B), "int64"), # type: ignore
            Z: T.Tensor((dim_size), "float64"), # type: ignore
    ):
        # T.func_attr({"tir.noalias": True})
        # T.func_attr({"tl.copy_vectorize": 4}) 

        with T.Kernel(T.ceildiv(B, block_B), threads=block_B) as (b_idx):
            t = T.thread_binding(0, block_B, "threadIdx.x")
            tid = b_idx * block_B + t

            if tid < B:
                i = tid // (B // dim_size)
                T.atomic_add(Z[i], X[tid] * Y[tid])

    return main

kernel = dot(B, dim_size)
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    dot_tilelang = torch.zeros(dim_size, dtype=dtype, device=device)

    # dot_tilelang = kernel(X, Y)
    kernel(X, Y, dot_tilelang) 

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"dot_tilelang_time: {(end - start):.4f} ms")

dot_tilelang_err = (dot_baseline - dot_tilelang)
print(f"dot_tilelang_error: {dot_tilelang_err.max().item()}")


### outer ###
print(f"\n*** outer ***")
B = 147
X = torch.randn(B, dtype=dtype, device=device)
Y = torch.randn(B, dtype=dtype, device=device)

## baseline ##
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    outer_baseline = torch.outer(X, Y)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"outer_baseline_time: {(end - start):.4f} ms")

## torch ##
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    outer_torch = X.unsqueeze(1) * Y.unsqueeze(0)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"outer_torch_time: {(end - start):.4f} ms")

outer_torch_err = (outer_baseline - outer_torch).abs()
print(f"outer_torch_error: {outer_torch_err.max().item()}")

## tilelang ##

