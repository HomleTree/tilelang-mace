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



# @tilelang.jit
# def forward1(B, U, V, W,
#             dtype="float32", 
#             accum_dtype="float32",
#             threads=128):
    
#     # tile
#     block_B = min(B, 32)
#     block_W = min(W, 32)
#     block_U = min(U, 32)
#     block_V = min(V, 32)
#     val = T.cast(0.03227486121839514, dtype)

#     @T.prim_func
#     def main(
#             X: T.Tensor((B, U), dtype), # type: ignore
#             Y: T.Tensor((B, V), dtype), # type: ignore
#             W_tensor: T.Tensor((U, V, W), dtype), # type: ignore
#             Z: T.Tensor((B, W), dtype), # type: ignore
#     ):
#         with T.Kernel(B, T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):
            
#             X_shared = T.alloc_shared((block_U,), dtype)
#             Y_shared = T.alloc_shared((V,), dtype)
#             W_shared = T.alloc_shared((block_U, V, block_W), dtype)

#             C_local = T.alloc_fragment((block_W,), accum_dtype)
#             T.clear(C_local)

#             for uo in T.Pipelined(T.ceildiv(U, block_U), num_stages=2):
#                 T.copy(X[b_idx, uo * block_U: (uo + 1) * block_U], X_shared)
#                 T.copy(Y[b_idx, :], Y_shared)

#                 for u, v, w in T.Parallel(block_U, V, block_W):
#                     C_local[w] += X_shared[u] * Y[b_idx, v] * W_tensor[u, v, w]

#             for w in T.Parallel(block_W):
#                 w_global = w_idx * block_W + w
#                 if w_global < W:
#                     Z[b_idx, w_global] = C_local[w]
            
#     return main

@tilelang.jit
def forward(B, U, V, W,
            dtype="float32", 
            accum_dtype="float32",
            threads=128):
    
    # tile
    block_B = block_W = block_UV = 32
    val = T.cast(0.03227486121839514, dtype)

    @T.prim_func
    def main(
            XY: T.Tensor((B, U*V), dtype), # type: ignore
            W_tensor: T.Tensor((U*V, W), dtype), # type: ignore
            Z: T.Tensor((B, W), dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):
            
            A_shared = T.alloc_shared((block_B, block_UV), dtype)
            B_shared = T.alloc_shared((block_UV, block_W), dtype)

            C_local = T.alloc_fragment((block_B, block_W), accum_dtype)
            T.clear(C_local)

            # data_shared
            for ko in T.Pipelined(T.ceildiv(U*V, block_UV), num_stages=3):
                T.copy(XY[b_idx * block_B, ko * block_UV], A_shared)
                T.copy(W_tensor[ko * block_UV, w_idx * block_W], B_shared)
                
                # gemm
                for i, j, k in T.Parallel(block_B, block_W, block_UV):
                    C_local[i, j] += A_shared[i, k] * B_shared[k, j]

            for i, j in T.Parallel(block_B, block_W):
                Z[b_idx * block_B + i, w_idx * block_W + j] = C_local[i, j] * val

    return main

### example ###
"FullyConnectedTensorProduct(96x0e x 10x0e -> 96x0e | 92160 paths | 92160 weights)"
### data ###
B = 5152
U = W = 96
V = 10
val = 0.03227486121839514

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

gen = torch.Generator(device=device)
gen.manual_seed(42)

irreps_in1 = o3.Irreps("96x0e")
irreps_in2 = o3.Irreps("10x0e")
irreps_out = o3.Irreps("96x0e")

tp_e3nn = o3.FullyConnectedTensorProduct(
    irreps_in1, irreps_in2, 
    irreps_out, 
    shared_weights=False,     
    internal_weights=False
)

X = torch.randn(B, irreps_in1.dim, dtype=dtype, device=device, requires_grad=True)
Y = torch.randn(B, irreps_in2.dim, dtype=dtype, device=device, requires_grad=True)
W_tensor = torch.randn(1, tp_e3nn.weight_numel, dtype=dtype, device=device, requires_grad=True)
Z = torch.zeros(B, W, device=device, dtype=dtype)

print(f"irreps_in1: {irreps_in1.dim}, irreps_in2: {irreps_in2.dim}, irreps_out: {irreps_out.dim}, weight: {tp_e3nn.weight_numel}")

# #### tilelang ####
# dtype1 = torch.float32
# X = torch.randn(B, irreps_in1.dim, dtype=dtype, device=device, requires_grad=True)
# Y = torch.randn(B, irreps_in2.dim, dtype=dtype, device=device, requires_grad=True)
# W_tensor = torch.randn(1, tp_e3nn.weight_numel, dtype=dtype, device=device, requires_grad=True)
# Z = torch.zeros(B, W, device=device, dtype=dtype)

# kernel1= forward1(B, U, V, W)
# W_reshaped = W_tensor.reshape(U, V, W)
# # torch.cuda.synchronize()
# # kernel1(X, Y, W_reshaped.contiguous(), Z)
# # torch.cuda.synchronize()

# # print(Z.shape)
# # print(Z)
# retry = 100
# for retry_id in range(0, retry):
#     torch.cuda.synchronize(device=device)
#     start = time.perf_counter() * 1000
#     kernel1(X, Y, W_reshaped.contiguous(), Z)
#     torch.cuda.synchronize(device=device)
#     end = time.perf_counter() * 1000

#     if retry_id == retry - 1:
#         print(f"tilelang time: {(end - start):.4f} ms")

# # outer = outer.reshape(B, -1)
# # # print(outer.shape)

# # # W_reshaped = W_tensor.reshape(U*V, W)
# # W_reshaped = W_tensor.reshape(U*V, W)
# # kernel = forward(B, U, V, W)

# # # outer = outer.contiguous()
# # W_tensor = W_tensor.contiguous()
# # Z = Z.contiguous()

# # retry = 100
# # for retry_id in range(0, retry):
# #     torch.cuda.synchronize(device=device)
# #     start = time.perf_counter() * 1000
# #     # outer = X.unsqueeze(-1) * Y.unsqueeze(-2)
# #     # outer = outer.reshape(B, -1)
# #     kernel(outer, W_reshaped, Z)
# #     torch.cuda.synchronize(device=device)
# #     end = time.perf_counter() * 1000

# #     if retry_id == retry - 1:
# #         print(f"tilelang time: {(end - start):.4f} ms") 

# # print(Z)

# out_e3nn = tp_e3nn(X, Y, W_tensor)
# print(out_e3nn)

# tp_cueq = cueq.FullyConnectedTensorProduct(
#     irreps_in1, irreps_in2, 
#     irreps_out, 
#     layout=cue.mul_ir,
#     use_fallback=True,
#     shared_weights=False,     
#     internal_weights=False,
#     device=device     
# )

# for retry_id in range(0, retry):
#     torch.cuda.synchronize(device=device)
#     start = time.perf_counter() * 1000
#     out_cueq = tp_cueq(X, Y, W_tensor)
#     torch.cuda.synchronize(device=device)
#     end = time.perf_counter() * 1000

#     if retry_id == retry - 1:
#         print(f"cueq time: {(end - start):.4f} ms")

# print(out_cueq)

# # # print(f"out: {torch.allclose(Z, out_e3nn, rtol=1e-02, atol=1e-02, equal_nan=False)}")

# # err = (out_e3nn - Z).abs()
# # print("max err: ", err.max().item())
# # # print("95% quantile: ", err.quantile(0.95).item())
# # # print("99% quantile: ", err.quantile(0.99).item())

## manual_seed ##
def make_X(B, dim, dtype, device, requires_grad):
    gen = torch.Generator(device=device)
    gen.manual_seed(42)

    X = torch.randn(B, dim,
                    dtype=dtype,
                    device=device,
                    requires_grad=requires_grad,
                    generator=gen)
    return X

### tilelang ###
def tilelang_forward_fctp(
    B,
    irreps_in1,
    irreps_in2,
    irreps_out,
    dtype,
):
    irreps_in1 = cue.Irreps("O3", irreps_in1)
    irreps_in2 = cue.Irreps("O3", irreps_in2)
    irreps_out = cue.Irreps("O3", irreps_out)

    tp_cueq = cueq.FullyConnectedTensorProduct(
        irreps_in1, irreps_in2, 
        irreps_out, 
        layout=cue.mul_ir,
        use_fallback=True,
        shared_weights=False,     
        internal_weights=False,
        device=device     
    )

    X = make_X(B, irreps_in1.dim, dtype=dtype, device=device, requires_grad=True)
    Y = make_X(B, irreps_in2.dim, dtype=dtype, device=device, requires_grad=True)
    W_tensor = make_X(1, tp_cueq.weight_numel, dtype=dtype, device=device, requires_grad=True)
    Z = torch.zeros(B, W, device=device, dtype=dtype)

    kernel = forward(B, U, V, W)
    W_reshaped = W_tensor.reshape(U * V, W).contiguous() 
    retry = 100
    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        XY = X.unsqueeze(-1) * Y.unsqueeze(-2)
        XY = XY.reshape(B, -1).contiguous() 
        kernel(XY, W_reshaped, Z)
        
        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        t = end - start
        if retry_id == retry - 1:
            print(f"tilelang_forward_cost: {t:.4f} ms")
    
    print(f"out_tilelang shapes: {tuple(Z.shape)}")
    return Z

### cueq ###
def cueq_forward_fctp(
    B,
    irreps_in1,
    irreps_in2,
    irreps_out,
    dtype,
):  
    irreps_in1 = cue.Irreps("O3", irreps_in1)
    irreps_in2 = cue.Irreps("O3", irreps_in2)
    irreps_out = cue.Irreps("O3", irreps_out)

    tp_cueq = cueq.FullyConnectedTensorProduct(
        irreps_in1, irreps_in2, 
        irreps_out, 
        layout=cue.mul_ir,
        use_fallback=True,
        shared_weights=False,     
        internal_weights=False,
        device=device     
    )

    X = make_X(B, irreps_in1.dim, dtype=dtype, device=device, requires_grad=True)
    Y = make_X(B, irreps_in2.dim, dtype=dtype, device=device, requires_grad=True)
    W = make_X(1, tp_cueq.weight_numel, dtype=dtype, device=device, requires_grad=True)

    retry = 100
    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        out_cueq = tp_cueq(X, Y, W)
        
        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        t = end - start
        if retry_id == retry - 1:
            print(f"cueq_forward_cost: {t:.4f} ms")
    
    print(f"out_cueq shapes: {tuple(out_cueq.shape)}")
    return out_cueq

### e3nn ###
def e3nn_forward_fctp(
    B,
    irreps_in1,
    irreps_in2,
    irreps_out,
    dtype,
):  
    irreps_in1 = o3.Irreps(irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)
    irreps_out = o3.Irreps(irreps_out)

    tp_e3nn = o3.FullyConnectedTensorProduct(
        irreps_in1, irreps_in2, 
        irreps_out,
        shared_weights=False, 
        internal_weights=False
    )

    X = make_X(B, irreps_in1.dim, dtype=dtype, device=device, requires_grad=True)
    Y = make_X(B, irreps_in2.dim, dtype=dtype, device=device, requires_grad=True)
    W = make_X(1, tp_e3nn.weight_numel, dtype=dtype, device=device, requires_grad=True)

    retry = 100
    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        out_e3nn = tp_e3nn(X, Y, W)
        
        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        t = end - start
        if retry_id == retry - 1:
            print(f"e3nn_forward_cost: {t:.4f} ms")
    
    print(f"out_e3nn shapes: {tuple(out_e3nn.shape)}")
    return out_e3nn

### einsum ###
def einsum_forward_fctp(
    B,
    irreps_in1,
    irreps_in2,
    irreps_out,
    dtype,
):  
    irreps_in1 = cue.Irreps("O3", irreps_in1)
    irreps_in2 = cue.Irreps("O3", irreps_in2)
    irreps_out = cue.Irreps("O3", irreps_out)

    tp = cueq.FullyConnectedTensorProduct(
        irreps_in1, irreps_in2, 
        irreps_out, 
        layout=cue.mul_ir,
        use_fallback=True,
        shared_weights=False,     
        internal_weights=False,
        device=device     
    )

    val = 0.03227486121839514
    X = make_X(B, irreps_in1.dim, dtype=dtype, device=device, requires_grad=True)
    Y = make_X(B, irreps_in2.dim, dtype=dtype, device=device, requires_grad=True)
    W = make_X(1, tp.weight_numel, dtype=dtype, device=device, requires_grad=True)
    W_reshaped = W.reshape(U, V, irreps_out.dim)

    retry = 100
    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        out_einsum = torch.einsum("bu,bv,uvw->bw", X, Y, W_reshaped)
        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        if retry_id == retry - 1:
            print(f"einsum_forward_time: {(end - start):.4f} ms") 

    out_einsum *= val
    print(f"out_einsum shapes: {tuple(out_einsum.shape)}")
    return out_einsum

#### main ####
def main():
    out_tilelang = tilelang_forward_fctp(B, "96x0e", "10x0e", "96x0e", dtype)
    out_cueq     = cueq_forward_fctp(B, "96x0e", "10x0e", "96x0e", dtype)
    out_e3nn     = e3nn_forward_fctp(B, "96x0e", "10x0e", "96x0e", dtype)
    out_einsum   = einsum_forward_fctp(B, "96x0e", "10x0e", "96x0e", dtype)

    err_tilelang = (out_e3nn - out_tilelang).abs()
    err_cueq = (out_e3nn - out_cueq).abs()
    err_einsum = (out_e3nn - out_einsum).abs()
    print(f"err_tilelang: {err_tilelang.max().item()}\n"
          f"err_cueq:     {err_cueq.max().item()}\n"
          f"err_einsum:   {err_einsum.max().item()}")

if __name__ == "__main__":
    main()

