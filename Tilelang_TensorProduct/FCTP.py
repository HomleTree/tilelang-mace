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
torch.manual_seed(42)

# @tilelang.jit
# def forward(B, U, V, W,
#                    dtype="float64", 
#                    accum_dtype="float64",
#                    threads=96):

#     @T.prim_func
#     def main(
#             X: T.Tensor((B, U), dtype), # type: ignore
#             Y: T.Tensor((B, V), dtype), # type: ignore
#             W_tensor: T.Tensor((U, V, W), dtype), # type: ignore
#             Z: T.Tensor((B, W), dtype), # type: ignore
#     ):
#         with T.Kernel(W, B, threads=1) as (w_idx, b_idx):
#             # 使用更高精度的累加器

#             Z_accum = T.alloc_fragment((1), accum_dtype)
#             T.clear(Z_accum)
            
#             # 在累加前转换为高精度
#             for u, v in T.Parallel(U, V):
#                 x_val = T.cast(X[b_idx, u], dtype)
#                 y_val = T.cast(Y[b_idx, v], dtype)
#                 w_val = T.cast(W_tensor[u, v, w_idx], dtype)
#                 Z_accum[0] += x_val * y_val * w_val
            
#             # 最后转换回输出精度
#             Z[b_idx, w_idx] = T.cast(Z_accum[0], dtype)
#             # T.atomic_add(Z[b_idx, w_idx], Z_accum[0])

#     return main

@tilelang.jit
def forward(B, U, V, W, val,
                   dtype="float32", 
                   accum_dtype="float32",
                   threads=96):

    @T.prim_func
    def main(
            X: T.Tensor((B, U), dtype), # type: ignore
            Y: T.Tensor((B, V), dtype), # type: ignore
            W_tensor: T.Tensor((U, V, W), dtype), # type: ignore
            Z: T.Tensor((B, W), dtype), # type: ignore
    ):
        with T.Kernel(W, B, threads=1) as (w_idx, b_idx):
            # 使用更高精度的累加器

            Z_accum = T.alloc_fragment((1), accum_dtype)
            T.clear(Z_accum)
            
            # 在累加前转换为高精度
            for u, v in T.Parallel(U, V):
                x_val = T.cast(X[b_idx, u], dtype)
                y_val = T.cast(Y[b_idx, v], dtype)
                w_val = T.cast(W_tensor[u, v, w_idx], dtype)
                Z_accum[0] += x_val * y_val * w_val * val
            
            # 最后转换回输出精度
            Z[b_idx, w_idx] = T.cast(Z_accum[0], dtype)
            # T.atomic_add(Z[b_idx, w_idx], Z_accum[0])

    return main

B = 368
U = W = 96
V = 10
val = 0.03227486121839514

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

block_B = block_W = 96

irreps_in1 = o3.Irreps("96x0e")
irreps_in2 = o3.Irreps("10x0e")
irreps_out = o3.Irreps("96x0e")

tp = cueq.FullyConnectedTensorProduct(
    irreps_in1, irreps_in2, 
    irreps_out, 
    layout=cue.mul_ir,
    use_fallback=True,
    shared_weights=False,     
    internal_weights=False,
    device=device     
)

instructions=[(0, 0, 0, "uvw", True)]
tp_e3nn = o3.TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions,
        shared_weights=False, internal_weights=False).to('cuda')

X = torch.randn(B, irreps_in1.dim, device=device, requires_grad=True)
Y = torch.randn(B, irreps_in2.dim, device=device, requires_grad=True)
W_tensor = torch.randn(1, tp.weight_numel, device=device, requires_grad=True)
Z = torch.randn(B, W, device=device, requires_grad=True)

# X = torch.randn(B, irreps_in1.dim, dtype=torch.float64, device=device, requires_grad=True)
# Y = torch.randn(B, irreps_in2.dim, dtype=torch.float64, device=device, requires_grad=True)
# W_tensor = torch.randn(1, tp.weight_numel, dtype=torch.float64, device=device, requires_grad=True)
# Z = torch.randn(B, W, device=device, dtype=torch.float64, requires_grad=True)

#### data ####
print(irreps_in1.dim)
print(irreps_in2.dim)
print(tp.weight_numel)

#### e3nn ####
retry = 100
for retry_id in range(0, retry):
    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    out_e3nn = tp_e3nn(X, Y, W_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter() * 1000
    
    if retry_id == retry - 1:
        print(f"e3nn_forward_cost: {(end-start):.4f} ms")

# print(Z.shape)
# print(Z)

#### cuequivarance ####
for retry_id in range(0, retry):
    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    out_cueq = tp(X, Y, W_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter() * 1000
    
    if retry_id == retry - 1:
        print(f"cueq_forward_cost: {(end-start):.4f} ms")

# print(out.shape)
# print(out)

#### torch.einsum ####
W_reshaped = W_tensor.reshape(U, V, W)
for retry_id in range(0, retry):
    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    output = torch.einsum("bu,bv,uvw->bw", X, Y, W_reshaped)
    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"einsum_forward_time: {(end - start):.4f} ms") 
output *= val
print(output)

#### tilelang ####
# X = torch.randn(B, irreps_in1.dim, dtype=torch.float64, device=device, requires_grad=True)
# Y = torch.randn(B, irreps_in2.dim, dtype=torch.float64, device=device, requires_grad=True)
# W_tensor = torch.randn(1, tp.weight_numel, dtype=torch.float64, device=device, requires_grad=True)
# Z = torch.randn(B, W, device=device, dtype=torch.float64, requires_grad=True)

# W_reshaped = W_tensor.reshape(U, V, W)
kernel = forward(B, U, V, W, val)
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    kernel(X, Y, W_reshaped, Z)
    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"tilelang time: {(end - start):.4f} ms") 

print(Z)

err = (out_cueq - Z).abs().max().item()
print(f"error: {err:.5g}")

checks = {
    "out": torch.allclose(out_cueq, Z, rtol=1e-02, atol=1e-02, equal_nan=False)
}
print(f"forward: {checks}")

# print("error:", (out - Z).abs().max().item())
# print(Z)
# print(out)

