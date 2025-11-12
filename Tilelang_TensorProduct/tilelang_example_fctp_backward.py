import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout

import cuequivariance as cue
import cuequivariance_torch as cueq
import e3nn.o3 as o3

import math
import time 
import torch
torch.manual_seed(42)
import numpy as np
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.jit.*"
)

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

# @tilelang.jit
# def backward(B, U, V, W,
#             dtype="float32", 
#             accum_dtype="float32",
#             threads=32):
    
#     # tile
#     block_B = 96
#     block_U = block_W = 32
#     val = T.cast(0.03227486121839514, dtype)

#     @T.prim_func
#     def main(
#             grad_out: T.Tensor((B, W), "float32"), # type: ignore
#             X: T.Tensor((B, U), "float32"), # type: ignore
#             Y: T.Tensor((B, V), "float32"), # type: ignore
#             W_tensor: T.Tensor((U, V, W), "float32"), # type: ignore
#             grad_X: T.Tensor((B, U), "float32"), # type: ignore
#             grad_Y: T.Tensor((B, V), "float32"), # type: ignore
#             grad_W: T.Tensor((U, V, W), "float32"), # type: ignore
#     ):
#         with T.Kernel(T.ceildiv(B, block_B), threads=threads) as (b_idx):

#             X_shared = T.alloc_shared((block_B, block_W), dtype)
#             grad_out_shared = T.alloc_shared((block_B, block_W), dtype)

#             for ko in T.Pipelined(T.ceildiv(W, block_W), num_stages=3):
#                 T.copy(X[b_idx * block_B, ko * block_W], X_shared)
#                 T.copy(grad_out[b_idx * block_B, ko * block_W], grad_out_shared)
                
#                 for b, u, v, w in T.Parallel(block_B, block_U, V, block_W):
#                     global_b = b_idx * block_B + b
#                     T.atomic_add(grad_X[global_b, u], grad_out_shared[b, w] * Y[global_b, v] * W_tensor[u, v, w] * val)
#                     T.atomic_add(grad_Y[global_b, v], grad_out_shared[b, w] * X_shared[b, u] * W_tensor[u, v, w] * val)
#                     T.atomic_add(grad_W[u, v, w],  grad_out_shared[b, w] * X_shared[b, u] * Y[global_b, v] * val)

#     return main

@tilelang.jit
def backward(B, U, V, W,
            dtype="float32", 
            accum_dtype="float32",
            threads=32):
    
    # tile
    block_B = 96
    block_U = block_W = 32
    val = T.cast(0.03227486121839514, dtype)

    @T.prim_func
    def main(
            grad_out: T.Tensor((B, W), "float32"), # type: ignore
            X: T.Tensor((B, U), "float32"), # type: ignore
            Y: T.Tensor((B, V), "float32"), # type: ignore
            W_tensor: T.Tensor((U, V, W), "float32"), # type: ignore
            grad_X: T.Tensor((B, U), "float32"), # type: ignore
            grad_Y: T.Tensor((B, V), "float32"), # type: ignore
            grad_W: T.Tensor((U, V, W), "float32"), # type: ignore
    ):
        with T.Kernel(B, threads=threads) as (b_idx):

            # X_shared = T.alloc_shared((block_B, block_W), dtype)
            # grad_out_shared = T.alloc_shared((block_B, block_W), dtype)

            # for ko in T.Pipelined(T.ceildiv(W, block_W), num_stages=3):
            #     T.copy(X[b_idx * block_B, ko * block_W], X_shared)
            #     T.copy(grad_out[b_idx * block_B, ko * block_W], grad_out_shared)
                
            for u, v, w in T.Parallel(U, V, W):

                T.atomic_add(grad_X[b_idx, u], grad_out[b_idx, w] * Y[b_idx, v] * W_tensor[u, v, w] * val)
                T.atomic_add(grad_Y[b_idx, v], grad_out[b_idx, w] * X[b_idx, u] * W_tensor[u, v, w] * val)
                T.atomic_add(grad_W[u, v, w],  grad_out[b_idx, w] * X[b_idx, u] * Y[b_idx, v] * val)

    return main

# W_reshaped = W_tensor.reshape(U, V, irreps_out.dim)
# out_einsum = torch.einsum("bu,bv,uvw->bw", X, Y, W_reshaped)
# grad_X = torch.einsum("bw,bv,uvw->bu", grad_out, Y, W_reshaped) * val
# grad_Y = torch.einsum("bw,bu,uvw->bv", grad_out, X, W_reshaped) * val
# grad_W = torch.einsum("bw,bu,bv->uvw", grad_out, X, Y) * val
# grad_W = grad_W.reshape(1, tp_e3nn.weight_numel)

### example ###
"FullyConnectedTensorProduct(96x0e x 10x0e -> 96x0e | 92160 paths | 92160 weights)"
### data ###
B = 368
U = W = 96
V = 10
val = 0.03227486121839514

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

gen = torch.Generator(device=device)
gen.manual_seed(42)

block_B = block_W = 96

irreps_in1 = o3.Irreps("96x0e")
irreps_in2 = o3.Irreps("10x0e")
irreps_out = o3.Irreps("96x0e")

tp_e3nn = o3.FullyConnectedTensorProduct(
    irreps_in1, irreps_in2, 
    irreps_out, 
    shared_weights=False,     
    internal_weights=False
)

#### tilelang ####
dtype1 = torch.float64
X = torch.randn(B, irreps_in1.dim, dtype=dtype, device=device, requires_grad=True)
Y = torch.randn(B, irreps_in2.dim, dtype=dtype, device=device, requires_grad=True)
W_tensor = torch.randn(1, tp_e3nn.weight_numel, dtype=dtype, device=device, requires_grad=True)
Z = torch.zeros(B, W, device=device, dtype=dtype)

W_reshaped = W_tensor.reshape(U, V, W)

retry = 100
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    out_e3nn = tp_e3nn(X, Y, W_tensor)
    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"e3nn_forward_time: {(end - start):.4f} ms")

# print(out_e3nn)

grad_out = torch.ones(B, W, device=device, dtype=dtype)
for retry_id in range(0, retry):
    if retry_id != 0:
        X.grad.zero_(), Y.grad.zero_(), W_tensor.grad.zero_()

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    out_e3nn.backward(grad_out, retain_graph=True)
    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"e3nn_backward_time: {(end - start):.4f} ms")

print(X.grad)
print(Y.grad)
print(W_tensor.grad)

tp_cueq = cueq.FullyConnectedTensorProduct(
    irreps_in1, irreps_in2, 
    irreps_out, 
    layout=cue.mul_ir,
    use_fallback=True,
    shared_weights=False,     
    internal_weights=False,
    device=device     
)

for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    out_cueq = tp_cueq(X, Y, W_tensor)
    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"cueq_forward_time: {(end - start):.4f} ms")

for retry_id in range(0, retry):
    if retry_id != 0:
        X.grad.zero_(), Y.grad.zero_(), W_tensor.grad.zero_()

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    out_cueq.backward(grad_out, retain_graph=True)
    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"cueq_backward_time: {(end - start):.4f} ms")

# print(X.grad)

### bakward ###
W_reshaped = W_tensor.reshape(U, V, irreps_out.dim)
out_einsum = torch.einsum("bu,bv,uvw->bw", X, Y, W_reshaped)
grad_X = torch.einsum("bw,bv,uvw->bu", grad_out, Y, W_reshaped) * val
grad_Y = torch.einsum("bw,bu,uvw->bv", grad_out, X, W_reshaped) * val
grad_W = torch.einsum("bw,bu,bv->uvw", grad_out, X, Y) * val
grad_W = grad_W.reshape(1, tp_e3nn.weight_numel)

# print(grad_X)
# print(grad_Y)
# print(grad_W)

# kernel = backward(B, U, V, W)

grad_X = torch.zeros(B, irreps_in1.dim, dtype=dtype, device=device)
grad_Y = torch.zeros(B, irreps_in2.dim, dtype=dtype, device=device)
grad_W = torch.zeros(tp_e3nn.weight_numel, dtype=dtype, device=device)
grad_W = grad_W.reshape(U, V, irreps_out.dim)

kernel = backward(B, U, V, W)

for retry_id in range(0, retry):
    grad_X = torch.zeros(B, irreps_in1.dim, dtype=dtype, device=device)
    grad_Y = torch.zeros(B, irreps_in2.dim, dtype=dtype, device=device)
    grad_W = torch.zeros(tp_e3nn.weight_numel, dtype=dtype, device=device)
    grad_W = grad_W.reshape(U, V, irreps_out.dim)

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    kernel(grad_out, X, Y, W_reshaped, grad_X, grad_Y, grad_W)
    grad_W = grad_W.reshape(1, tp_e3nn.weight_numel)
    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"tilelang_backward_time: {(end - start):.4f} ms")


# kernel(grad_out, X, Y, W_reshaped, grad_X, grad_Y, grad_W)
# grad_W = grad_W.reshape(1, tp_e3nn.weight_numel)

# print(grad_X)
# print(grad_Y)
# print(grad_W)

print(f"err_X: {(X.grad - grad_X).abs().max().item()}\n"
      f"err_Y: {(Y.grad - grad_Y).abs().max().item()}\n"
      f"err_W: {(W_tensor.grad - grad_W).abs().max().item()}")
# print(out_cueq)

# # print(f"out: {torch.allclose(Z, out_e3nn, rtol=1e-02, atol=1e-02, equal_nan=False)}")

# err = (out_e3nn - Z).abs()
# print("max err: ", err.max().item())
# # print("95% quantile: ", err.quantile(0.95).item())
# # print("99% quantile: ", err.quantile(0.99).item())

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

    grad_out = torch.ones(B, W, device=device, dtype=dtype)

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
def cueq_backward_fctp(
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

    out_cueq = tp_cueq(X, Y, W_tensor)
    grad_out = torch.ones(B, irreps_out.dim, device=device, dtype=dtype)

    retry = 100
    for retry_id in range(0, retry):
        if retry_id != 0:
            X.grad.zero_(), Y.grad.zero_(), W_tensor.grad.zero_()

        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        out_cueq.backward(grad_out, retain_graph=True)
        
        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        t = end - start
        if retry_id == retry - 1:
            print(f"cueq_backward_cost: {t:.4f} ms")
    
    # print(f"out_cueq shapes: {tuple(out_cueq.shape)}")
    return X.grad, Y.grad, W_tensor.grad

### e3nn ###
def e3nn_backward_fctp(
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
    W_tensor = make_X(1, tp_e3nn.weight_numel, dtype=dtype, device=device, requires_grad=True)

    out_e3nn = tp_e3nn(X, Y, W_tensor)
    grad_out = torch.ones(B, irreps_out.dim, device=device, dtype=dtype)

    retry = 100
    for retry_id in range(0, retry):
        if retry_id != 0:
            X.grad.zero_(), Y.grad.zero_(), W_tensor.grad.zero_()

        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        out_e3nn.backward(grad_out, retain_graph=True)
        
        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        t = end - start
        if retry_id == retry - 1:
            print(f"e3nn_backward_cost: {t:.4f} ms")

    # print(W_tensor.grad)
    
    # print(f"out_e3nn shapes: {tuple(out_e3nn.shape)}")
    return X.grad, Y.grad, W_tensor.grad

### einsum ###
def einsum_backward_fctp(
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

    grad_out = torch.ones(B, irreps_out.dim, device=device, dtype=dtype)

    retry = 100
    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        grad_X = torch.einsum("bw,bv,uvw->bu", grad_out, Y, W_reshaped) * val
        grad_Y = torch.einsum("bw,bu,uvw->bv", grad_out, X, W_reshaped) * val
        grad_W = torch.einsum("bw,bu,bv->uvw", grad_out, X, Y) * val
        grad_W = grad_W.reshape(1, tp.weight_numel)

        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        if retry_id == retry - 1:
            print(f"einsum_backward_time: {(end - start):.4f} ms") 

    # print(grad_W)

    # print(f"out_einsum shapes: {tuple(out_einsum.shape)}")
    return grad_X, grad_Y, grad_W

#### main ####
def main():
    # out_tilelang = tilelang_forward_fctp(B, "96x0e", "10x0e", "96x0e", dtype)
    # out_cueq     = cueq_forward_fctp(B, "96x0e", "10x0e", "96x0e", dtype)
    X.grad, Y.grad, W_tensor.grad = e3nn_backward_fctp(B, "96x0e", "10x0e", "96x0e", dtype)
    grad_X, grad_Y, grad_W = einsum_backward_fctp(B, "96x0e", "10x0e", "96x0e", dtype)

    # err_tilelang = (out_e3nn - out_tilelang).abs()
    # err_cueq = (out_e3nn - out_cueq).abs()
    
    # print(f"err_tilelang: {err_tilelang.max().item()}\n"
    #       f"err_cueq:     {err_cueq.max().item()}\n"
    #       f"err_einsum:   {err_einsum.max().item()}")

    print(f"err_X: {(X.grad - grad_X).abs().max().item()}\n"
          f"err_Y: {(Y.grad - grad_Y).abs().max().item()}\n"
          f"err_W: {(W_tensor.grad - grad_W).abs().max().item()}")

if __name__ == "__main__":
    main()

