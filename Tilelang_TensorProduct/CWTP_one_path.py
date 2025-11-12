import cuequivariance as cue
import cuequivariance_torch as cueq
import e3nn.o3 as o3

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout

import math 
import time
import torch
torch.manual_seed(42)

@tilelang.jit
def forward(B, U, dim, block_B, dtype="float64", accum_dtype="float16"):
    
    @T.prim_func
    def main(
        X: T.Tensor((B, U), dtype), # type: ignore
        Y: T.Tensor((B, dim), dtype), # type: ignore
        W: T.Tensor((U), dtype), # type: ignore
        Z: T.Tensor((B, U, dim), dtype), # type: ignore
    ):
        with T.Kernel(B, threads=U*dim) as b:
            tid = T.thread_binding(0, U*dim, thread="threadIdx.x")

            u = tid // dim
            d = tid % dim

            Z[b, u, d] = X[b, u] * Y[b, d] * W[u]  
    return main

@tilelang.jit
def backward(B, U, dim, block_B, threads, dtype="float64", accum_dtype="float64"):
    
    total = U * dim

    @T.prim_func
    def main(
        X: T.Tensor((B, U), dtype), # type: ignore
        Y: T.Tensor((B, dim), dtype), # type: ignore
        W: T.Tensor((U), dtype), # type: ignore
        grad_out: T.Tensor((B, U, dim), dtype), # type: ignore
        grad_X: T.Tensor((B, U), dtype), # type: ignore
        grad_Y: T.Tensor((B, dim), dtype), # type: ignore
        grad_W: T.Tensor((U), dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(total, threads), threads=threads) as (b_idx, t_idx):
            t = T.thread_binding(0, threads, thread="threadIdx.x")
            tid = t_idx * threads + t

            grad_W_shared = T.alloc_shared((U), accum_dtype)

            for i in T.serial(T.ceildiv(U, threads)):
                idx = i * threads + t
                if idx < U:
                    grad_W_shared[idx] = 0

            T.sync_threads()

            for bi in T.serial(block_B):
                b = b_idx * block_B + bi

                if tid < total:

                    tmp = tid
                    u = tmp // dim
                    i = tmp % dim

                    xv = T.cast(X[b, u], dtype)
                    yv = T.cast(Y[b, i], dtype)
                    wv = T.cast(W[u], dtype)
                    g = T.cast(grad_out[b, u, i], dtype)

                    T.atomic_add(grad_X[b, u], g * yv * wv)
                    T.atomic_add(grad_Y[b, i], g * xv * wv)
                    T.atomic_add(grad_W_shared[u], g * xv * yv)

            T.sync_threads()
            
            for i in T.serial(T.ceildiv(U, threads)):
                idx = i * threads + t
                if idx < U:
                    if grad_W_shared[idx] != 0:
                        T.atomic_add(grad_W[idx], grad_W_shared[idx])
    return main

# @tilelang.jit
# def backward(B, U, dim, block_B, threads, dtype="float64", accum_dtype="float64"):
    
#     total = U * dim

#     @T.prim_func
#     def main(
#         X: T.Tensor((B, U), dtype), # type: ignore
#         Y: T.Tensor((B, dim), dtype), # type: ignore
#         W: T.Tensor((U), dtype), # type: ignore
#         grad_out: T.Tensor((B, U, dim), dtype), # type: ignore
#         grad_X: T.Tensor((B, U), dtype), # type: ignore
#         grad_Y: T.Tensor((B, dim), dtype), # type: ignore
#         grad_W: T.Tensor((U), dtype), # type: ignore
#     ):
#         with T.Kernel(B, T.ceildiv(total, threads), threads=512) as (b_idx, t_idx):
#             # T.ceildiv(B, block_B)
#             t = T.thread_binding(0, threads, thread="threadIdx.x")
#             tid = t_idx * threads + t

#             grad_X_shared = T.alloc_shared((U), dtype)
#             grad_W_shared = T.alloc_shared((U), dtype)
            
#             for i in T.serial(T.ceildiv(U, threads)):
#                 idx = i * threads + t
#                 if idx < U:
#                     grad_W_shared[idx] = 0
#                     grad_X_shared[idx] = 0
                
#             T.sync_threads()

#             if tid < total:

#                 tmp = tid
#                 u = tmp // dim
#                 i = tmp % dim

#                 xv = X[b_idx, u]
#                 yv = Y[b_idx, i]
#                 wv = W[u]
#                 g = grad_out[b_idx, u, i]

#                 # T.atomic_add(grad_X[global_b, u], g * yv * wv)
#                 T.atomic_add(grad_X_shared[u], g * yv * wv)
#                 T.atomic_add(grad_Y[b_idx, i], g * xv * wv)
#                 T.atomic_add(grad_W_shared[u], g * xv * yv)

#             T.sync_threads()
            
#             for i in T.serial(T.ceildiv(2*U, threads)):
#                 idx = i * threads + t
#                 if idx < 2 * U:
#                     if idx < U:
#                         if grad_W_shared[idx] != 0:
#                             T.atomic_add(grad_W[idx], grad_W_shared[idx])
                    
#                         T.atomic_add(grad_X[b_idx, idx], grad_X_shared[idx])
#                 # elif idx < U + U:
#                 #     for bi in T.serial(block_B):
#                 #         global_b = b_idx * block_B + bi
#                 #         T.atomic_add(grad_X[global_b, idx], grad_X_shared[idx])

#     return main

B = 5152
U = W = 96
V = 1

dim_list = [7]
dim_sum = sum(dim_list)  # 16
path_num = len(dim_list) # 4

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

##### Forward #####

#### cuequivarance ####
irreps_in1 = cue.Irreps("O3", "96x0e")
irreps_in2 = cue.Irreps("O3", "1x0e")
irreps_out = cue.Irreps("O3", "96x0e")

tp = cueq.ChannelWiseTensorProduct(
    irreps_in1, irreps_in2, 
    filter_irreps_out = irreps_out, 
    layout=cue.mul_ir,
    shared_weights=False,     
    internal_weights=False,
    device=device     
)

x_all = torch.randn(B, U, dtype=torch.float64, device=device, requires_grad=True)  # 96×1=96
y_all = torch.randn(B, dim_sum * V, dtype=torch.float64, device=device, requires_grad=True)    # 1×16=16
w_all = torch.randn(1, tp.weight_numel, dtype=torch.float64, device=device, requires_grad=True)
Z = torch.zeros(B, W, dim_sum, dtype=torch.float64, device=device, requires_grad=True)    

#### tilelang ####
### Forward ###
retry = 100

block_B = 32
w_all1 = w_all.reshape(tp.weight_numel)
kernel_forward = forward(B, U, dim_sum, block_B)
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() *1000

    kernel_forward(x_all, y_all, w_all1, Z)
    torch.cuda.synchronize(device=device)
    end = time.perf_counter() *1000
    t = (end - start)
    if (retry_id == retry - 1):
        print(f"tile_forward_cost: {t:.4f} ms")
Z = Z.reshape(B, -1)
# print(Z)
# print(Z.shape)

### Backward ###
kernel_backward = backward(B, U, dim_sum, block_B, 96)
grad_Z = torch.ones(B, U, dim_sum, dtype=torch.float64, device=device)

for retry_id in range(0, retry):
    grad_X = torch.zeros(B, U, dtype=torch.float64, device=device)
    grad_Y = torch.zeros(B, dim_sum * V, dtype=torch.float64, device=device)
    grad_W = torch.zeros(tp.weight_numel, dtype=torch.float64, device=device)

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    kernel_backward(x_all, y_all, w_all1, grad_Z, grad_X, grad_Y, grad_W)
    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    t = (end - start)
    if (retry_id == retry - 1):
        print(f"tile_backward_cost: {t:.4f} ms")

# print(grad_X)
# print(grad_Y)
# print(grad_W)

#### cuequivarance ####
### Forward ###
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() *1000

    out = tp(x_all, y_all, w_all)
    torch.cuda.synchronize(device=device)
    end = time.perf_counter() *1000
    t = (end - start)
    if retry_id == retry - 1:
        print(f"cueq_forward_cost: {t:.4f} ms")

# print(out.shape)

### Backward ###
grad_out = torch.ones(B, U * dim_sum, dtype=torch.float64, device=device)
for retry_id in range(0, retry):
    if retry_id != 0:
        x_all.grad.zero_(), y_all.grad.zero_(), w_all.grad.zero_()
    
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    out.backward(grad_out, retain_graph=True)
    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    t = end - start
    if retry_id == retry - 1:
        print(f"cueq_backward_cost: {t:.4f} ms")

# print(w_all.grad)

#### torch.einsum ####
### Forward ###
for retry_id in range(0, retry):
    total_forward_time = 0
    dim_offset = 0
    outs = []
    
    for dim_idx in range(0, path_num):
        dim = dim_list[dim_idx]

        x = x_all
        y = y_all[:, dim_offset:dim_offset + dim]
        w = w_all[:, dim_idx * W:(dim_idx + 1) * W]
        w = w.reshape(U)

        torch.cuda.synchronize(device=device)
        start = time.perf_counter() * 1000

        ## torch.einsum的手动实现 ##
        # z = torch.empty(B, U, dim, dtype=x.dtype, device=x.device)
        # for b in range(B):
        #     for u in range(U):
        #         for i in range(dim):
        #             z[b, u, i] = x[b, u] * y[b, i] * w[u]
        z = torch.einsum("bu,bi->bui", x, y)
        z *= w.view(1, W, 1)
        
        torch.cuda.synchronize(device=device)
        end = time.perf_counter() * 1000
        if retry_id == retry - 1:
            total_forward_time += end - start
        z = z.reshape(B, -1)
        outs.append(z)
        # print(z.shape)
        dim_offset += dim

    output = torch.cat(outs, dim=1)
print(f"einsum_forward_time: {total_forward_time:.4f} ms")

### Backward ###
for retry_id in range(0, retry):
    total_backward_time = 0
    dim_offset = 0
    grad_z = torch.ones(B, U, dim_sum, dtype=torch.float64, device=device)
    
    for dim_idx in range(0, path_num):
        dim = dim_list[dim_idx]

        x = x_all
        y = y_all[:, dim_offset:dim_offset + dim]

        w = w_all[:, dim_idx * W:(dim_idx + 1) * W]
        w = w.reshape(U)

        torch.cuda.synchronize(device=device)
        start = time.perf_counter() * 1000

        grad_x = torch.einsum("bui,bi,u->bu", grad_z, y, w)
        grad_y = torch.einsum("bui,bu,u->bi", grad_z, x, w)
        grad_w = torch.einsum("bui,bu,bi->u", grad_z, x, y)

        torch.cuda.synchronize(device=device)
        end = time.perf_counter() * 1000
        if retry_id == retry - 1:
            total_backward_time += end - start
        
        dim_offset += dim
print(f"einsum_backward_time: {total_backward_time:.4f} ms")

checks = {
    "out": torch.allclose(out, output, rtol=1e-05, atol=1e-05, equal_nan=False)
}
print(f"forward: {checks}")
grad_checks = {
    "grad_x": torch.allclose(grad_X, grad_x, rtol=1e-05, atol=1e-05, equal_nan=False),
    "grad_y": torch.allclose(grad_Y, grad_y, rtol=1e-05, atol=1e-05, equal_nan=False),
    "grad_w": torch.allclose(grad_W, grad_w, rtol=1e-05, atol=1e-05, equal_nan=False)
}
print(f"backward: {grad_checks}")
# err = (out - Z1).abs().max().item()
# print(f"error: {err:.5g}")
# print(out)

# print(x_all.grad)
# print(y_all.grad)
# print(w_all.grad)


