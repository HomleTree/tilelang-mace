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
# print(torch.backends.cuda.matmul.allow_tf32)

# @tilelang.jit
# def forward(B, U, V, W,
#             dtype="float32", 
#             accum_dtype="float32",
#             threads=128):
    
#     # tile
#     block_B = block_W = block_UV = 32
#     val = T.cast(0.03227486121839514, dtype)

#     @T.prim_func
#     def main(
#             XY: T.Tensor((B, U*V), dtype), # type: ignore
#             W_tensor: T.Tensor((U*V, W), dtype), # type: ignore
#             Z: T.Tensor((B, W), dtype), # type: ignore
#     ):
#         with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):
            
#             A_shared = T.alloc_shared((block_B, block_UV), dtype)
#             B_shared = T.alloc_shared((block_UV, block_W), dtype)

#             C_local = T.alloc_local((block_B, block_W), accum_dtype)
#             T.clear(C_local)

#             # data_shared
#             for ko in T.Pipelined(T.ceildiv(U*V, block_UV), num_stages=3):
#                 T.copy(XY[b_idx * block_B, ko * block_UV], A_shared)
#                 T.copy(W_tensor[ko * block_UV, w_idx * block_W], B_shared)
                
#                 # gemm
#                 for i, j, k in T.Parallel(block_B, block_W, block_UV):
#                     C_local[i, j] += A_shared[i, k] * B_shared[k, j]

#             for i, j in T.Parallel(block_B, block_W):
#                 Z[b_idx * block_B + i, w_idx * block_W + j] = C_local[i, j] * val

#     return main

@tilelang.jit
def forward2(B, U, V, W, dim_sum, path_num,
            dtype="float32", 
            accum_dtype="float32",
            threads=256):
    
    # tile
    block_B = 32
    block_W = 32
    block_UV = 64
    val = T.cast(0.03227486121839514, dtype)

    @T.prim_func
    def main(
            X: T.Tensor((B*dim_sum, U), dtype), # type: ignore
            Y: T.Tensor((B, V), dtype), # type: ignore
            W_tensor: T.Tensor((U*V, W), dtype), # type: ignore
            Z: T.Tensor((B*dim_sum, W), dtype), # type: ignore
            kDims: T.Tensor((4,), "int32"), # type: ignore
            kOff: T.Tensor((4,), "int32"), # type: ignore
    ):
        with T.Kernel(T.ceildiv(B*dim, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):


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

@tilelang.jit
def outerproduct(B, U, V, W, dim, path_num,
            dtype="float32", 
            accum_dtype="float32",
            threads=96):
    
    # tile
    block_B = 32
    # block_W = 32
    block_U = 32
    block_V = 10
    val = T.cast(0.03227486121839514, dtype)

    @T.prim_func
    def main(
            X: T.Tensor((B, U, dim), "float32"), # type: ignore
            Y: T.Tensor((B, V), "float32"), # type: ignore
            XY: T.Tensor((B, U, V, dim), "float32"), # type: ignore
    ):
        with T.Kernel(B, dim, threads=threads) as (b_idx, d_idx):

            for u, v in T.Parallel(U, V):
                XY[b_idx, u, v, d_idx] += X[b_idx, u, d_idx] * Y[b_idx, v]

    return main

@tilelang.jit
def forward(B, U, V, W, dim, 
            dtype="float32", 
            accum_dtype="float32",
            threads=128):
    
    # tile
    block_B = 64
    block_W = 64
    block_UV = 64
    block_U = 32
    val = T.cast(0.03227486121839514, dtype)

    @T.prim_func
    def main(
            X: T.Tensor((B, U, dim), dtype), # type: ignore
            Y: T.Tensor((B, V), dtype), # type: ignore
            W_tensor: T.Tensor((U*V, W), dtype), # type: ignore
            Z: T.Tensor((B*dim, W), dtype), # type: ignore
    ):
        # T.use_swizzle(panel_size=10, enable=True)
        # T.func_attr({"tir.noalias": True})

        with T.Kernel(T.ceildiv(B*dim, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):
            
            X_shared = T.alloc_shared((block_B, block_U, dim), dtype)
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
            for ko in T.Pipelined(T.ceildiv(U*V, block_UV), num_stages=6):
                T.copy(W_tensor[ko * block_UV, w_idx * block_W], B_shared)

                base_k = ko * block_UV
                for i, j in T.Parallel(block_B, block_UV):
                    b = (base_i + i) // dim
                    d = (base_i + i) % dim
                    u = (base_k + j) // V
                    v = (base_k + j) % V

                    A_shared[i, j] = X[b, u, d] * Y_shared[i, v]
                
                # for i, j, k in T.Parallel(block_B, block_W, block_UV):
                #     C_local[i, j] += A_shared[i, k] * B_shared[k, j]

                T.gemm(A_shared, B_shared, C_local)

            # for i, j in T.Parallel(block_B, block_W):
            #     Z[b_idx * block_B + i, w_idx * block_W + j] = C_local[i, j] 

            T.copy(C_local, Z[b_idx * block_B, w_idx * block_W])

    return main

@tilelang.jit
def backward(B, U, V, W, dim, 
            dtype="float32", 
            accum_dtype="float32",
            threads=128):
    
    # tile
    block_B = 64
    block_W = 32
    block_UV = block_WV = 64
    block_U = 32
    val = T.cast(0.03227486121839514, dtype)

    @T.prim_func
    def main(
            grad_out: T.Tensor((B, W, dim), dtype), # type: ignore
            X: T.Tensor((B, U, dim), dtype), # type: ignore
            Y: T.Tensor((B, V), dtype), # type: ignore
            W_tensor: T.Tensor((W*V, U), dtype), # type: ignore
            grad_X: T.Tensor((B*dim, U), dtype), # type: ignore
            grad_Y: T.Tensor((B, V), dtype), # type: ignore
            grad_W: T.Tensor((U, V, W), dtype), # type: ignore
    ):
        # T.use_swizzle(panel_size=10, enable=True)
        # T.func_attr({"tir.noalias": True})

        with T.Kernel(T.ceildiv(B*dim, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):
            
            # X_shared = 
            Y_shared = T.alloc_shared((block_B, V), dtype)

            A_shared = T.alloc_shared((block_B, block_WV), dtype)
            B_shared = T.alloc_shared((block_WV, block_U), dtype)

            C_local = T.alloc_fragment((block_B, block_W), accum_dtype)
            T.clear(C_local)

            base_i = b_idx * block_B
            for i, v in T.Parallel(block_B, V):
                b = (base_i + i) // dim
                Y_shared[i, v] = Y[b, v]

            # data_shared
            for ko in T.Pipelined(T.ceildiv(W*V, block_WV), num_stages=6):
                # T.copy(XY[b_idx * block_B, ko * block_UV], A_shared)
                T.copy(W_tensor[ko * block_UV, w_idx * block_W], B_shared)

                base_k = ko * block_UV
                for i, j in T.Parallel(block_B, block_WV):
                    b = (base_i + i) // dim
                    d = (base_i + i) % dim
                    u = (base_k + j) // V
                    v = (base_k + j) % V

                    A_shared[i, j] = grad_out[b, u, d] * Y_shared[i, v]
                
                # for i, j, k in T.Parallel(block_B, block_W, block_UV):
                #     C_local[i, j] += A_shared[i, k] * B_shared[k, j]

                T.gemm(A_shared, B_shared, C_local)

            # for i, j in T.Parallel(block_B, block_W):
            #     Z[b_idx * block_B + i, w_idx * block_W + j] = C_local[i, j] 

            T.copy(C_local, grad_X[b_idx * block_B, w_idx * block_W])

    return main

### example ###
"FullyConnectedTensorProduct(96x0e+96x1o+96x2e+96x3o x 10x0e -> 96x0e+96x1o+96x2e+96x3o | 368640 paths | 368640 weights)"
### data ###
B = 368
U = W = 96
V = 10
val = 0.03227486121839514

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

gen = torch.Generator(device=device)
gen.manual_seed(42)

irreps_in1 = o3.Irreps("96x0e+96x1o+96x2e+96x3o")
irreps_in2 = o3.Irreps("10x0e")
irreps_out = o3.Irreps("96x0e+96x1o+96x2e+96x3o")

# irreps_in1 = o3.Irreps("96x0e")
# irreps_in2 = o3.Irreps("10x0e")
# irreps_out = o3.Irreps("96x0e")

tp_e3nn = o3.FullyConnectedTensorProduct(
    irreps_in1, irreps_in2, 
    irreps_out, 
    shared_weights=False,     
    internal_weights=False
)

X = torch.randn(B, irreps_in1.dim, dtype=dtype, device=device, requires_grad=True)
Y = torch.randn(B, irreps_in2.dim, dtype=dtype, device=device, requires_grad=True)
W_tensor = torch.randn(1, tp_e3nn.weight_numel, dtype=dtype, device=device, requires_grad=True)
Z = torch.zeros(B, irreps_out.dim, device=device, dtype=dtype)

print(f"irreps_in1: {irreps_in1.dim}, irreps_in2: {irreps_in2.dim}, irreps_out: {irreps_out.dim}, weight: {tp_e3nn.weight_numel}")

instr = [
    (i_1, i_2, i_out, 'uvw', True, 1.0)
    for i_1, (mul1, ir_1) in enumerate(irreps_in1)     # i:编号
    for i_2, (mul2, ir_2) in enumerate(irreps_in2)     # mul:通道数
    for i_out, (mul3, ir_out) in enumerate(irreps_out) # ir.l:角量
    if ir_out in ir_1 * ir_2
]

dim_list = [
    (2 * ir_1.l + 1)
    for i_1, (_, ir_1) in enumerate(irreps_in1)
    for i_2, (_, ir_2) in enumerate(irreps_in2)
    for i_out, (_, ir_out) in enumerate(irreps_out)
    if ir_out in ir_1 * ir_2
]

cg_list = [
    (ir_1.l, ir_2.l, ir_out.l)
    for i_1, (_, ir_1) in enumerate(irreps_in1)
    for i_2, (_, ir_2) in enumerate(irreps_in2)
    for i_out, (_, ir_out) in enumerate(irreps_out)
    if ir_out in ir_1 * ir_2
]

path_num = len(instr)

# test_list = [
#     (ir_1 * ir_2, ir_out)
#     for i_1, (_, ir_1) in enumerate(irreps_in1)
#     for i_2, (_, ir_2) in enumerate(irreps_in2)
#     for i_out, (_, ir_out) in enumerate(irreps_out)
#     if ir_out in ir_1 * ir_2
# ]

# print(test_list)

### e3nn ###
retry = 100
out_e3nn = tp_e3nn(X, Y, W_tensor)
grad_output = torch.ones_like(out_e3nn)
for retry_id in range(0, retry):
    
    if retry_id != 0:
        X.grad.zero_(), Y.grad.zero_(), W_tensor.grad.zero_() 

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    out_e3nn.backward(grad_output, retain_graph=True)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"e3nn_backward_time: {end - start} ms")

## load grad data ##
grad_X_e3nn = X.grad.clone() 
grad_Y_e3nn = Y.grad.clone()
grad_W_e3nn = W_tensor.grad.clone()

# print(grad_X_e3nn)

### cueq ###
irreps_in1 = cue.Irreps("O3", "96x0e+96x1o+96x2e+96x3o")
irreps_in2 = cue.Irreps("O3", "10x0e")
irreps_out = cue.Irreps("O3", "96x0e+96x1o+96x2e+96x3o")

# irreps_in1 = cue.Irreps("O3", "96x0e")
# irreps_in2 = cue.Irreps("O3", "10x0e")
# irreps_out = cue.Irreps("O3", "96x0e")

tp_cueq = cueq.FullyConnectedTensorProduct(
    irreps_in1, irreps_in2, 
    irreps_out, 
    layout=cue.mul_ir,
    use_fallback=True,
    shared_weights=False,     
    internal_weights=False,
    device=device     
)

out_cueq = tp_cueq(X, Y, W_tensor)
for retry_id in range(0, retry):
    
    if retry_id != 0:
        X.grad.zero_(), Y.grad.zero_(), W_tensor.grad.zero_() 

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    out_cueq.backward(grad_output, retain_graph=True)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"cueq_backward_time: {end - start} ms")

## load grad data ##
grad_X_cueq = X.grad.clone()
grad_Y_cueq = Y.grad.clone()
grad_W_cueq = W_tensor.grad.clone()

# print(grad_X_cueq)

### torch.einsum ###
W_reshaped = W_tensor.reshape(-1, U, V, W)
for retry_id in range(0, retry):

    dim_offset = 0

    grad_X = []
    grad_W = []
    outs = []

    grad_Y_einsum = torch.zeros_like(Y)

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    for i in range(path_num):

        dim = dim_list[i]

        grad_out = grad_output[:, dim_offset * W: (dim_offset + dim) * W].reshape(B, -1, dim)
        x = X[:, dim_offset * U: (dim_offset + dim) * U].reshape(B, -1, dim)

        y = Y
        w = W_reshaped[i]

        out = torch.einsum("bui,bv,uvw->bwi", x, y, w) * val
        grad_x = torch.einsum("bwi,bv,uvw->bui", grad_out, y, w) * val
        grad_y = torch.einsum("bwi,bui,uvw->bv", grad_out, x, w) * val
        grad_w = torch.einsum("bwi,bui,bv->uvw", grad_out, x, y) * val

        out = out.reshape(B,-1)
        grad_x = grad_x.reshape(B, -1)
        grad_w = grad_w.reshape(-1, U * V * W)

        outs.append(out)
        grad_X.append(grad_x)
        grad_W.append(grad_w)
        grad_Y_einsum += grad_y

        dim_offset += dim

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"einsum_backward_time: {end - start} ms")

    out_einsum = torch.cat(outs, dim=1)
    grad_X_einsum = torch.cat(grad_X, dim=1)
    grad_W_einsum = torch.cat(grad_W, dim=1)

# err = (grad_Y_e3nn - grad_Y_einsum).abs()
# print("tilelang err: ", err.max().item())

print(f"grad_X shape: {list(grad_X_einsum.shape)}")
print(f"grad_Y shape: {list(grad_Y_einsum.shape)}")
print(f"grad_W shape: {list(grad_W_einsum.shape)}")

### tilelang ###
kernel = backward(B, U, V, W, 3)
# for retry_id in range(0, retry):

#     dim_offset = 0
#     outs = []

#     # kDims = torch.tensor([1, 3, 5, 7], dtype=torch.int32, device=device)
#     # kOff = torch.tensor([0, 1, 4, 9], dtype=torch.int32, device=device)

#     torch.cuda.synchronize(device=device)
#     start = time.perf_counter() * 1000
for retry_id in range(0, retry):

    dim_offset = 0
    grad_X = []
    grad_W = []

    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    for i in range(path_num):
        dim = dim_list[i]
        kernel = backward(B, U, V, W, dim)

        grad_out = grad_output[:, dim_offset * W: (dim_offset + dim) * W].reshape(B, -1, dim).contiguous()
        x = X[:, dim_offset * U: (dim_offset + dim) * U].reshape(B, -1, dim).contiguous()
        y = Y
        w = W_reshaped[i]
        w = w.permute(2, 1, 0).reshape(W * V, U).contiguous()

        # print(grad_out.shape)

        grad_x = torch.zeros(B*dim, U, device=device, dtype=dtype)
        grad_y = torch.zeros(B, V, device=device, dtype=dtype)
        grad_w = torch.zeros(U, V, W, device=device, dtype=dtype)
        
        # torch.cuda.cudart().cudaProfilerStart()
        # torch.cuda.synchronize(device=device)

        kernel(grad_out, x, y, w, grad_x, grad_y, grad_w)

        # torch.cuda.synchronize(device=device)
        # # torch.cuda.cudart().cudaProfilerStop()

        grad_x = grad_x.reshape(B, -1, W).permute(0, 2, 1).reshape(B, -1)
        grad_X.append(grad_x)

        dim_offset += dim

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000
    if retry_id == retry - 1:
        print(f"tilelang_backward_time: {end - start} ms")

    grad_X_tilelang = torch.cat(grad_X, dim=1)

print(grad_X_e3nn)
print(grad_X_tilelang * val)


