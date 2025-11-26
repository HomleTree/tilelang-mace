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

# @tilelang.jit
# def forward(B, U, V, W, dim,
#             dtype="float16", 
#             accum_dtype="float32",
#             threads=128):
    
#     # tile
#     block_B = 64
#     block_W = 64
#     block_UV = 64
#     block_U = 64

#     @T.prim_func
#     def main(
#             X: T.Tensor((B, U, dim), dtype), # type: ignore
#             Y: T.Tensor((B, V), dtype), # type: ignore
#             W_tensor: T.Tensor((U*V, W), dtype), # type: ignore
#             Z: T.Tensor((B*dim, W), dtype), # type: ignore
#     ):
#         # T.use_swizzle(panel_size=10, enable=True)
#         T.func_attr({"tir.noalias": True})
#         T.func_attr({"tl.copy_vectorize": 4}) 

#         with T.Kernel(T.ceildiv(B*dim, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):
            
#             # X_shared = T.alloc_shared((block_B, block_U, dim), dtype)
#             Y_shared = T.alloc_shared((block_B, V), dtype)

#             A_shared = T.alloc_shared((block_B, block_UV), dtype)
#             B_shared = T.alloc_shared((block_UV, block_W), dtype)

#             C_local = T.alloc_fragment((block_B, block_W), accum_dtype)
#             T.clear(C_local)

#             base_i = b_idx * block_B
#             for i, v in T.Parallel(block_B, V):
#                 b = (base_i + i) // dim
#                 Y_shared[i, v] = Y[b, v]

#             # data_shared
#             for ko in T.Pipelined(T.ceildiv(U*V, block_UV), num_stages=6):
#                 # T.copy(XY[b_idx * block_B, ko * block_UV], A_shared)
#                 T.copy(W_tensor[ko * block_UV, w_idx * block_W], B_shared)

#                 base_k = ko * block_UV
#                 for i, j in T.Parallel(block_B, block_UV):
#                     b = (base_i + i) // dim
#                     d = (base_i + i) % dim
#                     u = (base_k + j) // V
#                     v = (base_k + j) % V

#                     A_shared[i, j] = X[b, u, d] * Y_shared[i, v]
                
#                 # for i, j, k in T.Parallel(block_B, block_W, block_UV):
#                 #     C_local[i, j] += A_shared[i, k] * B_shared[k, j]

#                 T.gemm(A_shared, B_shared, C_local)

#             # for i, j in T.Parallel(block_B, block_W):
#             #     Z[b_idx * block_B + i, w_idx * block_W + j] = C_local[i, j] 

#             T.copy(C_local, Z[b_idx * block_B, w_idx * block_W])

#     return main

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

### example ###
"FullyConnectedTensorProduct(96x0e+96x1o+96x2e+96x3o x 10x0e -> 96x0e+96x1o+96x2e+96x3o | 368640 paths | 368640 weights)"
### data ###
B = 368
U = W = 96
V = 10
val = 0.03227486121839514

retry = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

irreps_in1 = o3.Irreps("96x0e+96x1o+96x2e+96x3o")
irreps_in2 = o3.Irreps("10x0e")
irreps_out = o3.Irreps("96x0e+96x1o+96x2e+96x3o")

# irreps_in1 = o3.Irreps("96x2e")
# irreps_in2 = o3.Irreps("10x0e")
# irreps_out = o3.Irreps("96x2e")

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

### e3nn ###
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000

    out_e3nn = tp_e3nn(X, Y, W_tensor)

    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"e3nn time: {(end - start):.4f} ms") 

### cueq ###
irreps_in1 = cue.Irreps("O3", "96x0e+96x1o+96x2e+96x3o")
irreps_in2 = cue.Irreps("O3", "10x0e")
irreps_out = cue.Irreps("O3", "96x0e+96x1o+96x2e+96x3o")

# irreps_in1 = cue.Irreps("O3", "96x2e")
# irreps_in2 = cue.Irreps("O3", "10x0e")
# irreps_out = cue.Irreps("O3", "96x2e")

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
        print(f"cueq time: {(end - start):.4f} ms") 

err = (out_e3nn - out_cueq).abs()
print("cueq err: ", err.max().item())

### torch.einsum ###
W_reshaped = W_tensor.reshape(-1, U, V, W)
for retry_id in range(0, retry):

    dim_offset = 0
    outs = []
    total_einsum_time = 0
    for i in range(path_num):

        dim = dim_list[i]

        x = X[:, dim_offset * U: (dim_offset + dim) * U].reshape(B, -1, dim)
        x = x.reshape(B, -1, dim)
        y = Y
        w = W_reshaped[i]

        torch.cuda.synchronize(device=device)
        start = time.perf_counter() * 1000

        # out = torch.einsum("bui,bv,uvw->bwi", x, y, w) * val
        out1 = torch.einsum("bui,bv->buvi", x, y) * val
        out1 = out1.permute(0,3,1,2).reshape(-1,U,V)
        out = torch.einsum("cuv,uvw->cw", out1, w)

        out = out.reshape(B, -1,W).permute(0,2,1).reshape(B,-1)

        torch.cuda.synchronize(device=device)
        end = time.perf_counter() * 1000

        if retry_id == retry - 1:
            total_einsum_time += end - start 

        outs.append(out)

        dim_offset += dim
out_einsum = torch.cat(outs, dim=1)
print(f"einsum time: {total_einsum_time} ms")

err = (out_e3nn - out_einsum).abs()
print("einsum err: ", err.max().item())

### tilelang ###
W_reshaped = W_tensor.reshape(-1, U, V, W)
for retry_id in range(0, retry):

    dim_offset = 0
    outs = []

    # kDims = torch.tensor([1, 3, 5, 7], dtype=torch.int32, device=device)
    # kOff = torch.tensor([0, 1, 4, 9], dtype=torch.int32, device=device)

    # torch.cuda.synchronize(device=device)
    # start = time.perf_counter() * 1000
    for i in range(path_num):
        dim = dim_list[i]
        kernel1 = forward(B, U, V, W, dim)

        x = X[:, dim_offset * U: (dim_offset + dim) * U].reshape(B, U, dim).contiguous()

        y = Y
        w = W_reshaped[i]
        w = w.reshape(U*V, W).contiguous()
        z = torch.zeros(B*dim, W, device=device, dtype=dtype)
        
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.synchronize(device=device)

        kernel1(x, y, w, z)

        torch.cuda.synchronize(device=device)
        torch.cuda.cudart().cudaProfilerStop()

        z = z.reshape(B, -1, W).permute(0, 2, 1).reshape(B, -1)

        outs.append(z)
        dim_offset += dim

    # torch.cuda.synchronize(device=device)
    # end = time.perf_counter() * 1000
    # if retry_id == retry - 1:
    #     print(f"tilelang time: {end - start} ms")

output1 = torch.cat(outs, dim=1) * val

err = (out_e3nn - output1).abs()
print("tilelang err: ", err.max().item())

### manual_seed ###
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

    W_reshaped = W_tensor.reshape(-1, U, V, W)
    for retry_id in range(0, retry):
        dim_offset = 0
        outs = []

        torch.cuda.synchronize()
        start = time.perf_counter() * 1000
        for i in range(0, path_num):
            dim = dim_list[i]
            kernel = forward(B, U, V, W, dim)

            x = X[:, dim_offset * U: (dim_offset + dim) * U].reshape(B, -1, dim).contiguous()
            y = Y
            w = W_reshaped[i]
            w = w.reshape(U*V, W).contiguous()
            z = torch.zeros(B*dim, W, device=device, dtype=dtype)
            
            kernel(x, y, w, z)

            z = z.view(B, dim, W).transpose(1, 2).reshape(B, -1)

            outs.append(z)
            dim_offset += dim

        out_tilelang = torch.cat(outs, dim=1)
        out_tilelang *= val

        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        if retry_id == retry - 1:
            print(f"tilelang_forward_time: {(end - start):.4f} ms") 

    return out_tilelang

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

    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        out_cueq = tp_cueq(X, Y, W)
        
        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        t = end - start
        if retry_id == retry - 1:
            print(f"cueq_forward_cost: {t:.4f} ms")
    
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

    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        out_e3nn = tp_e3nn(X, Y, W)
        
        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        t = end - start
        if retry_id == retry - 1:
            print(f"e3nn_forward_cost: {t:.4f} ms")
    
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
    W_tensor = make_X(1, tp.weight_numel, dtype=dtype, device=device, requires_grad=True)

    W_reshaped = W_tensor.reshape(-1, U, V, W)
    for retry_id in range(0, retry):
        dim_offset = 0
        outs = []

        torch.cuda.synchronize()
        start = time.perf_counter() * 1000
        for i in range(0, path_num):
            dim = dim_list[i]

            x = X[:, dim_offset * U: (dim_offset + dim) * U].reshape(B, -1, dim)
            y = Y
            w = W_reshaped[i]

            out = torch.einsum("bui,bv,uvw->bwi", x, y, w) 
            out = out.reshape(B, -1)

            outs.append(out)
            dim_offset += dim

        out_einsum = torch.cat(outs, dim=1)
        out_einsum *= val

        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        if retry_id == retry - 1:
            print(f"einsum_forward_time: {(end - start):.4f} ms") 

    return out_einsum

#### main ####
def main():
    irreps_in1 = "96x0e+96x1o+96x2e+96x3o"
    irreps_in2 = "10x0e"
    irreps_out = "96x0e+96x1o+96x2e+96x3o"

    # irreps_in1 = "96x2e"
    # irreps_in2 = "10x0e"
    # irreps_out = "96x2e"

    out_tilelang = tilelang_forward_fctp(B, irreps_in1, irreps_in2, irreps_out, dtype)
    out_cueq     = cueq_forward_fctp(B, irreps_in1, irreps_in2, irreps_out, dtype)
    out_e3nn     = e3nn_forward_fctp(B, irreps_in1, irreps_in2, irreps_out, dtype)
    out_einsum   = einsum_forward_fctp(B, irreps_in1, irreps_in2, irreps_out, dtype)

    err_tilelang = (out_e3nn - out_tilelang).abs()
    err_cueq = (out_e3nn - out_cueq).abs()
    err_einsum = (out_e3nn - out_einsum).abs()
    print(f"err_tilelang: {err_tilelang.max().item()}\n"
          f"err_cueq:     {err_cueq.max().item()}\n"
          f"err_einsum:   {err_einsum.max().item()}")

if __name__ == "__main__":
    main()