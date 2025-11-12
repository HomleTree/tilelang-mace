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
# def forward(B, U, V, W,
#             dtype="float32", 
#             accum_dtype="float32",
#             threads=128):
    
#     # tile
#     block_B  = min(B, 32)
#     block_W  = min(W, 32)
#     block_UV = min(U*V, 32)
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

#             C_local = T.alloc_fragment((block_B, block_W), accum_dtype)
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

# dim_list = (2, 3, 5, 7) 
# path_num = 4

@tilelang.jit
def forward(B, U, V, W, path_num, dim_sum, dim_tup,
            dtype="float32", 
            accum_dtype="int32",
            threads=128):
    
    # tile
    block_B  = min(B, 32)
    block_W  = min(W, 32)
    block_UV = min(U*V, 32)
    val = T.cast(0.03227486121839514, dtype)
    # dim_list1 = (2, 3, 5, 7) 
    # path_nums = path_num

    @T.prim_func
    def main(
            XY: T.Tensor((B, U*V, dim_sum), dtype), # type: ignore
            W_tensor: T.Tensor((path_num, U*V, W), "float32"), # type: ignore
            Z: T.Tensor((B, W*dim_sum), dtype), # type: ignore
            dim_list: T.Tensor((path_num), "int32"), # type: ignore
            dim_offset_list: T.Tensor((path_num), "int32"), # type: ignore
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):

            for i in T.unroll(4):
                idx = T.alloc_local((1,), "int32")
                idx[0] = i

                dim = dim_list[i]
                dim_offset = dim_offset_list[i] 
                xy_base = dim_offset_list[i] * U * V

                A_shared = T.alloc_shared((block_B, U*V, dim), dtype)
                B_shared = T.alloc_shared((block_UV, block_W), dtype)

                for d, ko in T.Parallel(dim, T.ceildiv(U*V, block_UV)):

                    T.copy(XY[b_idx * block_B, ko * block_UV, dim_offset], A_shared)
                    T.copy(W_tensor[idx[0], ko * block_UV, w_idx * block_W], B_shared)

                # C_local = T.alloc_fragment((block_B, block_W*dim[0]), accum_dtype)
                # T.clear(C_local)

    return main

### example ###
"FullyConnectedTensorProduct(96x0e+96x1o+96x2e+96x3o x 10x0e -> 96x0e+96x1o+96x2e+96x3o | 368640 paths | 368640 weights)"
### data ###
B = 5152
U = W = 96
V = 10
val = 0.03227486121839514

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

gen = torch.Generator(device=device)
gen.manual_seed(42)

block_B = block_W = 96

irreps_in1 = o3.Irreps("96x0e+96x1o+96x2e+96x3o")
irreps_in2 = o3.Irreps("10x0e")
irreps_out = o3.Irreps("96x0e+96x1o+96x2e+96x3o")

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
Z = torch.zeros(B, W*16, device=device, dtype=dtype)

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
# print(cg_list)
path_num = len(instr)
retry = 2

path_num = 4
dim_sum = 16
dim_list = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
dim_offset_list = torch.tensor([0, 1, 4, 9], device=device, dtype=torch.int32)

W_reshaped = W_tensor.reshape(path_num, U * V, W)
 
kernel = forward(B, U, V, W, path_num, dim_sum)

retry = 100
for retry_id in range(0, retry):

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    dim_offset = 0
    dim_kffset = 0
    outers = []
    for path_idx in range(path_num):
        dim = dim_list[path_idx]

        x = X[:, dim_offset * U: (dim_offset + dim) * U]
        y = Y

        outer = x.unsqueeze(-2) * y.unsqueeze(-1)
        outer = outer.reshape(B, V, U, dim)

        outers.append(outer)

        dim_offset += dim
        dim_kffset += dim

    outers1 = torch.cat(outers, dim=3)

    dim_offset = 0
    dim_kffset = 0
    outs = []
    for path_idx in range(path_num):
        dim = dim_list[path_idx]

        W_reshaped = W_tensor.reshape(path_num, U, V, W)
        w = W_reshaped[path_idx]

        outer = outers1[:,:,:,dim_offset: (dim_offset + dim)]

        out = torch.einsum("bvui,uvw->bwi", outer, w) * val
        out = out.reshape(B, -1)
        outs.append(out)

        dim_offset += dim
        dim_kffset += dim

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"einsum_forward2_time: {(end - start):.4f} ms")

    output2 = torch.cat(outs, dim=1)




### einsum_forward ###
for retry_id in range(0, retry):

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    dim_offset = 0
    dim_kffset = 0
    outs = []

    for path_idx in range(path_num):
        dim = dim_list[path_idx]
        # print(dim)

        x = X[:, dim_offset * U: (dim_offset + dim) * U].reshape(B, U, dim)
        y = Y
        W_reshaped = W_tensor.reshape(path_num, U, V, W)
        w = W_reshaped[path_idx]

        out = torch.einsum("bui,bv,uvw->bwi", x, y, w) * val

        out = out.reshape(B, -1)
        # print(out.shape)
        outs.append(out)

        dim_offset += dim
        dim_kffset += dim

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"einsum_forward_time: {(end - start):.4f} ms")

    output = torch.cat(outs, dim=1)

print(output.shape)
print(output)

### einsum_forward1 ###
for retry_id in range(0, retry):

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    dim_offset = 0
    dim_kffset = 0
    outs = []
    for path_idx in range(path_num):
        dim = dim_list[path_idx]

        x = X[:, dim_offset * U: (dim_offset + dim) * U]
        y = Y
        W_reshaped = W_tensor.reshape(path_num, U, V, W)
        w = W_reshaped[path_idx]

        outer = x.unsqueeze(-2) * y.unsqueeze(-1)
        outer = outer.reshape(B, V, U, dim)

        out = torch.einsum("bvui,uvw->bwi", outer, w) * val

        out = out.reshape(B, -1)
        outs.append(out)

        dim_offset += dim
        dim_kffset += dim

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"einsum_forward1_time: {(end - start):.4f} ms")

    output1 = torch.cat(outs, dim=1)
print(output1.shape)
print(output1)

### einsum_forward2 ###
for retry_id in range(0, retry):

    torch.cuda.synchronize()
    start = time.perf_counter() * 1000

    dim_offset = 0
    dim_kffset = 0
    outers = []
    for path_idx in range(path_num):
        dim = dim_list[path_idx]

        x = X[:, dim_offset * U: (dim_offset + dim) * U]
        y = Y

        outer = x.unsqueeze(-2) * y.unsqueeze(-1)
        outer = outer.reshape(B, V, U, dim)

        outers.append(outer)

        dim_offset += dim
        dim_kffset += dim

    outers1 = torch.cat(outers, dim=3)

    dim_offset = 0
    dim_kffset = 0
    outs = []
    for path_idx in range(path_num):
        dim = dim_list[path_idx]

        W_reshaped = W_tensor.reshape(path_num, U, V, W)
        w = W_reshaped[path_idx]

        outer = outers1[:,:,:,dim_offset: (dim_offset + dim)]

        out = torch.einsum("bvui,uvw->bwi", outer, w) * val
        out = out.reshape(B, -1)
        outs.append(out)

        dim_offset += dim
        dim_kffset += dim

    torch.cuda.synchronize()
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"einsum_forward2_time: {(end - start):.4f} ms")

    output2 = torch.cat(outs, dim=1)
# print(output2.shape)
# print(output2)

### e3nn ###
for retry_id in range(0, retry):
    torch.cuda.synchronize(device=device)
    start = time.perf_counter() * 1000
    out_e3nn = tp_e3nn(X, Y, W_tensor)
    torch.cuda.synchronize(device=device)
    end = time.perf_counter() * 1000

    if retry_id == retry - 1:
        print(f"e3nn time: {(end - start):.4f} ms")

# print(out_e3nn.shape)
# print(out_e3nn)

### cueq ###
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

# print(out_cueq.shape)

## error ##
# print(f"out: {torch.allclose(Z, out_e3nn, rtol=1e-05, atol=1e-05, equal_nan=False)}")

err = (out_e3nn - output2).abs()
print("max err: ", err.max().item())

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

    path_num = 4
    dim_sum = 16
    dim_list = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    dim_offset_list = torch.tensor([0, 1, 4, 9], device=device, dtype=torch.int32)

    W_reshaped = W_tensor.reshape(4, U * V, W)
    w_slices = tuple(W_reshaped[p, :, :] for p in range(path_num))

    kernel = forward(B, U, V, W, path_num, dim_sum, w_slices)


    
    W_reshaped = W_tensor.reshape(U * V, W)

    retry = 100
    for retry_id in range(0, retry):
        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        XY = X.unsqueeze(-1) * Y.unsqueeze(-2)
        XY = XY.reshape(B, -1)
        kernel(XY, W_reshaped, Z, dim_list, dim_offset_list)
        
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
# def main():
#     out_tilelang = tilelang_forward_fctp(5152, "96x0e", "10x0e", "96x0e", dtype)
#     # out_cueq     = cueq_forward_fctp(5152, "96x0e", "10x0e", "96x0e", dtype)
#     out_e3nn     = e3nn_forward_fctp(5152, "96x0e", "10x0e", "96x0e", dtype)
#     # out_einsum   = einsum_forward_fctp(5152, "96x0e", "10x0e", "96x0e", dtype)

#     err_tilelang = (out_e3nn - out_tilelang).abs()
#     print(f"err_tilelang: {err_tilelang.max().item()}")


#     # err_cueq = (out_e3nn - out_cueq).abs()
#     # err_einsum = (out_e3nn - out_einsum).abs()
#     # print(f"err_tilelang: {err_tilelang.max().item()}\n"
#     #       f"err_cueq:     {err_cueq.max().item()}\n"
#     #       f"err_einsum:   {err_einsum.max().item()}")

# if __name__ == "__main__":
#     main()

