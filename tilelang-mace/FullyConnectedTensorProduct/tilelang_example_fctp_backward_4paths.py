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
# def backward(B, U, V, W, dim, 
#             dtype="float32", 
#             accum_dtype="float32",
#             threads=128):
    
#     # tile
#     block_B = 64
#     block_W = 32
#     block_UV = block_WV = 64
#     block_U = 32
#     val = T.cast(0.03227486121839514, dtype)

#     @T.prim_func
#     def main(
#             grad_out: T.Tensor((B, W, dim), dtype), # type: ignore
#             X: T.Tensor((B, U, dim), dtype), # type: ignore
#             Y: T.Tensor((B, V), dtype), # type: ignore
#             W_tensor: T.Tensor((W*V, U), dtype), # type: ignore
#             grad_X: T.Tensor((B*dim, U), dtype), # type: ignore
#             grad_Y: T.Tensor((B, V), dtype), # type: ignore
#             grad_W: T.Tensor((U, V, W), dtype), # type: ignore
#     ):
#         # T.use_swizzle(panel_size=10, enable=True)
#         # T.func_attr({"tir.noalias": True})

#         with T.Kernel(T.ceildiv(B*dim, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):
            
#             # X_shared = T.alloc_shared(())
#             Y_shared = T.alloc_shared((block_B, V), dtype)

#             A_shared = T.alloc_shared((block_B, block_WV), dtype)
#             B_shared = T.alloc_shared((block_WV, block_U), dtype)

#             grad_X_local = T.alloc_fragment((block_B, block_W), accum_dtype)
#             T.clear(grad_X_local)
#             grad_W_reshaped = T.Buffer((U*V, W), dtype, data=grad_W.data)
#             # grad_Y_local = T.alloc_fragment((block_B, V), accum_dtype)
#             # T.clear(grad_Y_local)

#             # T.customize.reshape(grad_W, (U*V, W))

#             base_i = b_idx * block_B
#             for i, v in T.Parallel(block_B, V):
#                 b = (base_i + i) // dim
#                 Y_shared[i, v] = Y[b, v]

#             # data_shared
#             for ko in T.Pipelined(T.ceildiv(W*V, block_WV), num_stages=6):
#                 # T.copy(XY[b_idx * block_B, ko * block_UV], A_shared)
#                 T.copy(W_tensor[ko * block_UV, w_idx * block_W], B_shared)

#                 base_k = ko * block_UV
#                 for i, j in T.Parallel(block_B, block_WV):
#                     b = (base_i + i) // dim
#                     d = (base_i + i) % dim
#                     u = (base_k + j) // V
#                     v = (base_k + j) % V

#                     A_shared[i, j] = grad_out[b, u, d] * Y_shared[i, v]
                
#                 # for i, j, k in T.Parallel(block_B, block_W, block_UV):
#                 #     C_local[i, j] += A_shared[i, k] * B_shared[k, j]

#                 T.gemm(A_shared, B_shared, grad_X_local)

#             # for i, j in T.Parallel(block_B, block_W):
#             #     Z[b_idx * block_B + i, w_idx * block_W + j] = C_local[i, j] 

#             T.copy(grad_X_local, grad_X[b_idx * block_B, w_idx * block_W])

#     return main

@tilelang.jit
def backward_x(B, U, V, W, dim, 
            dtype="float16", 
            accum_dtype="float32",
            threads=128):
    
    # tile
    block_B = 64
    block_W = 64
    block_UV = block_WV = 64
    block_U = 64
    val = T.cast(0.03227486121839514, dtype)

    @T.prim_func
    def main(
            grad_out: T.Tensor((B, W, dim), dtype), # type: ignore
            X: T.Tensor((B, U, dim), dtype), # type: ignore
            Y: T.Tensor((B, V), dtype), # type: ignore
            W_tensor: T.Tensor((W*V, U), dtype), # type: ignore
            grad_X: T.Tensor((B*dim, U), dtype), # type: ignore
    ):
        # T.use_swizzle(panel_size=10, enable=True)
        # T.func_attr({"tir.noalias": True})

        with T.Kernel(T.ceildiv(B*dim, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):
            
            Y_shared = T.alloc_shared((block_B, V), dtype)

            A_shared = T.alloc_shared((block_B, block_WV), dtype)
            B_shared = T.alloc_shared((block_WV, block_U), dtype)

            grad_X_local = T.alloc_fragment((block_B, block_W), accum_dtype)
            T.clear(grad_X_local)

            base_i = b_idx * block_B
            for i, v in T.Parallel(block_B, V):
                b = (base_i + i) // dim
                Y_shared[i, v] = Y[b, v]

            # data_shared
            for ko in T.Pipelined(T.ceildiv(W*V, block_WV), num_stages=6):
                # T.copy(XY[b_idx * block_B, ko * block_UV], A_shared)
                T.copy(W_tensor[ko * block_UV, w_idx * block_W], B_shared)

                base_k = ko * block_WV
                for i, j in T.Parallel(block_B, block_WV):
                    b = (base_i + i) // dim
                    d = (base_i + i) % dim
                    w = (base_k + j) // V
                    v = (base_k + j) % V

                    A_shared[i, j] = grad_out[b, w, d] * Y_shared[i, v]
                
                # for i, j, k in T.Parallel(block_B, block_W, block_UV):
                #     C_local[i, j] += A_shared[i, k] * B_shared[k, j]

                T.gemm(A_shared, B_shared, grad_X_local)

            # for i, j in T.Parallel(block_B, block_W):
            #     Z[b_idx * block_B + i, w_idx * block_W + j] = C_local[i, j] 

            T.copy(grad_X_local, grad_X[b_idx * block_B, w_idx * block_W])

    return main

### example ###
"FullyConnectedTensorProduct(96x0e+96x1o+96x2e+96x3o x 10x0e -> 96x0e+96x1o+96x2e+96x3o | 368640 paths | 368640 weights)"
"目前只考虑grad_X"
### data ###
B = 5152
U = W = 96
V = 10
val = 0.03227486121839514

retry = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

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
def tilelang_backward_fctp(
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
    grad_output = torch.ones(B, irreps_out.dim, dtype=dtype, device=device)
    for retry_id in range(0, retry):

        dim_offset = 0
        grad_X = []
        grad_W = []

        # grad_Y_tilelang = torch.zeros_like(Y)

        torch.cuda.synchronize()
        start = time.perf_counter() * 1000
        for i in range(0, path_num):
            dim = dim_list[i]
            kernel = backward_x(B, U, V, W, dim)
            
            grad_out = grad_output[:, dim_offset * W: (dim_offset + dim) * W].reshape(B, -1, dim).contiguous()
            x = X[:, dim_offset * U: (dim_offset + dim) * U].reshape(B, -1, dim).contiguous()
            y = Y
            w = W_reshaped[i].contiguous()
            w = w.permute(2, 1, 0).reshape(W * V, U).contiguous()
            # w_2 = w.permute(2, 1, 0).reshape(W * V, U).contiguous()

            grad_x = torch.zeros(B*dim, U, device=device, dtype=dtype)
            # grad_y = torch.zeros(B, V, device=device, dtype=dtype)
            # grad_w = torch.zeros(U, V, W, device=device, dtype=dtype)
            
            kernel(grad_out, x, y, w, grad_x)

            grad_x = grad_x.reshape(B, -1, W).permute(0, 2, 1).reshape(B, -1)
            grad_X.append(grad_x)

            # grad_Y_tilelang += grad_y

            dim_offset += dim

        grad_X_tilelang = torch.cat(grad_X, dim=1)
        grad_X_tilelang *= val
        # grad_Y_tilelang *= val

        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        if retry_id == retry - 1:
            print(f"tilelang_backward_time: {(end - start):.4f} ms") 

    return grad_X_tilelang, None, None

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
    W = make_X(1, tp_cueq.weight_numel, dtype=dtype, device=device, requires_grad=True)

    out_cueq = tp_cueq(X, Y, W)
    grad_output = torch.ones_like(out_cueq)
    for retry_id in range(0, retry):

        if retry_id != 0:
            X.grad.zero_(), Y.grad.zero_(), W.grad.zero_()

        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        out_cueq.backward(grad_output, retain_graph=True)
        
        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        t = end - start
        if retry_id == retry - 1:
            print(f"cueq_backward_cost: {t:.4f} ms")

    grad_X_cueq = X.grad.clone()
    grad_Y_cueq = Y.grad.clone()
    grad_W_cueq = W.grad.clone()
    
    return grad_X_cueq, grad_Y_cueq, grad_W_cueq

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
    W = make_X(1, tp_e3nn.weight_numel, dtype=dtype, device=device, requires_grad=True)

    out_e3nn = tp_e3nn(X, Y, W)
    grad_output = torch.ones_like(out_e3nn)
    for retry_id in range(0, retry):

        if retry_id != 0:
            X.grad.zero_(), Y.grad.zero_(), W.grad.zero_() 

        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        out_e3nn.backward(grad_output, retain_graph=True)
        
        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        t = end - start
        if retry_id == retry - 1:
            print(f"e3nn_backward_cost: {t:.4f} ms")

    grad_X_e3nn = X.grad.clone() 
    grad_Y_e3nn = Y.grad.clone()
    grad_W_e3nn = W.grad.clone()
        
    return grad_X_e3nn, grad_Y_e3nn, grad_W_e3nn

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
    W_tensor = make_X(1, tp.weight_numel, dtype=dtype, device=device, requires_grad=True)

    W_reshaped = W_tensor.reshape(-1, U, V, W)
    grad_output = torch.ones(B, irreps_out.dim, dtype=dtype, device=device)
    for retry_id in range(0, retry):

        dim_offset = 0
        grad_X = []
        grad_W = []
        outs = []

        grad_Y_einsum = torch.zeros_like(Y)

        torch.cuda.synchronize()
        start = time.perf_counter() * 1000
        for i in range(0, path_num):
            dim = dim_list[i]

            grad_out = grad_output[:, dim_offset * W: (dim_offset + dim) * W].reshape(B, -1, dim)
            x = X[:, dim_offset * U: (dim_offset + dim) * U].reshape(B, -1, dim)
            y = Y
            w = W_reshaped[i]

            grad_x = torch.einsum("bwi,bv,uvw->bui", grad_out, y, w) * val
            grad_y = torch.einsum("bwi,bui,uvw->bv", grad_out, x, w) * val
            # grad_y1 = torch.einsum("bwi,bui->bwu", grad_out, x) * val
            # grad_y = torch.einsum("bwu,uvw->bv", grad_y1, w)

            grad_w = torch.einsum("bwi,bui,bv->uvw", grad_out, x, y) * val
            # grad_w1 = torch.einsum("bwi,bui->bwu", grad_out, x) * val
            # grad_w = torch.einsum("bwu,bv->uvw", grad_w1, y)

            grad_x = grad_x.reshape(B, -1)
            grad_w = grad_w.reshape(-1, U * V * W)

            grad_X.append(grad_x)
            grad_W.append(grad_w)
            grad_Y_einsum += grad_y

            dim_offset += dim

        grad_X_einsum = torch.cat(grad_X, dim=1)
        grad_W_einsum = torch.cat(grad_W, dim=1)

        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        if retry_id == retry - 1:
            print(f"einsum_backward_time: {(end - start):.4f} ms") 

    return grad_X_einsum, grad_Y_einsum, grad_W_einsum

#### main ####
def main():
    irreps_in1 = "96x0e+96x1o+96x2e+96x3o"
    irreps_in2 = "10x0e"
    irreps_out = "96x0e+96x1o+96x2e+96x3o"

    grad_X_tilelang, _, _ = tilelang_backward_fctp(B, irreps_in1, irreps_in2, irreps_out, dtype)
    grad_X_cueq, grad_Y_cueq, grad_W_cueq     = cueq_backward_fctp(B, irreps_in1, irreps_in2, irreps_out, dtype)
    grad_X_e3nn, grad_Y_e3nn, grad_W_e3nn     = e3nn_backward_fctp(B, irreps_in1, irreps_in2, irreps_out, dtype)
    grad_X_einsum, grad_Y_einsum, grad_W_einsum   = einsum_backward_fctp(B, irreps_in1, irreps_in2, irreps_out, dtype)

    err_tilelang = (grad_X_e3nn - grad_X_tilelang).abs()
    err_cueq = (grad_X_e3nn - grad_X_cueq).abs()
    err_einsum = (grad_X_e3nn - grad_X_einsum).abs()
    print(f"err_tilelang: {err_tilelang.max().item()}\n"
          f"err_cueq:     {err_cueq.max().item()}\n"
          f"err_einsum:   {err_einsum.max().item()}")

if __name__ == "__main__":
    main()
