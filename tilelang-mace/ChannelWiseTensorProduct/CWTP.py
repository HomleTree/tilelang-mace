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

# @tilelang.jit
# def einsum(B, U, V, W, dtype="float64", accum_dtype="float64", threads=96):

#     @T.prim_func
#     def main(
#         X: T.Tensor((B, U), dtype), # type: ignore
#         Y: T.Tensor((B, 1), dtype), # type: ignore
#         W_tensor: T.Tensor((1, 96), dtype), # type: ignore
#         Z: T.Tensor((B, 1*W), dtype), # type: ignore
#     ):
#         with T.Kernel(W, B, threads = 1) as (w_idx, b_idx):
#             Z_accum = T.alloc_fragment((1), accum_dtype)
#             T.clear(Z_accum)

#             for u, v in T.Parallel(U, 1):
#                 x_val = T.cast(X[b_idx, u], dtype)
#                 y_val = T.cast(Y[b_idx, v], dtype)
#                 w_val = T.cast(W_tensor[:, w_idx], dtype)
#                 Z_accum[0] += x_val * y_val * w_val

#             Z[b_idx, w_idx] = T.cast(Z_accum[0], dtype)
        
#     return main

@tilelang.jit
def einsum_super_simple(B, U, dim_sum, path_num, block_B, threads = 256, dtype="float64", accum_dtype="float16"):
    
    total = U * dim_sum

    @T.prim_func
    def main(
        X: T.Tensor((B, U), dtype), # type: ignore
        Y: T.Tensor((B, dim_sum), dtype), # type: ignore
        W_tensor: T.Tensor((B, path_num * U), dtype), # type: ignore
        Z: T.Tensor((B, U * dim_sum), dtype), # type: ignore
    ):  
        with T.Kernel(T.ceildiv(B, block_B), threads=threads) as b_idx:
            tid = T.thread_binding(0, threads, thread="threadIdx.x")

            for bi in T.serial(block_B):
                b = b_idx * block_B + bi

                for t in T.serial((total + threads - 1) // threads):
                    idx = t * threads + tid
                    # if idx < total:

                    # === 同 CUDA while 拆段 ===
                    path = 0
                    tmp = idx
                    # while path < path_num and tmp >= U * dp[path]:
                    #     tmp -= U * dp[path]
                    #     path += 1
                    # dp_val = dp[path]
                    # dim_off = of[path]
                    u = tmp // dim_sum
                    i = tmp % dim_sum

                    # === 同 CUDA 三取一乘 ===
                    xv = X[b, u]
                    yv = Y[b, dim_off + i]
                    wv = W_tensor[b, path * U + u]
                    Z[b, idx] = xv * yv * wv
      
    return main

@tilelang.jit
def example(B, U, W, dim, block_B, dtype="float64", accum_dtype="float16"):
    
    @T.prim_func
    def main(
        X: T.Tensor((B, U), dtype),
        Y: T.Tensor((B, dim), dtype),
        W_tensor: T.Tensor((W), dtype),
        # Z: T.Tensor((B, W * dim), dtype),
        Z: T.Tensor((B, W, dim), dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_B), threads=U*dim) as b:
            tx = T.thread_binding(0, U*dim, thread="threadIdx.x")

            u = tx // dim
            d = tx % dim

            for bi in T.serial(block_B):       
                global_b = b * block_B + bi

                Z[global_b, u, d] = X[global_b, u] * Y[global_b, d] * W_tensor[u]
            
    return main

@tilelang.jit
def forward(B, U, V, W, 
            dtype="float32", 
            accum_dtype="float32",
            threads=256):
    
    block_B = block_W = block_UV = 32

    @T.prim_func
    def main(
        XY: T.Tensor((B, U*V), dtype), # type: ignore
        W_tensor: T.Tensor((W), dtype), # type: ignore
        Z: T.Tensor((B, W), dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(W, block_W), threads=threads) as (b_idx, w_idx):

            A_shared = T.alloc_shared((block_B, block_UV), dtype)
            B_shared = T.alloc_shared((block_W), dtype)
            # T.copy(W_tensor[w_idx * block_W], B_shared)

            C_local = T.alloc_fragment((block_B, block_W), accum_dtype)
            T.clear(C_local)

            # data_shared
            # for ko in T.Pipelined(T.ceildiv(U*V, block_UV), num_stages=3):       
            T.copy(XY[b_idx * block_B, w_idx * block_UV], A_shared)
            T.copy(W_tensor[w_idx * block_W], B_shared)
            
            for i, k in T.Parallel(block_B, block_W):
                C_local[i, k] += A_shared[i, k] * B_shared[k]
                global_b = b_idx * block_B + i
                global_w = w_idx * block_W + k
                Z[global_b, global_w] = C_local[i, k]

            # for i, j in T.Parallel(block_B, block_W):
            #     Z[b_idx * block_B + i, w_idx * block_W + j] = C_local[i, j]
            
    return main

B = 5152
U = W = 96
V = 1

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

#### cuequivarance ####
# irreps_in1 = cue.Irreps("O3", "96x0e")
# irreps_in2 = cue.Irreps("O3", "1x0e+1x1o+1x2e+1x3o")
# irreps_out = cue.Irreps("O3", "96x0e+96x1o+96x2e+96x3o")

irreps_in1 = cue.Irreps("O3", "96x0e")
irreps_in2 = cue.Irreps("O3", "1x0e")
irreps_out = cue.Irreps("O3", "96x0e")

tp_cueq = cueq.ChannelWiseTensorProduct(
    irreps_in1, irreps_in2, 
    irreps_out, 
    layout=cue.mul_ir,
    use_fallback=True,
    shared_weights=False,     
    internal_weights=False,
    device=device 
)

X = torch.randn(B, irreps_in1.dim, dtype=dtype, device=device, requires_grad=True)
Y = torch.randn(B, irreps_in2.dim, dtype=dtype, device=device, requires_grad=True)
W_tensor = torch.randn(1, tp_cueq.weight_numel, dtype=dtype, device=device, requires_grad=True)
Z = torch.zeros(B, W, device=device, dtype=dtype)

print(X.shape)
print(Y.shape)

print(f"irreps_in1: {irreps_in1.dim}, irreps_in2: {irreps_in2.dim}, irreps_out: {irreps_out.dim}, weight: {tp_cueq.weight_numel}")

instr = [
    (i_1, i_2, i_out, 'uvu', True, 1.0)
    for i_1, (mul1, ir_1) in enumerate(irreps_in1)     # i:编号
    for i_2, (mul2, ir_2) in enumerate(irreps_in2)     # mul:通道数
    for i_out, (mul3, ir_out) in enumerate(irreps_out) # ir.l:角量
    if ir_out in ir_1 * ir_2
]

dim_list = [
    (2 * ir_2.l + 1)
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

print(dim_list)
print(cg_list)
print(path_num)

dim_sum = 0
for i in range(path_num):
    dim_sum += dim_list[i]

print(dim_sum)

#### tilelang ####
retry = 100
# kernel = einsum(B, U, V, W, 96, 96, 96, 3)
# for retry_id in range(0, retry):
#     torch.cuda.synchronize()
#     start_total = time.perf_counter() *1000
#     kernel(x_all, y_all, w_all, Z)
#     torch.cuda.synchronize()
#     end_total = time.perf_counter() *1000
#     t = (end_total - start_total)
#     if retry_id == retry - 1:
#         print(f"tilelang cost: {t:.4f} ms")

block_B = 96
W_reshaped = W_tensor.reshape(tp_cueq.weight_numel)
#### cuequivarance ####
for retry_id in range(0, retry):
    torch.cuda.synchronize()
    start_total = time.perf_counter() *1000
    out = tp_cueq(X, Y, W_tensor)
    torch.cuda.synchronize()
    end_total = time.perf_counter() *1000
    t = (end_total - start_total)
    if retry_id == retry - 1:
        print(f"cueq cost: {t:.4f} ms")

print(out.shape)

#### torch.einsum ####
for retry_id in range(0, retry):
    total_einsum_time = 0
    dim_offset = 0
    outs = []
    
    for dim_idx in range(0, path_num):
        dim = dim_list[dim_idx]

        x = X
        y = Y[:, dim_offset:dim_offset + dim]
        w = W_tensor[:, dim_idx * W:(dim_idx + 1) * W]

        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        ## torch.einsum的手动实现 ##
        # z = torch.empty(B, U, dim, dtype=x.dtype, device=x.device)
        # for b in range(B):
        #     for u in range(U):
        #         for i in range(dim):
        #             z[b, u, i] = x[b, u] * y[b, i]
        z = torch.einsum("bu,bi->bui", x, y)
       
        z *= w.view(1, W, 1)

        torch.cuda.synchronize()
        end = time.perf_counter() * 1000
        if retry_id == retry - 1:
            total_einsum_time += end - start
        z = z.reshape(B, -1)
        outs.append(z)
        # print(z.shape)
        dim_offset += dim

    output = torch.cat(outs, dim=1)
    # output = output.reshape(B, -1)
    # print(output.shape)


#### torch.einsum1 ####
kernel = forward(B, U, V, W)

for retry_id in range(0, retry):
    # total_einsum_time1 = 0
    # dim_offset = 0
    # outs = []
    
    for dim_idx in range(0, path_num):
        dim = dim_list[dim_idx]

        torch.cuda.synchronize()
        start = time.perf_counter() * 1000

        # XY = torch.einsum("bu,bv->buv", X, Y)
        XY = X.unsqueeze(-1) * Y.unsqueeze(-2)
        XY = XY.reshape(B, -1).contiguous()

        kernel(XY, W_reshaped, Z)

        torch.cuda.synchronize()
        end = time.perf_counter() * 1000

        t = end - start
        if retry_id == retry - 1:
            print(f"tilelang_forward_cost: {t:.4f} ms")

print(f"einsum time: {total_einsum_time:.4f} ms")
# print(output.shape)
err = (out - output).abs().max().item()
print(f"error: {err:.5g}")

# print(out.shape)
# print(Z.shape)
# print(out)
# print(Z)

        
