import torch
import os, time
import ast
from torch.utils.cpp_extension import load

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout

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

fused_message_passing_fwd = load(
    name="fused_message_passing_fwd",
    sources=["fused_message_passing_fwd.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=sm_90"
    ]
)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype     = torch.float64
dtype_int = torch.int32

def generate_proper_sorted_graph(E=96, N=368, device=device):
    """生成正确排序的图数据"""
    dtype_int = torch.int32
    
    # 生成sender和receiver
    sender = torch.randint(low=0, high=N, size=(E,), dtype=dtype_int, device=device)
    receiver = torch.randint(low=0, high=N, size=(E,), dtype=dtype_int, device=device)
    
    # 关键修正：先按receiver排序，再按sender排序，确保相同receiver的边连续
    # 创建复合键 (receiver, sender) 进行排序
    composite_key = receiver * N + sender  # 确保唯一排序
    sorted_indices = torch.argsort(composite_key)
    
    receiver_sorted = receiver[sorted_indices]
    sender_sorted = sender[sorted_indices]
    
    print(f"排序验证:")
    print(f"  Receiver是否排序: {torch.all(receiver_sorted[:-1] <= receiver_sorted[1:])}")
    
    # 检查相同receiver的边是否连续
    for i in range(E-1):
        if receiver_sorted[i] == receiver_sorted[i+1]:
            assert sender_sorted[i] <= sender_sorted[i+1], "相同receiver的边不连续!"
    
    return sender_sorted, receiver_sorted

def compute_csr_indices_corrected(receiver, N, device='cuda'):
    """修正的CSR索引计算"""
    start_idx = torch.zeros(N, dtype=torch.int32, device=device)
    end_idx = torch.zeros(N, dtype=torch.int32, device=device)
    
    E = receiver.shape[0]
    
    # 处理没有边的节点
    # 首先将所有end_idx设置为start_idx，表示没有边
    # end_idx[i] = start_idx[i] 表示节点i没有边
    
    for i in range(E):
        r = receiver[i].item()
        
        # 运行开始：如果是第一条边或者前一条边的receiver不同
        if i == 0 or receiver[i-1].item() != r:
            start_idx[r] = i
        
        # 运行结束：如果是最后一条边或者下一条边的receiver不同
        if i == E-1 or receiver[i+1].item() != r:
            end_idx[r] = i + 1
    
    print(f"CSR索引统计:")
    print(f"  有入边的节点数: {(end_idx > start_idx).sum().item()}")
    print(f"  最大边数 per 节点: {(end_idx - start_idx).max().item()}")
    
    return start_idx, end_idx

def simulate_receiver_major_corrected(node_feats, edge_attrs, tp_weights, sender, receiver, dim_list, offs):
    """完全修正的模拟函数"""
    N, U = node_feats.shape
    E, DIM_SUM = edge_attrs.shape
    P = len(dim_list)
    
    # 计算正确的CSR索引
    start_idx_torch, end_idx_torch = compute_csr_indices_corrected(receiver, N, device=node_feats.device)
    
    # 初始化输出 [N, DIM_SUM, U]
    out_nodes_torch = torch.zeros(N, DIM_SUM, U, device=node_feats.device, dtype=node_feats.dtype)
    
    print(f"\n开始模拟计算...")
    
    # 为每个接收节点计算
    for r in range(N):
        st = start_idx_torch[r].item()
        ed = end_idx_torch[r].item()
        
        # 只有当 st < ed 时，该节点才有入边
        if st >= ed:
            continue
            
        num_edges = ed - st
        # print(f"节点{r}: {num_edges}条入边")
        
        # 这些边的索引范围
        edges_to_r = torch.arange(st, ed, device=node_feats.device)
        
        # 这些边的发送者索引
        senders_idx = sender[edges_to_r]
        
        # 发送者的节点特征 [num_edges, U]
        sender_feats = node_feats[senders_idx]
        
        # 为每个路径计算
        for p in range(P):
            d = dim_list[p].item()
            o = offs[p].item()
            
            # 当前路径的权重 [num_edges, U]
            weights_p = tp_weights[edges_to_r, p, :]
            
            # 当前路径的边属性 [num_edges, d]
            edge_attrs_p = edge_attrs[edges_to_r, o:o+d]
            
            # 计算消息
            weighted_feats = sender_feats * weights_p
            messages = torch.einsum('eu,ed->edu', weighted_feats, edge_attrs_p)
            aggregated = messages.sum(dim=0)
            
            # 存储到输出
            out_nodes_torch[r, o:o+d, :] += aggregated
    
    return out_nodes_torch, start_idx_torch, end_idx_torch

# 你的参数
N = 96
U = 96
E = 368
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64
dtype_int = torch.int32

dim_list = [1, 3, 5, 7]
P = len(dim_list)
dim_sum = sum(dim_list)
offs = [0, 1, 4, 9]

dim_list = torch.tensor(dim_list, dtype=dtype_int, device=device)
offs = torch.tensor(offs, dtype=dtype_int, device=device)

# 生成正确的排序数据
sender, receiver = generate_proper_sorted_graph(E, N, device)

# 其他张量
node_feats = torch.randn(N, U, dtype=dtype, device=device, requires_grad=True)
edge_attrs = torch.randn(E, dim_sum, dtype=dtype, device=device, requires_grad=True)
tp_weights = torch.randn(E, P, U, dtype=dtype, device=device, requires_grad=True)

# print(f"\n=== 调试信息 ===")
# print(f"节点数: {N}, 边数: {E}, 特征维度: {U}")
# print(f"路径配置: dim_list={dim_list.cpu().numpy()}, offs={offs.cpu().numpy()}")
# print(f"Receiver分布:")
# unique_receivers, counts = torch.unique(receiver, return_counts=True)
# print(f"  有入边的节点数: {len(unique_receivers)}")
# print(f"  最大入边数: {counts.max().item()}")
# print(f"  平均入边数: {counts.float().mean().item():.2f}")

# 运行修正的模拟
out_nodes_torch, start_idx_torch, end_idx_torch = simulate_receiver_major_corrected(
    node_feats, edge_attrs, tp_weights, sender, receiver, dim_list, offs
)

out_nodes_torch = out_nodes_torch.reshape(N, -1)

receiver_major = True
out_nodes, start_idx, end_idx = fused_message_passing_fwd.forward(node_feats, edge_attrs, tp_weights, sender, receiver, dim_list, offs, receiver_major)

err = (out_nodes - out_nodes_torch).abs()
print(f"error: {err.max().item()}")

err = (start_idx - start_idx_torch).abs()
print(f"error: {err.max().item()}")

