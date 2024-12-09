# 这是附加题模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现triton的深度学习推理过程，请严格保持输出格式输出
import os
from typing import Any

import h5py
import time
import numpy as np
import torch
import triton
import triton.language as tl


def read_params(dir):
    # 列出所有txt文件
    files = [f for f in os.listdir(dir) if f.endswith('.txt')]
    params = {}
    for fileName in files:
        data = []
        with open(os.path.join(dir, fileName), 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                value = float(line)
                data.append(value)
        modelName = fileName.replace(".txt", "")
        params[modelName] = data
    return params


def read_h5_file(dataPath):
    list_of_points = []
    list_of_labels = []
    with h5py.File(dataPath, "r") as hf:
        for k in hf.keys():
            # list_of_points.append(hf[k]["points"][:].astype(np.float32)) #每个points是（N,3）的二维数组ndarray
            list_of_points.append(hf[k]["points"][:].astype(np.float32).flatten())  # 每个points是N*3的一维ndarray
            list_of_labels.append(hf[k].attrs["label"])
    return list_of_points, list_of_labels


# 示例triton函数
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector`.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.`
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output






#### triton矩阵乘法函数
@triton.autotune(  ## 自动调优配置
    configs=[triton.Config({"BLOCK_SIZE": size}) for size in ( 32, 64, 128)],
    key=["m", "n", "k"],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, m, n, k, BLOCK_SIZE: tl.constexpr):
    # A: (m, k), B: (k, n), C: (m, n)
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offsets_m = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_n = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 初始化 C 的结果块
    c = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # 遍历 K 维度（分块加载 A 和 B）
    for block_id_k in range(0, tl.cdiv(k, BLOCK_SIZE)):
        offsets_k = block_id_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a_ptrs = a_ptr + offsets_m[:, None] * k + offsets_k[None, :]
        b_ptrs = b_ptr + offsets_k[:, None] * n + offsets_n[None, :]

        a = tl.load(a_ptrs, mask=(offsets_m[:, None] < m) & (offsets_k[None, :] < k), other=0.0)
        b = tl.load(b_ptrs, mask=(offsets_k[:, None] < k) & (offsets_n[None, :] < n), other=0.0)

        c += tl.dot(a, b)

    # 存储 C 的结果分块
    c_ptrs = c_ptr + offsets_m[:, None] * n + offsets_n[None, :]
    tl.store(c_ptrs, c, mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n))
def matmul(a: torch.Tensor, b: torch.Tensor):
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (
            triton.cdiv(a.shape[0], meta["BLOCK_SIZE"]),
            triton.cdiv(b.shape[1], meta["BLOCK_SIZE"]),
        )

    matmul_kernel[grid](a, b, out, a.shape[0], b.shape[1], a.shape[1])
    return out




# triton卷积函数
# @triton.autotune(  ## 自动调优配置
#     configs=[triton.Config({"BLOCK_SIZE": size}) for size in ( 32, 64, 128)],
#     key=["in_channels", "out_channels", "num_points"],
# )
# @triton.jit
# def convolution_kernel(
#     input_ptr,        # 输入张量的指针
#     output_ptr,       # 输出张量的指针
#     weight_ptr,       # 权重张量的指针
#     bias_ptr,         # 偏置张量的指针
#     num_points,       # 点的数量
#     in_channels,      # 输入通道数
#     out_channels,     # 输出通道数
#     BLOCK_SIZE: tl.constexpr,  # 并行计算的块大小
# ):
#     pid0 = tl.program_id(0)  # 输出的行
#     pid1 = tl.program_id(1)  # 输出的列
#
#     # 偏移量（行列索引）
#     offsets_row = pid0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)#第几行
#     offsets_col = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)#第几列
#
#     # 初始化 output 的结果块
#     output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
#
#     # 遍历输入通道，计算卷积
#     for block_id_in_channels in range(0, tl.cdiv(in_channels, BLOCK_SIZE)):
#         offsets_in_channels = block_id_in_channels * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#
#         # 计算权重和输入的偏移量
#         weight_ptrs = weight_ptr + offsets_row[:, None] * in_channels + offsets_in_channels[None, :]
#         input_ptrs = input_ptr + offsets_in_channels[:, None] * num_points + offsets_col[None, :]
#
#         # 加载权重和输入数据
#         weight = tl.load(weight_ptrs, mask=(offsets_row[:, None] < out_channels) & (offsets_in_channels[None, :] < in_channels), other=0.0)
#         input_data = tl.load(input_ptrs, mask=(offsets_in_channels[:, None] < in_channels) & (offsets_col[None, :] < num_points), other=0.0)
#
#         # 执行乘法并累加到 output
#         output += tl.dot(weight, input_data)
#
#     # 加上偏置
#     bias = tl.load(bias_ptr + offsets_row, mask=(offsets_row < out_channels), other=0.0)
#     output += bias[:, None]
#
#     # 将输出写回到 output_ptr
#     output_ptrs = output_ptr + offsets_row[:, None] * num_points + offsets_col[None, :]
#     tl.store(output_ptrs, output, mask=(offsets_row[:, None] < out_channels) & (offsets_col[None, :] < num_points))
#
# # 网格和块配置
# def convolution(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,num_points: int, in_channels: int, out_channels: int):
#
#
#     out = torch.empty((out_channels, num_points), device=input.device, dtype=input.dtype)
#
#     def grid(meta):
#         return (
#             triton.cdiv(out_channels, meta["BLOCK_SIZE"]),
#             triton.cdiv(num_points, meta["BLOCK_SIZE"]),
#         )
#
#     convolution_kernel[grid](input, out, weight, bias, num_points, in_channels, out_channels)
#     return out


@triton.jit
def triton_convolution_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,
                         M: int, N: int, K: int,
                         stride_am: int, stride_ak: int,
                         stride_bk: int, stride_bn: int,
                         stride_cm: int, stride_cn: int,
                         stride_bias: int,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                         ):

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n


    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # 加载偏置
    bias_ptrs = bias_ptr + offs_am * stride_bias
    bias = tl.load(bias_ptrs, mask=offs_am < M, other=0)


    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # GEMM loop:

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        #  load of A and B:
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(c_ptr.type.element_ty)


    # Add BIAS:
    c += bias[:, None]

    # Compute C pointers and store C:
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# 网格和块配置
def convolution(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):

    # 定义块大小
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    M: int
    N: int
    K: int
    M, K = a.shape
    _, N = b.shape

    c: torch.Tensor = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(args: dict[str, Any]) -> tuple[int]:
        return (triton.cdiv(M, args["BLOCK_SIZE_M"]) * triton.cdiv(N, args["BLOCK_SIZE_N"]), )

    triton_convolution_kernel[grid](a,b,c,bias,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1),bias.stride(0),BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K)

    return c






# triton批归一化函数
@triton.jit
def batch_norm_kernel(
        input_ptr,  # 输入指针
        output_ptr,  # 输出指针
        weight_ptr,  # 权重指针
        bias_ptr,  # 偏置指针
        running_mean_ptr,  # 运行均值指针
        running_var_ptr,  # 运行方差指针
        channels,  # 通道数
        num_points,  # 点的数量
        BLOCK_SIZE: tl.constexpr,  # 块大小
):
    pid = tl.program_id(0)  # 输出的行

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_row = offsets // num_points  # 第几行


    # 加载均值、方差、权重和偏置
    mean = tl.load(running_mean_ptr + offsets_row, mask=(offsets_row < channels), other=0.0)
    var = tl.load(running_var_ptr + offsets_row, mask=(offsets_row < channels), other=0.0)
    weight = tl.load(weight_ptr + offsets_row, mask=(offsets_row < channels), other=1.0)
    bias = tl.load(bias_ptr + offsets_row, mask=(offsets_row < channels), other=0.0)

    # 加载输入张量 x
    input = tl.load(input_ptr + offsets, mask=(offsets < num_points * channels), other=0.0)


    output = ((input - mean) / tl.sqrt(var + 1e-5)) * weight + bias

    # 将结果存储到输出张量
    tl.store(output_ptr + offsets, output, mask=(offsets < num_points * channels))


def batchnorm(input: torch.Tensor, weight:torch.Tensor, bias:torch.Tensor, mean:torch.Tensor, var:torch.Tensor, channels:int, num_points:int):
    # 定义块大小
    BLOCK_SIZE = 1024

    # 计算 grid 的大小
    grid = lambda meta: (triton.cdiv(channels * num_points, meta['BLOCK_SIZE']),)

    # 启动 Triton 核心进行批归一化计算
    out = torch.empty((channels, num_points), device=input.device, dtype=input.dtype)
    batch_norm_kernel[grid](input, out, weight, bias, mean, var, channels, num_points, BLOCK_SIZE=BLOCK_SIZE)
    return out

# triton Relu激活函数
@triton.jit
def relu_kernel(x_ptr,
                output_ptr,
                n_elements,
                BLOCK_SIZE: tl.constexpr,
                ):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets, mask=(offsets < n_elements), other=0.0)

    output = tl.maximum(x, 0)

    tl.store(output_ptr + offsets, output, mask=(offsets < n_elements))


def relu(x: torch.Tensor):

    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=128)
    return output


# triton maxpool函数
@triton.jit
def maxpool_kernel(
        input_ptr,  # 输入指针
        output_ptr,  # 输出指针
        channels,  # 通道数
        num_points,  # 点的数量
        BLOCK_SIZE: tl.constexpr  # 块大小
):
    # 获取当前线程块的 ID
    pid = tl.program_id(0)
    offsets_row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) // num_points  # 第几行


    for row in range(channels):
        max_value = -float('inf')
        for offset in range(0, num_points, BLOCK_SIZE):
            offsets = row * num_points + offset + tl.arange(0, BLOCK_SIZE)
            input_data = tl.load(input_ptr + offsets, mask=(offsets < channels * num_points), other=-float('inf'))
            # 遍历当前线程块内的点，找出最大值
            if tl.max(input_data) > max_value:
                max_value = tl.max(input_data)

        tl.store(output_ptr + row, max_value)

def maxpool(x: torch.Tensor,channels:int, num_points:int):
    # 定义块大小
    BLOCK_SIZE = 1024


    # 计算 grid 的大小
    grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SIZE']),)

    out = torch.empty((channels, 1), device=x.device, dtype=x.dtype)
    maxpool_kernel[grid](x, out, channels, num_points, BLOCK_SIZE=BLOCK_SIZE)
    return out

# triton全连接函数
@triton.jit
def fc_kernel(input_ptr, output_ptr, weight_ptr, bias_ptr, in_channels, out_channels, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 初始化输出为偏置
    output_value = tl.load(bias_ptr + offsets, mask=(offsets < out_channels), other=0.0)

    for i in range(in_channels):
        input_data = tl.load(input_ptr + i,mask=(i<in_channels), other=0.0)

        w_offsets = offsets * in_channels + i
        weight_data = tl.load(weight_ptr + w_offsets, mask=(w_offsets < in_channels*out_channels), other=0.0)

        output_value += input_data * weight_data
    # 将结果存储到输出
    tl.store(output_ptr + offsets, output_value)

def fc(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,in_channels: int, out_channels: int):
    # 定义块大小
    BLOCK_SIZE = 256

    # 计算 grid 的大小
    grid = lambda meta: (triton.cdiv(out_channels, meta['BLOCK_SIZE']),)

    # 启动 Triton 核心进行全连接计算
    out = torch.empty(out_channels,1 ,device=input.device, dtype=input.dtype)
    fc_kernel[grid](input, out, weight, bias, in_channels, out_channels, BLOCK_SIZE=BLOCK_SIZE)
    return out



# triton Iden仿射变换
@triton.jit
def iden_kernel(input_ptr, output_ptr, out_h, out_w, BLOCK_SIZE: tl.constexpr):
    # 获取程序的ID，这里是用来处理矩阵中的每个元素
    point_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 计算当前点的行和列
    row = point_idx // out_w
    col = point_idx % out_w

    input_data = tl.load(input_ptr + point_idx, mask=(point_idx < out_h * out_w), other=0.0)
    # 仅对主对角线元素进行加1
    output_data = tl.where(row == col, input_data + 1, input_data)

    tl.store(output_ptr + point_idx, output_data, mask=(point_idx < out_h * out_w))

def iden(x: torch.Tensor, out_h: int, out_w: int):
    # 定义块大小
    BLOCK_SIZE = 128
    # 计算 grid 的大小
    grid = lambda meta: (triton.cdiv(out_h * out_w, meta['BLOCK_SIZE']),)
    out = torch.empty(out_h, out_w, device=x.device, dtype=x.dtype)
    iden_kernel[grid](x,out, out_h, out_w, BLOCK_SIZE=BLOCK_SIZE)
    return out


# def downsample_points(points, target_num_points=1024):
#     num_points = points.shape[0]
#
#     if num_points <= target_num_points:
#         # 如果原始点数小于目标点数，直接返回原数据
#         return points
#
#     # 计算均匀的采样间隔
#     step = num_points // target_num_points
#
#     # 选择步长内的点
#     sampled_points = points[::step][:target_num_points]  # 采样并截断多余的点
#
#     return sampled_points

def downsample_points(points, target_num_points=1024):
    num_points = points.shape[0]

    if num_points <= target_num_points:
        # 如果原始点数小于目标点数，直接返回原数据
        return points

    # 随机选择 target_num_points 个点
    indices = np.random.choice(num_points, target_num_points, replace=False)
    sampled_points = points[indices]

    return sampled_points





def do_inference(list_of_points, list_of_labels, params):  # 请在本函数下使用triton实现推理操作


    # 加载模型参数
    feat_stn_conv1_weight = torch.tensor(params["feat.stn.conv1.weight"]).reshape(64, 3).cuda()
    feat_stn_conv1_bias = torch.tensor(params["feat.stn.conv1.bias"]).reshape(64, 1).cuda()
    feat_stn_bn1_weight = torch.tensor(params["feat.stn.bn1.weight"]).reshape(64,1).cuda()
    feat_stn_bn1_bias = torch.tensor(params["feat.stn.bn1.bias"]).reshape(64,1).cuda()
    feat_stn_bn1_mean = torch.tensor(params["feat.stn.bn1.running_mean"]).reshape(64,1).cuda()
    feat_stn_bn1_var = torch.tensor(params["feat.stn.bn1.running_var"]).reshape(64,1).cuda()

    feat_stn_conv2_weight = torch.tensor(params["feat.stn.conv2.weight"]).reshape(128, 64).cuda()
    feat_stn_conv2_bias = torch.tensor(params["feat.stn.conv2.bias"]).reshape(128, 1).cuda()
    feat_stn_bn2_weight = torch.tensor(params["feat.stn.bn2.weight"]).reshape(128,1).cuda()
    feat_stn_bn2_bias = torch.tensor(params["feat.stn.bn2.bias"]).reshape(128,1).cuda()
    feat_stn_bn2_mean = torch.tensor(params["feat.stn.bn2.running_mean"]).reshape(128,1).cuda()
    feat_stn_bn2_var = torch.tensor(params["feat.stn.bn2.running_var"]).reshape(128,1).cuda()

    feat_stn_conv3_weight = torch.tensor(params["feat.stn.conv3.weight"]).reshape(1024, 128).cuda()
    feat_stn_conv3_bias = torch.tensor(params["feat.stn.conv3.bias"]).reshape(1024, 1).cuda()
    feat_stn_bn3_weight = torch.tensor(params["feat.stn.bn3.weight"]).reshape(1024,1).cuda()
    feat_stn_bn3_bias = torch.tensor(params["feat.stn.bn3.bias"]).reshape(1024,1).cuda()
    feat_stn_bn3_mean = torch.tensor(params["feat.stn.bn3.running_mean"]).reshape(1024,1).cuda()
    feat_stn_bn3_var = torch.tensor(params["feat.stn.bn3.running_var"]).reshape(1024,1).cuda()

    feat_stn_fc1_weight = torch.tensor(params["feat.stn.fc1.weight"]).reshape(512, 1024).cuda()
    feat_stn_fc1_bias = torch.tensor(params["feat.stn.fc1.bias"]).reshape(512,1).cuda()
    feat_stn_bn4_weight = torch.tensor(params["feat.stn.bn4.weight"]).reshape(512,1).cuda()
    feat_stn_bn4_bias = torch.tensor(params["feat.stn.bn4.bias"]).reshape(512,1).cuda()
    feat_stn_bn4_mean = torch.tensor(params["feat.stn.bn4.running_mean"]).reshape(512,1).cuda()
    feat_stn_bn4_var = torch.tensor(params["feat.stn.bn4.running_var"]).reshape(512,1).cuda()

    feat_stn_fc2_weight = torch.tensor(params["feat.stn.fc2.weight"]).reshape(256, 512).cuda()
    feat_stn_fc2_bias = torch.tensor(params["feat.stn.fc2.bias"]).reshape(256,1).cuda()
    feat_stn_bn5_weight = torch.tensor(params["feat.stn.bn5.weight"]).reshape(256,1).cuda()
    feat_stn_bn5_bias = torch.tensor(params["feat.stn.bn5.bias"]).reshape(256,1).cuda()
    feat_stn_bn5_mean = torch.tensor(params["feat.stn.bn5.running_mean"]).reshape(256,1).cuda()
    feat_stn_bn5_var = torch.tensor(params["feat.stn.bn5.running_var"]).reshape(256,1).cuda()

    feat_stn_fc3_weight = torch.tensor(params["feat.stn.fc3.weight"]).reshape(9, 256).cuda()
    feat_stn_fc3_bias = torch.tensor(params["feat.stn.fc3.bias"]).reshape(9,1).cuda()

    feat_conv1_weight = torch.tensor(params["feat.conv1.weight"]).reshape(64, 3).cuda()
    feat_conv1_bias = torch.tensor(params["feat.conv1.bias"]).reshape(64, 1).cuda()
    feat_bn1_weight = torch.tensor(params["feat.bn1.weight"]).reshape(64,1).cuda()
    feat_bn1_bias = torch.tensor(params["feat.bn1.bias"]).reshape(64,1).cuda()
    feat_bn1_mean = torch.tensor(params["feat.bn1.running_mean"]).reshape(64,1).cuda()
    feat_bn1_var = torch.tensor(params["feat.bn1.running_var"]).reshape(64,1).cuda()

    feat_fstn_conv1_weight = torch.tensor(params["feat.fstn.conv1.weight"]).reshape(64, 64).cuda()
    feat_fstn_conv1_bias = torch.tensor(params["feat.fstn.conv1.bias"]).reshape(64,1).cuda()
    feat_fstn_bn1_weight = torch.tensor(params["feat.fstn.bn1.weight"]).reshape(64,1).cuda()
    feat_fstn_bn1_bias = torch.tensor(params["feat.fstn.bn1.bias"]).reshape(64,1).cuda()
    feat_fstn_bn1_mean = torch.tensor(params["feat.fstn.bn1.running_mean"]).reshape(64,1).cuda()
    feat_fstn_bn1_var = torch.tensor(params["feat.fstn.bn1.running_var"]).reshape(64,1).cuda()

    feat_fstn_conv2_weight = torch.tensor(params["feat.fstn.conv2.weight"]).reshape(128, 64).cuda()
    feat_fstn_conv2_bias = torch.tensor(params["feat.fstn.conv2.bias"]).reshape(128,1).cuda()
    feat_fstn_bn2_weight = torch.tensor(params["feat.fstn.bn2.weight"]).reshape(128,1).cuda()
    feat_fstn_bn2_bias = torch.tensor(params["feat.fstn.bn2.bias"]).reshape(128,1).cuda()
    feat_fstn_bn2_mean = torch.tensor(params["feat.fstn.bn2.running_mean"]).reshape(128,1).cuda()
    feat_fstn_bn2_var = torch.tensor(params["feat.fstn.bn2.running_var"]).reshape(128,1).cuda()

    feat_fstn_conv3_weight = torch.tensor(params["feat.fstn.conv3.weight"]).reshape(1024, 128).cuda()
    feat_fstn_conv3_bias = torch.tensor(params["feat.fstn.conv3.bias"]).reshape(1024,1).cuda()
    feat_fstn_bn3_weight = torch.tensor(params["feat.fstn.bn3.weight"]).reshape(1024,1).cuda()
    feat_fstn_bn3_bias = torch.tensor(params["feat.fstn.bn3.bias"]).reshape(1024,1).cuda()
    feat_fstn_bn3_mean = torch.tensor(params["feat.fstn.bn3.running_mean"]).reshape(1024,1).cuda()
    feat_fstn_bn3_var = torch.tensor(params["feat.fstn.bn3.running_var"]).reshape(1024,1).cuda()

    feat_fstn_fc1_weight = torch.tensor(params["feat.fstn.fc1.weight"]).reshape(512, 1024).cuda()
    feat_fstn_fc1_bias = torch.tensor(params["feat.fstn.fc1.bias"]).reshape(512,1).cuda()
    feat_fstn_bn4_weight = torch.tensor(params["feat.fstn.bn4.weight"]).reshape(512,1).cuda()
    feat_fstn_bn4_bias = torch.tensor(params["feat.fstn.bn4.bias"]).reshape(512,1).cuda()
    feat_fstn_bn4_mean = torch.tensor(params["feat.fstn.bn4.running_mean"]).reshape(512,1).cuda()
    feat_fstn_bn4_var = torch.tensor(params["feat.fstn.bn4.running_var"]).reshape(512,1).cuda()

    feat_fstn_fc2_weight = torch.tensor(params["feat.fstn.fc2.weight"]).reshape(256, 512).cuda()
    feat_fstn_fc2_bias = torch.tensor(params["feat.fstn.fc2.bias"]).reshape(256,1).cuda()
    feat_fstn_bn5_weight = torch.tensor(params["feat.fstn.bn5.weight"]).reshape(256,1).cuda()
    feat_fstn_bn5_bias = torch.tensor(params["feat.fstn.bn5.bias"]).reshape(256,1).cuda()
    feat_fstn_bn5_mean = torch.tensor(params["feat.fstn.bn5.running_mean"]).reshape(256,1).cuda()
    feat_fstn_bn5_var = torch.tensor(params["feat.fstn.bn5.running_var"]).reshape(256,1).cuda()

    feat_fstn_fc3_weight = torch.tensor(params["feat.fstn.fc3.weight"]).reshape(4096, 256).cuda()
    feat_fstn_fc3_bias = torch.tensor(params["feat.fstn.fc3.bias"]).reshape(4096).cuda()

    feat_conv2_weight = torch.tensor(params["feat.conv2.weight"]).reshape(128, 64).cuda()
    feat_conv2_bias = torch.tensor(params["feat.conv2.bias"]).reshape(128,1).cuda()
    feat_bn2_weight = torch.tensor(params["feat.bn2.weight"]).reshape(128,1).cuda()
    feat_bn2_bias = torch.tensor(params["feat.bn2.bias"]).reshape(128,1).cuda()
    feat_bn2_mean = torch.tensor(params["feat.bn2.running_mean"]).reshape(128,1).cuda()
    feat_bn2_var = torch.tensor(params["feat.bn2.running_var"]).reshape(128,1).cuda()

    feat_conv3_weight = torch.tensor(params["feat.conv3.weight"]).reshape(1024, 128).cuda()
    feat_conv3_bias = torch.tensor(params["feat.conv3.bias"]).reshape(1024,1).cuda()
    feat_bn3_weight = torch.tensor(params["feat.bn3.weight"]).reshape(1024,1).cuda()
    feat_bn3_bias = torch.tensor(params["feat.bn3.bias"]).reshape(1024,1).cuda()
    feat_bn3_mean = torch.tensor(params["feat.bn3.running_mean"]).reshape(1024,1).cuda()
    feat_bn3_var = torch.tensor(params["feat.bn3.running_var"]).reshape(1024,1).cuda()

    fc1_weight = torch.tensor(params["fc1.weight"]).reshape(512, 1024).cuda()
    fc1_bias = torch.tensor(params["fc1.bias"]).reshape(512,1).cuda()
    bn1_weight = torch.tensor(params["bn1.weight"]).reshape(512,1).cuda()
    bn1_bias = torch.tensor(params["bn1.bias"]).reshape(512,1).cuda()
    bn1_mean = torch.tensor(params["bn1.running_mean"]).reshape(512,1).cuda()
    bn1_var = torch.tensor(params["bn1.running_var"]).reshape(512,1).cuda()

    fc2_weight = torch.tensor(params["fc2.weight"]).reshape(256, 512).cuda()
    fc2_bias = torch.tensor(params["fc2.bias"]).reshape(256,1).cuda()
    bn2_weight = torch.tensor(params["bn2.weight"]).reshape(256,1).cuda()
    bn2_bias = torch.tensor(params["bn2.bias"]).reshape(256,1).cuda()
    bn2_mean = torch.tensor(params["bn2.running_mean"]).reshape(256,1).cuda()
    bn2_var = torch.tensor(params["bn2.running_var"]).reshape(256,1).cuda()

    fc3_weight = torch.tensor(params["fc3.weight"]).reshape(10, 256).cuda()
    fc3_bias = torch.tensor(params["fc3.bias"]).reshape(10,1).cuda()



    n=0




    for i, points in enumerate(list_of_points):
        # TODO ...在这里实现利用triton对点云数据进行深度学习的推理过程，当然，你也可以改进for循环以使用batch推理提速...
        # 打印每一帧的数据，仅用于调试！

        # print(f"=========Points : {i}=========")
        # # print(f"type {type(points)},shape {points.shape}")
        # point_num = int(len(points) / 3)
        # print(f"point_num={point_num}")
        # for point_i in range(point_num):
        #     print(f"x:{points[point_i*3+0]} y:{points[point_i*3+1]} z:{points[point_i*3+2]}")
        # print(f"Label: {list_of_labels[i]}")






        num_points = int(len(points) / 3)
        points = torch.tensor(points).reshape(num_points, 3).cuda()

        downsampled_points = downsample_points(points, target_num_points=1024)
        num_points = 1024

        input=downsampled_points.cuda()

        T_input=input.t().contiguous()
        # 开始推理


        feat_stn_conv1_res=convolution(feat_stn_conv1_weight, T_input, feat_stn_conv1_bias)
        feat_stn_bn1_res=batchnorm(feat_stn_conv1_res, feat_stn_bn1_weight, feat_stn_bn1_bias, feat_stn_bn1_mean, feat_stn_bn1_var, 64,num_points)
        feat_stn_relu1_res=relu(feat_stn_bn1_res)
        feat_stn_conv2_res=convolution(feat_stn_conv2_weight, feat_stn_relu1_res, feat_stn_conv2_bias)
        feat_stn_bn2_res=batchnorm(feat_stn_conv2_res, feat_stn_bn2_weight, feat_stn_bn2_bias, feat_stn_bn2_mean, feat_stn_bn2_var, 128,num_points)
        feat_stn_relu2_res=relu(feat_stn_bn2_res)
        feat_stn_conv3_res=convolution(feat_stn_conv3_weight, feat_stn_relu2_res, feat_stn_conv3_bias)
        feat_stn_bn3_res=batchnorm(feat_stn_conv3_res, feat_stn_bn3_weight, feat_stn_bn3_bias, feat_stn_bn3_mean, feat_stn_bn3_var, 1024, num_points)
        feat_stn_relu3_res=relu(feat_stn_bn3_res)
        feat_stn_max_res=maxpool(feat_stn_relu3_res, 1024,num_points)
        feat_stn_fc1_res=fc(feat_stn_max_res, feat_stn_fc1_weight, feat_stn_fc1_bias,1024, 512)
        feat_stn_bn4_res=batchnorm(feat_stn_fc1_res, feat_stn_bn4_weight, feat_stn_bn4_bias, feat_stn_bn4_mean, feat_stn_bn4_var, 512,1)
        feat_stn_relu4_res=relu(feat_stn_bn4_res)
        feat_stn_fc2_res=fc(feat_stn_relu4_res, feat_stn_fc2_weight, feat_stn_fc2_bias,512, 256)
        feat_stn_bn5_res=batchnorm(feat_stn_fc2_res, feat_stn_bn5_weight, feat_stn_bn5_bias, feat_stn_bn5_mean, feat_stn_bn5_var, 256,1)
        feat_stn_relu5_res=relu(feat_stn_bn5_res)
        feat_stn_fc3_res=fc(feat_stn_relu5_res, feat_stn_fc3_weight, feat_stn_fc3_bias, 256, 9)
        feat_stn_fc3_res_reshape = feat_stn_fc3_res.reshape(3, 3)

        matmul1_res=matmul(input, feat_stn_fc3_res_reshape)
        feat_conv1_res=convolution(feat_conv1_weight, matmul1_res.t().contiguous(), feat_conv1_bias)
        feat_bn1_res=batchnorm(feat_conv1_res, feat_bn1_weight, feat_bn1_bias, feat_bn1_mean, feat_bn1_var, 64,num_points)
        feat_relu1_res=relu(feat_bn1_res)
        feat_fstn_conv1_res=convolution(feat_fstn_conv1_weight, feat_relu1_res, feat_fstn_conv1_bias)
        feat_fstn_bn1_res=batchnorm(feat_fstn_conv1_res, feat_fstn_bn1_weight, feat_fstn_bn1_bias, feat_fstn_bn1_mean, feat_fstn_bn1_var, 64,num_points)
        feat_fstn_relu1_res=relu(feat_fstn_bn1_res)
        feat_fstn_conv2_res=convolution(feat_fstn_conv2_weight, feat_fstn_relu1_res, feat_fstn_conv2_bias)
        feat_fstn_bn2_res=batchnorm(feat_fstn_conv2_res, feat_fstn_bn2_weight, feat_fstn_bn2_bias, feat_fstn_bn2_mean, feat_fstn_bn2_var, 128, num_points)
        feat_fstn_relu2_res=relu(feat_fstn_bn2_res)
        feat_fstn_cov3_res=convolution(feat_fstn_conv3_weight, feat_fstn_relu2_res, feat_fstn_conv3_bias)
        feat_fstn_bn3_res=batchnorm(feat_fstn_cov3_res, feat_fstn_bn3_weight, feat_fstn_bn3_bias, feat_fstn_bn3_mean, feat_fstn_bn3_var, 1024, num_points)
        feat_fstn_relu3_res=relu(feat_fstn_bn3_res)
        feat_fstn_max_res=maxpool(feat_fstn_relu3_res, 1024,num_points)
        feat_fstn_fc1_res=fc(feat_fstn_max_res, feat_fstn_fc1_weight, feat_fstn_fc1_bias,1024, 512)
        feat_fstn_bn4_res=batchnorm(feat_fstn_fc1_res, feat_fstn_bn4_weight, feat_fstn_bn4_bias, feat_fstn_bn4_mean, feat_fstn_bn4_var, 512,1)
        feat_fstn_relu4_res=relu(feat_fstn_bn4_res)
        feat_fstn_fc2_res=fc(feat_fstn_relu4_res, feat_fstn_fc2_weight, feat_fstn_fc2_bias,512, 256)
        feat_fstn_bn5_res=batchnorm(feat_fstn_fc2_res, feat_fstn_bn5_weight, feat_fstn_bn5_bias, feat_fstn_bn5_mean, feat_fstn_bn5_var, 256,1)
        feat_fstn_relu5_res=relu(feat_fstn_bn5_res)
        feat_fstn_fc3_res=fc(feat_fstn_relu5_res, feat_fstn_fc3_weight, feat_fstn_fc3_bias,256, 64*64)
        feat_fstn_iden_res=iden(feat_fstn_fc3_res, 64, 64)

        matmul2_res=matmul(feat_relu1_res.t().contiguous(), feat_fstn_iden_res)

        feat_conv2_res=convolution(feat_conv2_weight, matmul2_res.t().contiguous(), feat_conv2_bias)
        feat_bn2_res=batchnorm(feat_conv2_res, feat_bn2_weight, feat_bn2_bias, feat_bn2_mean, feat_bn2_var, 128,num_points)
        feat_relu2_res=relu(feat_bn2_res)
        feat_conv3_res=convolution(feat_conv3_weight, feat_relu2_res, feat_conv3_bias)
        feat_bn3_res=batchnorm(feat_conv3_res, feat_bn3_weight, feat_bn3_bias, feat_bn3_mean, feat_bn3_var, 1024, num_points)
        feat_max_res=maxpool(feat_bn3_res, 1024,num_points)
        fc1_res=fc(feat_max_res, fc1_weight, fc1_bias,1024, 512)
        bn1_res=batchnorm(fc1_res, bn1_weight, bn1_bias, bn1_mean, bn1_var, 512,1)
        relu1_res=relu(bn1_res)
        fc2_res=fc(relu1_res, fc2_weight, fc2_bias,512, 256)
        bn2_res=batchnorm(fc2_res, bn2_weight, bn2_bias, bn2_mean, bn2_var, 256,1)
        relu2_res=relu(bn2_res)
        fc3_res=fc(relu2_res, fc3_weight, fc3_bias, 256, 10)
        res=fc3_res.cpu().reshape(10)
        # 验证结果


        lable=0
        max=0
        for k in range(10):
            # print(res[k].data)
            if res[k].data>max:
                max=res[k].data
                lable=k
        if lable==list_of_labels[i]:
            # print(f"Correct! Label: {list_of_labels[i]}")
            n+=1







    accuracy_rate =n/len(list_of_labels)
    return accuracy_rate


if __name__ == '__main__':
    dir = os.path.dirname(__file__)  # 保存模型参数文件(.txt)的文件夹路径

    # 读取模型参数
    params = read_params(dir)
    # #示例，获取feat.stn.fc3.bias的参数
    # feat_stn_fc3_bias = params["feat.stn.fc3.bias"]
    # print(f"feat.stn.fc3.bias:")
    # for i in feat_stn_fc3_bias:
    #     print(i)

    # 读取训练集数据
    dataPath = "./data/test_point_clouds.h5"
    list_of_points, list_of_labels = read_h5_file(dataPath)

    # 开始计时
    start = time.time()
    accuracy_rate = do_inference(list_of_points, list_of_labels, params)
    # 结束计时
    end = time.time()
    ms = end - start

    # 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    print(f"{ms:.4f}:{accuracy_rate:.4f}")










