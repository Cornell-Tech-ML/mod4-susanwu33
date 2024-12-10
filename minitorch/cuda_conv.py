from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function for execution on the CUDA device.

    Args:
    ----
        fn: The function to be compiled.
        **kwargs: Additional keyword arguments for the JIT compiler.

    Returns:
    -------
        The JIT-compiled function that can be executed on the CUDA device.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Callable[..., FakeCUDAKernel], **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a function for execution on the CUDA device.

    Args:
    ----
        fn: The function to be compiled.
        **kwargs: Additional keyword arguments for the JIT compiler.

    Returns:
    -------
        A JIT-compiled function that can be executed on the CUDA device.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32

def _cuda_tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """CUDA kernel for 1D convolution.

    Args:
        out: Output storage.
        out_shape: Output shape.
        out_strides: Output strides.
        out_size: Total number of output elements.
        input: Input storage.
        input_shape: Input shape.
        input_strides: Input strides.
        weight: Weight storage.
        weight_shape: Weight shape.
        weight_strides: Weight strides.
        reverse: Whether to reverse the kernel during convolution.

    """
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx >= out_size:
        return

    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(idx, out_shape, out_index)
    out_pos = index_to_position(out_index, out_strides)

    batch, out_channels, out_width = out_shape
    _, in_channels, in_width = input_shape
    _, in_channels, kernel_width = weight_shape

    acc = 0.0
    for in_c in range(in_channels):
        for k in range(kernel_width):
            if reverse:
                iw = out_index[2] - k
            else:
                iw = out_index[2] + k

            if 0 <= iw < in_width:
                input_idx = (
                    out_index[0] * input_strides[0]
                    + in_c * input_strides[1]
                    + iw * input_strides[2]
                )
                weight_idx = (
                    out_index[1] * weight_strides[0]
                    + in_c * weight_strides[1]
                    + k * weight_strides[2]
                )
                acc += input[input_idx] * weight[weight_idx]

    out[out_pos] = acc


tensor_conv1d_cuda = cuda.jit()(_cuda_tensor_conv1d)


def _cuda_tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """CUDA kernel for 2D convolution.

    Args:
        out: Output storage.
        out_shape: Output shape.
        out_strides: Output strides.
        out_size: Total number of output elements.
        input: Input storage.
        input_shape: Input shape.
        input_strides: Input strides.
        weight: Weight storage.
        weight_shape: Weight shape.
        weight_strides: Weight strides.
        reverse: Whether to reverse the kernel during convolution.
        
    """
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx >= out_size:
        return

    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(idx, out_shape, out_index)
    out_pos = index_to_position(out_index, out_strides)

    batch, out_channels, out_height, out_width = out_shape
    _, in_channels, in_height, in_width = input_shape
    _, in_channels, kernel_height, kernel_width = weight_shape

    acc = 0.0
    for in_c in range(in_channels):
        for k_h in range(kernel_height):
            for k_w in range(kernel_width):
                if reverse:
                    ih = out_index[2] - k_h
                    iw = out_index[3] - k_w
                else:
                    ih = out_index[2] + k_h
                    iw = out_index[3] + k_w

                if 0 <= ih < in_height and 0 <= iw < in_width:
                    input_idx = (
                        out_index[0] * input_strides[0]
                        + in_c * input_strides[1]
                        + ih * input_strides[2]
                        + iw * input_strides[3]
                    )
                    weight_idx = (
                        out_index[1] * weight_strides[0]
                        + in_c * weight_strides[1]
                        + k_h * weight_strides[2]
                        + k_w * weight_strides[3]
                    )
                    acc += input[input_idx] * weight[weight_idx]

    out[out_pos] = acc


tensor_conv2d_cuda = cuda.jit()(_cuda_tensor_conv2d)
