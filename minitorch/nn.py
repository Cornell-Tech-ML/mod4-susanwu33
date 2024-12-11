from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.

    # Compute the new dimensions
    new_height = height // kh
    new_width = width // kw

    input = input.contiguous()

    # Reshape the input tensor for tiling
    tiled = (
        input.view(batch, channel, new_height, kh, new_width, kw)
        # Rearrange dimensions to group the kernel region
        .permute(0, 1, 2, 4, 3, 5)
        # Flatten the kernel region
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw)
    )
    return tiled, new_height, new_width


max_reduce = FastOps.reduce(operators.max, -1e9)


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling on the input tensor.

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: Tuple of pooling kernel size (height, width)

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    tiled, new_height, new_width = tile(input, kernel)
    return tiled.mean(-1).view(input.shape[0], input.shape[1], new_height, new_width)


def argmax(input: Tensor, dim: Tensor) -> Tensor:
    """Compute the argmax as a 1-hot tensor along a specified dimension."""
    return input == max_reduce(input, int(dim.item()))


class Max(Function):
    """Custom Function for max operator with autograd support."""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for the Max function.

        Args:
        ----
            ctx: The context object to save information for backward pass.
            input: Input tensor for which to compute the max.
            dim: Dimension along which to compute the max.

        Returns:
        -------
            Tensor containing the max values along the specified dimension.

        """
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the Max function.

        Args:
        ----
            ctx: The context object containing saved values from the forward pass.
            grad_output: Gradient of the loss with respect to the output of the Max function.

        Returns:
        -------
            Tuple of gradients with respect to the input tensor and the dimension.

        """
        input, dim = ctx.saved_values
        return (argmax(input, dim)) * grad_output, dim


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max value and indices along a given dimension."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor."""
    a = input.exp()
    return a / a.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor."""
    max_vals = max(input, dim)
    logsumexp = (input - max_vals).exp().sum(dim).log()
    return input - max_vals - logsumexp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling on the input tensor.

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: Tuple of pooling kernel size (height, width)

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    # Reshape the input tensor for pooling
    tiled, new_height, new_width = tile(input, kernel)

    # Use max_reduce to compute the max over the kernel region
    max_pooled = max_reduce(tiled, -1)

    # Reshape the result to match the pooled dimensions
    return max_pooled.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout only if input is a Tensor."""
    if ignore or rate <= 0.0:
        # If ignore is True or rate is 0, return the input unchanged
        return input
    if rate >= 1.0:
        # If rate is 1, all values are dropped (result is zero tensor)
        return input * 0

    # Generate random noise and create a binary mask
    mask = rand(input.shape) > rate
    # Apply the mask and scale the output to maintain expected value
    return input * mask / (1 - rate)
