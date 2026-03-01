import math
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.nn import Linear


def constant_bias_init(value: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return an initializer that fills a tensor with a constant value."""

    def init_fn(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.constant_(tensor, value)

    return init_fn


def small_init(dim: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates an initialization function that fills a tensor with values sampled from a normal distribution **in_place**.
    The standard deviation is calculated based on the method described in the paper:
    "Transformers without Tears: Improving the Normalization of Self-Attention" by Nguyen, T. & Salazar, J. (2010).
    The formula for standard deviation (std) is:
        std = sqrt(2 / (5 * dim))
    Args:
        dim (int): The dimensionality of the tensor to be initialized. This typically corresponds to the
                   number of input features or the size of the hidden layer in a neural network.
    Returns:
        function: A function that takes a tensor as input and initializes it with a normal distribution
                  having mean 0.0 and the calculated standard deviation.
    """
    std = math.sqrt(2 / (5 * dim))

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def wang_init(dim: int, num_layers: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates an initialization function that fills a tensor with values sampled from a normal distribution
    based on Wang's initialization method  **in_place**.
    The standard deviation is calculated using the formula:
        std = 2 / (num_layers * sqrt(dim))
    This method accounts for the number of layers and the dimensionality to maintain appropriate scaling
    of weights, which can help in stabilizing training for deep neural networks.
    Args:
        dim (int): The dimensionality of the tensor to be initialized. Typically corresponds to the
                   number of input features or the size of the hidden layer.
        num_layers (int): The total number of layers in the neural network. This helps in scaling the
                        standard deviation appropriately as the network depth increases.
    Returns:
        function: A function that takes a tensor as input and initializes it with a normal distribution
                  having mean 0.0 and the calculated standard deviation.
    """
    std = 2 / num_layers / math.sqrt(dim)

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class CausalConv1D(torch.nn.Conv1d):
    """1D convolution with causal padding (left-only).
    This layer behaves like ``torch.nn.Conv1d`` but guarantees causality by
    padding only on the left with ``(kernel_size - 1) * dilation`` elements.
    Notes
    - Stride > 1 reduces the sequence length as in standard convs.
    - ``padding`` passed to the constructor is ignored in ``forward``; causality
      always uses computed left padding. ``padding_mode`` is respected.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Extract effective kernel size and dilation (ints)
        kernel_size = (
            self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
        )
        dilation = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation

        # Left-only padding for causality
        left_pad = (kernel_size - 1) * dilation

        # Perform convolution without additional padding (already applied)
        return F.conv1d(
            F.pad(input, (left_pad, 0)),
            self.weight,
            self.bias,
            self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )


class MLP(nn.Module):
    """A flexible Multi-Layer Perceptron that supports various activation functions including GLU variants.
    Always uses exactly two layers with equal input and output dimensions.
    """

    def __init__(
        self,
        dim: int,
        activation: Literal["relu", "gelu", "silu", "glu", "swiglu"],
        dropout: float = 0.0,
        expansion_factor: float = 2.0,
        bias: bool = False,
        init_method_in: Callable[[torch.Tensor], torch.Tensor] | None = None,
        init_method_out: Callable[[torch.Tensor], torch.Tensor] | None = None,
        bias_method_init_in: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.zeros_,
        bias_method_init_out: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.zeros_,
        out_dim: int | None = None,
    ):
        """Initialize the MLP.
        Args:
            dim (int): Input and output dimension.
            activation (Literal["relu", "gelu", "silu", "glu", "swiglu"]): Activation function to use.
            dropout (float): Dropout rate.
            expansion_factor (float): Factor to expand the hidden dimension.
            bias (bool): Whether to use bias in linear layers.
            init_method_in (Callable[[torch.Tensor], torch.Tensor]): Optional initialization method for the first layer.
            init_method_out (Callable[[torch.Tensor], torch.Tensor]): Optional initialization method for the second layer.
            out_dim (int | None): Optional output dimension. Defaults to ``dim`` when ``None``.
        """
        assert (
            isinstance(out_dim, int) or out_dim is None
        ), f"Output dimension must be an integer or None. Got {out_dim}."

        super().__init__()
        self.hidden_dim = int(dim * expansion_factor)
        self.output_dim = dim if out_dim is None else int(out_dim)
        self.activation = activation

        # Check if using GLU variants which require double width for hidden layers
        self.is_glu_variant = activation in ["glu", "swiglu"]
        glu_factor = 2 if self.is_glu_variant else 1

        # Construct first layer
        self.layer1 = Linear(dim, self.hidden_dim * glu_factor, bias=bias)
        if init_method_in is not None:
            init_method_in(self.layer1.weight)
        if self.layer1.bias is not None:
            bias_method_init_in(self.layer1.bias)

        # Construct dropout
        self.dropout = nn.Dropout(p=dropout)

        # Construct second layer
        self.layer2 = Linear(self.hidden_dim, self.output_dim, bias=bias)
        if init_method_out is not None:
            init_method_out(self.layer2.weight)
        if self.layer2.bias is not None:
            bias_method_init_out(self.layer2.bias)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation function to the input."""
        if self.activation in ["relu", "gelu", "silu"]:
            return getattr(F, self.activation)(x)
        elif self.activation == "glu":
            # Use torch's built-in GLU function
            return F.glu(x, dim=-1)
        elif self.activation == "swiglu":
            # SwiGLU: SiLU(x) * y
            a, b = torch.chunk(x, 2, dim=-1)
            return b * F.silu(a)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        # Apply first layer
        x = self.layer1(x)
        # Apply activation
        x = self._apply_activation(x)
        # Apply dropout
        x = self.dropout(x)
        # Second layer
        x = self.layer2(x)
        return x


class ConvFirstLayerMLP(nn.Module):
    """Two-layer MLP variant whose first layer is a 1D convolution."""

    def __init__(
        self,
        dim: int,
        activation: Literal["relu", "gelu", "silu", "glu", "swiglu"],
        kernel_size: int,
        dropout: float = 0.0,
        expansion_factor: float = 2.0,
        bias: bool = False,
        init_method_in: Callable[[torch.Tensor], torch.Tensor] | None = None,
        init_method_out: Callable[[torch.Tensor], torch.Tensor] | None = None,
        bias_method_init_in: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.zeros_,
        bias_method_init_out: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.zeros_,
        out_dim: int | None = None,
        causal: bool = True,
    ):
        """Initialize the ConvFirstLayerMLP.
        Args:
            dim (int): Input dimension.
            activation (Literal["relu", "gelu", "silu", "glu", "swiglu"]): Activation function to use.
            kernel_size (int): Kernel size for the convolution.
            dropout (float): Dropout rate.
            expansion_factor (float): Factor to expand the hidden dimension.
            bias (bool): Whether to use bias in linear layers.
            init_method_in (Callable[[torch.Tensor], torch.Tensor]): Optional initialization method for the first layer.
            init_method_out (Callable[[torch.Tensor], torch.Tensor]): Optional initialization method for the second layer.
            out_dim (int | None): Optional output dimension. Defaults to ``dim`` when ``None``.
            causal (bool): Whether to use causal padding. Defaults to True.
        """
        assert (
            isinstance(out_dim, int) or out_dim is None
        ), f"Output dimension must be an integer or None. Got {out_dim}."
        assert kernel_size > 0, f"Kernel size must be > 0, got {kernel_size}."

        super().__init__()
        self.input_dim = dim
        self.hidden_dim = int(dim * expansion_factor)
        self.output_dim = dim if out_dim is None else int(out_dim)
        self.activation = activation
        self.causal = causal
        # Check if using GLU variants which require double width for hidden layers
        self.is_glu_variant = activation in ["glu", "swiglu"]
        glu_factor = 2 if self.is_glu_variant else 1

        if init_method_in is None:
            init_method_in = small_init(self.input_dim)
        if init_method_out is None:
            init_method_out = wang_init(self.hidden_dim * glu_factor, 7)

        # Construct first layer
        if causal:
            self.conv1 = CausalConv1D(
                in_channels=dim,
                out_channels=self.hidden_dim * glu_factor,
                kernel_size=kernel_size,
                bias=bias,
            )
        else:
            padding = kernel_size // 2
            self.conv1 = nn.Conv1d(
                in_channels=dim,
                out_channels=self.hidden_dim * glu_factor,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )
        if init_method_in is not None:
            init_method_in(self.conv1.weight)
        if self.conv1.bias is not None:
            bias_method_init_in(self.conv1.bias)

        # Construct dropout
        self.dropout = nn.Dropout(p=dropout)

        # Construct second layer
        self.layer2 = Linear(self.hidden_dim, self.output_dim, bias=bias)
        if init_method_out is not None:
            init_method_out(self.layer2.weight)
        if self.layer2.bias is not None:
            bias_method_init_out(self.layer2.bias)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation function to the input."""
        if self.activation in ["relu", "gelu", "silu"]:
            return getattr(F, self.activation)(x)
        elif self.activation == "glu":
            return F.glu(x, dim=-1)
        elif self.activation == "swiglu":
            # SwiGLU: SiLU(x) * y
            a, b = torch.chunk(x, 2, dim=-1)
            return b * F.silu(a)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError("ConvFirstLayerMLP expects input with shape (batch, length, dim).")
        assert (
            x.shape[-1] == self.input_dim
        ), f"Input dimension must be {self.input_dim}, got {x.shape[-1]}."

        # Reorder to channels-first for the convolution while keeping batch dims intact.
        x_conv = x.transpose(-1, -2).contiguous()
        x_conv = self.conv1(x_conv)
        x_conv = x_conv.transpose(-1, -2).contiguous()
        x_conv = self._apply_activation(x_conv)
        x_conv = self.dropout(x_conv)
        x_out = self.layer2(x_conv)
        return x_out
