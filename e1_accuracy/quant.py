"""Simulated ("fake") 4-bit weight quantization: quantize, then convert straight back to float.

Every quantizer takes blocks shaped (..., block_size) and returns the dequantized
blocks plus a mask of the values that were clamped.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch

E2M1_MAX = 6.0
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX_EXPONENT = 2
E4M3_MAX = 448.0
E4M3_MIN_EXPONENT = -6
E4M3_MANTISSA_BITS = 3
E8M0_EXPONENT_RANGE = (-127, 127)
INT4_MIN, INT4_MAX = -8, 7
OCP_V1_1_DOUBLING_THRESHOLD = 7.0
MXATTENTION_DOUBLING_THRESHOLD = 7.25
TSENG_PRESCALE = 0.75
FOUR_OVER_SIX_LOW_TARGET = 4.0
FOUR_OVER_SIX_MAX_STORED_SCALE = 256.0

Quantized = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class WeightFormat:
    name: str
    block_size: int
    bits_per_weight: float
    quantize_blocks: Callable[[torch.Tensor], Quantized]


def quantize_matrix(weight: torch.Tensor, weight_format: WeightFormat) -> Quantized:
    """Quantize a (outputs, inputs) matrix in blocks that run along the input axis."""
    outputs, inputs = weight.shape
    padding = -inputs % weight_format.block_size
    padded = torch.nn.functional.pad(weight, (0, padding))
    blocks = padded.reshape(outputs, -1, weight_format.block_size)
    dequantized, clamped = weight_format.quantize_blocks(blocks)
    return (dequantized.reshape(outputs, -1)[:, :inputs],
            clamped.reshape(outputs, -1)[:, :inputs])


def quantize_fp16(blocks: torch.Tensor) -> Quantized:
    return blocks.half().to(blocks.dtype), torch.zeros_like(blocks, dtype=torch.bool)


def quantize_int4(blocks: torch.Tensor) -> Quantized:
    scale = (block_max(blocks) / INT4_MAX).half().to(blocks.dtype)
    scaled = blocks / nonzero(scale)
    levels = torch.round(scaled).clamp(INT4_MIN, INT4_MAX)
    return levels * scale, scaled.abs() > INT4_MAX + 0.5


def quantize_mxfp4_floor(blocks: torch.Tensor) -> Quantized:
    """OCP v1.0 rule: the power-of-two scale is rounded down, so the block maximum may clamp."""
    return quantize_e2m1_with_scale(blocks, mxfp4_floor_scale(blocks))


def quantize_mxfp4_up(blocks: torch.Tensor) -> Quantized:
    """Smallest power-of-two scale that puts the block maximum at or below 6: nothing clamps."""
    return quantize_mxfp4_doubling_where(blocks, scaled_block_max(blocks) > E2M1_MAX)


def quantize_mxfp4_threshold_7(blocks: torch.Tensor) -> Quantized:
    """OCP v1.1 Equation 4: double the scale only when the maximum would round up past 6."""
    return quantize_mxfp4_doubling_where(blocks, scaled_block_max(blocks) >= OCP_V1_1_DOUBLING_THRESHOLD)


def quantize_mxfp4_threshold_7_25(blocks: torch.Tensor) -> Quantized:
    return quantize_mxfp4_doubling_where(blocks, scaled_block_max(blocks) >= MXATTENTION_DOUBLING_THRESHOLD)


def quantize_mxfp4_best(blocks: torch.Tensor) -> Quantized:
    """Try the round-down scale and twice that scale; keep the lower-error one per block."""
    scale = mxfp4_floor_scale(blocks)
    return lower_error_choice(blocks, quantize_e2m1_with_scale(blocks, scale),
                              quantize_e2m1_with_scale(blocks, 2 * scale))


def quantize_mxfp4_three_quarters(blocks: torch.Tensor) -> Quantized:
    """Tseng et al.: shrink values by 3/4 before rounding so the maximum stays below 6, undo after."""
    scale = mxfp4_floor_scale(blocks) / TSENG_PRESCALE
    return quantize_e2m1_with_scale(blocks, scale)


def quantize_mxfp4_doubling_where(blocks: torch.Tensor, needs_doubling: torch.Tensor) -> Quantized:
    scale = mxfp4_floor_scale(blocks)
    return quantize_e2m1_with_scale(blocks, torch.where(needs_doubling, 2 * scale, scale))


def scaled_block_max(blocks: torch.Tensor) -> torch.Tensor:
    """The block maximum under the round-down scale: always in [4, 8)."""
    return block_max(blocks) / nonzero(mxfp4_floor_scale(blocks))


def mxfp4_floor_scale(blocks: torch.Tensor) -> torch.Tensor:
    exponent = floor_log2(block_max(blocks)) - E2M1_MAX_EXPONENT
    return torch.exp2(exponent.clamp(*E8M0_EXPONENT_RANGE))


def quantize_nvfp4(blocks: torch.Tensor) -> Quantized:
    tensor_scale = blocks.abs().max() / (E2M1_MAX * E4M3_MAX)
    return quantize_nvfp4_with_tensor_scale(blocks, nonzero(tensor_scale))


def quantize_nvfp4_four_over_six(blocks: torch.Tensor) -> Quantized:
    """Four Over Six: map the block maximum to 6 and to 4; keep the lower-error one per block."""
    tensor_scale = nonzero(blocks.abs().max() / (E2M1_MAX * FOUR_OVER_SIX_MAX_STORED_SCALE))
    return lower_error_choice(
        blocks,
        quantize_nvfp4_with_tensor_scale(blocks, tensor_scale, E2M1_MAX),
        quantize_nvfp4_with_tensor_scale(blocks, tensor_scale, FOUR_OVER_SIX_LOW_TARGET),
    )


def quantize_nvfp4_with_tensor_scale(blocks: torch.Tensor, tensor_scale, target: float = E2M1_MAX) -> Quantized:
    """The block scale maps the block maximum to the target and is stored in E4M3, relative to the tensor scale."""
    stored_scale = round_to_e4m3(block_max(blocks) / target / tensor_scale)
    return quantize_e2m1_with_scale(blocks, stored_scale * tensor_scale)


@dataclass(frozen=True)
class ScaleFormat:
    """An unsigned floating-point block scale with no sign bit.

    The largest exponent is 0: a per-tensor scale places the range, as in NVFP4.
    """
    exponent_bits: int
    mantissa_bits: int

    @property
    def bits(self) -> int:
        return self.exponent_bits + self.mantissa_bits

    @property
    def min_exponent(self) -> int:
        return 1 - 2 ** self.exponent_bits

    @property
    def min_value(self) -> float:
        return 2.0 ** self.min_exponent

    @property
    def max_value(self) -> float:
        return 2.0 - 2.0 ** -self.mantissa_bits


def quantize_with_scale_format(blocks: torch.Tensor, scale_format: ScaleFormat) -> Quantized:
    """E2M1 elements; the block scale is the stored-scale neighbour (below or above) with the lower error."""
    tensor_scale = nonzero(blocks.abs().max() / (E2M1_MAX * scale_format.max_value))
    below, above = scale_format_neighbours(block_max(blocks) / E2M1_MAX / tensor_scale, scale_format)
    return lower_error_choice(blocks,
                              quantize_e2m1_with_scale(blocks, below * tensor_scale),
                              quantize_e2m1_with_scale(blocks, above * tensor_scale))


def scale_format_neighbours(positive_values: torch.Tensor, scale_format: ScaleFormat):
    """The nearest representable scales at or below, and at or above, each value."""
    exponent = floor_log2(positive_values).clamp(scale_format.min_exponent, 0)
    step = torch.exp2(exponent - scale_format.mantissa_bits)
    steps = positive_values / step
    return ((torch.floor(steps) * step).clamp(scale_format.min_value, scale_format.max_value),
            (torch.ceil(steps) * step).clamp(scale_format.min_value, scale_format.max_value))


def lower_error_choice(blocks: torch.Tensor, first: Quantized, second: Quantized) -> Quantized:
    first_is_better = block_squared_error(blocks, first[0]) <= block_squared_error(blocks, second[0])
    return (torch.where(first_is_better, first[0], second[0]),
            torch.where(first_is_better, first[1], second[1]))


def block_squared_error(blocks: torch.Tensor, dequantized: torch.Tensor) -> torch.Tensor:
    return torch.sum((dequantized - blocks) ** 2, dim=-1, keepdim=True)


def quantize_e2m1_with_scale(blocks: torch.Tensor, scale: torch.Tensor) -> Quantized:
    scaled = blocks / nonzero(scale)
    return round_to_e2m1(scaled) * scale, scaled.abs() > E2M1_MAX


def round_to_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 value, ties to the even mantissa: steps of 0.5 below 2, 1 below 4, 2 up to 6."""
    magnitude = values.abs()
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    rounded = (torch.round(magnitude / step) * step).clamp(max=E2M1_MAX)
    return torch.sign(values) * rounded


def round_to_e4m3(positive_values: torch.Tensor) -> torch.Tensor:
    exponent = floor_log2(positive_values).clamp(min=E4M3_MIN_EXPONENT)
    step = torch.exp2(exponent - E4M3_MANTISSA_BITS)
    return (torch.round(positive_values / step) * step).clamp(max=E4M3_MAX)


def block_max(blocks: torch.Tensor) -> torch.Tensor:
    return blocks.abs().amax(dim=-1, keepdim=True)


def floor_log2(positive_values: torch.Tensor) -> torch.Tensor:
    """Exact floor(log2(x)) from the float's own exponent; zero maps to a very small exponent."""
    _, exponent = torch.frexp(positive_values)
    exact = (exponent - 1).to(positive_values.dtype)
    return torch.where(positive_values > 0, exact, torch.full_like(exact, E8M0_EXPONENT_RANGE[0]))


def nonzero(scale: torch.Tensor) -> torch.Tensor:
    """An all-zero block has scale 0; dividing by 1 instead keeps its values at 0."""
    return torch.where(scale == 0, torch.ones_like(scale), scale)


MXFP4_BITS = 4 + 8 / 32
NVFP4_BITS = 4 + 8 / 16

E1_FORMATS = (
    WeightFormat("fp16", 1, 16.0, quantize_fp16),
    WeightFormat("int4_g128", 128, 4 + 16 / 128, quantize_int4),
    WeightFormat("int4_g32", 32, 4 + 16 / 32, quantize_int4),
    WeightFormat("mxfp4", 32, MXFP4_BITS, quantize_mxfp4_floor),
    WeightFormat("nvfp4", 16, NVFP4_BITS, quantize_nvfp4),
)
E4_FORMATS = (
    WeightFormat("mxfp4_floor", 32, MXFP4_BITS, quantize_mxfp4_floor),
    WeightFormat("mxfp4_up", 32, MXFP4_BITS, quantize_mxfp4_up),
    WeightFormat("mxfp4_best", 32, MXFP4_BITS, quantize_mxfp4_best),
    WeightFormat("mxfp4_threshold_7", 32, MXFP4_BITS, quantize_mxfp4_threshold_7),
    WeightFormat("mxfp4_threshold_7_25", 32, MXFP4_BITS, quantize_mxfp4_threshold_7_25),
    WeightFormat("mxfp4_three_quarters", 32, MXFP4_BITS, quantize_mxfp4_three_quarters),
    WeightFormat("nvfp4", 16, NVFP4_BITS, quantize_nvfp4),
    WeightFormat("nvfp4_four_over_six", 16, NVFP4_BITS, quantize_nvfp4_four_over_six),
)
E5_BLOCK_SIZES = (8, 16, 32, 64)
E5_SCALE_FORMATS = (
    ScaleFormat(0, 0),
    ScaleFormat(1, 1), ScaleFormat(2, 0),
    ScaleFormat(2, 2), ScaleFormat(3, 1), ScaleFormat(4, 0),
    ScaleFormat(3, 3), ScaleFormat(4, 2),
    ScaleFormat(4, 4), ScaleFormat(5, 3),
)


def scale_sweep_format(block_size: int, scale_format: ScaleFormat) -> WeightFormat:
    name = f"e2m1_b{block_size}_scale_e{scale_format.exponent_bits}m{scale_format.mantissa_bits}"
    return WeightFormat(name, block_size, 4 + scale_format.bits / block_size,
                        partial(quantize_with_scale_format, scale_format=scale_format))


E5_FORMATS_BY_BLOCK = {
    block_size: tuple(scale_sweep_format(block_size, scale_format) for scale_format in E5_SCALE_FORMATS)
    for block_size in E5_BLOCK_SIZES
}
E5_CORE_BLOCK_SIZES = (16, 32)
E5_CORE_MIN_SCALE_BITS = 4
E5_CORE_FORMATS = tuple(
    scale_sweep_format(block_size, scale_format)
    for block_size in E5_CORE_BLOCK_SIZES
    for scale_format in E5_SCALE_FORMATS if scale_format.bits >= E5_CORE_MIN_SCALE_BITS
)
SMOKE_TEST_FORMATS = (E1_FORMATS[0], E5_CORE_FORMATS[0])
EXPERIMENTS = {
    "e1": E1_FORMATS,
    "e4": E4_FORMATS,
    "e5_core": E5_CORE_FORMATS,
    "smoke": SMOKE_TEST_FORMATS,
    **{f"e5_b{block_size}": formats for block_size, formats in E5_FORMATS_BY_BLOCK.items()},
}
FORMATS = {weight_format.name: weight_format
           for formats in EXPERIMENTS.values() for weight_format in formats}
