import pytest
import torch

import quant


def block_of_32(values):
    return torch.tensor(values + [0.0] * (32 - len(values)))


def test_e2m1_grid_is_exactly_the_eight_magnitudes():
    candidates = torch.linspace(-8, 8, 3201)
    produced = quant.round_to_e2m1(candidates).abs().unique().tolist()
    assert produced == list(quant.E2M1_GRID)


def test_e2m1_ties_go_to_the_even_mantissa():
    ties = torch.tensor([0.25, 0.75, 1.75, 2.5, 3.5, 5.0])
    assert quant.round_to_e2m1(ties).tolist() == [0.0, 1.0, 2.0, 2.0, 4.0, 4.0]


def test_homework3_question3_mxfp4_dot_product():
    a, _ = quant.quantize_mxfp4_floor(block_of_32([24.0, -12.0, 6.0, 2.0]))
    b, _ = quant.quantize_mxfp4_floor(block_of_32([0.25, 1.0, -0.5, 2.0]))
    assert (a[:4] / 4).tolist() == [6.0, -3.0, 1.5, 0.5]
    assert (b[:4] / 0.5).tolist() == [0.5, 2.0, -1.0, 4.0]
    assert torch.dot(a, b).item() == -5.0


def test_four_over_six_table2_nvfp4_block():
    block = torch.tensor([10.0, 20.0, 30.0, 40.0])
    dequantized, _ = quant.quantize_nvfp4_with_tensor_scale(block, tensor_scale=1.0)
    assert dequantized.tolist() == [9.75, 19.5, 26.0, 39.0]
    assert torch.mean((dequantized - block) ** 2).item() == pytest.approx(4.33, abs=0.005)


def test_mxfp4_floor_rule_clamps_a_maximum_between_6_and_8():
    dequantized, clamped = quant.quantize_mxfp4_floor(block_of_32([7.5, 1.0]))
    assert dequantized[0].item() == 6.0
    assert clamped.tolist()[:2] == [True, False]


def test_int4_levels_stay_in_range_and_reach_the_maximum():
    block = torch.linspace(-1, 1, 32)
    dequantized, clamped = quant.quantize_int4(block)
    assert dequantized.max().item() == pytest.approx(1.0, rel=1e-3)
    assert not clamped.any()


def test_all_zero_block_stays_zero_in_every_format():
    zeros = torch.zeros(2, 128)
    for weight_format in quant.FORMATS.values():
        dequantized, _ = quant.quantize_matrix(zeros, weight_format)
        assert torch.equal(dequantized, zeros), weight_format.name


def test_quantize_matrix_pads_inputs_that_do_not_fill_a_block():
    weight = torch.randn(3, 40, generator=torch.Generator().manual_seed(0))
    dequantized, clamped = quant.quantize_matrix(weight, quant.FORMATS["mxfp4"])
    assert dequantized.shape == clamped.shape == weight.shape


def scaled_maximum_after(quantize, maximum):
    dequantized, clamped = quantize(block_of_32([maximum, 1.0]))
    return dequantized[0].item(), bool(clamped[0])


def test_mxfp4_up_never_clamps():
    assert scaled_maximum_after(quant.quantize_mxfp4_up, 6.5) == (6.0, False)
    assert scaled_maximum_after(quant.quantize_mxfp4_up, 7.5) == (8.0, False)
    assert scaled_maximum_after(quant.quantize_mxfp4_up, 6.0) == (6.0, False)


def test_mxfp4_threshold_7_doubles_only_from_7():
    assert scaled_maximum_after(quant.quantize_mxfp4_threshold_7, 6.9) == (6.0, True)
    assert scaled_maximum_after(quant.quantize_mxfp4_threshold_7, 7.0) == (8.0, False)


def test_mxfp4_threshold_7_25_doubles_only_from_7_25():
    assert scaled_maximum_after(quant.quantize_mxfp4_threshold_7_25, 7.2) == (6.0, True)
    assert scaled_maximum_after(quant.quantize_mxfp4_threshold_7_25, 7.25) == (8.0, False)


def test_mxfp4_best_is_never_worse_than_floor_or_up():
    blocks = torch.randn(64, 32, generator=torch.Generator().manual_seed(1))
    error = lambda quantize: quant.block_squared_error(blocks, quantize(blocks)[0])
    best = error(quant.quantize_mxfp4_best)
    assert torch.all(best <= error(quant.quantize_mxfp4_floor))
    assert torch.all(best <= error(quant.quantize_mxfp4_up))


def test_mxfp4_three_quarters_keeps_the_maximum_below_6_steps():
    dequantized, clamped = quant.quantize_mxfp4_three_quarters(block_of_32([7.6, 2.6, -0.9, 0.2]))
    assert not clamped.any()
    assert dequantized[0].item() == pytest.approx(8.0)


def test_four_over_six_picks_the_exact_scale_for_table2_block():
    block = torch.tensor([10.0, 20.0, 30.0, 40.0])
    six = quant.quantize_nvfp4_with_tensor_scale(block, 1.0, 6.0)
    four = quant.quantize_nvfp4_with_tensor_scale(block, 1.0, 4.0)
    chosen, _ = quant.lower_error_choice(block, six, four)
    assert chosen.tolist() == [10.0, 20.0, 30.0, 40.0]


def test_scale_format_bits_and_range():
    scale_format = quant.ScaleFormat(exponent_bits=2, mantissa_bits=2)
    assert scale_format.bits == 4
    assert scale_format.min_value == 0.125
    assert scale_format.max_value == 1.75
    assert quant.scale_sweep_format(16, scale_format).bits_per_weight == 4.25


def test_scale_format_neighbours_bracket_the_value():
    scale_format = quant.ScaleFormat(exponent_bits=2, mantissa_bits=2)
    below, above = quant.scale_format_neighbours(torch.tensor([1.3, 0.3, 1.0]), scale_format)
    assert below.tolist() == [1.25, 0.25, 1.0]
    assert above.tolist() == [1.5, 0.3125, 1.0]


def test_scale_format_neighbours_stay_inside_the_representable_range():
    scale_format = quant.ScaleFormat(exponent_bits=1, mantissa_bits=1)
    below, above = quant.scale_format_neighbours(torch.tensor([0.01, 1.9]), scale_format)
    assert below.tolist() == [0.5, 1.5]
    assert above.tolist() == [0.5, 1.5]


def test_zero_bit_scale_is_one_scale_for_the_whole_tensor():
    blocks = torch.tensor([[[12.0, 3.0], [1.0, -0.4]]])
    dequantized, _ = quant.quantize_with_scale_format(blocks, quant.ScaleFormat(0, 0))
    expected, _ = quant.quantize_e2m1_with_scale(blocks, torch.tensor(2.0))
    assert torch.equal(dequantized, expected)


def test_scale_format_keeps_the_tensor_maximum_exact_and_zero_blocks_zero():
    blocks = torch.tensor([[[10.5, 1.0], [0.0, 0.0], [0.02, -0.01]]])
    for scale_format in quant.E5_SCALE_FORMATS:
        dequantized, clamped = quant.quantize_with_scale_format(blocks, scale_format)
        assert dequantized[0, 0, 0].item() == 10.5
        assert dequantized[0, 1].tolist() == [0.0, 0.0]
        assert not clamped[0, 0].any()


def test_finer_scale_lowers_the_error_of_a_small_block():
    blocks = torch.tensor([[[6.0, 1.0], [0.37, -0.11]]])
    coarse, _ = quant.quantize_with_scale_format(blocks, quant.ScaleFormat(0, 0))
    fine, _ = quant.quantize_with_scale_format(blocks, quant.ScaleFormat(4, 4))
    assert torch.sum((fine - blocks) ** 2) < torch.sum((coarse - blocks) ** 2)
