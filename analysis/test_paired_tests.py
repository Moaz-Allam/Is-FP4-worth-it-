import math

import pytest

import paired_tests


def test_paired_difference_of_known_values():
    result = paired_tests.paired_difference([3.0, 5.0, 8.0], [1.0, 2.0, 4.0])
    assert result["mean"] == pytest.approx(3.0)
    assert result["standard_error"] == pytest.approx(1.0 / math.sqrt(3))
    assert result["z"] == pytest.approx(3.0 * math.sqrt(3))


def test_identical_lists_give_zero_without_dividing_by_zero():
    assert paired_tests.paired_difference([1.0, 2.0], [1.0, 2.0]) == {"mean": 0.0, "standard_error": 0.0, "z": 0.0}


def test_lists_of_different_length_are_rejected():
    with pytest.raises(ValueError):
        paired_tests.paired_difference([1.0, 2.0], [1.0])
