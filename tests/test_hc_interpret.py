"""Tests for hc_interpret.py's argument reading and result formatting. Both
must agree with the generated C harness, because full_compiler.sh compares
the interpreter's printed result with the executable's as text. One test
covers the command line hc_interpret.py shares with hc_main.py.
"""
import pytest

from conftest import parse, _entry_fn
from hc_interpret import args_from_argv, format_result
from hc_main import parse_args


# One scalar and one 2x2 tensor: three of the harness's argument cases.
FUNC = parse("""
func.func @my_func(%s: i32, %t: tensor<2x2xi32>) -> i32 {
  func.return %s : i32
}""")


@pytest.mark.parametrize("values, expected", [
    ([7, 1, 2, 3, 4], [7, [[1, 2], [3, 4]]]),      # every value given
    ([7, 1], [7, [[1, 12], [13, 14]]]),            # the rest use the defaults
    ([], [10, [[11, 12], [13, 14]]]),              # 10*(i+1) and 10*i + k + 1
])
def test_args_are_read_in_order_then_defaulted(values, expected):
    assert args_from_argv(_entry_fn(FUNC), values) == expected


def test_too_many_argument_values_are_rejected():
    with pytest.raises(ValueError, match="takes only 5"):
        args_from_argv(_entry_fn(FUNC), [1, 2, 3, 4, 5, 6])


@pytest.mark.parametrize("value, expected", [
    (450, "450"),
    ([0, 2, 27], "[0, 2, 27]"),
    ([[1, 2], [3, 4]], "[1, 2]\n[3, 4]"),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], "[1, 2]\n[3, 4]\n[5, 6]\n[7, 8]"),
])
def test_result_is_formatted_like_the_harness_prints_it(value, expected):
    assert format_result(value) == expected


def test_command_line_has_flags_then_model_then_values():
    """The loop-pass flags come first. A negative value is an argument value,
    not a flag."""
    options = parse_args(["--interchange", "--tile", "2", "--vectorize", "4", "m.onnx", "7", "-5"])
    assert (options.interchange, options.tile, options.vectorize) == (True, 2, 4)
    assert (options.model, options.values) == ("m.onnx", [7, -5])
