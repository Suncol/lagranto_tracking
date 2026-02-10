import numpy as np
import pytest

from lagranto_track import interpolate_4d_time, interpolate_time


def test_interpolate_4d_time_int_input_returns_float64_with_fraction():
	data = np.array(
		[
			[[[0]]],
			[[[10]]],
		],
		dtype=np.int32,
	)
	time_known = np.array([0.0, 1.0], dtype=np.float64)
	time_target = np.array([0.25, 0.5, 0.75], dtype=np.float64)

	out = interpolate_4d_time(data, time_known, time_target)
	assert out.dtype == np.float64
	assert out[:, 0, 0, 0] == pytest.approx([2.5, 5.0, 7.5])


def test_interpolate_time_rejects_non_monotonic_time_known():
	data = np.array([0.0, 0.0, 10.0], dtype=np.float64)
	time_known = np.array([0.0, 2.0, 1.0], dtype=np.float64)
	time_target = np.array([1.5], dtype=np.float64)

	with pytest.raises(ValueError, match='strictly increasing'):
		interpolate_time(data, time_known, time_target)


def test_interpolate_time_rejects_repeated_time_known():
	data = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
	time_known = np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64)
	time_target = np.array([0.5], dtype=np.float64)

	with pytest.raises(ValueError, match='strictly increasing'):
		interpolate_time(data, time_known, time_target)


def test_interpolate_4d_time_rejects_repeated_time_known():
	data = np.arange(4, dtype=np.float64).reshape(4, 1, 1, 1)
	time_known = np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64)
	time_target = np.array([0.5], dtype=np.float64)

	with pytest.raises(ValueError, match='strictly increasing'):
		interpolate_4d_time(data, time_known, time_target)


@pytest.mark.parametrize(
	'invalid_time_known',
	[
		np.array([0.0, np.nan, 2.0], dtype=np.float64),
		np.array([0.0, np.inf, 2.0], dtype=np.float64),
	],
)
def test_interpolate_time_rejects_non_finite_time_known(invalid_time_known):
	data = np.array([0.0, 10.0, 20.0], dtype=np.float64)
	time_target = np.array([0.5], dtype=np.float64)

	with pytest.raises(ValueError, match='finite'):
		interpolate_time(data, invalid_time_known, time_target)


def test_interpolate_4d_time_rejects_non_finite_time_known():
	data = np.arange(3, dtype=np.float64).reshape(3, 1, 1, 1)
	time_known = np.array([0.0, np.nan, 2.0], dtype=np.float64)
	time_target = np.array([0.5], dtype=np.float64)

	with pytest.raises(ValueError, match='finite'):
		interpolate_4d_time(data, time_known, time_target)


def test_interpolate_time_keeps_boundary_and_nan_target_behavior():
	data = np.array([0.0, 10.0, 20.0], dtype=np.float64)
	time_known = np.array([0.0, 1.0, 2.0], dtype=np.float64)
	time_target = np.array([-1.0, 0.0, 0.5, 1.5, 2.0, 3.0, np.nan], dtype=np.float64)

	out = interpolate_time(data, time_known, time_target)
	expected = np.array([0.0, 0.0, 5.0, 15.0, 20.0, 20.0, np.nan], dtype=np.float64)

	assert out[:-1] == pytest.approx(expected[:-1])
	assert np.isnan(out[-1])


def test_interpolate_4d_time_valid_float_input_non_regression_values():
	data = np.zeros((3, 1, 1, 2), dtype=np.float32)
	data[:, 0, 0, 0] = np.array([0.0, 10.0, 20.0], dtype=np.float32)
	data[:, 0, 0, 1] = np.array([100.0, 110.0, 120.0], dtype=np.float32)
	time_known = np.array([0.0, 1.0, 2.0], dtype=np.float64)
	time_target = np.array([0.5, 1.5], dtype=np.float64)

	out = interpolate_4d_time(data, time_known, time_target)
	expected = np.array(
		[
			[[[5.0, 105.0]]],
			[[[15.0, 115.0]]],
		],
		dtype=np.float64,
	)

	assert out.dtype == np.float64
	assert out == pytest.approx(expected)
