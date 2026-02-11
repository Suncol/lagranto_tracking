import numpy as np
import pytest

from lagranto_track import (
	get_trace_time_midpoint,
	get_trace_time_midpoint_backward,
	sample_scalar_on_midpoint_steps,
	track_particles_midpoint,
	track_particles_midpoint_backward,
)


RADIUS_MARS = 3396200.0


def _build_case(time_values):
	time = np.asarray(time_values, dtype=np.float64)
	alt = np.array([0.0, 1000.0], dtype=np.float64)
	lat = np.array([-30.0, 0.0, 30.0], dtype=np.float64)
	lon = np.array([-180.0, -60.0, 60.0, 180.0], dtype=np.float64)

	shape = (time.size, alt.size, lat.size, lon.size)
	zero = np.zeros(shape, dtype=np.float64)
	scalar = np.empty(shape, dtype=np.float64)
	for i, t in enumerate(time):
		scalar[i, :, :, :] = t

	start = (0.0, 0.0, 500.0)
	return time, alt, lat, lon, zero, scalar, start


def _sample_values(samples, key):
	return [step[key] for step in samples]


def test_forward_trace_time_alignment():
	time, alt, lat, lon, zero, scalar, start = _build_case([0.0, 1.0, 2.0, 3.0])
	mid_steps = track_particles_midpoint(
		time, zero, zero, zero, [start],
		alt, lat, lon, -1e9, 1e9,
		radius=RADIUS_MARS, lon_mode='-180_180', periodic_lon=True
	)
	trace_time = get_trace_time_midpoint(time)

	samples = sample_scalar_on_midpoint_steps(
		mid_steps, scalar, alt, lat, lon,
		trace_time=trace_time, time_grid=time,
		lon_mode='-180_180', periodic_lon=True
	)
	assert _sample_values(samples, 0) == pytest.approx([1.0, 2.0, 3.0])


def test_backward_full_range_trace_time_alignment():
	time, alt, lat, lon, zero, scalar, start = _build_case([0.0, 1.0, 2.0, 3.0])
	mid_steps = track_particles_midpoint_backward(
		time, zero, zero, zero, [start],
		alt, lat, lon, -1e9, 1e9,
		radius=RADIUS_MARS, lon_mode='-180_180', periodic_lon=True,
		start_index=3, n_steps=3
	)
	trace_time = get_trace_time_midpoint_backward(time, start_index=3, n_steps=3)

	samples = sample_scalar_on_midpoint_steps(
		mid_steps, scalar, alt, lat, lon,
		trace_time=trace_time, time_grid=time,
		lon_mode='-180_180', periodic_lon=True
	)
	assert _sample_values(samples, 0) == pytest.approx([2.0, 1.0, 0.0])


def test_backward_subset_trace_time_alignment():
	time, alt, lat, lon, zero, scalar, start = _build_case([0.0, 1.0, 2.0, 3.0, 4.0])
	mid_steps = track_particles_midpoint_backward(
		time, zero, zero, zero, [start],
		alt, lat, lon, -1e9, 1e9,
		radius=RADIUS_MARS, lon_mode='-180_180', periodic_lon=True,
		start_index=4, n_steps=2
	)
	trace_time = get_trace_time_midpoint_backward(time, start_index=4, n_steps=2)

	samples = sample_scalar_on_midpoint_steps(
		mid_steps, scalar, alt, lat, lon,
		trace_time=trace_time, time_grid=time,
		lon_mode='-180_180', periodic_lon=True
	)
	assert _sample_values(samples, 0) == pytest.approx([3.0, 2.0])


def test_missing_trace_time_raises():
	time, alt, lat, lon, _, scalar, _ = _build_case([0.0, 1.0])
	with pytest.raises(ValueError, match='trace_time'):
		sample_scalar_on_midpoint_steps([], scalar, alt, lat, lon, time_grid=time)


def test_missing_time_grid_raises():
	_, alt, lat, lon, _, scalar, _ = _build_case([0.0, 1.0])
	with pytest.raises(ValueError, match='time_grid'):
		sample_scalar_on_midpoint_steps([], scalar, alt, lat, lon, trace_time=np.array([], dtype=np.float64))


def test_trace_time_length_mismatch_raises():
	time, alt, lat, lon, _, scalar, _ = _build_case([0.0, 1.0, 2.0])
	with pytest.raises(ValueError, match='trace_time length'):
		sample_scalar_on_midpoint_steps(
			[{}], scalar, alt, lat, lon,
			trace_time=np.array([], dtype=np.float64), time_grid=time
		)


def test_time_grid_not_strictly_increasing_raises():
	time, alt, lat, lon, _, scalar, _ = _build_case([0.0, 1.0, 2.0])
	with pytest.raises(ValueError, match='strictly increasing'):
		sample_scalar_on_midpoint_steps(
			[], scalar, alt, lat, lon,
			trace_time=np.array([], dtype=np.float64),
			time_grid=np.array([0.0, 2.0, 1.0], dtype=np.float64)
		)


def test_trace_time_value_without_match_raises():
	time, alt, lat, lon, _, scalar, _ = _build_case([0.0, 1.0, 2.0])
	with pytest.raises(ValueError, match='has no match in time_grid'):
		sample_scalar_on_midpoint_steps(
			[{}], scalar, alt, lat, lon,
			trace_time=np.array([5.0], dtype=np.float64), time_grid=time
		)
