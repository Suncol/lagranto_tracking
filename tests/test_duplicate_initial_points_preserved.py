import numpy as np

from lagranto_track import (
	track_particles_heun,
	track_particles_heun_backward,
	track_particles_midpoint,
	track_particles_midpoint_backward,
)


RADIUS_MARS = 3396200.0


def _build_zero_wind_case():
	time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
	alt = np.array([0.0, 1000.0], dtype=np.float64)
	lat = np.array([-30.0, 0.0, 30.0], dtype=np.float64)
	lon = np.array([-180.0, -60.0, 60.0, 180.0], dtype=np.float64)
	shape = (time.size, alt.size, lat.size, lon.size)
	zero = np.zeros(shape, dtype=np.float64)
	return time, alt, lat, lon, zero


def _assert_duplicate_particles_preserved(step_dicts):
	assert len(step_dicts) == 2
	for step_dict in step_dicts:
		assert len(step_dict) == 2
		assert set(step_dict.keys()) == {0, 1}
		assert np.allclose(np.asarray(step_dict[0]), np.asarray(step_dict[1]))


def test_heun_forward_preserves_duplicate_initial_points():
	time, alt, lat, lon, zero = _build_zero_wind_case()
	start = (0.0, 0.0, 500.0)
	steps = track_particles_heun(
		time, zero, zero, zero, [start, start],
		alt, lat, lon, -1e9, 1e9,
		radius=RADIUS_MARS, lon_mode='-180_180', periodic_lon=True
	)
	_assert_duplicate_particles_preserved(steps)


def test_heun_backward_preserves_duplicate_initial_points():
	time, alt, lat, lon, zero = _build_zero_wind_case()
	start = (0.0, 0.0, 500.0)
	steps = track_particles_heun_backward(
		time, zero, zero, zero, [start, start],
		alt, lat, lon, -1e9, 1e9,
		radius=RADIUS_MARS, lon_mode='-180_180', periodic_lon=True,
		start_index=2, n_steps=2
	)
	_assert_duplicate_particles_preserved(steps)


def test_midpoint_forward_preserves_duplicate_initial_points():
	time, alt, lat, lon, zero = _build_zero_wind_case()
	start = (0.0, 0.0, 500.0)
	steps = track_particles_midpoint(
		time, zero, zero, zero, [start, start],
		alt, lat, lon, -1e9, 1e9,
		radius=RADIUS_MARS, lon_mode='-180_180', periodic_lon=True
	)
	_assert_duplicate_particles_preserved(steps)


def test_midpoint_backward_preserves_duplicate_initial_points():
	time, alt, lat, lon, zero = _build_zero_wind_case()
	start = (0.0, 0.0, 500.0)
	steps = track_particles_midpoint_backward(
		time, zero, zero, zero, [start, start],
		alt, lat, lon, -1e9, 1e9,
		radius=RADIUS_MARS, lon_mode='-180_180', periodic_lon=True,
		start_index=2, n_steps=2
	)
	_assert_duplicate_particles_preserved(steps)
