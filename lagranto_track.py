import numpy as np
from numba import jit, prange
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed

def pres2alt(pres,p0=710,H=10): # return altitude in km
	return H*np.log(p0/pres)

def alt2pres(alt,p0=710,H=10): # return pressure in Pa
	return p0*np.exp(-alt/H)

def change_vertical_wind_unit_from_m_per_s_to_Pa_per_s(w, rho, g=3.71):
	"""
	Convert vertical wind speed from meters per second to Pascals per second.

	Parameters:
	w (float): Vertical wind speed in meters per second. Positive if downward (LMD style).
	rho (float): Air density in kg/m^3.
	g (float): Gravitational acceleration in m/s^2. Default is 3.71.

	Returns:
	float: Vertical wind speed in Pascals per second.
	"""
	return w * rho * g

@jit(nopython=True)
def interpolate_time(data, time_known, time_target):
    T = data.shape[0]
    if time_known.shape[0] != T:
        raise ValueError("time_known length must equal data.shape[0]")

    # 可选：严格递增性检查（如需，改为显式循环比较以适配 numba）
    # for k in range(1, T):
    #     if not (time_known[k] > time_known[k-1]):
    #         raise ValueError("time_known must be strictly increasing")

    spatial_shape = data.shape[1:]
    T_target = time_target.shape[0]

    out_dtype = np.float64  # 或 np.result_type(data.dtype, np.float64)（Numba 下直接写 float64 更稳）
    result = np.empty((T_target,) + spatial_shape, dtype=out_dtype)

    for idx in range(T_target):
        t = time_target[idx]

        # 处理 NaN：全部比较为 False，直接设为边界或 NaN（此处设为 NaN）
        if not (t == t):
            result[idx] = np.nan
            continue

        if t <= time_known[0]:
            result[idx] = data[0]
            continue
        if t >= time_known[T - 1]:
            result[idx] = data[T - 1]
            continue

        found = False
        for j in range(T - 1):
            if (time_known[j] <= t) and (t <= time_known[j + 1]):
                denom = time_known[j + 1] - time_known[j]
                if denom == 0.0:
                    # 重复时间点：取后一帧（或取前一帧/平均，按需求定）
                    result[idx] = data[j + 1]
                else:
                    w = (t - time_known[j]) / denom
                    result[idx] = (1.0 - w) * data[j] + w * data[j + 1]
                found = True
                break

        if not found:
            # 若因浮点误差漏判，做一次就近钳制（也可选择报错/外推）
            # 这里按就近点钳制
            # 简单线性扫描求最近点
            min_dist = abs(t - time_known[0])
            min_k = 0
            for k in range(1, T):
                d = abs(t - time_known[k])
                if d < min_dist:
                    min_dist = d
                    min_k = k
            result[idx] = data[min_k]

    return result


@jit(nopython=True, parallel=True)
def interpolate_4d_time(data, time_known, time_target):
	"""
		Interpolation of the time dimension for four-dimensional data (parallel implementation).
		
		Parameters:
		- data: Four-dimensional data at known time points, shape (T, Z, Y, X).
		- time_known: Array of known time points, shape (T,).
		- time_target: Array of target time points for interpolation, shape (T_target,).
		
		Returns:
		- Interpolated four-dimensional data, shape (T_target, Z, Y, X).
	"""
	T, Z, Y, X = data.shape
	T_target = len(time_target)
	
	result = np.empty((T_target, Z, Y, X), dtype=data.dtype)
	
	for z in prange(Z):  
		for y in range(Y):
			for x in range(X):
				result[:, z, y, x] = interpolate_time(
					data[:, z, y, x],
					time_known,
					time_target
				)
	
	return result

def mod_longitude(lon):
	return ((lon + 180) % 360) - 180


def normalize_lat_lon(lat, lon, lon_mode=None):
	if lon_mode is None:
		lon = mod_longitude(lon)
	else:
		lon = normalize_lon_to_mode(lon, lon_mode)
	while lat > 90 or lat < -90:
		if lat > 90:
			lat = 180 - lat
			lon = normalize_lon_to_mode(lon + 180, lon_mode) if lon_mode is not None else mod_longitude(lon + 180)
		elif lat < -90:
			lat = -180 - lat
			lon = normalize_lon_to_mode(lon + 180, lon_mode) if lon_mode is not None else mod_longitude(lon + 180)
	return lat, lon

	
def calculate_longitude_offset(latitude_deg, distance_m, radius):
	R = radius
	latitude_rad = np.deg2rad(latitude_deg)

	cos_lat = np.cos(latitude_rad)
	if np.isnan(cos_lat):
		return np.nan
	min_abs_cos = 1e-6
	if cos_lat >= 0:
		cos_lat = max(cos_lat, min_abs_cos)
	else:
		cos_lat = min(cos_lat, -min_abs_cos)

	delta_lambda_rad = distance_m / (R * cos_lat)

	# 将弧度转换为度数
	delta_lambda_deg = np.rad2deg(delta_lambda_rad)

	return delta_lambda_deg

def calculate_latitude_offset(distance_m, radius):
	return calculate_longitude_offset(0, distance_m, radius)


def infer_lon_mode(lon_grid):
	lon_grid = np.asarray(lon_grid)
	if lon_grid.ndim != 1:
		raise ValueError('lon grid must be one-dimensional')
	min_lon = float(np.nanmin(lon_grid))
	max_lon = float(np.nanmax(lon_grid))
	if min_lon >= 0.0 and max_lon <= 360.0:
		return '0_360'
	return '-180_180'


def normalize_lon_to_mode(lon, mode):
	if mode == '0_360':
		return lon % 360.0
	return mod_longitude(lon)

def get_interpolated_winds(alt, lat, lon, u_field, v_field, w_field, alt_grid, lat_grid, lon_grid, lon_mode=None):
	"""Helper function to get interpolated winds at a point"""
	if lon_mode is not None:
		lon = normalize_lon_to_mode(lon, lon_mode)
	interp_point = np.array([[alt, lat, lon]], dtype=np.float64)

	u_interp = RegularGridInterpolator(
		(alt_grid, lat_grid, lon_grid),
		u_field,
		method='linear',
		bounds_error=False,
		fill_value=np.nan
	)
	v_interp = RegularGridInterpolator(
		(alt_grid, lat_grid, lon_grid),
		v_field,
		method='linear',
		bounds_error=False,
		fill_value=np.nan
	)
	w_interp = RegularGridInterpolator(
		(alt_grid, lat_grid, lon_grid),
		w_field,
		method='linear',
		bounds_error=False,
		fill_value=np.nan
	)
	
	u = float(u_interp(interp_point)[0])
	v = float(v_interp(interp_point)[0])
	w = float(w_interp(interp_point)[0])
	return np.array([u, v, w], dtype=np.float64)


def get_next_position_alt(lon, lat, alt_m, wind, dt, radius, lower_boundary, upper_boundary, lon_mode=None, w_positive_up=True):
	"""Propagate a particle in space using altitude (m) as vertical coordinate."""
	dx = wind[0] * dt
	dy = wind[1] * dt
	dz = wind[2] * dt if w_positive_up else -wind[2] * dt

	lon_offset = calculate_longitude_offset(lat, dx, radius)
	lat_offset = calculate_latitude_offset(dy, radius)
	if not np.isfinite(lon_offset) or not np.isfinite(lat_offset) or not np.isfinite(dz):
		return None

	new_alt = alt_m + dz
	if (lower_boundary is not None and new_alt < lower_boundary) or (upper_boundary is not None and new_alt > upper_boundary):
		return None

	new_lon = lon + lon_offset
	new_lat = lat + lat_offset
	new_lat, new_lon = normalize_lat_lon(new_lat, new_lon, lon_mode)

	return (new_lon, new_lat, new_alt)

def get_trace_time_heun(new_time_points):
	return new_time_points[1:]

def get_trace_time_heun_backward(time, start_index=None, start_time=None, n_steps=None):
	"""Return time stamps associated with backward Heun integration steps."""
	time = np.asarray(time)
	if time.ndim != 1:
		raise ValueError('time must be one-dimensional')
	if time.size < 2:
		raise ValueError('time array must contain at least two entries')
	if not np.all(np.diff(time) > 0):
		raise ValueError('time array must be strictly increasing')

	if start_index is not None and start_time is not None:
		raise ValueError('Provide only one of start_index or start_time')

	if start_index is not None:
		start_idx = int(start_index)
		if start_idx < 1 or start_idx >= time.size:
			raise ValueError('start_index must satisfy 1 <= start_index < len(time)')
	else:
		if start_time is None:
			start_idx = time.size - 1
		else:
			matches = np.where(np.isclose(time, start_time, rtol=0.0, atol=0.0))[0]
			if matches.size == 0:
				raise ValueError('start_time must match one of the entries in time array')
			start_idx = int(matches[-1])
			if start_idx < 1:
				raise ValueError('start_time corresponds to the earliest sample; cannot step backward')

	if n_steps is not None:
		n_steps = int(n_steps)
		if n_steps < 1:
			raise ValueError('n_steps must be >= 1 when provided')
		end_idx = max(start_idx - n_steps, 0)
	else:
		end_idx = 0

	if start_idx == end_idx:
		return np.array([], dtype=time.dtype)

	indices = np.arange(start_idx - 1, end_idx - 1, -1, dtype=int)
	return time[indices]

def track_particles_heun(time, u_time_interp, v_time_interp, w_time_interp, need_track_initial_points,
				alt_grid_m, lat_grid, lon_grid, lower_boundary, upper_boundary,
				radius=3396200, verbose=False, w_positive_up=True, lon_mode=None):
	"""
	Track particles using Heun's method with altitude (m) and vertical velocity (m/s).
	Particles stop when altitude crosses the provided boundaries or when any interpolated
	wind component becomes NaN (indicating they have moved outside the data domain).
	"""
	time = np.asarray(time)
	if time.ndim != 1:
		raise ValueError('time must be one-dimensional')
	if time.size < 2:
		raise ValueError('time array must have at least two points for Heun integration')
	if not np.all(np.diff(time) > 0):
		raise ValueError('time array must be strictly increasing')

	lat_grid = np.asarray(lat_grid)
	lon_grid = np.asarray(lon_grid)
	alt_grid = np.asarray(alt_grid_m)
	u_time_interp = np.asarray(u_time_interp)
	v_time_interp = np.asarray(v_time_interp)
	w_time_interp = np.asarray(w_time_interp)

	if u_time_interp.shape != v_time_interp.shape or u_time_interp.shape != w_time_interp.shape:
		raise ValueError('u, v, w interpolated arrays must have identical shapes')
	if u_time_interp.shape[0] != time.size:
		raise ValueError('time dimension mismatch between wind fields and time array')
	if u_time_interp.shape[1] != alt_grid.size:
		raise ValueError('altitude dimension mismatch between wind fields and altitude grid')
	if u_time_interp.shape[2] != lat_grid.size or u_time_interp.shape[3] != lon_grid.size:
		raise ValueError('horizontal dimensions mismatch between wind fields and latitude/longitude grids')

	if lat_grid.ndim != 1 or lon_grid.ndim != 1 or alt_grid.ndim != 1:
		raise ValueError('alt_grid, lat_grid, lon_grid must be one-dimensional')
	if not np.all(np.diff(lat_grid) > 0):
		raise ValueError('lat grid must be strictly increasing')
	if not np.all(np.diff(lon_grid) > 0):
		raise ValueError('lon grid must be strictly increasing')

	sort_idx = np.argsort(alt_grid)
	alt_sorted = alt_grid[sort_idx]
	if not np.all(np.diff(alt_sorted) > 0):
		raise ValueError('altitude grid must be strictly increasing after sorting')

	u_sorted = np.take(u_time_interp, sort_idx, axis=1)
	v_sorted = np.take(v_time_interp, sort_idx, axis=1)
	w_sorted = np.take(w_time_interp, sort_idx, axis=1)

	if lon_mode is None:
		lon_mode = infer_lon_mode(lon_grid)

	positions = {}
	active = {}
	for point in need_track_initial_points:
		if len(point) != 3:
			raise ValueError('Each initial point must be a (lon, lat, altitude_m) tuple')
		lon0, lat0, alt0 = point
		if not np.isfinite(alt0):
			raise ValueError('Initial altitude must be finite')
		lat0, lon0 = normalize_lat_lon(lat0, lon0, lon_mode)
		positions[point] = (lon0, lat0, alt0)
		active[point] = True

	new_position_dict_list = []
	iterator = range(time.size - 1)
	if verbose:
		iterator = tqdm(iterator, desc='Heun integration along time axis')

	for itime in iterator:
		dt = float(time[itime + 1] - time[itime])
		if not np.isfinite(dt) or dt <= 0:
			raise ValueError('time array must be strictly increasing with finite step')
		u_slice_now = u_sorted[itime, :, :, :]
		v_slice_now = v_sorted[itime, :, :, :]
		w_slice_now = w_sorted[itime, :, :, :]
		u_slice_next = u_sorted[itime + 1, :, :, :]
		v_slice_next = v_sorted[itime + 1, :, :, :]
		w_slice_next = w_sorted[itime + 1, :, :, :]

		next_positions = {}
		for point, (lon_now, lat_now, alt_now) in positions.items():
			if not active[point]:
				next_positions[point] = (lon_now, lat_now, alt_now)
				continue

			k1_wind = get_interpolated_winds(
				alt_now, lat_now, lon_now,
				u_slice_now, v_slice_now, w_slice_now,
				alt_sorted, lat_grid, lon_grid, lon_mode=lon_mode
			)
			if np.any(np.isnan(k1_wind)):
				active[point] = False
				next_positions[point] = (lon_now, lat_now, alt_now)
				continue

			predicted = get_next_position_alt(
				lon_now, lat_now, alt_now, k1_wind, dt, radius,
				lower_boundary, upper_boundary, lon_mode=lon_mode, w_positive_up=w_positive_up
			)
			if predicted is None:
				active[point] = False
				next_positions[point] = (lon_now, lat_now, alt_now)
				continue

			pred_lon, pred_lat, pred_alt = predicted
			k2_wind = get_interpolated_winds(
				pred_alt, pred_lat, pred_lon,
				u_slice_next, v_slice_next, w_slice_next,
				alt_sorted, lat_grid, lon_grid, lon_mode=lon_mode
			)
			if np.any(np.isnan(k2_wind)):
				active[point] = False
				next_positions[point] = (lon_now, lat_now, alt_now)
				continue

			final_wind = 0.5 * (k1_wind + k2_wind)
			new_pos = get_next_position_alt(
				lon_now, lat_now, alt_now, final_wind, dt, radius,
				lower_boundary, upper_boundary, lon_mode=lon_mode, w_positive_up=w_positive_up
			)
			if new_pos is None:
				active[point] = False
				next_positions[point] = (lon_now, lat_now, alt_now)
			else:
				next_positions[point] = new_pos

		positions = next_positions
		new_position_dict_list.append(next_positions)

	return new_position_dict_list


def track_particles_heun_backward(time, u_time_interp, v_time_interp, w_time_interp, need_track_initial_points,
					alt_grid_m, lat_grid, lon_grid, lower_boundary, upper_boundary,
					radius=3396200, verbose=False, w_positive_up=True, lon_mode=None,
					start_index=None, start_time=None, n_steps=None):
	"""
	Backward Heun integration using altitude (m) and vertical velocity (m/s).
	Particles are traced from later to earlier times without negating the wind fields.
	Particles stop when altitude crosses the supplied boundaries or interpolated winds are NaN.
	"""
	time = np.asarray(time)
	if time.ndim != 1:
		raise ValueError('time must be one-dimensional')
	if time.size < 2:
		raise ValueError('time array must have at least two points for Heun integration')
	if not np.all(np.diff(time) > 0):
		raise ValueError('time array must be strictly increasing')

	if start_index is not None and start_time is not None:
		raise ValueError('Provide only one of start_index or start_time')

	lat_grid = np.asarray(lat_grid)
	lon_grid = np.asarray(lon_grid)
	alt_grid = np.asarray(alt_grid_m)
	u_time_interp = np.asarray(u_time_interp)
	v_time_interp = np.asarray(v_time_interp)
	w_time_interp = np.asarray(w_time_interp)

	if u_time_interp.shape != v_time_interp.shape or u_time_interp.shape != w_time_interp.shape:
		raise ValueError('u, v, w interpolated arrays must have identical shapes')
	if u_time_interp.shape[0] != time.size:
		raise ValueError('time dimension mismatch between wind fields and time array')
	if u_time_interp.shape[1] != alt_grid.size:
		raise ValueError('altitude dimension mismatch between wind fields and altitude grid')
	if u_time_interp.shape[2] != lat_grid.size or u_time_interp.shape[3] != lon_grid.size:
		raise ValueError('horizontal dimensions mismatch between wind fields and latitude/longitude grids')

	if lat_grid.ndim != 1 or lon_grid.ndim != 1 or alt_grid.ndim != 1:
		raise ValueError('alt_grid, lat_grid, lon_grid must be one-dimensional')
	if not np.all(np.diff(lat_grid) > 0):
		raise ValueError('lat grid must be strictly increasing')
	if not np.all(np.diff(lon_grid) > 0):
		raise ValueError('lon grid must be strictly increasing')

	sort_idx = np.argsort(alt_grid)
	alt_sorted = alt_grid[sort_idx]
	if not np.all(np.diff(alt_sorted) > 0):
		raise ValueError('altitude grid must be strictly increasing after sorting')

	u_sorted = np.take(u_time_interp, sort_idx, axis=1)
	v_sorted = np.take(v_time_interp, sort_idx, axis=1)
	w_sorted = np.take(w_time_interp, sort_idx, axis=1)

	if lon_mode is None:
		lon_mode = infer_lon_mode(lon_grid)

	if start_index is not None:
		start_idx = int(start_index)
		if start_idx < 1 or start_idx >= time.size:
			raise ValueError('start_index must satisfy 1 <= start_index < len(time)')
	else:
		if start_time is None:
			start_idx = time.size - 1
		else:
			matches = np.where(np.isclose(time, start_time, rtol=0.0, atol=0.0))[0]
			if matches.size == 0:
				raise ValueError('start_time must match one of the entries in time array')
			start_idx = int(matches[-1])
			if start_idx < 1:
				raise ValueError('start_time corresponds to the earliest sample; cannot step backward')

	if n_steps is not None:
		n_steps = int(n_steps)
		if n_steps < 1:
			raise ValueError('n_steps must be >= 1 when provided')
		end_idx = max(start_idx - n_steps, 0)
	else:
		end_idx = 0

	if start_idx == end_idx:
		return []

	positions = {}
	active = {}
	for point in need_track_initial_points:
		if len(point) != 3:
			raise ValueError('Each initial point must be a (lon, lat, altitude_m) tuple')
		lon0, lat0, alt0 = point
		if not np.isfinite(alt0):
			raise ValueError('Initial altitude must be finite')
		lat0, lon0 = normalize_lat_lon(lat0, lon0, lon_mode)
		positions[point] = (lon0, lat0, alt0)
		active[point] = True

	new_position_dict_list = []
	iterator = range(start_idx, end_idx, -1)
	if verbose:
		iterator = tqdm(iterator, desc='Heun backward integration along time axis')

	for itime in iterator:
		delta = float(time[itime] - time[itime - 1])
		if not np.isfinite(delta) or delta <= 0:
			raise ValueError('time array must be strictly increasing with finite step')
		dt = -delta
		u_slice_now = u_sorted[itime, :, :, :]
		v_slice_now = v_sorted[itime, :, :, :]
		w_slice_now = w_sorted[itime, :, :, :]
		u_slice_prev = u_sorted[itime - 1, :, :, :]
		v_slice_prev = v_sorted[itime - 1, :, :, :]
		w_slice_prev = w_sorted[itime - 1, :, :, :]

		next_positions = {}
		for point, (lon_now, lat_now, alt_now) in positions.items():
			if not active[point]:
				next_positions[point] = (lon_now, lat_now, alt_now)
				continue

			k1_wind = get_interpolated_winds(
				alt_now, lat_now, lon_now,
				u_slice_now, v_slice_now, w_slice_now,
				alt_sorted, lat_grid, lon_grid, lon_mode=lon_mode
			)
			if np.any(np.isnan(k1_wind)):
				active[point] = False
				next_positions[point] = (lon_now, lat_now, alt_now)
				continue

			predicted = get_next_position_alt(
				lon_now, lat_now, alt_now, k1_wind, dt, radius,
				lower_boundary, upper_boundary, lon_mode=lon_mode, w_positive_up=w_positive_up
			)
			if predicted is None:
				active[point] = False
				next_positions[point] = (lon_now, lat_now, alt_now)
				continue

			pred_lon, pred_lat, pred_alt = predicted
			k2_wind = get_interpolated_winds(
				pred_alt, pred_lat, pred_lon,
				u_slice_prev, v_slice_prev, w_slice_prev,
				alt_sorted, lat_grid, lon_grid, lon_mode=lon_mode
			)
			if np.any(np.isnan(k2_wind)):
				active[point] = False
				next_positions[point] = (lon_now, lat_now, alt_now)
				continue

			final_wind = 0.5 * (k1_wind + k2_wind)
			new_pos = get_next_position_alt(
				lon_now, lat_now, alt_now, final_wind, dt, radius,
				lower_boundary, upper_boundary, lon_mode=lon_mode, w_positive_up=w_positive_up
			)
			if new_pos is None:
				active[point] = False
				next_positions[point] = (lon_now, lat_now, alt_now)
			else:
				next_positions[point] = new_pos

		positions = next_positions
		new_position_dict_list.append(next_positions)

	return new_position_dict_list
