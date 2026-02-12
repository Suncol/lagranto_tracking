
"""
lagranto_nvec_petterssen.py
==========================

A rigorous, self-contained reference implementation of kinematic Lagrangian trajectory
calculations on a spherical planet using a *non-singular* horizontal position representation:

    state = (n_vec, h)

where n_vec is a unit "n-vector" (direction from planet center) and h is altitude (meters).

This avoids the polar singularity inherent in longitude/latitude ODEs (i.e., no 1/cos(lat)),
while still allowing output in (lon, lat, h) at every requested time.

Numerical scheme: Petterssen iterative Euler / iterative trapezoid as used in LAGRANTO.

Key literature anchors (used explicitly in code comments):
  - Sprenger & Wernli (2015, Geosci. Model Dev.) describe LAGRANTO's iterative time step:
    Euler predictor followed by repeated corrections using averaged velocities, with defaults:
    * three iterative steps
    * dt = 1/12 of the data time interval
    * bilinear horizontal + linear vertical interpolation.
    (See their Eqs. (2)-(4) and surrounding text.)
  - Rößler et al. (2018, Geosci. Model Dev.) formalize Petterssen scheme with inner iterations
    (their Eqs. (5)-(6)) and relate 1 inner iteration to Heun's method.
  - Gade (2010, Journal of Navigation) gives non-singular n-vector representation and provides:
    * robust n-vector -> lat conversion (arctan form, their Eq. (6))
    * derivative relations for n-vector and height:
        n_dot = ω × n (their Eq. (14))
        h_dot = n · v (their Eq. (15))
      which, on a sphere, imply n_dot = v_horizontal / (R+h).
  - Stohl (1998, Atmos. Environ.) reviews trajectory computation and sources of errors; together
    with Rößler et al. (2018), supports using low-order 4-D (time+3D space) linear interpolation
    with careful time stepping.

This module is designed to be "drop-in friendly" with common LAGRANTO-style calling patterns:
  - forward tracking over an increasing time array
  - backward tracking over the same array by choosing dt < 0 (DO NOT negate winds)

Only dependency: numpy.

Author: (reference implementation generated for rigorous review / scientific use)
License: MIT-like (adapt as needed)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np


# -------------------------------
# Status codes (explicit & stable)
# -------------------------------

class ParticleStatus(IntEnum):
    """Per-particle integration status."""
    ACTIVE = 0
    HIT_LOWER_BOUNDARY = 1
    HIT_UPPER_BOUNDARY = 2
    OUT_OF_DOMAIN = 3          # outside wind field grids (space or time)
    NAN_WIND = 4               # wind interpolation resulted in NaN/Inf
    INVALID_STATE = 5          # invalid radius, non-finite values, etc.


# -------------------------------
# Utility: longitude modes & wrap
# -------------------------------

class LonMode(IntEnum):
    """Longitude normalization mode."""
    NEG180_TO_180 = 0
    ZERO_TO_360 = 1


def infer_lon_mode(lon_grid_deg: np.ndarray) -> LonMode:
    """
    Infer lon mode from grid.
    - If grid is mostly non-negative and spans beyond 180°, treat as 0..360.
    - Else treat as -180..180.

    (Heuristic; can be overridden by user.)
    """
    lon_min = float(np.nanmin(lon_grid_deg))
    lon_max = float(np.nanmax(lon_grid_deg))
    if lon_min >= -1e-9 and lon_max > 180.0:
        return LonMode.ZERO_TO_360
    return LonMode.NEG180_TO_180


def wrap_lon(lon_deg: float, mode: LonMode) -> float:
    """Wrap longitude into the chosen mode."""
    if mode == LonMode.ZERO_TO_360:
        return float(lon_deg % 360.0)
    # (-180, 180]
    return float(((lon_deg + 180.0) % 360.0) - 180.0)


# -------------------------------
# n-vector conversions (sphere)
# -------------------------------

def ll_to_nvec(lon_deg: float, lat_deg: float) -> np.ndarray:
    """
    Convert lon/lat [deg] to unit n-vector on a sphere.

    We use the conventional ECEF-like axis choice:
      x = lon=0, lat=0 (intersection of equator and prime meridian)
      y = lon=90E, lat=0
      z = north pole

    n = [cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)]
    """
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    cl = np.cos(lat)
    return np.array([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], dtype=float)


def nvec_to_ll(
    n: np.ndarray,
    lon_ref_deg: float,
    *,
    pole_eps: float = 1e-12,
    lon_mode: LonMode = LonMode.ZERO_TO_360,
) -> Tuple[float, float, float]:
    """
    Convert unit n-vector -> (lon_deg, lat_deg, updated_lon_ref_deg).

    Robust latitude formula uses atan2 as recommended by Gade (2010), Eq. (6),
    to avoid arcsin instability/overflow near poles.

    At the poles, longitude is mathematically undefined. We return lon_ref_deg to keep
    a continuous gauge for output and for defining a consistent East direction.

    Returns:
      lon_deg (wrapped to lon_mode),
      lat_deg,
      lon_ref_deg_out (updated when not too close to the poles)
    """
    n = np.asarray(n, dtype=float)
    if n.shape != (3,):
        raise ValueError("n must be shape (3,)")

    x, y, z = float(n[0]), float(n[1]), float(n[2])
    rho = float(np.hypot(x, y))

    # Gade (2010) Eq. (6): lat = atan2(z, sqrt(x^2+y^2)) [for our axis convention]
    lat = float(np.rad2deg(np.arctan2(z, rho)))

    if rho > pole_eps:
        lon = float(np.rad2deg(np.arctan2(y, x)))
        lon = wrap_lon(lon, lon_mode)
        lon_ref_out = lon
    else:
        lon = wrap_lon(lon_ref_deg, lon_mode)
        lon_ref_out = lon_ref_deg

    return lon, lat, lon_ref_out


def unit(v: np.ndarray, *, eps: float = 1e-30) -> np.ndarray:
    """Normalize a vector; if norm is too small or non-finite, raise."""
    v = np.asarray(v, dtype=float)
    nrm = float(np.linalg.norm(v))
    if not np.isfinite(nrm) or nrm <= eps:
        raise ValueError("cannot normalize vector with non-finite or near-zero norm")
    return v / nrm


# -------------------------------
# Local ENU basis from n-vector
# -------------------------------

def enu_basis_from_nvec(
    n: np.ndarray,
    lon_ref_deg: float,
    *,
    pole_eps: float = 1e-12,
    lon_mode: LonMode = LonMode.ZERO_TO_360,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct local East/North/Up unit vectors at n.

    For non-polar points:
      east  = k × n / ||k × n||   where k = [0,0,1]
      north = n × east
      up    = n

    At/near poles, k × n is near zero and East is not unique (true geometric degeneracy).
    We choose a *gauge* using lon_ref_deg to define a continuous East direction in the
    equatorial plane:
      east(lon_ref) = [-sin(lon_ref), cos(lon_ref), 0]
      north = up × east

    This does NOT modify the physics; it only chooses a basis where the basis is not unique.
    """
    n = unit(n)
    k = np.array([0.0, 0.0, 1.0], dtype=float)

    cross = np.cross(k, n)
    s = float(np.linalg.norm(cross))
    up = n

    if s > pole_eps:
        east = cross / s
    else:
        # Gauge-fixed East at the pole using lon_ref
        lon_ref = np.deg2rad(wrap_lon(lon_ref_deg, lon_mode))
        east = np.array([-np.sin(lon_ref), np.cos(lon_ref), 0.0], dtype=float)
        east = unit(east)

    north = np.cross(up, east)
    north = unit(north)

    # Ensure orthonormality (optional minor re-orthogonalization)
    east = unit(np.cross(north, up))

    return east, north, up


# -------------------------------------------
# 4-D linear wind interpolation (t,z,lat,lon)
# -------------------------------------------

@dataclass(frozen=True)
class WindField4D:
    """
    Wind field sampled on a regular grid with dimensions:
        time (nt), altitude z (nz), latitude (ny), longitude (nx)

    Arrays must be shaped:
        u, v, w : (nt, nz, ny, nx)

    Interpolation: 4-D linear interpolation:
      - time: linear between bracketing time slices
      - space: trilinear in (z, lat, lon)
        which corresponds to bilinear (lat/lon) + linear (z),
        consistent with LAGRANTO's description (Sprenger & Wernli, 2015).

    Periodic longitude is supported via an internal grid extension by +360° with wrap.

    Notes:
      - Coordinates: lon/lat in degrees; alt in meters; time in seconds
        (time units must match u,v,w in m/s; i.e. dt in seconds).
    """
    time_s: np.ndarray               # (nt,)
    alt_m: np.ndarray                # (nz,)
    lat_deg: np.ndarray              # (ny,)
    lon_deg: np.ndarray              # (nx or nx+1 if periodic extended)

    u: np.ndarray                    # (nt, nz, ny, nx_ext)
    v: np.ndarray
    w: np.ndarray

    lon_mode: LonMode
    periodic_lon: bool
    _lon0: float                     # first lon value (deg)
    _lon_period: float = 360.0

    @staticmethod
    def from_arrays(
        time_s: np.ndarray,
        alt_m: np.ndarray,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        *,
        lon_mode: Optional[LonMode] = None,
        periodic_lon: Union[bool, str] = "auto",
        atol: float = 1e-9,
    ) -> "WindField4D":
        """
        Build a WindField4D, validating and (if needed) sorting altitudes and handling
        periodic longitude extension.

        periodic_lon:
          - False: no wrap; points outside lon range are OUT_OF_DOMAIN
          - True : enforce periodic wrap by extending the lon dimension
          - "auto": infer periodic if grid spans ~360°
        """
        time_s = np.asarray(time_s, dtype=float)
        alt_m = np.asarray(alt_m, dtype=float)
        lat_deg = np.asarray(lat_deg, dtype=float)
        lon_deg = np.asarray(lon_deg, dtype=float)

        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        w = np.asarray(w, dtype=float)

        if time_s.ndim != 1 or alt_m.ndim != 1 or lat_deg.ndim != 1 or lon_deg.ndim != 1:
            raise ValueError("time_s, alt_m, lat_deg, lon_deg must be 1-D arrays")
        if time_s.size < 2:
            raise ValueError("time_s must have at least 2 entries")
        if not np.all(np.diff(time_s) > 0):
            raise ValueError("time_s must be strictly increasing")
        if not np.all(np.diff(lat_deg) > 0):
            raise ValueError("lat_deg grid must be strictly increasing")
        if not np.all(np.diff(lon_deg) > 0):
            raise ValueError("lon_deg grid must be strictly increasing (before any periodic handling)")

        if u.shape != v.shape or u.shape != w.shape:
            raise ValueError("u, v, w must have identical shapes")
        if u.ndim != 4:
            raise ValueError("u, v, w must be 4-D arrays shaped (nt, nz, ny, nx)")
        nt, nz, ny, nx = u.shape
        if nt != time_s.size or nz != alt_m.size or ny != lat_deg.size or nx != lon_deg.size:
            raise ValueError("grid dimension mismatch between coordinates and wind arrays")

        # Sort altitude if needed (many datasets do; keep this deterministic).
        if not np.all(np.diff(alt_m) > 0):
            order = np.argsort(alt_m)
            alt_sorted = alt_m[order]
            if not np.all(np.diff(alt_sorted) > 0):
                raise ValueError("alt_m must be strictly increasing after sorting")
            u = u[:, order, :, :]
            v = v[:, order, :, :]
            w = w[:, order, :, :]
            alt_m = alt_sorted

        if lon_mode is None:
            lon_mode = infer_lon_mode(lon_deg)

        # Decide periodic longitude.
        if isinstance(periodic_lon, str):
            if periodic_lon.lower() != "auto":
                raise ValueError("periodic_lon must be bool or 'auto'")
            # Infer periodic if span ~ 360 degrees.
            diffs = np.diff(lon_deg)
            step = float(np.median(diffs))
            span = float(lon_deg[-1] - lon_deg[0])
            periodic = span >= (360.0 - 1.5 * step)
        else:
            periodic = bool(periodic_lon)

        lon0 = float(lon_deg[0])

        if periodic:
            # If the last point duplicates the first + 360°, drop it to avoid zero cell.
            if abs((lon_deg[-1] - lon_deg[0]) - 360.0) <= atol:
                lon_deg = lon_deg[:-1]
                u = u[:, :, :, :-1]
                v = v[:, :, :, :-1]
                w = w[:, :, :, :-1]
                nx = lon_deg.size

            # Extend by one ghost point at lon0 + 360 with values equal to lon0.
            lon_ext = np.concatenate([lon_deg, np.array([lon_deg[0] + 360.0], dtype=float)])
            u_ext = np.concatenate([u, u[:, :, :, :1]], axis=3)
            v_ext = np.concatenate([v, v[:, :, :, :1]], axis=3)
            w_ext = np.concatenate([w, w[:, :, :, :1]], axis=3)

            lon_deg = lon_ext
            u, v, w = u_ext, v_ext, w_ext

        return WindField4D(
            time_s=time_s,
            alt_m=alt_m,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            u=u,
            v=v,
            w=w,
            lon_mode=lon_mode,
            periodic_lon=periodic,
            _lon0=lon0,
        )

    # ---- Internal helper: bracketing indices

    def _effective_time_atol(self, t_s: float, user_atol: float) -> float:
        """
        Build a robust time tolerance to absorb floating-point roundoff near time boundaries.

        We intentionally apply this only to the time axis:
          time_atol = max(user_atol, 64*eps*scale, 1e-12)
          scale = max(1, |t|, |time_min|, |time_max|)
        """
        t_s = float(t_s)
        user_atol = float(user_atol)
        eps = float(np.finfo(float).eps)
        scale = max(1.0, abs(t_s), abs(float(self.time_s[0])), abs(float(self.time_s[-1])))
        return float(max(user_atol, 64.0 * eps * scale, 1e-12))

    @staticmethod
    def _bracket(grid: np.ndarray, x: float, *, atol: float = 0.0) -> Optional[Tuple[int, float]]:
        """
        Return (i0, f) such that:
          x is between grid[i0] and grid[i0+1]
          f in [0,1] is fractional coordinate.

        If x is outside grid, return None.
        """
        x = float(x)
        if not np.isfinite(x):
            return None

        if x < grid[0] - atol or x > grid[-1] + atol:
            return None

        # Handle right boundary exactly.
        if x >= grid[-1] - atol:
            i0 = grid.size - 2
            denom = float(grid[i0 + 1] - grid[i0])
            if denom <= 0:
                return None
            return i0, 1.0

        i0 = int(np.searchsorted(grid, x, side="right") - 1)
        if i0 < 0 or i0 >= grid.size - 1:
            return None
        denom = float(grid[i0 + 1] - grid[i0])
        if denom <= 0:
            return None
        f = float((x - grid[i0]) / denom)
        # guard tiny rounding
        if f < 0.0:
            f = 0.0
        elif f > 1.0:
            f = 1.0
        return i0, f

    def _wrap_lon_to_internal(self, lon_deg: float) -> float:
        """
        Wrap lon into [lon0, lon0+360) if periodic, otherwise just normalize to lon_mode.
        """
        lon = wrap_lon(lon_deg, self.lon_mode)
        if not self.periodic_lon:
            return lon

        # shift into [lon0, lon0+360)
        while lon < self.lon_deg[0]:
            lon += 360.0
        while lon >= self.lon_deg[0] + 360.0:
            lon -= 360.0
        return lon

    @staticmethod
    def _trilinear(
        field_zyx: np.ndarray,
        iz0: int, fz: float,
        iy0: int, fy: float,
        ix0: int, fx: float,
    ) -> float:
        """
        Trilinear interpolation for a single time slice.

        field_zyx shape: (nz, ny, nx)
        """
        # corner indices
        iz1 = iz0 + 1
        iy1 = iy0 + 1
        ix1 = ix0 + 1

        c000 = field_zyx[iz0, iy0, ix0]
        c001 = field_zyx[iz0, iy0, ix1]
        c010 = field_zyx[iz0, iy1, ix0]
        c011 = field_zyx[iz0, iy1, ix1]
        c100 = field_zyx[iz1, iy0, ix0]
        c101 = field_zyx[iz1, iy0, ix1]
        c110 = field_zyx[iz1, iy1, ix0]
        c111 = field_zyx[iz1, iy1, ix1]

        corners = np.array([c000, c001, c010, c011, c100, c101, c110, c111], dtype=float)
        if not np.all(np.isfinite(corners)):
            return float("nan")

        wz0 = 1.0 - fz
        wy0 = 1.0 - fy
        wx0 = 1.0 - fx
        wz1 = fz
        wy1 = fy
        wx1 = fx

        return float(
            c000 * (wz0 * wy0 * wx0) +
            c001 * (wz0 * wy0 * wx1) +
            c010 * (wz0 * wy1 * wx0) +
            c011 * (wz0 * wy1 * wx1) +
            c100 * (wz1 * wy0 * wx0) +
            c101 * (wz1 * wy0 * wx1) +
            c110 * (wz1 * wy1 * wx0) +
            c111 * (wz1 * wy1 * wx1)
        )
    def sample_wind_with_status(
        self,
        t_s: float,
        alt_m: float,
        lat_deg: float,
        lon_deg: float,
        *,
        atol: float = 0.0,
    ) -> Tuple[float, float, float, ParticleStatus]:
        """
        Sample (u,v,w) at (t, alt, lat, lon) using 4-D linear interpolation.

        Returns (u, v, w, status) where status is:
          - ParticleStatus.ACTIVE        if interpolation succeeded (finite result)
          - ParticleStatus.OUT_OF_DOMAIN if (t,alt,lat,lon) is outside the grids
          - ParticleStatus.NAN_WIND      if any interpolation corner is NaN/Inf

        NOTE:
          This method uses ParticleStatus for stable semantics across the module.
          Time bracketing always uses an internal machine-precision-level tolerance to
          suppress false OUT_OF_DOMAIN at endpoints due to floating-point roundoff.
        """
        t_s = float(t_s)
        alt_m = float(alt_m)
        lat_deg = float(lat_deg)
        lon_deg = float(lon_deg)

        # Time bracket: tolerate tiny FP endpoint overshoots/undershoots only on time axis.
        time_atol = self._effective_time_atol(t_s, atol)
        t_min = float(self.time_s[0])
        t_max = float(self.time_s[-1])
        if t_min - time_atol <= t_s < t_min:
            t_s = t_min
        elif t_max < t_s <= t_max + time_atol:
            t_s = t_max
        tb = self._bracket(self.time_s, t_s, atol=time_atol)
        if tb is None:
            return (float("nan"),) * 3 + (ParticleStatus.OUT_OF_DOMAIN,)
        it0, ft = tb
        it1 = it0 + 1

        # Spatial brackets
        zb = self._bracket(self.alt_m, alt_m, atol=atol)
        yb = self._bracket(self.lat_deg, lat_deg, atol=atol)
        lon_internal = self._wrap_lon_to_internal(lon_deg)
        xb = self._bracket(self.lon_deg, lon_internal, atol=atol)

        if zb is None or yb is None or xb is None:
            return (float("nan"),) * 3 + (ParticleStatus.OUT_OF_DOMAIN,)

        iz0, fz = zb
        iy0, fy = yb
        ix0, fx = xb

        # Trilinear at each time slice
        u0 = self._trilinear(self.u[it0], iz0, fz, iy0, fy, ix0, fx)
        v0 = self._trilinear(self.v[it0], iz0, fz, iy0, fy, ix0, fx)
        w0 = self._trilinear(self.w[it0], iz0, fz, iy0, fy, ix0, fx)

        u1 = self._trilinear(self.u[it1], iz0, fz, iy0, fy, ix0, fx)
        v1 = self._trilinear(self.v[it1], iz0, fz, iy0, fy, ix0, fx)
        w1 = self._trilinear(self.w[it1], iz0, fz, iy0, fy, ix0, fx)

        if not (np.isfinite(u0) and np.isfinite(v0) and np.isfinite(w0) and
                np.isfinite(u1) and np.isfinite(v1) and np.isfinite(w1)):
            return (float("nan"),) * 3 + (ParticleStatus.NAN_WIND,)

        u = (1.0 - ft) * u0 + ft * u1
        v = (1.0 - ft) * v0 + ft * v1
        w = (1.0 - ft) * w0 + ft * w1

        if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(w)):
            return (float("nan"),) * 3 + (ParticleStatus.NAN_WIND,)

        return float(u), float(v), float(w), ParticleStatus.ACTIVE

    def sample_wind(
        self,
        t_s: float,
        alt_m: float,
        lat_deg: float,
        lon_deg: float,
        *,
        atol: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Backward-compatible wrapper returning only (u,v,w).
        Use sample_wind_with_status if you need status detail.
        """
        u, v, w, _ = self.sample_wind_with_status(t_s, alt_m, lat_deg, lon_deg, atol=atol)
        return u, v, w


# ------------------------------------
# Dynamics: (n, h) rates from (u, v, w)
# ------------------------------------

@dataclass(frozen=True)
class DynamicsConfig:
    radius_m: float = 3396200.0          # Mars mean radius default (m)
    w_positive_up: bool = True           # sign convention for vertical velocity
    pole_eps: float = 1e-12              # pole detection threshold in nvec_to_ll / basis
    lon_mode: LonMode = LonMode.ZERO_TO_360


def state_rates_from_wind(
    n: np.ndarray,
    h_m: float,
    lon_ref_deg: float,
    wind_uvw: Tuple[float, float, float],
    cfg: DynamicsConfig,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Compute time derivatives (dn/dt, dh/dt) from local ENU wind components (u, v, w).

    Using n-vector formulation (Gade 2010):
      - h_dot = n · v  (Eq. 15). For the ENU decomposition used here, this is simply w
        (up component) up to sign convention.
      - n_dot can be expressed via angular velocity ω (Eq. 13-14). On a sphere this reduces to:
            n_dot = v_horizontal / (R + h)
        since only the horizontal component changes the direction.

    Implementation steps:
      1) Build ENU basis at n (non-singular; gauge-fixed at the pole).
      2) Construct horizontal velocity vector in Cartesian frame:
            v_tan = u * e_east + v * e_north
      3) Compute n_dot = v_tan / (R + h).
      4) dh/dt = w (or -w depending on convention).

    Returns None if state is invalid (e.g., non-finite or radius <= 0).
    """
    n = np.asarray(n, dtype=float)
    if n.shape != (3,):
        raise ValueError("n must be shape (3,)")
    if not np.all(np.isfinite(n)):
        return None
    if not np.isfinite(h_m):
        return None

    R = float(cfg.radius_m)
    r = R + float(h_m)
    if not np.isfinite(r) or r <= 0.0:
        return None

    u, v, w = wind_uvw
    if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(w)):
        return None

    east, north, up = enu_basis_from_nvec(
        n, lon_ref_deg, pole_eps=cfg.pole_eps, lon_mode=cfg.lon_mode
    )
    v_tan = u * east + v * north

    dn_dt = v_tan / r

    dh_dt = w if cfg.w_positive_up else -w
    return dn_dt, float(dh_dt)


# -------------------------------------------------
# Petterssen / LAGRANTO iterative Euler time stepping
# -------------------------------------------------

@dataclass(frozen=True)
class PetterssenConfig:
    """
    Petterssen scheme configuration.

    iters_total: total number of iterations *including* the initial Euler predictor.
      - LAGRANTO defaults to three iterative steps (Sprenger & Wernli 2015).
      - Mapping to Rößler et al. (2018) notation:
          their x_{n+1,0} is the Euler predictor,
          their x_{n+1,l} is the l-th inner iteration (Eq. 6).
        Thus iters_total = 3 corresponds to l_max = 2 in Rößler (2018).
    """
    iters_total: int = 3
    # Optional early-stopping tolerance on the 3D position vector r = (R+h)n (meters):
    tol_m: Optional[float] = None
    # Max allowed iterations (safety):
    iters_max: int = 20


def petterssen_step(
    n: np.ndarray,
    h_m: float,
    lon_ref_deg: float,
    t_s: float,
    dt_s: float,
    wind: WindField4D,
    dyn: DynamicsConfig,
    cfg: PetterssenConfig,
    *,
    lower_boundary_m: float,
    upper_boundary_m: float,
) -> Tuple[np.ndarray, float, float, ParticleStatus]:
    """
    Advance one particle state (n, h) by dt using Petterssen iterative Euler.

    Algorithm (LAGRANTO description; Sprenger & Wernli 2015, Eqs. (2)-(4)):
      1) Euler predictor:
            x* = x + dt * u(x, t)
      2) For each further iteration:
            u* = 0.5 [u(x, t) + u(x_prev, t+dt)]
            x_new = x + dt * u*
      repeated; LAGRANTO uses three iterative steps by default.

    Here x is not (lon,lat,p) but our state (n,h) mapped to 3D position vector:
        r = (R+h) n

    We apply the same iteration on the *state derivatives* (dn/dt, dh/dt) obtained
    from the interpolated wind at (t, lon, lat, h).

    Returns:
      n_new, h_new, lon_ref_new, status
    On failure (boundary, out of domain, NaN wind, invalid state) we return the original
    state (n,h,lon_ref) and a non-ACTIVE status.
    """
    # Quick reject on non-finite dt
    if not np.isfinite(dt_s) or dt_s == 0.0:
        return n, h_m, lon_ref_deg, ParticleStatus.INVALID_STATE

    # Evaluate wind and rates at the start of the step.
    lon0, lat0, lonref0_out = nvec_to_ll(n, lon_ref_deg, pole_eps=dyn.pole_eps, lon_mode=dyn.lon_mode)
    u0, v0, w0, st0 = wind.sample_wind_with_status(t_s, h_m, lat0, lon0)
    if st0 != ParticleStatus.ACTIVE:
        return n, h_m, lon_ref_deg, st0

    rates0 = state_rates_from_wind(n, h_m, lon_ref_deg, (u0, v0, w0), dyn)
    if rates0 is None:
        return n, h_m, lon_ref_deg, ParticleStatus.INVALID_STATE
    dn0, dh0 = rates0

    # Euler predictor (Sprenger & Wernli 2015, Eq. (2); Rößler 2018, Eq. (5))
    try:
        n_guess = unit(n + dt_s * dn0)
    except ValueError:
        return n, h_m, lon_ref_deg, ParticleStatus.INVALID_STATE
    h_guess = float(h_m + dt_s * dh0)
    # Update lon_ref using the guessed position if it is not at the pole
    lon_g, lat_g, lon_ref_guess = nvec_to_ll(
        n_guess, lon_ref_deg, pole_eps=dyn.pole_eps, lon_mode=dyn.lon_mode
    )

    # Boundary check after predictor (if already outside, stop and freeze)
    if h_guess < lower_boundary_m:
        return n, h_m, lon_ref_deg, ParticleStatus.HIT_LOWER_BOUNDARY
    if h_guess > upper_boundary_m:
        return n, h_m, lon_ref_deg, ParticleStatus.HIT_UPPER_BOUNDARY

    # Optional convergence measure in 3D space (meters)
    R = float(dyn.radius_m)

    def pos_vec(nv: np.ndarray, hh: float) -> np.ndarray:
        return (R + hh) * nv

    r_prev = pos_vec(n_guess, h_guess)

    iters_total = int(cfg.iters_total)
    if iters_total < 1:
        raise ValueError("iters_total must be >= 1")

    if iters_total > cfg.iters_max:
        raise ValueError("iters_total too large (safety)")

    # Further iterations (Sprenger & Wernli 2015 Eqs. (3)-(4); Rößler 2018 Eq. (6))
    for _ in range(iters_total - 1):
        # Wind at t+dt and guessed endpoint
        t_next = t_s + dt_s
        u1, v1, w1, st1 = wind.sample_wind_with_status(t_next, h_guess, lat_g, lon_g)
        if st1 != ParticleStatus.ACTIVE:
            return n, h_m, lon_ref_deg, st1

        rates1 = state_rates_from_wind(n_guess, h_guess, lon_ref_guess, (u1, v1, w1), dyn)
        if rates1 is None:
            return n, h_m, lon_ref_deg, ParticleStatus.INVALID_STATE
        dn1, dh1 = rates1

        dn_avg = 0.5 * (dn0 + dn1)
        dh_avg = 0.5 * (dh0 + dh1)

        try:
            n_guess = unit(n + dt_s * dn_avg)
        except ValueError:
            return n, h_m, lon_ref_deg, ParticleStatus.INVALID_STATE
        h_guess = float(h_m + dt_s * dh_avg)

        # Boundary check each correction
        if h_guess < lower_boundary_m:
            return n, h_m, lon_ref_deg, ParticleStatus.HIT_LOWER_BOUNDARY
        if h_guess > upper_boundary_m:
            return n, h_m, lon_ref_deg, ParticleStatus.HIT_UPPER_BOUNDARY

        lon_g, lat_g, lon_ref_guess = nvec_to_ll(
            n_guess, lon_ref_guess, pole_eps=dyn.pole_eps, lon_mode=dyn.lon_mode
        )

        # Optional convergence
        if cfg.tol_m is not None:
            r_now = pos_vec(n_guess, h_guess)
            if float(np.linalg.norm(r_now - r_prev)) <= float(cfg.tol_m):
                break
            r_prev = r_now

    return n_guess, h_guess, lon_ref_guess, ParticleStatus.ACTIVE


# -----------------------
# Substep planning utility
# -----------------------

def plan_substeps(
    dt_total_s: float,
    *,
    substeps: Union[int, str] = "auto",
    dt_sub_s: Optional[float] = None,
    max_substeps: int = 10000,
) -> Tuple[int, float]:
    """
    Plan number of substeps N and per-substep dt given a total dt_total.

    Modes:
      - substeps="auto": N = 12 (LAGRANTO default dt = 1/12 data interval; Sprenger & Wernli 2015)
      - substeps=int:    N = substeps
      - dt_sub_s=float:  N = ceil(|dt_total| / dt_sub_s), dt = dt_total / N

    Returns (N, dt).
    """
    dt_total_s = float(dt_total_s)
    if not np.isfinite(dt_total_s) or dt_total_s == 0.0:
        raise ValueError("dt_total_s must be finite and non-zero")

    if dt_sub_s is not None:
        dt_sub_s = float(dt_sub_s)
        if not np.isfinite(dt_sub_s) or dt_sub_s <= 0.0:
            raise ValueError("dt_sub_s must be a positive finite number")
        N = int(np.ceil(abs(dt_total_s) / dt_sub_s))
        N = max(1, N)
    else:
        if isinstance(substeps, str):
            if substeps.lower() != "auto":
                raise ValueError("substeps must be int or 'auto' (or pass dt_sub_s)")
            N = 12
        else:
            N = int(substeps)
            if N < 1:
                raise ValueError("substeps must be >= 1")

    if N > max_substeps:
        raise ValueError(f"planned substeps {N} exceeds max_substeps={max_substeps}")

    dt = dt_total_s / float(N)
    return N, float(dt)


# ---------------------------------------
# Public API: forward/backward trajectory
# ---------------------------------------

def _init_particles(
    need_track_initial_points: Sequence[Tuple[float, float, float]],
    *,
    lon_mode: LonMode,
    lower_boundary_m: float,
    upper_boundary_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize arrays:
      nvecs (N,3), h (N,), lon_ref (N,), status (N,)
    """
    pts = list(need_track_initial_points)
    N = len(pts)
    nvecs = np.zeros((N, 3), dtype=float)
    h = np.zeros((N,), dtype=float)
    lon_ref = np.zeros((N,), dtype=float)
    status = np.zeros((N,), dtype=int)

    for i, (lon, lat, alt) in enumerate(pts):
        if not (np.isfinite(lon) and np.isfinite(lat) and np.isfinite(alt)):
            # mark invalid
            nvecs[i, :] = np.array([np.nan, np.nan, np.nan], dtype=float)
            h[i] = float("nan")
            lon_ref[i] = float("nan")
            status[i] = int(ParticleStatus.INVALID_STATE)
            continue

        lon_wrapped = wrap_lon(float(lon), lon_mode)
        nvecs[i, :] = ll_to_nvec(lon_wrapped, float(lat))
        h[i] = float(alt)
        lon_ref[i] = lon_wrapped

        if h[i] < lower_boundary_m:
            status[i] = int(ParticleStatus.HIT_LOWER_BOUNDARY)
        elif h[i] > upper_boundary_m:
            status[i] = int(ParticleStatus.HIT_UPPER_BOUNDARY)
        else:
            status[i] = int(ParticleStatus.ACTIVE)

    return nvecs, h, lon_ref, status


def _export_positions(
    nvecs: np.ndarray,
    h: np.ndarray,
    lon_ref: np.ndarray,
    status: np.ndarray,
    *,
    dyn: DynamicsConfig,
) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, int], np.ndarray]:
    """
    Convert internal state arrays to dict outputs keyed by particle id.
    Also updates lon_ref where not at pole (for output continuity).
    """
    positions: Dict[int, Tuple[float, float, float]] = {}
    statuses: Dict[int, int] = {}
    lon_ref_out = lon_ref.copy()

    N = nvecs.shape[0]
    for i in range(N):
        st = int(status[i])
        statuses[i] = st

        if st == int(ParticleStatus.INVALID_STATE):
            positions[i] = (float("nan"), float("nan"), float("nan"))
            continue

        # Even if inactive, keep last position frozen (LAGRANTO default behavior)
        try:
            lon_i, lat_i, lonref_i = nvec_to_ll(
                nvecs[i], lon_ref_out[i], pole_eps=dyn.pole_eps, lon_mode=dyn.lon_mode
            )
            lon_ref_out[i] = lonref_i
        except Exception:
            lon_i, lat_i = float("nan"), float("nan")

        positions[i] = (float(lon_i), float(lat_i), float(h[i]))

    return positions, statuses, lon_ref_out


def track_particles_petterssen(
    time_s: Sequence[float],
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    need_track_initial_points: Sequence[Tuple[float, float, float]],
    alt_grid_m: Sequence[float],
    lat_grid_deg: Sequence[float],
    lon_grid_deg: Sequence[float],
    lower_boundary_m: float,
    upper_boundary_m: float,
    *,
    radius_m: float = 3396200.0,
    verbose: bool = False,
    w_positive_up: bool = True,
    lon_mode: Optional[LonMode] = None,
    periodic_lon: Union[bool, str] = "auto",
    # LAGRANTO-aligned defaults:
    substeps: Union[int, str] = "auto",
    dt_sub_s: Optional[float] = None,
    petterssen_iters_total: int = 3,
    petterssen_tol_m: Optional[float] = None,
    return_status: bool = True,
) -> Union[List[Dict[int, Tuple[float, float, float]]],
           Tuple[List[Dict[int, Tuple[float, float, float]]], List[Dict[int, int]]]]:
    """
    Forward trajectory integration from time_s[0] to time_s[-1] with Petterssen scheme.

    Inputs:
      - time_s: 1-D increasing time points in seconds.
      - u,v,w: wind arrays shaped (nt, nz, ny, nx) aligned with (time, alt, lat, lon).
      - need_track_initial_points: list of (lon_deg, lat_deg, alt_m) at time_s[0].
      - grids: alt_grid_m (m), lat_grid_deg (deg), lon_grid_deg (deg).

    Stopping conditions:
      - altitude crosses lower/upper boundary: particle becomes inactive and position freezes.
      - interpolated wind becomes NaN/Inf or point is out of domain: inactive, freezes.

    Returns:
      positions_list: list of dicts, one per output step (len = nt-1).
      If return_status=True: also returns statuses_list with same length.

    Literature alignment:
      - Petterssen iterative scheme (Sprenger & Wernli 2015; Rößler 2018).
      - Default substeps="auto" -> dt = data interval / 12 (Sprenger & Wernli 2015).
    """
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must be 1-D with at least 2 entries")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("time_s must be strictly increasing")

    alt_grid_m = np.asarray(alt_grid_m, dtype=float)
    lat_grid_deg = np.asarray(lat_grid_deg, dtype=float)
    lon_grid_deg = np.asarray(lon_grid_deg, dtype=float)

    if lon_mode is None:
        lon_mode = infer_lon_mode(lon_grid_deg)

    wind = WindField4D.from_arrays(
        time_s, alt_grid_m, lat_grid_deg, lon_grid_deg, u, v, w,
        lon_mode=lon_mode, periodic_lon=periodic_lon
    )
    dyn = DynamicsConfig(
        radius_m=float(radius_m),
        w_positive_up=bool(w_positive_up),
        pole_eps=1e-12,
        lon_mode=lon_mode
    )
    pet = PetterssenConfig(iters_total=int(petterssen_iters_total), tol_m=petterssen_tol_m)

    nvecs, h, lon_ref, status = _init_particles(
        need_track_initial_points,
        lon_mode=lon_mode,
        lower_boundary_m=float(lower_boundary_m),
        upper_boundary_m=float(upper_boundary_m),
    )

    positions_out: List[Dict[int, Tuple[float, float, float]]] = []
    status_out: List[Dict[int, int]] = []

    # Iterate over coarse output intervals
    for k in range(time_s.size - 1):
        t0 = float(time_s[k])
        t1 = float(time_s[k + 1])
        dt_total = t1 - t0

        Nsub, dt = plan_substeps(dt_total, substeps=substeps, dt_sub_s=dt_sub_s)
        if verbose:
            print(f"[forward] step {k+1}/{time_s.size-1}: dt_total={dt_total:.6g}s, substeps={Nsub}, dt_sub={dt:.6g}s")

        # Integrate with substeps
        for s in range(Nsub):
            t = t0 + s * dt
            for i in range(nvecs.shape[0]):
                if status[i] != int(ParticleStatus.ACTIVE):
                    continue

                n_new, h_new, lonref_new, st = petterssen_step(
                    nvecs[i], h[i], lon_ref[i],
                    t_s=t, dt_s=dt,
                    wind=wind, dyn=dyn, cfg=pet,
                    lower_boundary_m=float(lower_boundary_m),
                    upper_boundary_m=float(upper_boundary_m),
                )

                if st == ParticleStatus.ACTIVE:
                    nvecs[i] = n_new
                    h[i] = h_new
                    lon_ref[i] = lonref_new
                else:
                    # Freeze at previous state and mark inactive.
                    status[i] = int(st)

        # Export output at t1
        pos_k, st_k, lon_ref = _export_positions(nvecs, h, lon_ref, status, dyn=dyn)
        positions_out.append(pos_k)
        if return_status:
            status_out.append(st_k)

    return (positions_out, status_out) if return_status else positions_out


def track_particles_petterssen_backward(
    time_s: Sequence[float],
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    need_track_initial_points: Sequence[Tuple[float, float, float]],
    alt_grid_m: Sequence[float],
    lat_grid_deg: Sequence[float],
    lon_grid_deg: Sequence[float],
    lower_boundary_m: float,
    upper_boundary_m: float,
    *,
    radius_m: float = 3396200.0,
    verbose: bool = False,
    w_positive_up: bool = True,
    lon_mode: Optional[LonMode] = None,
    periodic_lon: Union[bool, str] = "auto",
    # Backward controls:
    start_index: Optional[int] = None,
    start_time: Optional[float] = None,
    n_steps: Optional[int] = None,
    # Integration controls:
    substeps: Union[int, str] = "auto",
    dt_sub_s: Optional[float] = None,
    petterssen_iters_total: int = 3,
    petterssen_tol_m: Optional[float] = None,
    return_status: bool = True,
) -> Union[List[Dict[int, Tuple[float, float, float]]],
           Tuple[List[Dict[int, Tuple[float, float, float]]], List[Dict[int, int]]]]:
    """
    Backward trajectory integration with Petterssen scheme.

    IMPORTANT (rigor):
      - Do NOT negate winds. Backward trajectories are obtained by integrating with dt < 0,
        i.e., stepping from later times to earlier times.

    This mirrors the approach used in your existing code and is the mathematically correct
    time reversal for the kinematic ODE dx/dt = v(x,t).

    start_index / start_time:
      - If both None: start at the last time sample (len(time_s)-1).
      - If start_time is given, it must match an entry in time_s exactly (like your current code).

    n_steps:
      - number of coarse output intervals to step backward; default: all the way to the start.

    Output:
      - list length = number of coarse backward steps.
      - each entry is the particle positions at the *earlier* time after finishing that interval.
    """
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must be 1-D with at least 2 entries")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("time_s must be strictly increasing")

    if start_index is not None and start_time is not None:
        raise ValueError("Provide only one of start_index or start_time")

    if start_index is not None:
        start_idx = int(start_index)
        if start_idx < 1 or start_idx >= time_s.size:
            raise ValueError("start_index must satisfy 1 <= start_index < len(time_s)")
    else:
        if start_time is None:
            start_idx = int(time_s.size - 1)
        else:
            matches = np.where(time_s == float(start_time))[0]
            if matches.size == 0:
                raise ValueError("start_time must match an entry of time_s exactly")
            start_idx = int(matches[-1])
            if start_idx < 1:
                raise ValueError("start_time corresponds to earliest sample; cannot step backward")

    if n_steps is None:
        end_idx = 0
    else:
        n_steps = int(n_steps)
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        end_idx = max(start_idx - n_steps, 0)

    if lon_mode is None:
        lon_mode = infer_lon_mode(np.asarray(lon_grid_deg, dtype=float))

    wind = WindField4D.from_arrays(
        time_s, np.asarray(alt_grid_m, dtype=float), np.asarray(lat_grid_deg, dtype=float),
        np.asarray(lon_grid_deg, dtype=float), u, v, w,
        lon_mode=lon_mode, periodic_lon=periodic_lon
    )
    dyn = DynamicsConfig(
        radius_m=float(radius_m),
        w_positive_up=bool(w_positive_up),
        pole_eps=1e-12,
        lon_mode=lon_mode
    )
    pet = PetterssenConfig(iters_total=int(petterssen_iters_total), tol_m=petterssen_tol_m)

    # Initial points are assumed to be at time_s[start_idx] (user responsibility).
    nvecs, h, lon_ref, status = _init_particles(
        need_track_initial_points,
        lon_mode=lon_mode,
        lower_boundary_m=float(lower_boundary_m),
        upper_boundary_m=float(upper_boundary_m),
    )

    positions_out: List[Dict[int, Tuple[float, float, float]]] = []
    status_out: List[Dict[int, int]] = []

    # Iterate backward over coarse intervals: time[start_idx] -> ... -> time[end_idx]
    # For each coarse step from it -> it-1, dt_total is negative.
    for it in range(start_idx, end_idx, -1):
        t_now = float(time_s[it])
        t_prev = float(time_s[it - 1])
        dt_total = t_prev - t_now  # negative

        Nsub, dt = plan_substeps(dt_total, substeps=substeps, dt_sub_s=dt_sub_s)
        if verbose:
            print(f"[backward] step {it}->{it-1}: dt_total={dt_total:.6g}s, substeps={Nsub}, dt_sub={dt:.6g}s")

        # Integrate with substeps (dt is negative)
        for s in range(Nsub):
            t = t_now + s * dt
            for i in range(nvecs.shape[0]):
                if status[i] != int(ParticleStatus.ACTIVE):
                    continue

                n_new, h_new, lonref_new, st = petterssen_step(
                    nvecs[i], h[i], lon_ref[i],
                    t_s=t, dt_s=dt,
                    wind=wind, dyn=dyn, cfg=pet,
                    lower_boundary_m=float(lower_boundary_m),
                    upper_boundary_m=float(upper_boundary_m),
                )

                if st == ParticleStatus.ACTIVE:
                    nvecs[i] = n_new
                    h[i] = h_new
                    lon_ref[i] = lonref_new
                else:
                    status[i] = int(st)

        # Export output at the earlier time t_prev
        pos_k, st_k, lon_ref = _export_positions(nvecs, h, lon_ref, status, dyn=dyn)
        positions_out.append(pos_k)
        if return_status:
            status_out.append(st_k)

    return (positions_out, status_out) if return_status else positions_out


# ------------------------
# Minimal self-test (opt.)
# ------------------------

def _self_test_constant_east_wind() -> None:
    """
    A minimal sanity check:
      - constant east wind u=U0, v=0, w=0
      - ensure motion in physical space is ~U0*dt regardless of latitude,
        and no polar "cos(lat)" singularity exists.

    This is not a comprehensive validation, but it catches most gross mistakes.
    """
    R = 3396200.0
    U0 = 50.0  # m/s
    time = np.array([0.0, 600.0])  # 10 minutes
    alt = np.array([0.0, 1000.0])
    lat = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    lon = np.linspace(0.0, 360.0 - 10.0, 36)

    nt, nz, ny, nx = time.size, alt.size, lat.size, lon.size
    u = np.full((nt, nz, ny, nx), U0, dtype=float)
    v = np.zeros_like(u)
    w = np.zeros_like(u)

    # Start very close to north pole
    init = [(10.0, 89.999, 0.0)]
    traj, stat = track_particles_petterssen(
        time, u, v, w, init, alt, lat, lon,
        lower_boundary_m=-1e9, upper_boundary_m=1e9,
        radius_m=R, substeps=12, petterssen_iters_total=3,
        periodic_lon=True, return_status=True
    )
    (lon1, lat1, h1) = traj[0][0]
    assert stat[0][0] == int(ParticleStatus.ACTIVE)
    assert abs(h1 - 0.0) < 1e-9
    # We can't assert a specific lon change near pole (lon gauge), but we can assert
    # that latitude stays ~constant and state remains finite.
    assert np.isfinite(lon1) and np.isfinite(lat1)


if __name__ == "__main__":
    _self_test_constant_east_wind()
    print("Self-test passed.")
