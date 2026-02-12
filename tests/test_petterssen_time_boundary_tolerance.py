import numpy as np

import lagranto_nvec_petterssen as pet


def _build_zero_wind_case(time_values):
    time_s = np.asarray(time_values, dtype=np.float64)
    alt_m = np.array([0.0, 1000.0], dtype=np.float64)
    lat_deg = np.array([-30.0, 0.0, 30.0], dtype=np.float64)
    lon_deg = np.array([-180.0, -60.0, 60.0, 180.0], dtype=np.float64)

    shape = (time_s.size, alt_m.size, lat_deg.size, lon_deg.size)
    zero = np.zeros(shape, dtype=np.float64)
    return time_s, alt_m, lat_deg, lon_deg, zero


def test_forward_substeps12_no_false_out_of_domain_at_time_end():
    time_s, alt_m, lat_deg, lon_deg, zero = _build_zero_wind_case([0.0, 10.0])
    init = [(0.0, 0.0, 500.0)]

    traj, status = pet.track_particles_petterssen(
        time_s,
        zero,
        zero,
        zero,
        init,
        alt_m,
        lat_deg,
        lon_deg,
        lower_boundary_m=-1e9,
        upper_boundary_m=1e9,
        substeps=12,
        periodic_lon=True,
        lon_mode=pet.LonMode.NEG180_TO_180,
        return_status=True,
    )

    assert status[-1][0] == int(pet.ParticleStatus.ACTIVE)
    assert np.allclose(np.asarray(traj[-1][0]), np.asarray(init[0]))


def test_backward_substeps12_no_false_out_of_domain_at_time_end():
    time_s, alt_m, lat_deg, lon_deg, zero = _build_zero_wind_case([0.0, 10.0, 20.0])
    init = [(0.0, 0.0, 500.0)]

    traj, status = pet.track_particles_petterssen_backward(
        time_s,
        zero,
        zero,
        zero,
        init,
        alt_m,
        lat_deg,
        lon_deg,
        lower_boundary_m=-1e9,
        upper_boundary_m=1e9,
        start_index=2,
        n_steps=2,
        substeps=12,
        periodic_lon=True,
        lon_mode=pet.LonMode.NEG180_TO_180,
        return_status=True,
    )

    assert len(traj) == 2
    assert [step[0] for step in status] == [int(pet.ParticleStatus.ACTIVE), int(pet.ParticleStatus.ACTIVE)]
    assert np.allclose(np.asarray(traj[-1][0]), np.asarray(init[0]))


def test_sample_wind_true_time_out_of_domain_still_rejected():
    time_s, alt_m, lat_deg, lon_deg, zero = _build_zero_wind_case([0.0, 10.0])
    wf = pet.WindField4D.from_arrays(
        time_s,
        alt_m,
        lat_deg,
        lon_deg,
        zero,
        zero,
        zero,
        lon_mode=pet.LonMode.NEG180_TO_180,
        periodic_lon=True,
    )

    u, v, w, st = wf.sample_wind_with_status(10.0 + 1e-3, 500.0, 0.0, 0.0)
    assert st == pet.ParticleStatus.OUT_OF_DOMAIN
    assert np.isnan(u) and np.isnan(v) and np.isnan(w)


def test_sample_wind_tiny_time_endpoint_overshoot_is_absorbed():
    time_s, alt_m, lat_deg, lon_deg, zero = _build_zero_wind_case([0.0, 10.0])
    wf = pet.WindField4D.from_arrays(
        time_s,
        alt_m,
        lat_deg,
        lon_deg,
        zero,
        zero,
        zero,
        lon_mode=pet.LonMode.NEG180_TO_180,
        periodic_lon=True,
    )

    tiny_overshoots = [
        np.nextafter(time_s[-1], np.inf),
        time_s[-1] + 1e-15,
    ]
    for t_s in tiny_overshoots:
        u, v, w, st = wf.sample_wind_with_status(t_s, 500.0, 0.0, 0.0)
        assert st == pet.ParticleStatus.ACTIVE
        assert np.isfinite(u) and np.isfinite(v) and np.isfinite(w)
