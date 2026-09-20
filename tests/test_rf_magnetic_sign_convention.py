"""Pin the azimuthal-B sign convention at the RF-Track boundary.

RF-Track's ``RF_FieldMap_2d`` takes its azimuthal magnetic argument along ``-e_theta``, the
opposite of the right-handed convention the qualified artifact stores (the artifact's sign is
the physical one: it satisfies the m=0 Faraday relation dEr/dz - dEz/dr = -i*omega*B_theta).
``build_volume`` therefore negates ``Bt_grid`` exactly once, at construction.

If that negation is removed or duplicated, the transverse RF Lorentz force
``q(E_r - beta_z c B_theta)`` flips sign and the field RF-Track integrates stops being a
Maxwell solution -- silently, because E_r, E_z and B_z all still read back correctly.

Probes are taken on the +x axis, where ``e_theta = +y_hat``, so the azimuthal component of the
field RF-Track reports is exactly its ``B_y``.

Note the phasors here are deliberately complex: RF-Track discards the magnetic map outright
when every supplied array is purely real (see the guard in ``build_volume``).
"""

import numpy as np
import pytest

rft = pytest.importorskip("RF_Track")

import rf_gun as rg  # noqa: E402

F_HZ = 2.865417220e9
BT = 3.0e-3
BZ = 4.0e-4
ER = 1.0e6
EZ = 2.0e6
PHASE = np.exp(0.3j)  # any genuine phase; keeps RF-Track from dropping the magnetic map


def _volume(bt=BT, bz=BZ, with_b=True):
    nz, nr = 161, 61
    hr, hz = 1.0e-4, 5.0e-5
    shape = (nz, nr)
    Er = np.full(shape, ER * PHASE)
    Ez = np.full(shape, EZ * PHASE)
    params = rg.VolumeBuildParams(
        f_hz=F_HZ, map_z0_m=0.0, z_min_m=0.0, z_max_m=(nz - 1) * hz,
        hr_m=hr, hz_m=hz, dt_mm=0.1,
    )
    kwargs = {}
    if with_b:
        kwargs = dict(Bt_grid=np.full(shape, bt * PHASE), Bz_grid=np.full(shape, bz * PHASE))
    volume = rg.build_volume(rft, Er, Ez, 0.0, params, **kwargs)
    return volume, hr, hz


def _probe(volume, hr, hz, r_m=2.0e-3, z_m=2.0e-3, t_mm_c=0.0):
    E, B = volume.get_field(r_m * 1e3, 0.0, z_m * 1e3, t_mm_c)
    return np.asarray(E).ravel(), np.asarray(B).ravel()


def test_azimuthal_b_reaches_rftrack_with_the_artifact_sign():
    """On +x (e_theta = +y_hat), RF-Track must report B_y = +Re(B_theta phasor)."""
    volume, hr, hz = _volume()
    expected = (BT * PHASE).real
    for r_m, z_m in [(2.0e-3, 2.0e-3), (1.0e-3, 4.0e-3), (3.0e-3, 1.0e-3)]:
        _, B = _probe(volume, hr, hz, r_m=r_m, z_m=z_m)
        ratio = B[1] / expected
        assert ratio == pytest.approx(1.0, abs=1e-5), (
            f"azimuthal B sign flipped at (r={r_m*1e3} mm, z={z_m*1e3} mm): "
            f"B_y/Re(B_theta) = {ratio:+.6f}, expected +1. RF-Track's RF_FieldMap_2d Bt "
            "argument is along -e_theta; build_volume must negate it exactly once."
        )


def test_other_components_are_not_negated():
    """Guard against 'fixing' the sign with direction=-1, which also conjugates E."""
    volume, hr, hz = _volume()
    E, B = _probe(volume, hr, hz)
    assert E[0] / (ER * PHASE).real == pytest.approx(1.0, abs=1e-5), "E_r must not be negated"
    assert E[2] / (EZ * PHASE).real == pytest.approx(1.0, abs=1e-5), "E_z must not be negated"
    assert B[2] / (BZ * PHASE).real == pytest.approx(1.0, abs=1e-5), "B_z must not be negated"


def test_transverse_magnetic_force_opposes_radial_electric_force():
    """The physical consequence: for beta_z > 0 the force is q(E_r - beta_z c B_theta).

    With E_r and B_theta in phase and both positive, the magnetic term must *reduce* the
    outward radial force. A flipped B_theta would add to it -- the actual failure mode.
    """
    volume, hr, hz = _volume()
    E, B = _probe(volume, hr, hz)
    assert E[0] > 0.0 and B[1] > 0.0, "test precondition: probe where E_r and B_theta are +"
    f_magnetic = -0.9 * 299792458.0 * B[1]
    assert f_magnetic < 0.0, (
        "with E_r>0 and B_theta>0 the magnetic term must oppose the radial electric force; "
        f"got {f_magnetic:+.6g} V/m alongside E_r = {E[0]:+.6g} V/m"
    )


def test_e_only_control_reports_zero_magnetic_field():
    """The --rf-magnetic-field zero control must really deliver no RF B."""
    volume, hr, hz = _volume(with_b=False)
    _, B = _probe(volume, hr, hz)
    assert np.allclose(B, 0.0), f"E-only control leaked a magnetic field: {B}"


def test_all_real_phasors_are_rejected_rather_than_silently_dropping_b():
    """RF-Track returns B=0 for all-real maps; build_volume must refuse instead."""
    nz, nr = 81, 41
    hr, hz = 1.0e-4, 5.0e-5
    shape = (nz, nr)
    params = rg.VolumeBuildParams(
        f_hz=F_HZ, map_z0_m=0.0, z_min_m=0.0, z_max_m=(nz - 1) * hz,
        hr_m=hr, hz_m=hz, dt_mm=0.1,
    )
    with pytest.raises(ValueError, match="purely real"):
        rg.build_volume(
            rft,
            np.full(shape, ER + 0j), np.full(shape, EZ + 0j), 0.0, params,
            Bt_grid=np.full(shape, BT + 0j), Bz_grid=np.full(shape, BZ + 0j),
        )


@pytest.mark.parametrize("weak_complex_phase", [True, False])
def test_nonzero_imaginary_parts_and_explicit_e_only_maps_are_not_rejected(weak_complex_phase):
    shape = (81, 41)
    params = rg.VolumeBuildParams(
        f_hz=F_HZ, map_z0_m=0, z_min_m=0, z_max_m=0.004,
        hr_m=1e-4, hz_m=5e-5, dt_mm=0.1,
    )
    # 1e-9 was wrongly classified as zero by allclose's default tolerance.
    ez = np.full(shape, EZ + (1e-9j if weak_complex_phase else 0j))
    bt = BT if weak_complex_phase else 0.0
    volume = rg.build_volume(rft, np.full(shape, ER), ez, 0.0, params,
                            Bt_grid=np.full(shape, bt), Bz_grid=np.zeros(shape))
    _, b = volume.get_field(2.0, 0, 2.0, 0)
    assert np.asarray(b).ravel()[1] == pytest.approx(bt, abs=1e-10)
