import numpy as np

from gpu_dcopf_hpr.hpr_generic import (
    HPRState,
    construct_spectral_proximal,
    halpern_update,
    hpr_step,
    reflect_state,
)
from gpu_dcopf_hpr.toy_problems import analytic_toy_case


def state(scale: float) -> HPRState:
    return HPRState(
        y=scale * np.array([1.0, -2.0]),
        z=scale * np.array([3.0, -4.0]),
        x=scale * np.array([5.0, -6.0]),
    )


def test_reflection_is_two_times_proximal_minus_current() -> None:
    current = state(1.0)
    proximal = state(3.0)

    reflected = reflect_state(current, proximal)

    np.testing.assert_allclose(reflected.y, [5.0, -10.0], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(reflected.z, [15.0, -20.0], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(reflected.x, [25.0, -30.0], rtol=0.0, atol=1e-14)


def test_k_zero_halpern_weights_are_one_half_each() -> None:
    anchor = state(1.0)
    reflected = state(3.0)

    result = halpern_update(anchor, reflected, iteration=0)

    np.testing.assert_allclose(result.y, state(2.0).y, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(result.z, state(2.0).z, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(result.x, state(2.0).x, rtol=0.0, atol=1e-14)


def test_large_k_halpern_weight_approaches_reflected_state() -> None:
    anchor = state(1.0)
    reflected = state(3.0)
    iteration = 1_000_000

    result = halpern_update(anchor, reflected, iteration=iteration)

    expected = (anchor.x + (iteration + 1) * reflected.x) / (iteration + 2)
    np.testing.assert_allclose(result.x, expected, rtol=1e-14, atol=1e-14)
    assert np.linalg.norm(result.x - reflected.x) < 2e-5


def test_anchor_is_detached_and_not_corrupted_by_result_mutation() -> None:
    source = np.array([1.0, 2.0])
    anchor = HPRState(y=source, z=source, x=source)
    reflected = HPRState(y=[3.0, 4.0], z=[3.0, 4.0], x=[3.0, 4.0])

    result = halpern_update(anchor, reflected, iteration=0)
    source[:] = -99.0
    result.x[:] = 99.0

    np.testing.assert_array_equal(anchor.x, [1.0, 2.0])
    np.testing.assert_array_equal(anchor.y, [1.0, 2.0])


def test_spectral_proximal_is_psd_and_total_metric_is_positive_definite() -> None:
    lp = analytic_toy_case().lp
    proximal = construct_spectral_proximal(lp)
    gram = lp.dense_A() @ lp.dense_A().T

    assert np.linalg.eigvalsh(proximal.matrix)[0] >= -1e-12
    np.testing.assert_allclose(
        gram + proximal.matrix,
        proximal.tau * np.eye(lp.m),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.linalg.eigvalsh(gram + proximal.matrix)[0] > 0.0


def test_one_step_preserves_box_projection_identity() -> None:
    lp = analytic_toy_case().lp
    initial = HPRState(y=np.zeros(lp.m), z=np.zeros(lp.n), x=np.zeros(lp.n))
    proximal = construct_spectral_proximal(lp)

    step = hpr_step(lp, initial, initial.detached_copy(), proximal, iteration=0, sigma=1.0)

    expected_x = np.clip(initial.x + lp.dense_A().T @ initial.y - lp.c, 0.0, 1.0)
    np.testing.assert_allclose(step.proximal.x, expected_x, rtol=0.0, atol=1e-14)
