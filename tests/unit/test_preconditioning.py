from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from gpu_dcopf_hpr.canonical_lp import CanonicalLP
from gpu_dcopf_hpr.hpr_generic import HPRState
from gpu_dcopf_hpr.preconditioning import LPPreconditioner, precondition_lp


def _example_lp(*, sparse_input: bool = True) -> CanonicalLP:
    A1 = np.array([[4.0, 0.0, 1.0]], dtype=np.float64)
    A2 = np.array([[0.0, 9.0, -2.0], [1.0, 3.0, 0.0]], dtype=np.float64)
    if sparse_input:
        A1 = sparse.csr_matrix(A1)
        A2 = sparse.csr_matrix(A2)
    return CanonicalLP(
        c=[3.0, -4.0, 2.0],
        A1=A1,
        b1=[7.0],
        A2=A2,
        b2=[-2.0, 1.0],
        lower=[-2.0, -3.0, 0.0],
        upper=[5.0, 4.0, 6.0],
    )


def _dense(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _assert_lp_data_equal(left: CanonicalLP, right: CanonicalLP, *, atol: float) -> None:
    np.testing.assert_allclose(_dense(left.A1), _dense(right.A1), rtol=0.0, atol=atol)
    np.testing.assert_allclose(_dense(left.A2), _dense(right.A2), rtol=0.0, atol=atol)
    np.testing.assert_allclose(left.b1, right.b1, rtol=0.0, atol=atol)
    np.testing.assert_allclose(left.b2, right.b2, rtol=0.0, atol=atol)
    np.testing.assert_allclose(left.c, right.c, rtol=0.0, atol=atol)
    np.testing.assert_allclose(left.lower, right.lower, rtol=0.0, atol=atol)
    np.testing.assert_allclose(left.upper, right.upper, rtol=0.0, atol=atol)


def test_one_simultaneous_ruiz_step_matches_hand_calculation() -> None:
    lp = CanonicalLP(
        c=[1.0, 2.0, 3.0],
        A1=sparse.csr_matrix([[4.0, 0.0, 1.0]]),
        b1=[2.0],
        A2=sparse.csr_matrix([[0.0, 9.0, 0.0]]),
        b2=[3.0],
        lower=[-1.0, -1.0, -1.0],
        upper=[1.0, 1.0, 1.0],
    )

    result = precondition_lp(lp, ruiz_iterations=1)

    np.testing.assert_array_equal(result.row_denominator, [2.0, 3.0])
    np.testing.assert_array_equal(result.column_denominator, [2.0, 3.0, 1.0])
    np.testing.assert_allclose(
        _dense(result.scaled_lp.A),
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]],
        rtol=0.0,
        atol=0.0,
    )
    step = result.diagnostics.iterations[0]
    assert step.method == "ruiz"
    assert step.iteration == 1
    assert step.norm == "infinity"
    assert step.row_before.maximum == 9.0
    assert step.column_before.minimum_positive == 1.0


def test_pock_chambolle_alpha_one_step_matches_hand_calculation() -> None:
    lp = CanonicalLP(
        c=[1.0, 1.0],
        A1=[[4.0, 1.0]],
        b1=[1.0],
        A2=[[0.0, 9.0]],
        b2=[0.0],
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
    )

    result = precondition_lp(lp, pock_chambolle=True)
    expected_rows = np.sqrt([5.0, 9.0])
    expected_columns = np.sqrt([4.0, 10.0])
    expected_matrix = np.array(
        [
            [
                4.0 / (expected_rows[0] * expected_columns[0]),
                1.0 / (expected_rows[0] * expected_columns[1]),
            ],
            [0.0, 9.0 / (expected_rows[1] * expected_columns[1])],
        ]
    )

    np.testing.assert_allclose(result.row_denominator, expected_rows, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        result.column_denominator,
        expected_columns,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        _dense(result.scaled_lp.A),
        expected_matrix,
        rtol=0.0,
        atol=1e-15,
    )
    step = result.diagnostics.iterations[0]
    assert step.method == "pock_chambolle"
    assert step.norm == "l1"


def test_ten_ruiz_iterations_are_recorded_in_order_and_preserve_nnz() -> None:
    result = precondition_lp(
        _example_lp(),
        ruiz_iterations=10,
        pock_chambolle=True,
    )

    assert result.diagnostics.ruiz_iterations == 10
    assert len(result.diagnostics.iterations) == 11
    assert [step.iteration for step in result.diagnostics.iterations[:10]] == list(range(1, 11))
    assert all(step.method == "ruiz" for step in result.diagnostics.iterations[:10])
    assert result.diagnostics.iterations[-1].method == "pock_chambolle"
    assert result.diagnostics.nnz_preserved
    assert result.diagnostics.original_nnz == _example_lp().A.nnz
    assert sparse.issparse(result.scaled_lp.A1)
    assert sparse.issparse(result.scaled_lp.A2)


def test_data_and_state_round_trip_through_complete_pipeline() -> None:
    lp = _example_lp()
    result = precondition_lp(
        lp,
        ruiz_iterations=10,
        pock_chambolle=True,
        normalize=True,
    )
    original_state = HPRState(
        x=[0.25, -0.5, 1.5],
        y=[-0.75, 0.4, 1.25],
        z=[2.0, -1.5, 0.125],
    )

    scaled_state = result.scale_state(original_state)
    recovered_state = result.recover_state(scaled_state)
    recovered_lp = result.recover_lp()

    np.testing.assert_allclose(recovered_state.x, original_state.x, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(recovered_state.y, original_state.y, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(recovered_state.z, original_state.z, rtol=0.0, atol=2e-15)
    _assert_lp_data_equal(recovered_lp, lp, atol=2e-14)
    with pytest.raises(ValueError):
        result.row_denominator[0] = 2.0
    with pytest.raises(ValueError):
        result.column_denominator[0] = 2.0


def test_primal_stationarity_and_objective_identities_hold() -> None:
    lp = _example_lp()
    result = precondition_lp(
        lp,
        ruiz_iterations=3,
        pock_chambolle=True,
        normalize=True,
    )
    original = HPRState(
        x=[0.2, -0.3, 1.1],
        y=[-0.4, 0.8, 0.6],
        z=[0.7, -0.2, 0.5],
    )
    scaled = result.scale_state(original)
    original_primal = np.asarray(lp.A @ original.x).reshape(-1) - lp.b
    scaled_primal = np.asarray(result.scaled_lp.A @ scaled.x).reshape(-1) - result.scaled_lp.b
    original_stationarity = lp.c - np.asarray(lp.A.T @ original.y).reshape(-1) - original.z
    scaled_stationarity = (
        result.scaled_lp.c - np.asarray(result.scaled_lp.A.T @ scaled.y).reshape(-1) - scaled.z
    )
    original_objective = float(lp.c @ original.x)
    scaled_objective = float(result.scaled_lp.c @ scaled.x)

    np.testing.assert_allclose(
        scaled_primal,
        original_primal / (result.row_denominator * result.b_scale),
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        scaled_stationarity,
        original_stationarity / (result.column_denominator * result.c_scale),
        rtol=0.0,
        atol=2e-15,
    )
    assert result.original_objective_from_scaled(scaled_objective) == pytest.approx(
        original_objective,
        rel=0.0,
        abs=2e-15,
    )
    assert result.objective_factor == pytest.approx(result.b_scale * result.c_scale)


def test_normalization_uses_full_b_and_c_norms_and_zero_norms_are_neutral() -> None:
    result = precondition_lp(
        _example_lp(),
        ruiz_iterations=2,
        pock_chambolle=True,
        normalize=True,
    )
    diagonal_b = result.source_lp.b / result.row_denominator
    diagonal_c = result.source_lp.c / result.column_denominator

    assert result.b_scale == pytest.approx(1.0 + np.linalg.norm(diagonal_b))
    assert result.c_scale == pytest.approx(1.0 + np.linalg.norm(diagonal_c))
    np.testing.assert_allclose(result.scaled_lp.b, diagonal_b / result.b_scale)
    np.testing.assert_allclose(result.scaled_lp.c, diagonal_c / result.c_scale)
    np.testing.assert_allclose(
        result.scaled_lp.lower,
        result.source_lp.lower * result.column_denominator / result.b_scale,
    )
    np.testing.assert_allclose(
        result.scaled_lp.upper,
        result.source_lp.upper * result.column_denominator / result.b_scale,
    )

    zero = CanonicalLP(
        c=[0.0, 0.0],
        A1=[[1.0, 0.0]],
        b1=[0.0],
        A2=np.empty((0, 2)),
        b2=[],
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
    )
    neutral = precondition_lp(zero, normalize=True)
    assert neutral.b_scale == 1.0
    assert neutral.c_scale == 1.0


def test_pock_chambolle_alpha_one_bounds_the_scaled_spectral_norm() -> None:
    result = precondition_lp(
        _example_lp(),
        ruiz_iterations=10,
        pock_chambolle=True,
    )

    assert np.linalg.norm(result.scaled_lp.dense_A(), ord=2) <= 1.0 + 5e-15


def test_dense_and_sparse_inputs_produce_identical_transforms() -> None:
    settings = {
        "ruiz_iterations": 10,
        "pock_chambolle": True,
        "normalize": True,
    }
    dense = precondition_lp(_example_lp(sparse_input=False), **settings)
    sparse_result = precondition_lp(_example_lp(sparse_input=True), **settings)

    np.testing.assert_array_equal(dense.row_denominator, sparse_result.row_denominator)
    np.testing.assert_array_equal(dense.column_denominator, sparse_result.column_denominator)
    assert dense.b_scale == sparse_result.b_scale
    assert dense.c_scale == sparse_result.c_scale
    _assert_lp_data_equal(dense.scaled_lp, sparse_result.scaled_lp, atol=0.0)
    assert dense.diagnostics == sparse_result.diagnostics


def test_zero_rows_and_columns_receive_neutral_denominators() -> None:
    lp = CanonicalLP(
        c=[1.0, 0.0, 2.0],
        A1=sparse.csr_matrix([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        b1=[0.0, 1.0],
        A2=sparse.csr_matrix((0, 3)),
        b2=[],
        lower=[-1.0, -1.0, -1.0],
        upper=[1.0, 1.0, 1.0],
    )

    result = precondition_lp(lp, ruiz_iterations=10, pock_chambolle=True)

    assert result.row_denominator[0] == 1.0
    assert result.column_denominator[1] == 1.0
    assert result.column_denominator[2] == 1.0
    assert np.all(np.isfinite(result.row_denominator))
    assert np.all(np.isfinite(result.column_denominator))
    assert result.diagnostics.nnz_preserved
    assert all(step.row_before.zero_count >= 1 for step in result.diagnostics.iterations)
    assert all(step.column_before.zero_count >= 2 for step in result.diagnostics.iterations)


def test_lp_with_no_constraint_rows_can_still_be_preconditioned() -> None:
    lp = CanonicalLP(
        c=[1.0, -1.0],
        A1=np.empty((0, 2)),
        b1=[],
        A2=np.empty((0, 2)),
        b2=[],
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
    )

    result = precondition_lp(
        lp,
        ruiz_iterations=10,
        pock_chambolle=True,
        normalize=True,
    )

    assert result.row_denominator.shape == (0,)
    np.testing.assert_array_equal(result.column_denominator, [1.0, 1.0])
    assert result.scaled_lp.A.shape == (0, 2)
    assert result.diagnostics.original_nnz == result.diagnostics.scaled_nnz == 0
    assert result.diagnostics.nnz_preserved


@pytest.mark.parametrize("value", (-1, 1.5, True))
def test_invalid_ruiz_iteration_options_are_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="ruiz_iterations"):
        precondition_lp(_example_lp(), ruiz_iterations=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("pock_chambolle", 1), ("normalize", "yes")),
)
def test_nonboolean_feature_options_are_rejected(keyword: str, value: object) -> None:
    with pytest.raises(TypeError):
        precondition_lp(_example_lp(), **{keyword: value})


def test_invalid_transform_factors_states_and_objectives_are_rejected() -> None:
    result = precondition_lp(_example_lp())

    with pytest.raises(ValueError, match="row_denominator"):
        replace(result, row_denominator=[1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="column_denominator"):
        replace(result, column_denominator=[1.0, np.inf, 1.0])
    with pytest.raises(ValueError, match="b_scale"):
        replace(result, b_scale=0.0)
    with pytest.raises(ValueError, match="c_scale"):
        replace(result, c_scale=np.nan)
    with pytest.raises(ValueError, match="state.x"):
        result.scale_state(HPRState(x=[0.0], y=[0.0], z=[0.0]))
    with pytest.raises(ValueError, match="scaled objective"):
        result.original_objective_from_scaled(np.inf)


def test_builder_returns_required_public_contract() -> None:
    lp = _example_lp()
    result = precondition_lp(lp)

    assert isinstance(result, LPPreconditioner)
    assert result.source_lp is lp
    assert result.source_lp is not result.scaled_lp
    assert result.row_denominator.shape == (result.source_lp.m,)
    assert result.column_denominator.shape == (result.source_lp.n,)
