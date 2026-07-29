import numpy as np
import pytest

from gpu_dcopf_hpr.projections import (
    project_box,
    project_dual_set,
    project_nonnegative,
)


def test_box_projection_handles_below_inside_and_above_without_mutation() -> None:
    values = np.array([-2.0, 0.25, 3.0])
    original = values.copy()

    projected = project_box(values, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    np.testing.assert_array_equal(projected, [0.0, 0.25, 1.0])
    np.testing.assert_array_equal(values, original)


def test_nonnegative_projection() -> None:
    np.testing.assert_array_equal(
        project_nonnegative([-1.0, -0.0, 0.5]),
        [0.0, 0.0, 0.5],
    )


def test_dual_set_projection_preserves_free_rows_and_clips_inequality_rows() -> None:
    projected = project_dual_set([-2.0, 3.0, -4.0, 5.0], equality_rows=2)

    np.testing.assert_array_equal(projected, [-2.0, 3.0, 0.0, 5.0])


def test_dual_set_projection_supports_empty_equality_or_inequality_blocks() -> None:
    np.testing.assert_array_equal(project_dual_set([-1.0, 2.0], 0), [0.0, 2.0])
    np.testing.assert_array_equal(project_dual_set([-1.0, 2.0], 2), [-1.0, 2.0])
    np.testing.assert_array_equal(project_dual_set([], 0), [])


def test_box_projection_handles_finite_floating_point_neighbors() -> None:
    just_below_zero = np.nextafter(0.0, -1.0)
    just_above_one = np.nextafter(1.0, 2.0)

    projected = project_box(
        [just_below_zero, np.nextafter(1.0, 0.0), just_above_one],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    )

    np.testing.assert_array_equal(projected, [0.0, np.nextafter(1.0, 0.0), 1.0])


@pytest.mark.parametrize(
    "call",
    [
        lambda: project_box([0.0], [0.0, 0.0], [1.0, 1.0]),
        lambda: project_box([0.0], [1.0], [0.0]),
        lambda: project_nonnegative([np.nan]),
        lambda: project_dual_set([0.0], -1),
        lambda: project_dual_set([0.0], 2),
    ],
)
def test_invalid_projection_inputs_are_rejected(call: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()
