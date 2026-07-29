import numpy as np
import pytest
from scipy import sparse

from gpu_dcopf_hpr.canonical_lp import CanonicalLP


def make_lp(**overrides: object) -> CanonicalLP:
    data: dict[str, object] = {
        "c": [2.0, 1.0],
        "A1": [[1.0, 1.0]],
        "b1": [1.0],
        "A2": [[1.0, -1.0]],
        "b2": [-0.2],
        "lower": [0.0, 0.0],
        "upper": [1.0, 1.0],
    }
    data.update(overrides)
    return CanonicalLP(**data)


def test_dense_blocks_remain_distinguishable_and_stack_in_order() -> None:
    lp = make_lp()

    assert (lp.n, lp.m1, lp.m2, lp.m) == (2, 1, 1, 2)
    np.testing.assert_array_equal(lp.A1, [[1.0, 1.0]])
    np.testing.assert_array_equal(lp.A2, [[1.0, -1.0]])
    np.testing.assert_array_equal(lp.A, [[1.0, 1.0], [1.0, -1.0]])
    np.testing.assert_array_equal(lp.b, [1.0, -0.2])


def test_sparse_blocks_are_accepted_and_stacked_as_sparse() -> None:
    lp = make_lp(
        A1=sparse.csr_matrix([[1.0, 1.0]]),
        A2=sparse.coo_matrix([[1.0, -1.0]]),
    )

    assert sparse.issparse(lp.A1)
    assert sparse.issparse(lp.A2)
    assert sparse.issparse(lp.A)
    np.testing.assert_array_equal(lp.A.toarray(), [[1.0, 1.0], [1.0, -1.0]])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"c": []}, "at least one"),
        ({"c": [[1.0, 2.0]]}, "one-dimensional"),
        ({"A1": [1.0, 1.0]}, "two-dimensional"),
        ({"A1": [[1.0, 1.0, 1.0]]}, "2 columns"),
        ({"b1": []}, "length 1"),
        ({"A2": [[1.0, -1.0], [0.0, 1.0]], "b2": [-0.2]}, "length 2"),
        ({"lower": [0.0]}, "length 2"),
        ({"upper": [1.0]}, "length 2"),
        ({"lower": [0.0, 2.0], "upper": [1.0, 1.0]}, "must not exceed"),
        ({"c": [np.nan, 1.0]}, "finite"),
        ({"A1": [[np.inf, 1.0]]}, "finite"),
        ({"b2": [np.nan]}, "finite"),
        ({"upper": [1.0, np.inf]}, "finite"),
    ],
)
def test_invalid_lp_data_is_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_lp(**overrides)


def test_empty_equality_and_inequality_blocks_keep_two_dimensional_shapes() -> None:
    no_equalities = make_lp(A1=[], b1=[])
    no_inequalities = make_lp(A2=[], b2=[])

    assert no_equalities.A1.shape == (0, 2)
    assert no_equalities.m1 == 0
    assert no_inequalities.A2.shape == (0, 2)
    assert no_inequalities.m2 == 0
