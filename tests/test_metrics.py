from shared.interfaces import Execution
from shared.metrics import bootstrap, cell_equal, execution_equal, paired_bootstrap


def e(rows, columns=None):
    return Execution(columns or ["a", "b"], rows)


def test_multiset():
    assert execution_equal(
        e([(1, None), (1, None), (2, "x")]), e([(2, "x"), (1, None), (1, None)])
    )
    assert not execution_equal(e([(1, None), (2, None)]), e([(1, None), (1, None)]))


def test_order_and_shape():
    assert not execution_equal(e([(1, 2), (3, 4)]), e([(3, 4), (1, 2)]), True)
    assert not execution_equal(e([(1, 2)]), e([(2, 1)]))
    assert not execution_equal(e([], ["a"]), e([], ["a", "b"]))
    assert execution_equal(e([]), e([]))


def test_ties():
    assert execution_equal(
        e([(1, "b"), (1, "a"), (2, "c")]),
        e([(1, "a"), (1, "b"), (2, "c")]),
        True,
        [(0, 2), (2, 3)],
    )


def test_tolerance_null_types():
    assert (
        cell_equal(1.0, 1.00000001)
        and not cell_equal(None, 0)
        and not cell_equal("1", 1)
    )
    assert execution_equal(e([(1.0, None)]), e([(1.00000001, None)]))


def test_truncation():
    x = e([(1, 2)])
    x.truncated = True
    assert execution_equal(x, x) is None


def test_bootstrap():
    assert bootstrap([0, 1]) == bootstrap([0, 1])
    assert bootstrap([]) is None
    assert paired_bootstrap({"q": 1}, {"q": 0})["mean"] == 1
