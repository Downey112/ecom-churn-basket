import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlxtend.frequent_patterns import association_rules  # noqa: E402

from ecb.basket import frequent_itemsets, min_orders, one_hot  # noqa: E402
from ecb.catalog import DEFAULT, categorise  # noqa: E402
from ecb.stats import hosmer_lemeshow, vif_table  # noqa: E402


@pytest.mark.parametrize("desc, cat", [
    ("WHITE HANGING HEART T-LIGHT HOLDER", "Candles & Lighting"),
    ("CHRISTMAS CANDLE HOLDER", "Christmas & Seasonal"),          # first rule wins
    ("JUMBO BAG RED RETROSPOT", "Bags & Storage"),
    ("PINK SPOTTY CUP", "Kitchen & Dining"),
    ("CHOCOLATE HOT WATER BOTTLE", "Home Textiles"),               # before Kitchen's BOTTLE
    ("EARRINGS RING SET", "Jewellery & Accessories"),
    ("STRING OF PEARLS", DEFAULT),                                 # 'RING' only as a whole word
    (None, DEFAULT),
])
def test_categorise(desc, cat):
    assert categorise(desc) == cat


def test_hosmer_lemeshow_accepts_calibrated_and_rejects_shifted():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20_000)
    y = rng.binomial(1, p)
    _, dof, pval, table = hosmer_lemeshow(y, p)
    assert dof == 8 and len(table) == 10
    assert pval > 0.01
    _, _, pval_bad, _ = hosmer_lemeshow(y, np.clip(p + 0.1, 0, 0.999))
    assert pval_bad < 1e-6


def test_hosmer_lemeshow_matches_hand_computation():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    p = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
    chi2, dof, _, t = hosmer_lemeshow(y, p, groups=5)
    e1, o1, n = t["expected"], t["observed"], t["n"]
    manual = (((o1 - e1) ** 2) / e1 + (((n - o1) - (n - e1)) ** 2) / (n - e1)).sum()
    assert dof == 3 and chi2 == pytest.approx(manual)


def test_vif_independent_vs_collinear():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(5000, 3)), columns=list("abc"))
    assert vif_table(X).max() < 1.05
    X["d"] = X["a"] + X["b"] + rng.normal(scale=0.05, size=5000)
    v = vif_table(X)
    assert v["d"] > 100 and v["c"] < 1.05


def test_min_orders_is_exact_at_float_boundaries():
    assert 0.07 * 100 > 7                       # the float trap
    assert min_orders(100, 0.07) == 7
    assert min_orders(36_594, 0.01) == 366       # 365.94 rounds up
    assert min_orders(200, 0.01) == 2
    assert min_orders(100, 70 * 0.01) == 70      # computed support carries float noise


def _boundary_lines():
    """100 orders; items A and B together in exactly 7 orders (support 0.07), C alone in 6."""
    rows = [(o, i) for o in range(7) for i in "AB"] + [(o, "C") for o in range(7, 13)]
    rows += [(o, "Z%d" % o) for o in range(13, 100)]          # filler orders, all infrequent
    return pd.DataFrame(rows, columns=["order_id", "product_id"])


def test_one_hot_keeps_boundary_items_and_empty_baskets():
    b = one_hot(_boundary_lines(), 0.07)
    assert list(b.columns) == ["A", "B"] and len(b) == 100   # C (6 orders) dropped, empty baskets kept


@pytest.mark.parametrize("method", ["apriori", "fpgrowth"])
def test_frequent_itemsets_keep_itemsets_exactly_at_threshold(method):
    b = one_hot(_boundary_lines(), 0.07)
    f = frequent_itemsets(b, 0.07, max_len=3, method=method)
    assert f["key"].tolist() == [("A",), ("B",), ("A", "B")]
    assert f["support"].tolist() == pytest.approx([0.07, 0.07, 0.07])


def test_triples_and_rule_metrics_on_a_hand_built_basket():
    """20 orders: A+B+C in 4, A+B in 2 more, A alone in 2, D in 12 (filler, also frequent)."""
    rows = [(o, i) for o in range(4) for i in "ABC"] + [(o, i) for o in range(4, 6) for i in "AB"]
    rows += [(o, "A") for o in range(6, 8)] + [(o, "D") for o in range(8, 20)]
    b = one_hot(pd.DataFrame(rows, columns=["order_id", "product_id"]), 0.2)
    f = frequent_itemsets(b, 0.2, max_len=3)
    assert ("A", "B", "C") in f["key"].tolist()
    assert dict(zip(f["key"], f["support"]))[("A", "B", "C")] == pytest.approx(4 / 20)
    rules = association_rules(f, metric="confidence", min_threshold=0.5)
    r = rules[(rules["antecedents"] == frozenset("AB")) & (rules["consequents"] == frozenset("C"))].iloc[0]
    assert r["confidence"] == pytest.approx(4 / 6)                 # P(C | A,B)
    assert r["lift"] == pytest.approx((4 / 6) / (4 / 20))          # / P(C)
    assert not any(a & c for a, c in zip(rules["antecedents"], rules["consequents"]))
