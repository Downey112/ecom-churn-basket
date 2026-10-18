import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
