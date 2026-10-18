"""Statistical helpers not provided (in the needed form) by statsmodels / scikit-learn."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif_table(X: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor per column (an intercept is added, as VIF assumes)."""
    X = pd.DataFrame(X)
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    return pd.Series([variance_inflation_factor(Xc, i + 1) for i in range(X.shape[1])],
                     index=X.columns, name="vif")


def hosmer_lemeshow(y, p, groups: int = 10) -> tuple[float, int, float, pd.DataFrame]:
    """Hosmer-Lemeshow goodness-of-fit test on deciles of predicted risk.

    Returns (chi2 statistic, degrees of freedom = groups - 2, p-value, per-group table).
    A small p-value means observed and predicted event counts disagree (poor calibration).
    """
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["g"] = pd.qcut(df["p"].rank(method="first"), groups, labels=False)
    t = df.groupby("g").agg(n=("y", "size"), observed=("y", "sum"), expected=("p", "sum"),
                            mean_p=("p", "mean"))
    exp1, exp0 = t["expected"], t["n"] - t["expected"]
    obs1, obs0 = t["observed"], t["n"] - t["observed"]
    chi2 = float((((obs1 - exp1) ** 2) / exp1 + ((obs0 - exp0) ** 2) / exp0).sum())
    dof = groups - 2
    return chi2, dof, float(stats.chi2.sf(chi2, dof)), t
