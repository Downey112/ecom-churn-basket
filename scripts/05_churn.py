"""30-day churn: binary logistic regression (inference) + decision tree, out-of-time validation.

Snapshots (features use only orders before the cut-off; label = no order in the next 30 days):
  train  2011-10-11  (label window 11 Oct - 9 Nov 2011)
  test   2011-11-10  (label window 10 Nov - 9 Dec 2011, the last 30 days of data)
  score  2011-12-10  (live scoring: the future is unknown; models refit on the test snapshot)
Population at each cut-off: customers with at least one order in the previous 365 days.
Customers with no order in that year are already gone; they are written as risk_tier 'Dormant'.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecb import plotstyle as ps  # noqa: E402
from ecb.db import ROOT, connect, engine  # noqa: E402
from ecb.stats import hosmer_lemeshow, vif_table  # noqa: E402

T_TRAIN, T_TEST, T_SCORE = "2011-10-11", "2011-11-10", "2011-12-10"
HORIZON = 30
SEED = 241
TABLES = ROOT / "docs" / "tables"
OUT = ROOT / "data" / "processed"
TIERS = [(0.8, "High"), (0.5, "Medium"), (0.0, "Low")]

# candidate predictors (all computed by fn_customer_rfm); the full set shows the collinearity problem
FULL = ["log_recency", "log_frequency", "log_purchase_rate", "log_monetary", "log_aov", "log_items_per_order",
        "log_distinct_products", "tenure_days", "orders_90d", "log_monetary_90d",
        "ipt_mean_days", "log_ipt_sd", "one_purchase_day", "lt3_purchase_days"]
# final model: drop terms that are (near-)linear combinations of others and use purchase rate
# (orders per 30 days of tenure) instead of raw frequency, which overlaps tenure; every VIF < 5
FINAL = ["log_recency", "log_purchase_rate", "log_aov", "log_distinct_products", "tenure_days",
         "orders_90d", "ipt_mean_days", "log_ipt_sd", "one_purchase_day"]
FLAGS = {"one_purchase_day", "lt3_purchase_days"}


def snapshot(as_of: str, labelled: bool = True) -> pd.DataFrame:
    q = f"""
        SELECT r.*{', l.churned' if labelled else ''}
        FROM fn_customer_rfm(%(t)s::timestamp) r
        {'JOIN fn_churn_label(%(t)s::timestamp, %(h)s) l USING (customer_id)' if labelled else ''}
        WHERE r.last_order >= %(t)s::timestamp - interval '365 days'
    """
    df = pd.read_sql(q, engine(), params={"t": as_of, "h": HORIZON})
    num = df.columns.difference(["customer_id", "as_of", "first_order", "last_order"])
    df[num] = df[num].astype(float)
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c, src in [("log_recency", "recency_days"), ("log_frequency", "frequency"),
                   ("log_monetary", "monetary"), ("log_aov", "avg_order_value"),
                   ("log_items_per_order", "avg_items_per_order"),
                   ("log_distinct_products", "distinct_products"), ("log_monetary_90d", "monetary_90d")]:
        df[c] = np.log1p(df[src])
    df["log_purchase_rate"] = np.log1p(df["frequency"] / np.maximum(df["tenure_days"], 1) * 30)
    df["one_purchase_day"] = (df["purchase_days"] < 2).astype(float)
    df["lt3_purchase_days"] = (df["purchase_days"] < 3).astype(float)
    # rhythm is undefined without repeat visits; impute 0 and let the indicator carry the effect
    df["ipt_mean_days"] = df["ipt_mean_days"].fillna(0)
    df["log_ipt_sd"] = np.log1p(np.sqrt(df["ipt_var_days"].fillna(0)))
    return df


class Standardiser:
    """z-score continuous columns with training-set moments, so odds ratios are per 1 SD."""

    def fit(self, df, cols):
        self.cont = [c for c in cols if c not in FLAGS]
        self.mu, self.sd = df[self.cont].mean(), df[self.cont].std()
        self.cols = cols
        return self

    def transform(self, df):
        X = df[self.cols].copy()
        X[self.cont] = (X[self.cont] - self.mu) / self.sd
        return X


def fit_logit(train: pd.DataFrame, cols: list[str]):
    st = Standardiser().fit(train, cols)
    model = sm.Logit(train["churned"], sm.add_constant(st.transform(train))).fit(disp=0)
    return model, st


def logit_table(model) -> pd.DataFrame:
    ci = model.conf_int()
    wald = (model.params / model.bse) ** 2
    return pd.DataFrame({
        "coef": model.params, "std_err": model.bse,
        "odds_ratio": np.exp(model.params), "or_ci_low": np.exp(ci[0]), "or_ci_high": np.exp(ci[1]),
        "wald_chi2": wald, "p_value": stats.chi2.sf(wald, 1),
    })


def metrics(y, p, name: str) -> dict:
    hl, dof, hl_p, _ = hosmer_lemeshow(y, p)
    order = np.argsort(-p)
    top = np.asarray(y)[order][: len(y) // 10].mean()
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5).ravel()
    return {"model": name, "auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "accuracy@0.5": (tp + tn) / len(y),
            "precision@0.5": tp / max(tp + fp, 1), "recall@0.5": tp / max(tp + fn, 1),
            "top_decile_churn_rate": top, "hl_chi2": hl, "hl_df": dof, "hl_p": hl_p}


def tune_tree(train: pd.DataFrame, cols: list[str]) -> DecisionTreeClassifier:
    grid = GridSearchCV(DecisionTreeClassifier(random_state=SEED),
                        {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [20, 50, 100, 200]},
                        scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    grid.fit(train[cols], train["churned"])
    print(f"  tree CV best {grid.best_params_} (CV AUC {grid.best_score_:.3f})")
    return grid.best_estimator_


def tier(p: float) -> str:
    return next(name for cut, name in TIERS if p >= cut)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    train, test = snapshot(T_TRAIN), snapshot(T_TEST)
    for name, d in [("train", train), ("test", test)]:
        print(f"{name}: {len(d):,} active customers, 30-day churn rate {d['churned'].mean():.1%}")

    # 1. full model: diagnose multicollinearity
    vif_full = vif_table(train[FULL])
    print("VIF, full candidate set:\n" + vif_full.round(1).to_string())
    vif_final = vif_table(train[FINAL])
    print("VIF, final set:\n" + vif_final.round(2).to_string())
    pd.DataFrame({"vif_full_set": vif_full, "vif_final_set": vif_final}).round(2).to_csv(TABLES / "churn_vif.csv")

    # 2. logistic regression on the final set
    logit, st = fit_logit(train, FINAL)
    coef = logit_table(logit)
    print(coef.round(4).to_string())
    print(f"  McFadden pseudo-R2 {logit.prsquared:.3f}; LR chi2 {logit.llr:.1f} (p={logit.llr_pvalue:.2g}); "
          f"n={int(logit.nobs)}")
    coef.round(5).to_csv(TABLES / "churn_logit_coefficients.csv")
    hl_in = hosmer_lemeshow(train["churned"], logit.predict())
    print(f"  Hosmer-Lemeshow in-sample: chi2 {hl_in[0]:.2f}, df {hl_in[1]}, p {hl_in[2]:.3f}")
    pd.Series({"chi2": hl_in[0], "df": hl_in[1], "p_value": hl_in[2], "pseudo_r2_mcfadden": logit.prsquared,
               "lr_chi2": logit.llr, "lr_p": logit.llr_pvalue, "n": logit.nobs}).round(5).to_csv(
        TABLES / "churn_logit_fit_train.csv", header=["value"])
    p_logit = logit.predict(sm.add_constant(st.transform(test), has_constant="add")).to_numpy()

    # baseline: recency alone
    base, st_b = fit_logit(train, ["log_recency"])
    p_base = base.predict(sm.add_constant(st_b.transform(test), has_constant="add")).to_numpy()

    # 3. decision tree (no scaling needed; same candidate set as the full model)
    tree = tune_tree(train, FULL)
    p_tree = tree.predict_proba(test[FULL])[:, 1]
    print(export_text(tree, feature_names=FULL, max_depth=3))

    y = test["churned"].to_numpy()
    res = pd.DataFrame([metrics(y, p_base, "Recency-only logistic (baseline)"),
                        metrics(y, p_logit, "Logistic regression"),
                        metrics(y, p_tree, "Decision tree")])
    print(res.round(3).to_string(index=False))
    print(f"  base-rate shift: mean predicted {p_logit.mean():.3f} vs observed {y.mean():.3f} in the test month")
    res.round(4).to_csv(TABLES / "churn_test_metrics.csv", index=False)
    _, _, _, hl_tab = hosmer_lemeshow(y, p_logit)
    hl_tab.round(3).to_csv(TABLES / "churn_hosmer_lemeshow_test.csv")
    imp = pd.Series(tree.feature_importances_, index=FULL).sort_values(ascending=False)
    imp.round(4).rename("importance").to_csv(TABLES / "churn_tree_importance.csv")

    # champion: logistic unless the tree beats it by > 0.01 AUC (interpretability tie-break)
    auc_l, auc_t = res.loc[1, "auc"], res.loc[2, "auc"]
    champion = "decision_tree" if auc_t > auc_l + 0.01 else "logistic_regression"
    print(f"champion: {champion}")

    # 4. live scoring: refit champion on the most recent labelled snapshot (test), score at T_SCORE
    live = snapshot(T_SCORE, labelled=False)
    if champion == "logistic_regression":
        m, s = fit_logit(test, FINAL)
        p_live = m.predict(sm.add_constant(s.transform(live), has_constant="add")).to_numpy()
    else:
        t2 = DecisionTreeClassifier(random_state=SEED, **{k: tree.get_params()[k]
                                    for k in ("max_depth", "min_samples_leaf")}).fit(test[FULL], test["churned"])
        p_live = t2.predict_proba(live[FULL])[:, 1]
    live["churn_probability"] = p_live
    live["risk_tier"] = live["churn_probability"].map(tier)

    seg = pd.read_parquet(OUT / "segments.parquet")
    pred = seg.merge(live[["customer_id", "churn_probability", "risk_tier"]], on="customer_id", how="left")
    pred["risk_tier"] = pred["risk_tier"].fillna("Dormant")
    pred["model_name"] = np.where(pred["risk_tier"] == "Dormant", None, f"{champion}_30d_v1")
    pred["features_as_of"] = pd.Timestamp(T_SCORE)
    print(pred["risk_tier"].value_counts().to_string())
    print(pd.crosstab(pred["persona"], pred["risk_tier"]).to_string())
    cols = ["customer_id", "cluster_id", "persona", "churn_probability", "risk_tier", "model_name",
            "features_as_of"]
    with connect() as conn:
        conn.execute("TRUNCATE customer_predictions")
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO customer_predictions ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})",
                [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row)
                 for row in pred[cols].itertuples(index=False)])
    print(f"wrote {len(pred):,} rows to customer_predictions")

    # figures
    fig, ax = ps.plt.subplots(figsize=(5.6, 5))
    for i, (name, p) in enumerate([("Logistic regression", p_logit), ("Decision tree", p_tree),
                                   ("Recency only", p_base)]):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=ps.SERIES[i], label=f"{name} (AUC {roc_auc_score(y, p):.3f})",
                linewidth=2 if i < 2 else 1.5, linestyle="-" if i < 2 else "--")
    ax.plot([0, 1], [0, 1], color=ps.AXIS, linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("ROC on the out-of-time test month")
    ps.save(fig, "churn_roc.png")

    fig, ax = ps.plt.subplots(figsize=(5.6, 5))
    for i, (name, p) in enumerate([("Logistic regression", p_logit), ("Decision tree", p_tree)]):
        _, _, _, t = hosmer_lemeshow(y, p)
        ax.plot(t["mean_p"], t["observed"] / t["n"], color=ps.SERIES[i], marker="o", markersize=6, label=name)
    ax.plot([0, 1], [0, 1], color=ps.AXIS, linewidth=1)
    ax.set_xlabel("mean predicted churn probability (decile)")
    ax.set_ylabel("observed churn rate")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Calibration by risk decile (test month)")
    ps.save(fig, "churn_calibration.png")

    c = coef.drop("const").sort_values("odds_ratio")
    fig, ax = ps.plt.subplots(figsize=(7, 4.6))
    yy = np.arange(len(c))
    ax.hlines(yy, c["or_ci_low"], c["or_ci_high"], color=ps.INK2, linewidth=1.5)
    ax.scatter(c["odds_ratio"], yy, s=55, color=ps.SERIES[0], edgecolor=ps.SURFACE, linewidth=2, zorder=3)
    ax.axvline(1, color=ps.MUTED, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(ps.plt.NullFormatter())
    ax.set_yticks(yy, [f"{n}{'' if n in FLAGS else ' (per SD)'}" for n in c.index], fontsize=8)
    ax.set_xlabel("odds ratio for churn, 95% CI (log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Logistic regression: odds ratios")
    ps.save(fig, "churn_odds_ratios.png")

    fig, ax = ps.plt.subplots(figsize=(12, 5.5))
    plot_tree(tree, feature_names=FULL, class_names=["stays", "churns"], filled=False, max_depth=3,
              impurity=False, proportion=True, rounded=True, fontsize=7, ax=ax)
    ax.set_title("Decision tree (top 3 levels)")
    ps.save(fig, "churn_tree.png")


if __name__ == "__main__":
    main()
