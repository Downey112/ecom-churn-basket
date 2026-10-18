"""Shopper personas: standardise behavioural variables, PCA to remove multicollinearity, K-Means.

Population: customers in vw_customer_rfm (as of the dataset's last day) with >= 3 purchase days,
because inter-purchase variance needs at least two gaps. Customers with 1-2 purchase days get
cluster 0 ("Too little history") rather than an imputed rhythm.
Output: data/processed/segments.parquet (customer_id, cluster_id, persona), figures, result tables.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecb import plotstyle as ps  # noqa: E402
from ecb.db import ROOT, engine  # noqa: E402
from ecb.stats import vif_table  # noqa: E402

OUT = ROOT / "data" / "processed"
TABLES = ROOT / "docs" / "tables"
SEED = 241
FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value", "avg_items_per_order",
            "distinct_products", "tenure_days", "ipt_mean_days", "ipt_sd_days"]
LOGGED = [f for f in FEATURES if f != "tenure_days"]   # right-skewed -> log1p


def name_personas(profile: pd.DataFrame) -> dict[int, str]:
    """Name clusters from their median profile relative to the other clusters (z-scored medians)."""
    z = (profile - profile.mean()) / profile.std(ddof=0)
    names = {}
    for cid, row in z.iterrows():
        if row["recency_days"] > 0.8:
            names[cid] = "Lapsing"
        elif row["tenure_days"] < -0.8:
            names[cid] = "New regular"
        elif row["monetary"] > 0.8 and row["frequency"] > 0.8:
            names[cid] = "Loyal high-value"
        elif row["avg_order_value"] > 0.8:
            names[cid] = "Bulk occasional"
        elif row["frequency"] > 0.3:
            names[cid] = "Regular"
        else:
            names[cid] = "Low-spend casual"
    seen: dict[str, int] = {}
    for cid in sorted(names):
        seen[names[cid]] = seen.get(names[cid], 0) + 1
        if seen[names[cid]] > 1:
            names[cid] = f"{names[cid]} ({seen[names[cid]]})"
    return names


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rfm = pd.read_sql("SELECT * FROM vw_customer_rfm", engine())
    num = rfm.columns.difference(["customer_id", "as_of", "first_order", "last_order"])
    rfm[num] = rfm[num].astype(float)
    rfm["ipt_sd_days"] = np.sqrt(rfm["ipt_var_days"])
    rep = rfm[rfm["purchase_days"] >= 3].copy()
    print(f"customers {len(rfm):,}; clustered (>= 3 purchase days) {len(rep):,}")

    X = rep[FEATURES].copy()
    X[LOGGED] = np.log1p(X[LOGGED])
    Z = pd.DataFrame(StandardScaler().fit_transform(X), columns=FEATURES, index=X.index)
    vif = vif_table(Z)
    corr = Z.corr()
    print("VIF of standardised inputs:\n" + vif.round(2).to_string())

    pca = PCA(random_state=SEED).fit(Z)
    evr = pca.explained_variance_ratio_
    n_pc = int(np.searchsorted(np.cumsum(evr), 0.85) + 1)
    kaiser = int((pca.explained_variance_ > 1).sum())
    scores = pca.transform(Z)[:, :n_pc]
    print(f"PCA: keep {n_pc} PCs ({np.cumsum(evr)[n_pc-1]:.1%} variance; Kaiser rule says {kaiser})")
    loadings = pd.DataFrame(pca.components_[:n_pc].T, index=FEATURES,
                            columns=[f"PC{i+1}" for i in range(n_pc)])
    print(loadings.round(2).to_string())
    print(f"max VIF after PCA: {vif_table(pd.DataFrame(scores)).max():.3f}")

    ks = list(range(2, 9))
    sil, inertia, fits = [], [], {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(scores)
        fits[k] = km
        inertia.append(km.inertia_)
        sil.append(silhouette_score(scores, km.labels_))
        print(f"  k={k}: silhouette {sil[-1]:.3f}")
    best_k = ks[int(np.argmax(sil))]
    rep["cluster_id"] = fits[best_k].labels_ + 1
    names = name_personas(rep.groupby("cluster_id")[FEATURES].median())
    rep["persona"] = rep["cluster_id"].map(names)
    print(f"chosen k={best_k} (max silhouette {max(sil):.3f})")

    rest = rfm[rfm["purchase_days"] < 3].assign(cluster_id=0, persona="Too little history")
    seg = pd.concat([rep, rest])
    summary = (seg.groupby(["cluster_id", "persona"])
                  .agg(customers=("customer_id", "size"),
                       median_recency_days=("recency_days", "median"), median_orders=("frequency", "median"),
                       median_spend=("monetary", "median"), median_order_value=("avg_order_value", "median"),
                       median_gap_days=("ipt_mean_days", "median"), revenue=("monetary", "sum"))
                  .reset_index())
    summary["revenue_share"] = summary["revenue"] / summary["revenue"].sum()
    print(summary.round(2).to_string(index=False))

    seg[["customer_id", "cluster_id", "persona"]].to_parquet(OUT / "segments.parquet", index=False)
    summary.round(4).to_csv(TABLES / "segments_summary.csv", index=False)
    loadings.round(3).to_csv(TABLES / "pca_loadings.csv")
    vif.round(2).rename("vif").to_csv(TABLES / "segment_input_vif.csv")
    pd.DataFrame({"k": ks, "silhouette": sil, "inertia": inertia}).round(4).to_csv(
        TABLES / "silhouette.csv", index=False)

    fig, ax = ps.plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    ax.grid(False)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else ps.INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation of standardised behavioural inputs")
    ps.save(fig, "seg_correlation.png")

    fig, axes = ps.plt.subplots(1, 2, figsize=(10.5, 3.8))
    pcs = np.arange(1, len(evr) + 1)
    axes[0].bar(pcs, evr, color=ps.SERIES[0], width=0.6, label="per component")
    axes[0].plot(pcs, np.cumsum(evr), color=ps.INK2, marker="o", markersize=5, linewidth=1.5,
                 label="cumulative")
    axes[0].axhline(0.85, color=ps.MUTED, linestyle="--", linewidth=1)
    axes[0].text(0.6, 0.86, "85% threshold", color=ps.MUTED, va="bottom", fontsize=8)
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("share of variance")
    axes[0].legend(loc="center right")
    axes[0].set_title(f"PCA: {n_pc} components keep {np.cumsum(evr)[n_pc-1]:.0%}")
    axes[1].plot(ks, sil, color=ps.SERIES[0], marker="o", markersize=6)
    axes[1].scatter([best_k], [max(sil)], s=90, facecolor=ps.SERIES[1], edgecolor=ps.SURFACE,
                    linewidth=2, zorder=3)
    axes[1].annotate(f"k = {best_k}", (best_k, max(sil)), xytext=(8, 4), textcoords="offset points")
    axes[1].set_xlabel("number of clusters k")
    axes[1].set_ylabel("silhouette score")
    axes[1].set_title("K-Means: silhouette by k")
    ps.save(fig, "seg_pca_silhouette.png")

    fig, ax = ps.plt.subplots(figsize=(7, 5.2))
    for i, cid in enumerate(sorted(rep["cluster_id"].unique())):
        m = rep["cluster_id"].values == cid
        ax.scatter(scores[m, 0], scores[m, 1], s=9, alpha=0.55, color=ps.SERIES[i], linewidths=0,
                   label=f"{cid}: {names[cid]} (n={m.sum():,})")
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%} of variance)")
    ax.legend(markerscale=2.5, fontsize=8)
    ax.set_title("Shopper personas in principal-component space")
    ps.save(fig, "seg_clusters.png")


if __name__ == "__main__":
    main()
