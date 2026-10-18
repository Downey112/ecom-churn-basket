"""Market-basket analysis: Apriori (mlxtend) -> support, confidence, lift; cross-checked against SQL.

Each order is a basket of distinct products. Minimum support 1% of orders (about 366 orders),
the same threshold as vw_basket_pairs, so every frequent pair must match the SQL view exactly.
Two more invariants: FP-Growth must find the same itemsets (all sizes) as Apriori, and every rule's
support, confidence and lift must match a direct recount on the basket matrix.
Rules are written to table basket_rules (defined in sql/01_schema.sql) for BI tools.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecb import plotstyle as ps  # noqa: E402
from ecb.basket import frequent_itemsets, one_hot  # noqa: E402
from ecb.db import ROOT, connect, engine  # noqa: E402

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5
TABLES = ROOT / "docs" / "tables"


def main() -> None:
    eng = engine()
    lines = pd.read_sql("SELECT DISTINCT order_id, product_id FROM order_items", eng)
    products = pd.read_sql("SELECT product_id, description, category, unit_price, cost_price FROM products",
                           eng).set_index("product_id")
    basket = one_hot(lines, MIN_SUPPORT)
    print(f"baskets {len(basket):,}; frequent single items {basket.shape[1]:,}")

    itemsets = frequent_itemsets(basket, MIN_SUPPORT, max_len=3)
    print(itemsets["size"].value_counts().sort_index().rename("frequent itemsets by size").to_string())

    # invariant: FP-Growth (no candidate generation) finds the same itemsets and supports
    fp = frequent_itemsets(basket, MIN_SUPPORT, max_len=3, method="fpgrowth")
    assert fp["key"].tolist() == itemsets["key"].tolist(), "Apriori and FP-Growth itemsets differ"
    assert np.allclose(fp["support"], itemsets["support"], rtol=0, atol=1e-12)
    print(f"check: {len(itemsets)} itemsets identical in Apriori and FP-Growth")

    # invariant: Apriori pair supports == SQL self-join view
    sql_pairs = pd.read_sql("SELECT product_a, product_b, support, pair_orders FROM vw_basket_pairs", eng)
    py_pairs = itemsets[itemsets["size"] == 2].copy()
    sql_pairs["key"] = list(zip(sql_pairs["product_a"], sql_pairs["product_b"]))
    merged = sql_pairs.merge(py_pairs[["key", "support"]], on="key", how="outer", suffixes=("_sql", "_py"))
    assert merged["support_sql"].notna().all() and merged["support_py"].notna().all(), "pair sets differ"
    max_diff = (merged["support_sql"].astype(float) - merged["support_py"]).abs().max()
    assert max_diff < 1e-5, max_diff
    print(f"check: {len(merged)} frequent pairs identical in SQL and Apriori (max support diff {max_diff:.1e})")

    rules = association_rules(itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift",
                   "antecedent support", "consequent support"]].copy()
    desc = products["description"]
    fmt = lambda s: " + ".join(sorted(desc[i] for i in s))  # noqa: E731
    rules["if_bought"] = rules["antecedents"].map(fmt)
    rules["then_also"] = rules["consequents"].map(fmt)
    rules["antecedent_ids"] = rules["antecedents"].map(lambda s: ",".join(sorted(s)))
    rules["consequent_ids"] = rules["consequents"].map(lambda s: ",".join(sorted(s)))
    rules["n_items"] = rules["antecedents"].map(len) + rules["consequents"].map(len)
    rules["orders"] = (rules["support"] * len(basket)).round().astype(int)
    cats = products["category"]
    rules["same_category"] = [len({cats[i] for i in a | c}) == 1
                              for a, c in zip(rules["antecedents"], rules["consequents"])]

    # invariant: rule metrics equal a direct recount on the basket matrix
    X = basket.to_numpy()
    col = {c: i for i, c in enumerate(basket.columns)}
    share = lambda items: X[:, [col[i] for i in items]].all(axis=1).mean()  # noqa: E731
    for r in rules.itertuples():
        s_ab, s_a, s_b = share(r.antecedents | r.consequents), share(r.antecedents), share(r.consequents)
        assert (abs(r.support - s_ab) < 1e-12 and abs(r.confidence - s_ab / s_a) < 1e-9
                and abs(r.lift - s_ab / (s_a * s_b)) < 1e-9), (r.antecedent_ids, r.consequent_ids)
    print(f"check: support, confidence and lift of all {len(rules)} rules match a direct recount")

    # ids break ties so the row order is reproducible
    rules = rules.sort_values(["lift", "confidence", "antecedent_ids", "consequent_ids"],
                              ascending=[False, False, True, True]).reset_index(drop=True)
    out = rules[["if_bought", "then_also", "antecedent_ids", "consequent_ids", "n_items", "orders",
                 "support", "confidence", "lift", "same_category"]]
    with connect() as conn:
        conn.execute("TRUNCATE basket_rules")
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO basket_rules ({', '.join(out.columns)}) VALUES ({', '.join(['%s'] * out.shape[1])})",
                [tuple(v.item() if hasattr(v, "item") else v for v in row) for row in out.itertuples(index=False)])
    TABLES.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(TABLES / "basket_rules.csv", index=False)
    print(f"rules with confidence >= {MIN_CONFIDENCE}: {len(rules)}; "
          f"cross-category {(~rules['same_category']).sum()}; median lift {rules['lift'].median():.1f}")

    # bundles = distinct pairs, ranked by lift (one row per pair, best direction)
    pairs = rules[rules["n_items"] == 2].copy()
    pairs["pair"] = [tuple(sorted(a | c)) for a, c in zip(rules.loc[pairs.index, "antecedents"],
                                                          rules.loc[pairs.index, "consequents"])]
    top = (pairs.sort_values(["confidence", "antecedent_ids"], ascending=[False, True]).drop_duplicates("pair")
           .sort_values(["lift", "confidence", "pair"], ascending=[False, False, True]).head(12))
    print(top[["if_bought", "then_also", "orders", "confidence", "lift"]].round(3).to_string(index=False))

    fig, ax = ps.plt.subplots(figsize=(9, 5.6))
    nice = lambda s: " ".join(w.capitalize() for w in s.split())  # noqa: E731
    labels = [f"{nice(a)}  →  {nice(c)}" for a, c in zip(top["if_bought"], top["then_also"])][::-1]
    y = np.arange(len(top))
    ax.barh(y, top["lift"][::-1], color=ps.SERIES[0], height=0.6)
    for yi, (l, cf) in enumerate(zip(top["lift"][::-1], top["confidence"][::-1])):
        ax.text(l + 0.6, yi, f"lift {l:.0f} · conf {cf:.0%}", va="center", fontsize=8, color=ps.INK2)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("lift (how many times more often bought together than by chance)")
    ax.set_xlim(0, top["lift"].max() * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_title("Top product bundles by lift (support ≥ 1% of orders)")
    ps.save(fig, "basket_top_bundles.png")

    fig, ax = ps.plt.subplots(figsize=(6.8, 4.8))
    sc = ax.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="Blues", s=16,
                    edgecolor=ps.AXIS, linewidth=0.3, vmin=0)
    fig.colorbar(sc, ax=ax, label="lift")
    ax.set_xscale("log")
    ax.set_xlabel("support (share of orders, log scale)")
    ax.set_ylabel("confidence")
    ax.set_title(f"All {len(rules)} association rules")
    ps.save(fig, "basket_rules_scatter.png")


if __name__ == "__main__":
    main()
