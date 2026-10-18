"""Market-basket helpers: exact support thresholds, one-hot baskets and frequent itemsets."""
import math
from decimal import Decimal

import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth


def min_orders(n_orders: int, min_support: float) -> int:
    """Smallest order count with support >= min_support, in exact decimal arithmetic like the SQL view.

    Float arithmetic can push the threshold over an integer (0.07 * 100 == 7.000000000000001)
    and silently drop an itemset that sits exactly on it. Rounding to 10 places first also absorbs
    float noise in a computed min_support (70 * 0.01 == 0.7000000000000001).
    """
    return math.ceil(Decimal(str(round(min_support, 10))) * n_orders)


def one_hot(lines: pd.DataFrame, min_support: float) -> pd.DataFrame:
    """Order x product boolean matrix, keeping only individually frequent items (downward closure)."""
    n = lines["order_id"].nunique()
    item_orders = lines.groupby("product_id")["order_id"].nunique()
    keep = item_orders[item_orders >= min_orders(n, min_support)].index
    sub = lines[lines["product_id"].isin(keep)]
    m = pd.crosstab(sub["order_id"], sub["product_id"]).gt(0)
    # baskets with no frequent item still count in the denominator
    empty = pd.DataFrame(False, index=pd.Index(sorted(set(lines["order_id"]) - set(m.index))), columns=m.columns)
    return pd.concat([m, empty])


def frequent_itemsets(basket: pd.DataFrame, min_support: float, max_len: int, method: str = "apriori") -> pd.DataFrame:
    """Frequent itemsets with an integer count threshold, sorted by (size, items) for comparison.

    mlxtend compares supports (or counts) against min_support in floating point; passing a support half
    an order below the exact integer threshold makes every comparison unambiguous.
    """
    n = len(basket)
    s = (min_orders(n, min_support) - 0.5) / n
    if method == "apriori":
        # dense apriori materialises orders x candidate pairs (~5 GiB here); low_memory scans per prefix
        out = apriori(basket, min_support=s, use_colnames=True, max_len=max_len, low_memory=True)
    elif method == "fpgrowth":
        out = fpgrowth(basket, min_support=s, use_colnames=True, max_len=max_len)
    else:
        raise ValueError(method)
    out["size"] = out["itemsets"].map(len)
    out["key"] = out["itemsets"].map(lambda x: tuple(sorted(x)))
    return out.sort_values(["size", "key"]).reset_index(drop=True)
