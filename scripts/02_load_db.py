"""Clean Online Retail II and load it into the PostgreSQL schema (sql/01_schema.sql + 02_views.sql).

Cleaning rules (each count is printed):
  * drop exact duplicate lines
  * drop guest lines (no Customer ID): they cannot be tied to a customer
  * drop cancellations / credit notes (Invoice 'C...') and bad-debt adjustments (Invoice 'A...')
  * product lines = StockCode starting with 5 digits, quantity > 0, price > 0;
    service codes (POST, DOT, M, C2, BANK CHARGES, AMAZONFEE, gift vouchers, ...) are not products
  * 'D' (Discount) lines on sales invoices become orders.discount_amount
Synthetic columns (age_group, payment_method, cost_price) use a fixed seed and are not used by any model.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecb.catalog import MARGIN, categorise  # noqa: E402
from ecb.db import ROOT, connect, run_sql_file  # noqa: E402

RAW = ROOT / "data" / "raw" / "online_retail_ii.parquet"
SEED = 241
FIRST_YEAR_END = pd.Timestamp("2010-12-01")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    log = [("raw lines", len(df))]
    df = df.drop_duplicates()
    log.append(("after dropping exact duplicates", len(df)))
    df = df[df["customer_id"].notna()]
    log.append(("after dropping guest lines", len(df)))
    df = df[~df["invoice"].str.startswith(("C", "A"))]
    log.append(("after dropping cancellations/adjustments", len(df)))
    disc = df[df["stock_code"] == "D"]
    prod = df[df["stock_code"].str.match(r"^\d{5}") & (df["quantity"] > 0) & (df["price"] > 0)]
    log.append(("product lines kept", len(prod)))
    for k, v in log:
        print(f"  {k:<42}{v:>10,}")
    return prod, disc


def build_tables(prod: pd.DataFrame, disc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    prod = prod.assign(customer_id=prod["customer_id"].astype(int))

    # products: modal description, median selling price as list price
    desc = (prod.dropna(subset=["description"])
                .groupby(["stock_code", "description"]).size().rename("n").reset_index()
                .sort_values(["stock_code", "n"], ascending=[True, False])
                .drop_duplicates("stock_code").set_index("stock_code")["description"])
    products = prod.groupby("stock_code")["price"].median().round(2).rename("unit_price").to_frame()
    products["description"] = desc.reindex(products.index).fillna("UNKNOWN").str.strip()
    products["category"] = products["description"].map(categorise)
    margin = products["category"].map(MARGIN) + rng.normal(0, 0.05, len(products))
    products["cost_price"] = (products["unit_price"] * (1 - margin.clip(0.2, 0.8))).round(2)
    products = products.reset_index(names="product_id")[
        ["product_id", "description", "category", "unit_price", "cost_price"]]

    # orders: one per invoice with at least one product line
    orders = (prod.groupby("invoice")
                  .agg(customer_id=("customer_id", "first"), order_timestamp=("invoice_date", "min"))
                  .reset_index(names="order_id"))
    disc_amt = (disc.assign(v=disc["quantity"].abs() * disc["price"].abs())
                    .groupby("invoice")["v"].sum())
    orders["discount_amount"] = orders["order_id"].map(disc_amt).fillna(0).round(2)
    orders["payment_method"] = rng.choice(["Card", "Bank Transfer", "PayPal"], len(orders),
                                          p=[0.6, 0.25, 0.15])

    # customers
    first = orders.groupby("customer_id")["order_timestamp"].min()
    country = prod.groupby("customer_id")["country"].agg(lambda s: s.mode().iat[0])
    yr1 = prod[prod["invoice_date"] < FIRST_YEAR_END]
    spend1 = (yr1["quantity"] * yr1["price"]).groupby(yr1["customer_id"]).sum()
    q90, q60 = spend1.quantile(0.90), spend1.quantile(0.60)
    tier = pd.Series("New", index=first.index)
    tier[spend1.index] = np.where(spend1 >= q90, "Gold", np.where(spend1 >= q60, "Silver", "Bronze"))
    customers = pd.DataFrame({
        "customer_id": first.index,
        "country": country.reindex(first.index).values,
        "signup_date": first.dt.date.values,
        "age_group": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], len(first),
                                p=[0.12, 0.28, 0.26, 0.20, 0.14]),
        "loyalty_tier": tier.values,
    })

    items = prod.rename(columns={"invoice": "order_id", "stock_code": "product_id", "price": "unit_price"})
    items = items[["order_id", "product_id", "quantity", "unit_price"]].round({"unit_price": 2})
    return {"customers": customers, "products": products,
            "orders": orders[["order_id", "customer_id", "order_timestamp", "payment_method", "discount_amount"]],
            "order_items": items}


def copy_frame(conn, table: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    cols = ", ".join(df.columns)
    with conn.cursor() as cur, cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv)") as cp:
        while chunk := buf.read(1 << 20):
            cp.write(chunk)
    print(f"  loaded {table:<14}{len(df):>10,}")


def main() -> None:
    print("cleaning")
    prod, disc = clean(pd.read_parquet(RAW))
    tables = build_tables(prod, disc)
    with connect() as conn:
        run_sql_file(conn, "01_schema.sql")
        for name in ("customers", "products", "orders", "order_items"):
            copy_frame(conn, name, tables[name])
        run_sql_file(conn, "02_views.sql")
        conn.execute("ANALYZE")
    print("done")


if __name__ == "__main__":
    main()
