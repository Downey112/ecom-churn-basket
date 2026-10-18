"""Download UCI Online Retail II (CC BY 4.0) and cache it as parquet.

Source: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = RAW / "online_retail_ii.parquet"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"[skip] {OUT} exists")
        return
    zpath = RAW / "online_retail_ii.zip"
    if not zpath.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        xlsx = io.BytesIO(z.read("online_retail_II.xlsx"))
    print("reading both sheets (2009-10, 2010-11); takes ~1 min")
    sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = ["invoice", "stock_code", "description", "quantity",
                  "invoice_date", "price", "customer_id", "country"]
    df["description"] = df["description"].astype("string")
    df["country"] = df["country"].astype("string")
    df.to_parquet(OUT, index=False)
    print(f"[ok] {len(df):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
