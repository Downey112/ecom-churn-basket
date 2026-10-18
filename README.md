# E-Commerce Customer Churn & Market-Basket Intelligence

This project turns raw retail transaction logs into two retention tools:
- a **retention engine**: shopper personas plus a 30-day churn score for every customer, written back to PostgreSQL;
- a **cross-selling engine**: product bundles mined with Apriori.

It combines database design (PostgreSQL schema, views and SQL functions) with multivariate statistics and data mining: PCA, K-Means, Apriori, logistic regression and a decision tree.

**Data:** [UCI Online Retail II](https://doi.org/10.24432/C5CG6D) (CC BY 4.0). It holds 1,067,371 invoice lines from a UK online gift wholesaler, covering 1 Dec 2009 to 9 Dec 2011. After cleaning: 5,852 customers in 41 countries, 4,619 products, 36,594 orders and 776,577 order lines.

## Headline results

| Question | Answer |
|---|---|
| Who are the shoppers? | Three personas among the 3,169 customers with at least 3 purchase days: **Loyal high-value** (1,050 customers, 72% of all revenue), **Lapsing** (1,590), **New regular** (529). Another 2,683 customers have too little history to profile. The silhouette score is 0.28, so the clusters overlap: treat them as a useful summary, not as sharply separated groups. |
| What sells together? | 873 frequent itemsets at 1% support and 249 rules at ≥ 50% confidence (median lift 11.6). The strongest rules complete a **design range**: buyers of *Poppy's Playhouse Livingroom* also buy the *Kitchen* piece 89% of the time (lift 53). Three *Regency* teacups → the third colour, 90% confidence. |
| Who will churn in the next 30 days? | Logistic regression scores **AUC 0.755** on an unseen later month, against 0.742 for a decision tree and 0.684 for recency alone. The riskiest 10% of customers churned at 90%, against a base rate of 67.5%. |
| Where should retention effort go? | 346 *Loyal high-value* customers now sit in the Medium tier (p = 0.5–0.8), with **£1.98M** of lifetime revenue. They are the save list: valuable, not yet gone. |

## Architecture

```
UCI xlsx ──01_fetch──▶ parquet ──02_load (clean + COPY)──▶ PostgreSQL
                                                            ├─ customers, products, orders, order_items
                                                            ├─ fn_customer_rfm(as_of) / vw_customer_rfm
                                                            ├─ fn_churn_label(as_of, horizon)
                                                            ├─ vw_basket_pairs (Apriori-pruned self-join)
                                                            └─ vw_customer_360
         03_segment: standardise → PCA → K-Means (silhouette) ─┐
         04_basket : Apriori → support/confidence/lift ────────┼─▶ basket_rules
         05_churn  : logistic + decision tree, out-of-time ────┴─▶ customer_predictions
```

## Database layer

`sql/01_schema.sql` holds the five tables from the brief, with keys, foreign keys, CHECK constraints and indexes. Two deliberate changes:

- **`order_items.unit_price` was added.** The same product sold at different prices over time and by volume. Revenue must use the price actually charged, not today's list price in `products.unit_price`.
- **`customers.state` became `country`.** The source records country only.

`sql/02_views.sql`:

- **`fn_customer_rfm(as_of)`** computes, using only orders before `as_of`:
  - Recency from `MAX(order_timestamp)`;
  - Frequency as `COUNT(DISTINCT order_id)`;
  - Monetary as `SUM(quantity * unit_price)`;
  - the mean and sample variance (`VAR_SAMP`) of the day gaps between purchases, via `LAG()` over distinct purchase days;
  - tenure, distinct products, and 90-day activity.

  Because it is parameterised by date, the churn model can be trained on past snapshots without seeing the future.
- **`vw_customer_rfm`** is that function evaluated at the day after the last order.
- **`fn_churn_label(as_of, 30)`** returns 1 if a customer places no order in `[as_of, as_of + 30 days)`.
- **`vw_basket_pairs`** self-joins `order_items` on `order_id` (with `a.product_id < b.product_id`) to count co-occurring pairs, then computes support, confidence in both directions, and lift. It applies Apriori's *downward-closure* rule in SQL: only items with ≥ 1% support can be in a frequent pair, which shrinks the join from millions of pairs to the 562 frequent items. It runs in about 5 s.
- **`vw_customer_360`** has one row per customer (RFM, persona, churn score) for DBeaver or Power BI.

`sql/03_dbeaver_workbook.sql` contains ready-to-run queries: RFM quintile scoring, cross-category pairs, the retention worklist, and persona × risk tier.

### Data provenance (read this before using a column)

| Column | Status |
|---|---|
| orders, timestamps, products, quantities, prices, countries | **Real** (source) |
| `signup_date` | Derived: first recorded purchase (the data starts Dec 2009, so earlier customers are left-censored) |
| `loyalty_tier` | Derived: first-year spend (Dec 2009 – Nov 2010); Gold = top 10%, Silver = next 30% |
| `category` | Derived: keyword rules on the description (`src/ecb/catalog.py`); 32% fall to "Home Decor" |
| `discount_amount` | Real but **always 0**: this wholesaler issued discounts as credit notes, which are removed with cancellations |
| `age_group`, `payment_method`, `cost_price` | **Synthetic** (seeded) placeholders so the schema is complete. Never used by any model. |

Cleaning (`scripts/02_load_db.py`):

| Step | Lines remaining |
|---|---|
| Raw | 1,067,371 |
| Exact duplicates removed | 1,033,036 |
| Guest lines removed (no customer ID) | 797,885 |
| Cancellations and adjustments removed | 779,495 |
| Non-product codes removed (postage, fees, vouchers) and non-positive lines dropped | 776,577 |

## Statistical and data-mining pipeline

### 1. PCA and K-Means personas (`scripts/03_segment.py`)

**Inputs.** Nine behavioural variables: recency, frequency, monetary, order value, items per order, distinct products, tenure, and the mean and SD of the inter-purchase gap. All are log-transformed where skewed, then standardised.

**Multicollinearity.** The inputs are badly collinear: VIF is 1,114 for monetary, 535 for frequency and 392 for order value, because monetary ≈ frequency × order value.

**PCA.** PCA removes the collinearity (VIF of the components = 1.00). Four components keep 88% of the variance; the Kaiser rule says 3.
- PC1 is overall engagement and spend.
- PC2 is a slow, irregular purchase rhythm.
- PC3 is tenure against basket size.
- PC4 is mostly recency.

**Clustering.** K-Means on those four component scores, k = 2–8 (`n_init=20`); the silhouette score peaks at k = 3.

![PCA and silhouette](docs/figures/seg_pca_silhouette.png)

| Persona | Customers | Median orders | Median spend | Median gap (days) | Revenue share |
|---|---:|---:|---:|---:|---:|
| Loyal high-value | 1,050 | 14 | £5,682 | 49 | 72.2% |
| Lapsing | 1,590 | 5 | £1,276 | 115 | 13.9% |
| New regular | 529 | 4 | £1,213 | 38 | 4.6% |
| Too little history (1–2 purchase days) | 2,683 | 1 | £326 | – | 9.3% |

Customers with fewer than 3 purchase days are kept out of K-Means rather than given an invented purchase rhythm: a gap variance needs at least two gaps.

![Clusters](docs/figures/seg_clusters.png)

### 2. Market-basket analysis (`scripts/04_basket.py`)

Apriori (`mlxtend`) runs on the order × product matrix at min support 1% (366 orders). It finds 562 frequent items, 259 pairs and 52 triples, giving 249 rules at confidence ≥ 0.5.

**Invariant check:** the script asserts that all 259 frequent pairs, and their supports, are identical in the SQL view and in Apriori. Two independent implementations agree.

![Top bundles](docs/figures/basket_top_bundles.png)

**Reading the result.** High lift here mostly means "completing a collection": same product, different colourway or room. That is still actionable as "complete the set" prompts and bundle pricing. True cross-category cross-sell is rare at 1% support. Note that some rules look cross-category only because the keyword categories split one design range (e.g. *Playhouse Kitchen* lands in Kitchen & Dining). All rules, including 3-item ones, are in table `basket_rules` and in `docs/tables/basket_rules.csv`.

### 3. 30-day churn (`scripts/05_churn.py`)

**Snapshots.**

| Snapshot | Features use orders before | Label window | Active customers | Churn rate |
|---|---|---|---:|---:|
| Train | 11 Oct 2011 | 11 Oct – 9 Nov | 4,378 | 72.9% |
| Test | 10 Nov 2011 | 10 Nov – 9 Dec (the last month of data) | 4,350 | 67.5% |
| Live scoring | 10 Dec 2011 | the future (unknown) | – | – |

"Active" means at least one order in the previous 365 days. The test month is later than the training month, so this is an out-of-time test, not a random split.

**Logistic regression (statsmodels).**

- **Multicollinearity.** With the full candidate set, VIF reaches 1,679 (log monetary), 815 (log frequency) and 464 (log order value). Dropping the redundant terms and replacing raw frequency with **purchase rate** (orders per 30 days of tenure) brings every VIF under 2.9 (`docs/tables/churn_vif.csv`).
- **Odds ratios** are per 1 SD of each continuous predictor:

| Predictor | Odds ratio | 95% CI | Wald χ² | p |
|---|---:|---|---:|---:|
| orders in the last 90 days | 0.41 | 0.34–0.49 | 95.7 | <0.001 |
| bought on only one day | 2.37 | 1.75–3.21 | 31.2 | <0.001 |
| distinct products bought (log) | 0.75 | 0.67–0.84 | 25.9 | <0.001 |
| mean inter-purchase gap | 1.35 | 1.20–1.53 | 24.5 | <0.001 |
| purchase rate (log) | 0.82 | 0.74–0.90 | 15.3 | <0.001 |
| recency (log) | 0.90 | 0.80–1.02 | 2.7 | 0.10 |
| order value (log) | 0.93 | 0.86–1.02 | 2.5 | 0.12 |
| gap SD (log) | 1.07 | 0.94–1.20 | 1.1 | 0.30 |
| tenure | 0.98 | 0.87–1.10 | 0.1 | 0.72 |

- **Overall fit.** McFadden pseudo-R² 0.149; likelihood-ratio χ² = 763 (p < 10⁻¹⁵⁰).
- **Hosmer-Lemeshow (training data):** χ² = 6.56, df = 8, **p = 0.59**, so the model fits the data it was trained on.

![Odds ratios](docs/figures/churn_odds_ratios.png)

Once 90-day activity is in the model, recency is no longer significant, because the two carry the same information. Breadth matters too: customers who buy many different products are less likely to churn. So does rhythm: customers with long usual gaps between purchases are more likely to have no order in any given 30-day window.

**Decision tree (scikit-learn).** Depth and minimum leaf size were tuned by 5-fold cross-validated AUC, giving depth 4 and at least 100 customers per leaf. It splits first on orders in the last 90 days (importance 0.54), then on purchase rate (0.27).

**Test-month results (`docs/tables/churn_test_metrics.csv`):**

| Model | AUC | PR-AUC | Brier | Churn rate in top 10% |
|---|---:|---:|---:|---:|
| Recency only (baseline) | 0.684 | 0.803 | 0.202 | 83.9% |
| **Logistic regression** | **0.755** | **0.841** | **0.178** | **89.7%** |
| Decision tree | 0.742 | 0.820 | 0.181 | 88.7% |

![ROC](docs/figures/churn_roc.png) ![Calibration](docs/figures/churn_calibration.png)

**Calibration caveat.** Hosmer-Lemeshow *rejects* both models on the test month (p < 0.001). The cause is a shift in the base rate, not the shape of the model: the test month is the pre-Christmas rush, so churn fell from 72.9% to 67.5%. Predictions average 71.6% against 67.5% observed, and nearly every decile sits below the diagonal in the calibration plot. The ranking holds up (AUC); the probability level drifts with the season. For this reason the live model is refit on the most recent labelled month before scoring. A production version would recalibrate monthly, or add month-of-year terms using several years of history.

**Choosing the live model.** Logistic regression wins on every test metric. The pre-set rule was to prefer the interpretable model unless the tree beat it by more than 0.01 AUC. The final model is refit on the test snapshot and scores the 4,261 customers active at 10 Dec 2011. Tiers: **High** p ≥ 0.8, **Medium** 0.5–0.8, **Low** < 0.5. The other 1,591 customers have had no order for a year and are marked **Dormant** without a score.

| Persona | Low | Medium | High | Dormant |
|---|---:|---:|---:|---:|
| Loyal high-value | 664 | 346 | 2 | 38 |
| Lapsing | 66 | 885 | 398 | 241 |
| New regular | 223 | 253 | 12 | 41 |
| Too little history | 55 | 446 | 911 | 1,271 |

## Run it

On Windows with PostgreSQL installed (no Docker needed: it starts a project-local cluster on port 5433):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\run_all.ps1          # fetch -> load -> segment -> basket -> churn -> tests (~5 min)
```

With Docker (tested with Docker Desktop 4.95 on Windows 11; gives the same results as the local cluster). Both use port 5433, so stop the local cluster first (`.\scripts\local_pg.ps1 stop`):

```bash
docker compose up -d
export DATABASE_URL=postgresql://postgres:retail@localhost:5433/retail
for s in 01_fetch_data 02_load_db 03_segment 04_basket 05_churn; do python scripts/$s.py; done
pytest -q tests
```

The same in PowerShell (`run_all.ps1` skips the local cluster when `DATABASE_URL` is set):

```powershell
docker compose up -d --wait
$env:DATABASE_URL = 'postgresql://postgres:retail@localhost:5433/retail'
.\run_all.ps1
```

**DBeaver:** add a PostgreSQL connection to `localhost:5433`, database `retail`, user `postgres` (no password for the local cluster; `retail` for Docker). Then open `sql/03_dbeaver_workbook.sql`.

**Tests** (18):
- unit tests for the keyword categoriser, Hosmer-Lemeshow (against a hand computation) and VIF;
- database invariant tests, including:
  - RFM totals reconcile with the base tables;
  - `fn_customer_rfm` does not look ahead past its cut-off;
  - churn labels match their definition;
  - basket metrics stay within bounds;
  - every customer has exactly one prediction, with a tier consistent with its probability.

## Limitations

- **One wholesaler, mostly UK, mostly business buyers.** "Churn" here means a reseller pausing orders, which is seasonal; a consumer shop would behave differently.
- **Thin training data.** There is one labelled training month and one test month. Stacking several monthly snapshots would give the tree more data, but repeated customers would break the independence that the logistic model's p-values rely on.
- **Censoring.** Customers active before Dec 2009 have understated tenure and frequency.
- **Synthetic columns** (`age_group`, `payment_method`, `cost_price`) are placeholders and carry no information.
