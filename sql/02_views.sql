-- Analytical layer: RFM features, churn labels, basket pairs, customer 360.

-- RFM + purchase-rhythm features using only orders strictly before p_as_of (no look-ahead).
CREATE OR REPLACE FUNCTION fn_customer_rfm(p_as_of timestamp)
RETURNS TABLE (
    customer_id          integer,
    as_of                timestamp,
    first_order          timestamp,
    last_order           timestamp,
    recency_days         numeric,   -- days since MAX(order_timestamp)
    frequency            bigint,    -- COUNT(DISTINCT order_id)
    monetary             numeric,   -- SUM(quantity * unit_price)
    avg_order_value      numeric,
    avg_items_per_order  numeric,
    distinct_products    bigint,
    tenure_days          numeric,
    purchase_days        bigint,    -- distinct calendar days with an order
    ipt_mean_days        numeric,   -- mean inter-purchase gap (needs >= 2 purchase days)
    ipt_var_days         numeric,   -- sample variance of inter-purchase gaps (needs >= 3 purchase days)
    orders_90d           bigint,
    monetary_90d         numeric
)
LANGUAGE sql STABLE AS $$
WITH lines AS (
    SELECT o.order_id, o.customer_id, o.order_timestamp, oi.product_id,
           oi.quantity, oi.quantity * oi.unit_price AS line_value
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_timestamp < p_as_of
),
per_customer AS (
    SELECT customer_id,
           MIN(order_timestamp)             AS first_order,
           MAX(order_timestamp)             AS last_order,
           COUNT(DISTINCT order_id)         AS frequency,
           SUM(line_value)                  AS monetary,
           SUM(quantity)                    AS items,
           COUNT(DISTINCT product_id)       AS distinct_products,
           COUNT(DISTINCT order_id) FILTER (WHERE order_timestamp >= p_as_of - interval '90 days') AS orders_90d,
           COALESCE(SUM(line_value) FILTER (WHERE order_timestamp >= p_as_of - interval '90 days'), 0) AS monetary_90d
    FROM lines
    GROUP BY customer_id
),
purchase_day AS (
    SELECT DISTINCT customer_id, order_timestamp::date AS day FROM lines
),
gaps AS (
    SELECT customer_id,
           day - LAG(day) OVER (PARTITION BY customer_id ORDER BY day) AS gap_days
    FROM purchase_day
),
rhythm AS (
    SELECT customer_id,
           COUNT(*)          AS purchase_days,
           AVG(gap_days)     AS ipt_mean_days,
           VAR_SAMP(gap_days) AS ipt_var_days
    FROM gaps
    GROUP BY customer_id
)
SELECT c.customer_id,
       p_as_of,
       c.first_order,
       c.last_order,
       ROUND((EXTRACT(EPOCH FROM p_as_of - c.last_order) / 86400)::numeric, 3),
       c.frequency,
       ROUND(c.monetary, 2),
       ROUND(c.monetary / c.frequency, 2),
       ROUND(c.items::numeric / c.frequency, 2),
       c.distinct_products,
       ROUND((EXTRACT(EPOCH FROM p_as_of - c.first_order) / 86400)::numeric, 3),
       r.purchase_days,
       ROUND(r.ipt_mean_days, 3),
       ROUND(r.ipt_var_days, 3),
       c.orders_90d,
       ROUND(c.monetary_90d, 2)
FROM per_customer c
JOIN rhythm r USING (customer_id);
$$;

-- RFM as of the day after the last recorded order (the "today" of the dataset).
CREATE OR REPLACE VIEW vw_customer_rfm AS
SELECT *
FROM fn_customer_rfm((SELECT date_trunc('day', MAX(order_timestamp)) + interval '1 day' FROM orders));

-- Churn label: 1 if a customer with history before p_as_of places NO order in [p_as_of, p_as_of + horizon).
CREATE OR REPLACE FUNCTION fn_churn_label(p_as_of timestamp, p_horizon_days integer DEFAULT 30)
RETURNS TABLE (customer_id integer, churned integer)
LANGUAGE sql STABLE AS $$
SELECT h.customer_id,
       CASE WHEN EXISTS (
           SELECT 1 FROM orders f
           WHERE f.customer_id = h.customer_id
             AND f.order_timestamp >= p_as_of
             AND f.order_timestamp <  p_as_of + make_interval(days => p_horizon_days)
       ) THEN 0 ELSE 1 END
FROM (SELECT DISTINCT customer_id FROM orders WHERE order_timestamp < p_as_of) h;
$$;

-- Co-occurring product pairs with support, confidence and lift.
-- Apriori pruning in SQL: only items whose own support >= 1% of orders can form a frequent pair
-- (downward closure), which keeps the self-join small.
CREATE OR REPLACE VIEW vw_basket_pairs AS
WITH basket AS (
    SELECT DISTINCT order_id, product_id FROM order_items
),
n AS (
    SELECT COUNT(DISTINCT order_id)::numeric AS n_orders FROM basket
),
item AS (
    SELECT product_id, COUNT(*) AS item_orders FROM basket GROUP BY product_id
),
frequent AS (
    SELECT i.product_id, i.item_orders FROM item i, n WHERE i.item_orders >= 0.01 * n.n_orders
),
pairs AS (
    SELECT a.product_id AS product_a, b.product_id AS product_b, COUNT(*) AS pair_orders
    FROM basket a
    JOIN basket b ON b.order_id = a.order_id AND a.product_id < b.product_id
    WHERE a.product_id IN (SELECT product_id FROM frequent)
      AND b.product_id IN (SELECT product_id FROM frequent)
    GROUP BY a.product_id, b.product_id
)
SELECT p.product_a,
       pa.description                                    AS description_a,
       p.product_b,
       pb.description                                    AS description_b,
       p.pair_orders,
       fa.item_orders                                    AS orders_a,
       fb.item_orders                                    AS orders_b,
       ROUND(p.pair_orders / n.n_orders, 5)              AS support,
       ROUND(p.pair_orders::numeric / fa.item_orders, 4) AS confidence_a_to_b,
       ROUND(p.pair_orders::numeric / fb.item_orders, 4) AS confidence_b_to_a,
       ROUND(p.pair_orders * n.n_orders / (fa.item_orders::numeric * fb.item_orders), 3) AS lift
FROM pairs p
CROSS JOIN n
JOIN frequent fa ON fa.product_id = p.product_a
JOIN frequent fb ON fb.product_id = p.product_b
JOIN products pa ON pa.product_id = p.product_a
JOIN products pb ON pb.product_id = p.product_b
WHERE p.pair_orders >= 0.01 * n.n_orders;

-- One row per customer for reporting tools (DBeaver, Power BI).
CREATE OR REPLACE VIEW vw_customer_360 AS
SELECT c.customer_id, c.country, c.signup_date, c.loyalty_tier,
       r.recency_days, r.frequency, r.monetary, r.avg_order_value, r.purchase_days,
       r.ipt_mean_days, r.ipt_var_days,
       p.cluster_id, p.persona, p.churn_probability, p.risk_tier, p.model_name, p.scored_at
FROM customers c
LEFT JOIN vw_customer_rfm r USING (customer_id)
LEFT JOIN customer_predictions p USING (customer_id);
