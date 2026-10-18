"""Invariant checks against the loaded database (skipped if PostgreSQL is not reachable)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecb.db import connect  # noqa: E402

try:
    _conn = connect()
except Exception:  # pragma: no cover
    pytest.skip("PostgreSQL not reachable; run scripts/local_pg.ps1 start", allow_module_level=True)


def one(sql, params=None):
    return _conn.execute(sql, params).fetchone()


def test_referential_integrity_and_cleaning():
    assert one("SELECT count(*) FROM order_items WHERE quantity <= 0 OR unit_price <= 0")[0] == 0
    assert one("SELECT count(*) FROM orders o WHERE NOT EXISTS "
               "(SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)")[0] == 0
    assert one("SELECT count(*) FROM orders WHERE order_id LIKE 'C%%' OR order_id LIKE 'A%%'")[0] == 0
    assert one("SELECT count(*) FROM customers c WHERE signup_date <> "
               "(SELECT min(order_timestamp)::date FROM orders o WHERE o.customer_id = c.customer_id)")[0] == 0


def test_rfm_totals_reconcile_with_base_tables():
    f, m, n = one("SELECT sum(frequency), sum(monetary), count(*) FROM vw_customer_rfm")
    orders, revenue, customers = one(
        "SELECT (SELECT count(*) FROM orders), (SELECT sum(quantity * unit_price) FROM order_items), "
        "(SELECT count(*) FROM customers)")
    assert f == orders and n == customers
    assert abs(float(m) - float(revenue)) < 0.01 * n   # per-customer rounding only


def test_rfm_has_no_look_ahead():
    t = "2011-01-01"
    f, m = one("SELECT sum(frequency), sum(monetary) FROM fn_customer_rfm(%s::timestamp)", (t,))
    orders, revenue = one(
        "SELECT count(DISTINCT o.order_id), sum(i.quantity * i.unit_price) FROM orders o "
        "JOIN order_items i USING (order_id) WHERE o.order_timestamp < %s::timestamp", (t,))
    assert f == orders and abs(float(m) - float(revenue)) < 50
    assert one("SELECT max(last_order) < %s::timestamp FROM fn_customer_rfm(%s::timestamp)", (t, t))[0]


def test_rhythm_fields_need_enough_purchase_days():
    bad = one("SELECT count(*) FROM vw_customer_rfm WHERE (purchase_days < 2 AND ipt_mean_days IS NOT NULL) "
              "OR (purchase_days < 3 AND ipt_var_days IS NOT NULL) OR (purchase_days >= 3 AND ipt_var_days IS NULL)")
    assert bad[0] == 0


def test_churn_label_matches_definition():
    t = "2011-11-10"
    churned, n = one("SELECT sum(churned), count(*) FROM fn_churn_label(%s::timestamp, 30)", (t,))
    active = one("SELECT count(DISTINCT customer_id) FROM orders WHERE order_timestamp >= %s::timestamp "
                 "AND order_timestamp < %s::timestamp + interval '30 days' AND customer_id IN "
                 "(SELECT customer_id FROM orders WHERE order_timestamp < %s::timestamp)", (t, t, t))[0]
    assert n - churned == active


def test_basket_pairs_metrics_are_consistent():
    bad = one("""SELECT count(*) FROM vw_basket_pairs
                 WHERE support < 0.01 OR confidence_a_to_b > 1 OR confidence_b_to_a > 1
                    OR pair_orders > LEAST(orders_a, orders_b) OR product_a >= product_b""")
    assert bad[0] == 0
    assert one("SELECT count(*) FROM vw_basket_pairs")[0] > 0


def test_basket_rules_match_a_recount_in_sql():
    n_rules = one("SELECT count(*) FROM basket_rules")[0]
    assert n_rules > 0
    # recount every rule's itemset, antecedent and consequent from order_items
    bad = one("""
        WITH basket AS (SELECT DISTINCT order_id, product_id FROM order_items),
        n AS (SELECT count(DISTINCT order_id)::float AS n FROM basket),
        r AS (SELECT *, string_to_array(NULLIF(antecedent_ids, ''), ',') AS a,
                     string_to_array(NULLIF(consequent_ids, ''), ',') AS c
              FROM basket_rules),
        sets AS (SELECT antecedent_ids, consequent_ids, 'ab' AS part, a || c AS items FROM r
                 UNION ALL SELECT antecedent_ids, consequent_ids, 'a', a FROM r
                 UNION ALL SELECT antecedent_ids, consequent_ids, 'c', c FROM r),
        members AS (SELECT antecedent_ids, consequent_ids, part, cardinality(items) AS k,
                           unnest(items) AS product_id FROM sets),
        hits AS (SELECT m.antecedent_ids, m.consequent_ids, m.part
                 FROM members m JOIN basket b USING (product_id)
                 GROUP BY m.antecedent_ids, m.consequent_ids, m.part, m.k, b.order_id
                 HAVING count(DISTINCT b.product_id) = m.k),   -- order contains every item of the set
        cnt AS (SELECT antecedent_ids, consequent_ids,
                       count(*) FILTER (WHERE part = 'ab') AS n_ab,
                       count(*) FILTER (WHERE part = 'a')  AS n_a,
                       count(*) FILTER (WHERE part = 'c')  AS n_c
                FROM hits GROUP BY antecedent_ids, consequent_ids)
        SELECT count(*) FROM r LEFT JOIN cnt USING (antecedent_ids, consequent_ids), n
        WHERE r.a IS NULL OR r.c IS NULL OR r.a && r.c   -- empty or overlapping sides
           OR cardinality(r.a || r.c) <> (SELECT count(DISTINCT x) FROM unnest(r.a || r.c) x)   -- repeated ids
           OR cnt.n_ab IS NULL OR r.orders <> cnt.n_ab OR r.n_items <> cardinality(r.a || r.c)
           OR abs(r.support - cnt.n_ab / n.n) > 1e-9
           OR abs(r.confidence - cnt.n_ab::float / NULLIF(cnt.n_a, 0)) > 1e-9
           OR abs(r.lift - cnt.n_ab * n.n / NULLIF(cnt.n_a::float * cnt.n_c, 0)) > 1e-6
           OR r.confidence < 0.5 OR r.support < 0.01""")
    assert bad[0] == 0
    # 2-item rules agree with the SQL pair view
    assert one("SELECT count(*) FROM basket_rules WHERE n_items = 2")[0] > 0
    assert one("""SELECT count(*) FROM basket_rules r JOIN vw_basket_pairs p
                  ON p.product_a = LEAST(r.antecedent_ids, r.consequent_ids)
                 AND p.product_b = GREATEST(r.antecedent_ids, r.consequent_ids)
                  WHERE r.n_items = 2 AND r.orders <> p.pair_orders""")[0] == 0
    assert one("SELECT count(*) FROM basket_rules WHERE n_items = 2")[0] == one(
        "SELECT count(*) FROM basket_rules r JOIN vw_basket_pairs p ON p.product_a = LEAST(r.antecedent_ids, "
        "r.consequent_ids) AND p.product_b = GREATEST(r.antecedent_ids, r.consequent_ids) WHERE r.n_items = 2")[0]


def test_predictions_cover_every_customer():
    n, cust, bad_p, dormant_with_p = one("""
        SELECT count(*), (SELECT count(*) FROM customers),
               count(*) FILTER (WHERE risk_tier <> 'Dormant' AND churn_probability IS NULL),
               count(*) FILTER (WHERE risk_tier = 'Dormant' AND churn_probability IS NOT NULL)
        FROM customer_predictions""")
    assert n == cust and bad_p == 0 and dormant_with_p == 0
    assert one("""SELECT count(*) FROM customer_predictions WHERE
                  (risk_tier = 'High' AND churn_probability < 0.8) OR
                  (risk_tier = 'Medium' AND (churn_probability < 0.5 OR churn_probability >= 0.8)) OR
                  (risk_tier = 'Low' AND churn_probability >= 0.5)""")[0] == 0
