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
