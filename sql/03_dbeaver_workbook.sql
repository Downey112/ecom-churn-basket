-- DBeaver workbook: open in DBeaver (connection: localhost:5433, db retail, user postgres)
-- and run statement by statement (Ctrl+Enter). Everything here is read-only.

-- 1. RFM snapshot for the latest day of data
SELECT * FROM vw_customer_rfm ORDER BY monetary DESC LIMIT 20;

-- 2. The same features as of any past cut-off (used for leak-free churn training)
SELECT * FROM fn_customer_rfm('2011-10-11') ORDER BY recency_days LIMIT 20;

-- 3. RFM quintile scores (classic 1-5 scoring; 5 = best)
SELECT customer_id, recency_days, frequency, monetary,
       6 - NTILE(5) OVER (ORDER BY recency_days)  AS r_score,
       NTILE(5) OVER (ORDER BY frequency)         AS f_score,
       NTILE(5) OVER (ORDER BY monetary)          AS m_score
FROM vw_customer_rfm;

-- 4. Strongest product pairs (Apriori-pruned self-join)
SELECT description_a, description_b, pair_orders, support, confidence_a_to_b, confidence_b_to_a, lift
FROM vw_basket_pairs ORDER BY lift DESC LIMIT 25;

-- 5. Pairs that span two keyword categories. Caveat: many are the same design range split by the
--    keyword rules (e.g. POPPY'S PLAYHOUSE KITCHEN vs ...LIVINGROOM), not true cross-category buys.
SELECT bp.description_a, pa.category AS category_a, bp.description_b, pb.category AS category_b,
       bp.support, bp.lift
FROM vw_basket_pairs bp
JOIN products pa ON pa.product_id = bp.product_a
JOIN products pb ON pb.product_id = bp.product_b
WHERE pa.category <> pb.category
ORDER BY bp.lift DESC LIMIT 25;

-- 6. Rules mined in Python (3-item rules included)
SELECT if_bought, then_also, orders, round(confidence::numeric, 3) AS confidence, round(lift::numeric, 1) AS lift
FROM basket_rules ORDER BY lift DESC LIMIT 25;

-- 7. Churn label as of a cut-off: share of active customers with no order in the next 30 days
SELECT round(avg(l.churned), 3) AS churn_rate, count(*) AS active_customers
FROM fn_churn_label('2011-11-10', 30) l
JOIN fn_customer_rfm('2011-11-10') r USING (customer_id)
WHERE r.last_order >= timestamp '2011-11-10' - interval '365 days';

-- 8. Retention worklist: high-risk customers worth saving, by spend
SELECT customer_id, persona, round(churn_probability, 3) AS p_churn, monetary, recency_days, frequency
FROM vw_customer_360
WHERE risk_tier = 'High'
ORDER BY monetary DESC LIMIT 50;

-- 9. Persona x risk tier: customers and revenue at stake
SELECT persona, risk_tier, count(*) AS customers, round(sum(monetary)) AS lifetime_revenue
FROM vw_customer_360
GROUP BY persona, risk_tier
ORDER BY persona, risk_tier;
