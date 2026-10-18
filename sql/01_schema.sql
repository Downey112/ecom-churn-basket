-- E-Commerce Churn & Market-Basket Intelligence: physical schema (PostgreSQL 14+)
-- Source: UCI Online Retail II (UK online gift wholesaler, Dec 2009 - Dec 2011).
DROP TABLE IF EXISTS basket_rules, customer_predictions, order_items, orders, products, customers CASCADE;

CREATE TABLE customers (
    customer_id   integer PRIMARY KEY,
    country       text    NOT NULL,          -- brief says "state"; the source records country
    signup_date   date    NOT NULL,          -- derived: date of first recorded purchase
    age_group     text    CHECK (age_group IN ('18-24','25-34','35-44','45-54','55+')),  -- SYNTHETIC
    loyalty_tier  text    NOT NULL CHECK (loyalty_tier IN ('Gold','Silver','Bronze','New'))
);
COMMENT ON COLUMN customers.age_group IS 'SYNTHETIC (seeded random); not in source. Not used in models.';
COMMENT ON COLUMN customers.loyalty_tier IS 'Derived from spend in the first year (Dec 2009 - Nov 2010): Gold top 10%, Silver next 30%, Bronze rest, New = no purchase that year.';

CREATE TABLE products (
    product_id   text          PRIMARY KEY,             -- source StockCode
    description  text          NOT NULL,
    category     text          NOT NULL,                -- derived from description keywords
    unit_price   numeric(10,2) NOT NULL CHECK (unit_price >= 0),   -- list price = median selling price
    cost_price   numeric(10,2) NOT NULL CHECK (cost_price >= 0)    -- SYNTHETIC: category margin assumption
);
COMMENT ON COLUMN products.cost_price IS 'SYNTHETIC: unit_price x (1 - assumed category margin +/- noise). Not in source.';

CREATE TABLE orders (
    order_id         text          PRIMARY KEY,         -- source Invoice
    customer_id      integer       NOT NULL REFERENCES customers,
    order_timestamp  timestamp     NOT NULL,
    payment_method   text          NOT NULL CHECK (payment_method IN ('Card','Bank Transfer','PayPal')),  -- SYNTHETIC
    discount_amount  numeric(10,2) NOT NULL DEFAULT 0 CHECK (discount_amount >= 0)
);
COMMENT ON COLUMN orders.payment_method IS 'SYNTHETIC (seeded random); not in source. Not used in models.';
COMMENT ON COLUMN orders.discount_amount IS 'From source "D" (Discount) lines on sales invoices; almost always 0 because the wholesaler issued discounts as credit notes.';

CREATE TABLE order_items (
    order_item_id  bigint        GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id       text          NOT NULL REFERENCES orders ON DELETE CASCADE,
    product_id     text          NOT NULL REFERENCES products,
    quantity       integer       NOT NULL CHECK (quantity > 0),
    unit_price     numeric(10,2) NOT NULL CHECK (unit_price >= 0)  -- price actually charged on this line
);
COMMENT ON COLUMN order_items.unit_price IS 'Added to the brief''s design: prices change over time and by volume, so revenue uses the price at time of sale, not products.unit_price.';

CREATE TABLE customer_predictions (
    customer_id        integer      PRIMARY KEY REFERENCES customers,
    cluster_id         integer,
    persona            text,
    churn_probability  numeric(6,5) CHECK (churn_probability BETWEEN 0 AND 1),
    risk_tier          text         CHECK (risk_tier IN ('Low','Medium','High','Dormant')),
    model_name         text,
    features_as_of     timestamp,   -- data cut-off the features were computed at
    scored_at          timestamptz  NOT NULL DEFAULT now()
);

CREATE TABLE basket_rules (                  -- written by scripts/04_basket.py
    if_bought       text             NOT NULL,
    then_also       text             NOT NULL,
    antecedent_ids  text             NOT NULL,   -- comma-separated sorted product_ids
    consequent_ids  text             NOT NULL,
    n_items         integer          NOT NULL CHECK (n_items >= 2),
    orders          integer          NOT NULL CHECK (orders > 0),
    support         double precision NOT NULL CHECK (support > 0 AND support <= 1),
    confidence      double precision NOT NULL CHECK (confidence >= 0.5 AND confidence <= 1),   -- MIN_CONFIDENCE
    lift            double precision NOT NULL CHECK (lift > 0),
    same_category   boolean          NOT NULL,
    PRIMARY KEY (antecedent_ids, consequent_ids),
    CHECK (antecedent_ids <> '' AND consequent_ids <> '' AND antecedent_ids <> consequent_ids)
);

COMMENT ON COLUMN customer_predictions.risk_tier IS 'Low < 0.5 <= Medium < 0.8 <= High on churn_probability; Dormant = no order in the 365 days before features_as_of (not scored).';

CREATE INDEX ix_orders_customer_ts ON orders (customer_id, order_timestamp);
CREATE INDEX ix_order_items_order   ON order_items (order_id);
CREATE INDEX ix_order_items_product ON order_items (product_id);
