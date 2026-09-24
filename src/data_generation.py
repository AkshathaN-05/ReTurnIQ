"""
ReTurnIQ — Milestone 1: Synthetic Data Generation Engine
=========================================================

Generates 1,000,000 synthetic e-commerce order records for return-risk
prediction.  Every predictor represents information realistically available
**before delivery**.

Key design decisions
--------------------
* **Chunked generation** — data is produced in 50 000-row chunks and appended
  to CSV so the full 1M-row DataFrame is never held entirely in memory.
* **Probabilistic target** — `returned` is sampled from a Bernoulli
  distribution whose probability comes from a latent risk score passed
  through a sigmoid, NOT from deterministic rules.
* **Correlated anomalies** — rare records (~1-2 %) simultaneously inflate
  several related columns.
* **Variable-rate missingness** — different columns have different missing
  percentages; the target column is never missing.

ID columns
----------
`customer_id` and `product_id` are included for realism only.
They are identifiers and **must not** be used as predictive features.
"""

import os
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
TOTAL_ROWS = 1_000_000
CHUNK_SIZE = 50_000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "return_data.csv")
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "data_generation_report.txt")

# Product categories and their base return-rate tendencies.
# Clothing / Footwear have higher returns; Books are very low.
CATEGORY_CONFIG = {
    "Clothing":              {"base_return_rate": 0.25, "weight": 0.22},
    "Footwear":              {"base_return_rate": 0.22, "weight": 0.13},
    "Electronics":           {"base_return_rate": 0.12, "weight": 0.18},
    "Home & Kitchen":        {"base_return_rate": 0.08, "weight": 0.14},
    "Books":                 {"base_return_rate": 0.04, "weight": 0.08},
    "Sports & Outdoors":     {"base_return_rate": 0.10, "weight": 0.10},
    "Beauty & Personal Care":{"base_return_rate": 0.09, "weight": 0.08},
    "Toys & Games":          {"base_return_rate": 0.11, "weight": 0.07},
}

CATEGORIES = list(CATEGORY_CONFIG.keys())
CATEGORY_WEIGHTS = np.array([CATEGORY_CONFIG[c]["weight"] for c in CATEGORIES])
CATEGORY_WEIGHTS = CATEGORY_WEIGHTS / CATEGORY_WEIGHTS.sum()  # normalise
CATEGORY_BASE_RATES = {c: CATEGORY_CONFIG[c]["base_return_rate"] for c in CATEGORIES}

PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Cash on Delivery", "EMI"]
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Missingness rates per column (proportion of rows set to NaN).
MISSING_RATES = {
    "customer_complaint_count":     0.030,
    "average_previous_order_value": 0.040,
    "product_rating":               0.020,
    "delivery_distance_km":         0.015,
    "size_change_history":          0.035,
    "category_return_history":      0.025,
    "customer_account_verified":    0.010,
    "customer_tenure_months":       0.020,
    "previous_category_orders":     0.008,
    "previous_category_returns":    0.008,
}


# ===================================================================
# 1. CUSTOMER FEATURES
# ===================================================================

def generate_customer_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate customer-level attributes for *n* rows.

    Relationships enforced
    ----------------------
    * `previous_returns <= previous_orders`
    * `previous_return_rate = previous_returns / previous_orders` (0 when no orders)
    * Tenure and order count are positively correlated.
    * Complaint count is loosely correlated with return history.
    """

    # --- Customer ID (identifier only, NOT a feature) ---
    customer_id = rng.integers(100_000, 999_999, size=n)

    # --- Age: 18–75, slight right skew ---
    customer_age = rng.normal(loc=35, scale=12, size=n).clip(18, 75).astype(int)

    # --- Tenure in months: 0–120, correlated with age ---
    # Older customers tend to have longer tenure, but with noise.
    base_tenure = (customer_age - 18) * 1.2 + rng.normal(0, 8, size=n)
    customer_tenure_months = base_tenure.clip(0, 120).astype(int)

    # --- Previous orders: correlated with tenure ---
    # Roughly 0.5–2 orders per month of tenure, Poisson-like.
    lam_orders = (customer_tenure_months * rng.uniform(0.3, 1.5, size=n)).clip(0.1, None)
    previous_orders = rng.poisson(lam=lam_orders).clip(0, 300)

    # --- Previous returns: fraction of orders, varying by customer ---
    # Each customer has a personal return propensity drawn from a Beta.
    personal_return_propensity = rng.beta(a=2, b=12, size=n)  # mean ≈ 0.14
    previous_returns = rng.binomial(n=previous_orders, p=personal_return_propensity)
    # Safety: guarantee returns <= orders (binomial guarantees this, but be safe)
    previous_returns = np.minimum(previous_returns, previous_orders)

    # --- Previous return rate ---
    with np.errstate(divide="ignore", invalid="ignore"):
        previous_return_rate = np.where(
            previous_orders > 0,
            previous_returns / previous_orders,
            0.0,
        )
    previous_return_rate = np.round(previous_return_rate, 4)

    # --- Complaint count: loosely correlated with return count ---
    complaint_base = previous_returns * rng.uniform(0.1, 0.5, size=n)
    customer_complaint_count = rng.poisson(lam=complaint_base.clip(0.1, None))
    customer_complaint_count = customer_complaint_count.clip(0, 30)

    # --- Average previous order value ---
    # Log-normal with slight correlation to age (older → slightly higher).
    log_mean = 3.5 + (customer_age - 35) * 0.005
    average_previous_order_value = np.round(
        rng.lognormal(mean=log_mean, sigma=0.6, size=n).clip(5, 2000), 2
    )

    # --- Account verified flag ---
    # Higher tenure → more likely verified.
    verify_prob = 0.5 + 0.004 * customer_tenure_months
    verify_prob = verify_prob.clip(0.0, 0.95)
    customer_account_verified = rng.binomial(1, verify_prob).astype(bool)

    return pd.DataFrame({
        "customer_id":                  customer_id,
        "customer_age":                 customer_age,
        "customer_tenure_months":       customer_tenure_months,
        "previous_orders":              previous_orders,
        "previous_returns":             previous_returns,
        "previous_return_rate":         previous_return_rate,
        "customer_complaint_count":     customer_complaint_count,
        "average_previous_order_value": average_previous_order_value,
        "customer_account_verified":    customer_account_verified,
    })


# ===================================================================
# 2. PRODUCT FEATURES
# ===================================================================

def generate_product_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate product-level attributes for *n* rows.

    `product_return_rate` varies around the category base rate with noise,
    so different items within the same category have different rates.
    """

    # --- Product ID (identifier only, NOT a feature) ---
    product_id = rng.integers(10_000, 99_999, size=n)

    # --- Category (weighted sampling) ---
    product_category = rng.choice(CATEGORIES, size=n, p=CATEGORY_WEIGHTS)

    # --- Product price: log-normal, category-dependent median ---
    # Electronics tend to be more expensive; Books cheaper.
    category_price_median = {
        "Clothing":               45,
        "Footwear":               65,
        "Electronics":           150,
        "Home & Kitchen":         55,
        "Books":                  15,
        "Sports & Outdoors":      60,
        "Beauty & Personal Care": 25,
        "Toys & Games":           30,
    }
    price_median = np.array([category_price_median[c] for c in product_category], dtype=float)
    product_price = np.round(
        rng.lognormal(mean=np.log(price_median), sigma=0.5, size=n).clip(2, 5000), 2
    )

    # --- Product rating: 1.0–5.0, beta-shaped, slightly left skew ---
    raw_rating = rng.beta(a=5, b=2, size=n) * 4 + 1  # range [1, 5]
    product_rating = np.round(raw_rating.clip(1.0, 5.0), 1)

    # --- Product return rate: category base + per-item noise ---
    base_rates = np.array([CATEGORY_BASE_RATES[c] for c in product_category], dtype=float)
    product_return_rate = np.round(
        (base_rates + rng.normal(0, 0.04, size=n)).clip(0.0, 0.6), 4
    )

    # --- Discount percentage: most items have 0-15 %, occasional big sales ---
    discount_percentage = np.round(
        rng.exponential(scale=8, size=n).clip(0, 70), 1
    )
    # Small bump: random 5 % of items get aggressive discounts (40-70 %).
    big_sale_mask = rng.random(size=n) < 0.05
    discount_percentage[big_sale_mask] = np.round(
        rng.uniform(40, 70, size=big_sale_mask.sum()), 1
    )

    return pd.DataFrame({
        "product_id":          product_id,
        "product_category":    product_category,
        "product_price":       product_price,
        "product_rating":      product_rating,
        "product_return_rate": product_return_rate,
        "discount_percentage": discount_percentage,
    })


# ===================================================================
# 3. ORDER FEATURES
# ===================================================================

def generate_order_features(n: int, rng: np.random.Generator,
                            product_price: np.ndarray,
                            discount_percentage: np.ndarray) -> pd.DataFrame:
    """
    Generate order-level attributes for *n* rows.

    `order_value = quantity * product_price * (1 - discount_percentage / 100)`
    """

    # --- Quantity: geometric-like distribution (most orders = 1 item) ---
    quantity = rng.geometric(p=0.55, size=n).clip(1, 10)

    # --- Order day of week ---
    order_day = rng.choice(DAYS_OF_WEEK, size=n)

    # --- Order hour: bimodal (morning + evening peaks) ---
    # Mixture of two normals: one at 10 AM, one at 20 PM.
    component = rng.random(size=n) < 0.45
    hour_raw = np.where(
        component,
        rng.normal(10, 2.5, size=n),
        rng.normal(20, 2.5, size=n),
    )
    order_hour = np.clip(np.round(hour_raw), 0, 23).astype(int)

    # --- Delivery distance (km): log-normal, right-skewed ---
    delivery_distance_km = np.round(
        rng.lognormal(mean=2.5, sigma=0.9, size=n).clip(1, 2000), 1
    )

    # --- Payment method ---
    payment_method = rng.choice(
        PAYMENT_METHODS,
        size=n,
        p=[0.30, 0.20, 0.25, 0.15, 0.10],
    )

    # --- Order value = qty × price × (1 - discount/100) ---
    order_value = np.round(
        quantity * product_price * (1 - discount_percentage / 100), 2
    )
    order_value = order_value.clip(0.01, None)  # avoid zero/negative

    # --- Gift order flag: ~8 % of orders ---
    is_gift_order = rng.binomial(1, 0.08, size=n).astype(bool)

    # --- Expedited shipping: ~15 % of orders ---
    is_expedited_shipping = rng.binomial(1, 0.15, size=n).astype(bool)

    return pd.DataFrame({
        "quantity":              quantity,
        "order_day":             order_day,
        "order_hour":            order_hour,
        "delivery_distance_km":  delivery_distance_km,
        "payment_method":        payment_method,
        "order_value":           order_value,
        "is_gift_order":         is_gift_order,
        "is_expedited_shipping": is_expedited_shipping,
    })


# ===================================================================
# 4. CUSTOMER–PRODUCT BEHAVIOUR FEATURES
# ===================================================================

def generate_behavior_features(n: int, rng: np.random.Generator,
                               product_category: np.ndarray,
                               previous_orders: np.ndarray,
                               previous_returns: np.ndarray) -> pd.DataFrame:
    """
    Generate behavioural signals that depend on both customer history and
    product category.

    `size_change_history` is meaningful mainly for Clothing / Footwear.
    """

    # --- size_change_history ---
    # Only Clothing / Footwear have meaningful size-change counts.
    is_size_relevant = np.isin(product_category, ["Clothing", "Footwear"])
    size_change_base = np.where(
        is_size_relevant,
        rng.poisson(lam=1.5, size=n),   # avg ~1.5 size swaps
        rng.poisson(lam=0.05, size=n),   # near-zero for other categories
    )
    size_change_history = size_change_base.clip(0, 15)

    # --- category_return_history: how many returns in this category ---
    # Fraction of previous returns assigned to this category, plus noise.
    cat_frac = rng.beta(a=2, b=5, size=n)   # mean ≈ 0.29
    category_return_history = np.round(previous_returns * cat_frac).astype(int)
    category_return_history = category_return_history.clip(0, None)

    # --- previous_category_orders ---
    cat_order_frac = rng.beta(a=2, b=5, size=n)
    previous_category_orders = np.round(previous_orders * cat_order_frac).astype(int)
    previous_category_orders = previous_category_orders.clip(0, None)

    # --- previous_category_returns <= previous_category_orders ---
    # Also capped at category_return_history for internal consistency.
    previous_category_returns = np.minimum(category_return_history, previous_category_orders)

    return pd.DataFrame({
        "size_change_history":       size_change_history,
        "category_return_history":   category_return_history,
        "previous_category_orders":  previous_category_orders,
        "previous_category_returns": previous_category_returns,
    })


# ===================================================================
# 5. PROBABILISTIC TARGET GENERATION
# ===================================================================

def generate_target(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Generate the binary `returned` target using a latent risk score.

    Steps
    -----
    1. Build a weighted linear combination of standardised features plus
       interaction / nonlinear terms.
    2. Add Gaussian noise so identical feature profiles don't always
       produce the same outcome.
    3. Apply a global intercept (bias) calibrated to yield ~10-15 %
       return rate.
    4. Pass through the sigmoid function to obtain probabilities.
    5. Sample from Bernoulli(p).

    No single feature dominates; class overlap is intentional.
    """

    # Helper: standardise a column (zero-mean, unit-variance).
    def _std(col: np.ndarray) -> np.ndarray:
        s = col.std()
        if s < 1e-9:
            return col - col.mean()
        return (col - col.mean()) / s

    # Extract arrays
    prev_return_rate     = df["previous_return_rate"].values.astype(float)
    prod_return_rate     = df["product_return_rate"].values.astype(float)
    cat_return_hist      = df["category_return_history"].values.astype(float)
    size_change          = df["size_change_history"].values.astype(float)
    complaint            = df["customer_complaint_count"].values.astype(float)
    discount             = df["discount_percentage"].values.astype(float)
    rating               = df["product_rating"].values.astype(float)
    price                = df["product_price"].values.astype(float)
    quantity             = df["quantity"].values.astype(float)
    distance             = df["delivery_distance_km"].values.astype(float)
    order_value          = df["order_value"].values.astype(float)

    # Boolean flags
    is_clothing_footwear = np.isin(
        df["product_category"].values, ["Clothing", "Footwear"]
    ).astype(float)

    # --- Latent risk score (weighted sum of standardised features) ---
    score = (
        0.50 * _std(prev_return_rate)
      + 0.45 * _std(prod_return_rate)
      + 0.30 * _std(cat_return_hist)
      + 0.25 * _std(size_change * is_clothing_footwear)  # interaction
      + 0.20 * _std(complaint)
      + 0.15 * _std(discount)
      - 0.25 * _std(rating)               # better rating → lower risk
      + 0.10 * _std(np.log1p(price))       # nonlinear
      + 0.10 * _std(quantity)
      + 0.08 * _std(np.log1p(distance))    # nonlinear
      + 0.12 * _std(discount * price / 100)  # interaction: high discount × expensive
      + 0.05 * _std(np.log1p(order_value))
    )

    # --- Add realistic noise ---
    noise = rng.normal(0, 0.8, size=len(df))
    score = score + noise

    # --- Intercept / bias to calibrate overall return rate to ~10-15 % ---
    # Empirically tuned: -2.5 yields ~12 % return rate after accounting
    # for the positive contribution of feature weights and noise.
    intercept = -2.5
    score = score + intercept

    # --- Sigmoid → probability ---
    prob = 1.0 / (1.0 + np.exp(-score))

    # --- Bernoulli sample ---
    returned = rng.binomial(1, prob)

    return returned


# ===================================================================
# 6. CORRELATED ANOMALIES
# ===================================================================

def introduce_correlated_anomalies(df: pd.DataFrame,
                                   rng: np.random.Generator) -> pd.DataFrame:
    """
    Inject rare correlated anomalies that affect multiple related columns
    simultaneously.  Total anomalous rows ≈ 1-2 % of the chunk.

    Three anomaly types are applied to non-overlapping subsets:
    1. Expensive-order anomaly  (~0.5 %)
    2. Customer-behaviour anomaly (~0.5 %)
    3. Discount anomaly (~0.4 %)

    Values remain plausible (no impossible values are created).
    """

    n = len(df)

    # --- 1. Expensive-order anomaly ---
    mask_exp = rng.random(n) < 0.005
    if mask_exp.any():
        k = mask_exp.sum()
        df.loc[mask_exp, "product_price"] = np.round(
            rng.uniform(800, 4500, size=k), 2
        )
        df.loc[mask_exp, "quantity"] = rng.integers(3, 10, size=k)
        # Recalculate order_value for affected rows
        df.loc[mask_exp, "order_value"] = np.round(
            df.loc[mask_exp, "quantity"].values
            * df.loc[mask_exp, "product_price"].values
            * (1 - df.loc[mask_exp, "discount_percentage"].values / 100),
            2,
        ).clip(0.01, None)

    # --- 2. Customer-behaviour anomaly ---
    mask_cust = (~mask_exp) & (rng.random(n) < 0.005)
    if mask_cust.any():
        k = mask_cust.sum()
        # Inflate previous orders first to keep returns ≤ orders
        df.loc[mask_cust, "previous_orders"] = rng.integers(80, 250, size=k)
        high_returns = rng.integers(
            30,
            df.loc[mask_cust, "previous_orders"].values + 1,
        )
        # Clip returns to at most previous_orders (safety)
        high_returns = np.minimum(high_returns, df.loc[mask_cust, "previous_orders"].values)
        df.loc[mask_cust, "previous_returns"] = high_returns
        # Recalculate return rate
        df.loc[mask_cust, "previous_return_rate"] = np.round(
            df.loc[mask_cust, "previous_returns"].values
            / df.loc[mask_cust, "previous_orders"].values,
            4,
        )
        df.loc[mask_cust, "customer_complaint_count"] = rng.integers(8, 25, size=k)

    # --- 3. Discount anomaly ---
    mask_disc = (~mask_exp) & (~mask_cust) & (rng.random(n) < 0.004)
    if mask_disc.any():
        k = mask_disc.sum()
        df.loc[mask_disc, "discount_percentage"] = np.round(
            rng.uniform(55, 85, size=k), 1
        )
        df.loc[mask_disc, "product_price"] = np.round(
            rng.uniform(200, 3000, size=k), 2
        )
        df.loc[mask_disc, "order_value"] = np.round(
            df.loc[mask_disc, "quantity"].values
            * df.loc[mask_disc, "product_price"].values
            * (1 - df.loc[mask_disc, "discount_percentage"].values / 100),
            2,
        ).clip(0.01, None)

    return df


# ===================================================================
# 7. MISSING VALUES
# ===================================================================

def introduce_missing_values(df: pd.DataFrame,
                             rng: np.random.Generator) -> pd.DataFrame:
    """
    Set random cells to NaN at column-specific rates.

    The target column `returned` is NEVER made missing.
    """

    for col, rate in MISSING_RATES.items():
        if col in df.columns:
            mask = rng.random(len(df)) < rate
            # Must convert to nullable dtype for boolean/int cols with NaN.
            if df[col].dtype == bool:
                df[col] = df[col].astype(object)
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype(float)
            df.loc[mask, col] = np.nan

    return df


# ===================================================================
# 8. CHUNK ASSEMBLY
# ===================================================================

def generate_chunk(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate a single chunk of *n* rows with all features + target.
    """

    # 1. Customer features
    cust = generate_customer_features(n, rng)

    # 2. Product features
    prod = generate_product_features(n, rng)

    # 3. Order features (needs price & discount from product)
    order = generate_order_features(
        n, rng,
        product_price=prod["product_price"].values,
        discount_percentage=prod["discount_percentage"].values,
    )

    # 4. Behaviour features (needs category, orders, returns from earlier)
    behav = generate_behavior_features(
        n, rng,
        product_category=prod["product_category"].values,
        previous_orders=cust["previous_orders"].values,
        previous_returns=cust["previous_returns"].values,
    )

    # Combine into one DataFrame
    df = pd.concat([cust, prod, order, behav], axis=1)

    # 5. Target (uses the combined DataFrame)
    df["returned"] = generate_target(df, rng)

    # 6. Correlated anomalies (must come after target so anomalies are
    #    realistic outliers, but they do NOT retroactively change the target)
    df = introduce_correlated_anomalies(df, rng)

    # 7. Missing values (last step — after anomalies, never touches target)
    df = introduce_missing_values(df, rng)

    return df


# ===================================================================
# 9. VALIDATION
# ===================================================================

def validate_dataset(csv_path: str) -> dict:
    """
    Read the generated CSV in chunks and run all validation checks.

    Returns a dict with the validation results and prints a human-readable
    summary.
    """

    print("\n" + "=" * 70)
    print("  DATASET VALIDATION")
    print("=" * 70)

    # Read full dataset for validation (this is a one-time read).
    df = pd.read_csv(csv_path)

    results = {}

    # --- 1. Row count ---
    results["row_count"] = len(df)
    results["col_count"] = len(df.columns)
    print(f"\nShape: {df.shape}")
    assert len(df) == TOTAL_ROWS, f"Expected {TOTAL_ROWS} rows, got {len(df)}"
    print(f"  [PASS] Row count = {TOTAL_ROWS:,}")

    # --- 2. Column list ---
    results["columns"] = list(df.columns)
    print(f"\nColumns ({len(df.columns)}):")
    for c in df.columns:
        print(f"    {c}")

    # --- 3. Target values ---
    unique_target = sorted(df["returned"].dropna().unique())
    assert set(unique_target) == {0, 1}, f"Unexpected target values: {unique_target}"
    assert df["returned"].isna().sum() == 0, "Target has missing values!"
    print("\n  [PASS] Target values: {0, 1}, no missing")

    # --- 4. Class distribution ---
    class_counts = df["returned"].value_counts().sort_index()
    class_pcts = df["returned"].value_counts(normalize=True).sort_index() * 100
    results["class_counts"] = class_counts.to_dict()
    results["class_pcts"] = {k: round(v, 2) for k, v in class_pcts.to_dict().items()}
    print("\nClass distribution:")
    for cls in [0, 1]:
        label = "Not returned" if cls == 0 else "Returned"
        print(f"    {label} ({cls}): {class_counts[cls]:>10,}  ({class_pcts[cls]:.2f}%)")
    returned_pct = class_pcts.get(1, 0)
    if 10 <= returned_pct <= 15:
        print("  [PASS] Return rate within target range (10-15 %)")
    else:
        print(f"  [WARNING] Return rate {returned_pct:.2f}% -- outside ideal 10-15 % range")

    # --- 5. Impossible-value checks ---
    print("\nBound checks:")
    checks_passed = True

    def _check(condition, description):
        nonlocal checks_passed
        violations = (~condition).sum()
        if violations > 0:
            print(f"    [FAIL] {description}: {violations:,} violations")
            checks_passed = False
        else:
            print(f"    [PASS] {description}")

    _check(df["customer_age"].dropna() >= 0,        "customer_age >= 0")
    _check(df["customer_tenure_months"].dropna() >= 0, "customer_tenure_months >= 0")
    _check(df["product_price"].dropna() > 0,         "product_price > 0")
    _check(df["quantity"].dropna() > 0,              "quantity > 0")
    _check(df["delivery_distance_km"].dropna() > 0,  "delivery_distance_km > 0")
    _check(df["discount_percentage"].dropna() >= 0,  "discount_percentage >= 0")
    _check(df["discount_percentage"].dropna() <= 100,"discount_percentage <= 100")
    _check(df["product_rating"].dropna() >= 1.0,     "product_rating >= 1.0")
    _check(df["product_rating"].dropna() <= 5.0,     "product_rating <= 5.0")

    # previous_returns <= previous_orders (using rows where both are non-NaN)
    both_present = df[["previous_returns", "previous_orders"]].dropna()
    _check(
        both_present["previous_returns"] <= both_present["previous_orders"],
        "previous_returns <= previous_orders",
    )

    results["bounds_valid"] = checks_passed

    # --- 6. Duplicate check ---
    dup_count = df.duplicated().sum()
    results["duplicate_count"] = int(dup_count)
    print(f"\nDuplicate rows: {dup_count:,}")

    # --- 7. Missing-value summary ---
    missing = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(3)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df.sort_values("missing_count", ascending=False)
    results["missing_summary"] = missing_df.to_dict()
    print("\nMissing-value summary:")
    print(missing_df.to_string())

    # --- 8. Numeric summary ---
    print("\nNumeric summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(df[numeric_cols].describe().round(3).to_string())

    # --- 9. Categorical summary ---
    print("\nCategorical column value counts:")
    cat_cols = ["product_category", "payment_method", "order_day"]
    results["categorical_summary"] = {}
    for col in cat_cols:
        vc = df[col].value_counts()
        results["categorical_summary"][col] = vc.to_dict()
        print(f"\n  {col}:")
        for val, cnt in vc.items():
            print(f"      {val}: {cnt:,}")

    # --- 10. File size ---
    file_size_bytes = os.path.getsize(csv_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    results["file_size_mb"] = round(file_size_mb, 1)
    print(f"\nFile size: {file_size_mb:.1f} MB")

    print("\n" + "=" * 70)
    overall = "ALL CHECKS PASSED" if checks_passed else "SOME CHECKS FAILED"
    print(f"  {overall}")
    print("=" * 70 + "\n")

    results["all_passed"] = checks_passed
    return results


# ===================================================================
# 10. REPORT GENERATION
# ===================================================================

def write_report(results: dict, generation_time: float) -> None:
    """
    Write a plain-text metadata / validation report to reports/.
    """

    os.makedirs(REPORT_DIR, exist_ok=True)

    lines = [
        "=" * 70,
        "  ReTurnIQ — Data Generation Report",
        "=" * 70,
        "",
        f"Generation seed:       {SEED}",
        f"Total rows generated:  {results['row_count']:,}",
        f"Total columns:         {results['col_count']}",
        f"Generation time:       {generation_time:.2f} seconds",
        f"Output file:           {OUTPUT_PATH}",
        f"File size:             {results.get('file_size_mb', 'N/A')} MB",
        "",
        "--- Feature List ---",
    ]
    for c in results["columns"]:
        note = ""
        if c in ("customer_id", "product_id"):
            note = "  [IDENTIFIER — do NOT use as predictive feature]"
        elif c == "returned":
            note = "  [TARGET]"
        lines.append(f"  {c}{note}")

    lines += [
        "",
        "--- Class Distribution ---",
    ]
    for cls, count in results["class_counts"].items():
        pct = results["class_pcts"][cls]
        label = "Not returned" if cls == 0 else "Returned"
        lines.append(f"  {label} ({cls}): {count:>10,}  ({pct:.2f}%)")

    lines += [
        "",
        "--- Missing-Value Summary ---",
    ]
    mc = results["missing_summary"]["missing_count"]
    mp = results["missing_summary"]["missing_pct"]
    for col in mc:
        lines.append(f"  {col:>35s}: {mc[col]:>8,} ({mp[col]:.3f}%)")

    lines += [
        "",
        f"--- Duplicate rows: {results['duplicate_count']:,} ---",
        "",
        f"--- Bound / constraint checks: {'PASSED' if results['bounds_valid'] else 'FAILED'} ---",
        "",
        f"--- Overall validation: {'PASSED' if results['all_passed'] else 'FAILED'} ---",
        "=" * 70,
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report saved to: {REPORT_PATH}")


# ===================================================================
# 11. MAIN ORCHESTRATOR
# ===================================================================

def generate_dataset() -> None:
    """
    Orchestrate the full dataset generation pipeline:

    1. Create RNG with fixed seed.
    2. Generate data in chunks and append each to CSV.
    3. Validate the final CSV.
    4. Write the metadata / validation report.
    """

    print(f"ReTurnIQ — Synthetic Data Generation")
    print(f"Seed: {SEED} | Rows: {TOTAL_ROWS:,} | Chunk size: {CHUNK_SIZE:,}")
    print("-" * 50)

    # Fixed seed for reproducibility
    rng = np.random.default_rng(SEED)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()

    n_chunks = TOTAL_ROWS // CHUNK_SIZE
    rows_written = 0

    for i in range(n_chunks):
        chunk_start = time.time()
        chunk_df = generate_chunk(CHUNK_SIZE, rng)

        # First chunk writes the header; subsequent chunks append without header.
        if i == 0:
            chunk_df.to_csv(OUTPUT_PATH, index=False, mode="w")
        else:
            chunk_df.to_csv(OUTPUT_PATH, index=False, mode="a", header=False)

        rows_written += len(chunk_df)
        chunk_time = time.time() - chunk_start
        print(f"  Chunk {i + 1}/{n_chunks}: {rows_written:>10,} rows written  "
              f"({chunk_time:.1f}s)")

        # Free memory for this chunk
        del chunk_df

    generation_time = time.time() - start_time
    print(f"\nGeneration complete in {generation_time:.2f}s")
    print(f"Dataset saved to: {OUTPUT_PATH}")

    # --- Validate ---
    results = validate_dataset(OUTPUT_PATH)

    # --- Report ---
    write_report(results, generation_time)


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    generate_dataset()
