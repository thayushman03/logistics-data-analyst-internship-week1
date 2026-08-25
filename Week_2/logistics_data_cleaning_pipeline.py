"""
Week 2 Task: Data Collection, Cleaning, and Preprocessing for Logistics Analysis
Logistics Data Analyst Intern — Yuva (YuvaIntern.com)
Author: Ayushman

Reference dataset: DataCo Smart Supply Chain for Big Data Analysis (Kaggle)
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

This script:
  1. Simulates the collection of a representative logistics working sample
     (same schema and same categories of data-quality issues documented for
     real-world logistics extracts of this kind).
  2. Runs that sample through a full cleaning & preprocessing pipeline:
     duplicate removal, missing-value treatment, categorical standardization,
     invalid-value correction, IQR-based outlier handling, and normalization.

Run with:  python logistics_data_cleaning_pipeline.py
Requires:  pandas, numpy
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 100)


# =====================================================================
# STEP 1: Data Collection Simulation
# =====================================================================
def generate_raw_sample(n=4000, seed=42):
    """Generate a representative logistics working sample with realistic,
    intentionally-injected data-quality issues."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    order_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.random.randint(0, 500, n), unit="D"
    )

    shipping_modes = ["Standard Class", "First Class", "Second Class", "Same Day"]
    markets = ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"]
    segments = ["Consumer", "Corporate", "Home Office"]
    categories = [
        "Cleats", "Men's Footwear", "Women's Apparel", "Fishing",
        "Camping & Hiking", "Electronics", "Indoor/Outdoor Games",
    ]
    statuses = [
        "COMPLETE", "PENDING", "CLOSED", "CANCELED",
        "PROCESSING", "ON_HOLD", "SUSPECTED_FRAUD",
    ]
    countries = [
        "United States", "Mexico", "Germany", "India",
        "Brazil", "Australia", "Nigeria", "Philippines",
    ]

    df = pd.DataFrame({
        "order_id": np.arange(100000, 100000 + n),
        "order_date": order_dates,
        "scheduled_shipping_days": np.random.choice([1, 2, 3, 4, 5], n, p=[0.15, 0.35, 0.25, 0.15, 0.10]),
        "shipping_mode": np.random.choice(shipping_modes, n, p=[0.6, 0.15, 0.15, 0.10]),
        "market": np.random.choice(markets, n),
        "order_country": np.random.choice(countries, n),
        "customer_segment": np.random.choice(segments, n),
        "category_name": np.random.choice(categories, n),
        "order_item_quantity": np.random.randint(1, 6, n),
        "sales": np.round(np.random.gamma(shape=2, scale=60, size=n), 2),
        "benefit_per_order": np.round(np.random.normal(20, 40, n), 2),
        "order_status": np.random.choice(statuses, n, p=[0.55, 0.10, 0.12, 0.06, 0.08, 0.05, 0.04]),
        "customer_zipcode": np.random.randint(10000, 99999, n).astype(float),
        "order_zipcode": np.random.randint(10000, 99999, n).astype(float),
    })

    noise = np.random.choice([-1, 0, 0, 0, 1, 1, 2, 3], n)
    df["actual_shipping_days"] = (df["scheduled_shipping_days"] + noise).clip(lower=1)
    df["shipping_date"] = df["order_date"] + pd.to_timedelta(df["actual_shipping_days"], unit="D")
    df["late_delivery_risk"] = (df["actual_shipping_days"] > df["scheduled_shipping_days"]).astype(int)

    # ---- inject realistic data-quality issues ----
    df.loc[np.random.choice(df.index, int(n * 0.42), replace=False), "order_zipcode"] = np.nan
    df.loc[np.random.choice(df.index, int(n * 0.06), replace=False), "customer_zipcode"] = np.nan
    df.loc[np.random.choice(df.index, int(n * 0.03), replace=False), "benefit_per_order"] = np.nan
    df.loc[np.random.choice(df.index, int(n * 0.015), replace=False), "category_name"] = None

    df = pd.concat([df, df.sample(int(n * 0.02), random_state=1)], ignore_index=True)

    def mess_case(x):
        r = np.random.rand()
        if r < 0.33:
            return x.upper()
        elif r < 0.66:
            return x.lower()
        return " " + x + "  "

    idx = np.random.choice(df.index, int(len(df) * 0.08), replace=False)
    df.loc[idx, "shipping_mode"] = df.loc[idx, "shipping_mode"].apply(mess_case)

    idx = np.random.choice(df.index, int(len(df) * 0.01), replace=False)
    df.loc[idx, "sales"] = df.loc[idx, "sales"] * np.random.uniform(15, 40, len(idx))

    idx = np.random.choice(df.index, int(len(df) * 0.008), replace=False)
    df.loc[idx, "actual_shipping_days"] = np.random.choice([45, 60, 90], len(idx))

    idx = np.random.choice(df.index, int(len(df) * 0.005), replace=False)
    df.loc[idx, "order_item_quantity"] = -1

    idx = np.random.choice(df.index, int(len(df) * 0.01), replace=False)
    df.loc[idx, "shipping_date"] = df.loc[idx, "order_date"] - pd.to_timedelta(
        np.random.randint(1, 5, len(idx)), unit="D"
    )

    return df


# =====================================================================
# STEP 2: Cleaning & Preprocessing Pipeline
# =====================================================================
def iqr_bounds(series):
    """Return (lower, upper) Tukey IQR bounds for outlier detection."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def clean_pipeline(df, verbose=True):
    raw_shape = df.shape
    log = lambda msg: print(msg) if verbose else None
    log(f"STEP 0 - Loaded raw data: {raw_shape[0]} rows x {raw_shape[1]} columns\n")

    # ---- 1. Duplicate removal ----
    dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    log(f"STEP 1 - Duplicates: {dupes} exact duplicate rows removed. Shape now {df.shape}\n")

    # ---- 2. Missing values ----
    df = df.drop(columns=["order_zipcode"])                       # too sparse (>40%) to impute
    df = df.dropna(subset=["customer_zipcode"])                   # low-rate identifier field
    df["benefit_per_order"] = df["benefit_per_order"].fillna(
        df["benefit_per_order"].median())                         # skewed numeric -> median
    df["category_name"] = df["category_name"].fillna(
        df["category_name"].mode()[0])                            # low-cardinality -> mode
    log(f"STEP 2 - Missing values resolved. Remaining nulls: {df.isnull().sum().sum()}. "
        f"Shape now {df.shape}\n")

    # ---- 3. Standardize categorical text ----
    canonical_map = {
        "standard class": "Standard Class", "first class": "First Class",
        "second class": "Second Class", "same day": "Same Day",
    }
    df["shipping_mode"] = df["shipping_mode"].str.strip().str.lower().map(canonical_map)
    log(f"STEP 3 - shipping_mode standardized to {df['shipping_mode'].nunique()} canonical labels\n")

    # ---- 4. Invalid / impossible values ----
    df = df[df["order_item_quantity"] > 0]                        # can't recover a true value
    mask = df["shipping_date"] < df["order_date"]
    df.loc[mask, "shipping_date"] = df.loc[mask, "order_date"] + pd.to_timedelta(
        df.loc[mask, "actual_shipping_days"], unit="D"
    )
    log(f"STEP 4 - Invalid values corrected. Shape now {df.shape}\n")

    # ---- 5. Outlier detection (IQR) ----
    low, high = iqr_bounds(df["sales"])
    df["sales_capped"] = df["sales"].clip(lower=low, upper=high)       # winsorize, keep the row
    _, high_d = iqr_bounds(df["actual_shipping_days"])
    df["shipping_delay_anomaly"] = (df["actual_shipping_days"] > high_d).astype(int)  # flag, don't drop
    log("STEP 5 - Outliers handled: sales capped, extreme delays flagged\n")

    # ---- 6. Normalization ----
    for col, new_col in [("sales_capped", "sales_norm"), ("actual_shipping_days", "shipping_days_norm")]:
        mn, mx = df[col].min(), df[col].max()
        df[new_col] = (df[col] - mn) / (mx - mn)
    log("STEP 6 - Min-Max normalization applied to sales_capped and actual_shipping_days\n")

    log("FINAL SUMMARY")
    log(f"   Raw shape:     {raw_shape[0]} rows x {raw_shape[1]} columns")
    log(f"   Cleaned shape: {df.shape[0]} rows x {df.shape[1]} columns")
    log(f"   Remaining missing values: {df.isnull().sum().sum()}")

    return df


if __name__ == "__main__":
    raw_df = generate_raw_sample()
    raw_df.to_csv("raw_logistics_data.csv", index=False)

    cleaned_df = clean_pipeline(raw_df)
    cleaned_df.to_csv("cleaned_logistics_data.csv", index=False)
