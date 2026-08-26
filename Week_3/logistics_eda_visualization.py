"""
Week 3 Task: Advanced Data Analysis and Visualization in Logistics
Logistics Data Analyst Intern — Yuva (YuvaIntern.com)
Author: Ayushman

This script is self-contained: it regenerates and cleans the same working
sample used in Week 2 (see logistics-data-analyst-internship-week2), adds
a transportation_cost variable, then runs the exploratory analysis and
produces the five figures used in the Week 3 report.

Reference dataset: DataCo Smart Supply Chain for Big Data Analysis (Kaggle)
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Run with:  python logistics_eda_visualization.py
Requires:  pandas, numpy, matplotlib, seaborn
Outputs:   figures/*.png, week3_dataset_with_cost.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold"})


# =====================================================================
# PART 1 (from Week 2): simulate + clean the working sample
# =====================================================================
def generate_raw_sample(n=4000, seed=42):
    np.random.seed(seed)
    order_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 500, n), unit="D")

    shipping_modes = ["Standard Class", "First Class", "Second Class", "Same Day"]
    markets = ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"]
    segments = ["Consumer", "Corporate", "Home Office"]
    categories = ["Cleats", "Men's Footwear", "Women's Apparel", "Fishing",
                  "Camping & Hiking", "Electronics", "Indoor/Outdoor Games"]
    statuses = ["COMPLETE", "PENDING", "CLOSED", "CANCELED", "PROCESSING", "ON_HOLD", "SUSPECTED_FRAUD"]
    countries = ["United States", "Mexico", "Germany", "India", "Brazil", "Australia", "Nigeria", "Philippines"]

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
    df.loc[idx, "shipping_date"] = df.loc[idx, "order_date"] - pd.to_timedelta(np.random.randint(1, 5, len(idx)), unit="D")

    return df


def iqr_bounds(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def clean_pipeline(df):
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=["order_zipcode"])
    df = df.dropna(subset=["customer_zipcode"])
    df["benefit_per_order"] = df["benefit_per_order"].fillna(df["benefit_per_order"].median())
    df["category_name"] = df["category_name"].fillna(df["category_name"].mode()[0])

    canonical_map = {"standard class": "Standard Class", "first class": "First Class",
                      "second class": "Second Class", "same day": "Same Day"}
    df["shipping_mode"] = df["shipping_mode"].str.strip().str.lower().map(canonical_map)

    df = df[df["order_item_quantity"] > 0]
    mask = df["shipping_date"] < df["order_date"]
    df.loc[mask, "shipping_date"] = df.loc[mask, "order_date"] + pd.to_timedelta(df.loc[mask, "actual_shipping_days"], unit="D")

    low, high = iqr_bounds(df["sales"])
    df["sales_capped"] = df["sales"].clip(lower=low, upper=high)
    _, high_d = iqr_bounds(df["actual_shipping_days"])
    df["shipping_delay_anomaly"] = (df["actual_shipping_days"] > high_d).astype(int)

    for col, new_col in [("sales_capped", "sales_norm"), ("actual_shipping_days", "shipping_days_norm")]:
        mn, mx = df[col].min(), df[col].max()
        df[new_col] = (df[col] - mn) / (mx - mn)

    return df


# =====================================================================
# PART 2 (Week 3): transportation cost, EDA, and visualizations
# =====================================================================
def add_transportation_cost(df, seed=7):
    np.random.seed(seed)
    base_rate = {"Standard Class": 8, "Second Class": 12, "First Class": 18, "Same Day": 25}
    market_surcharge = {"USCA": 3, "LATAM": 4, "Europe": 6, "Pacific Asia": 7, "Africa": 8}

    df["transportation_cost"] = (
        df["shipping_mode"].map(base_rate)
        + df["order_item_quantity"] * 2.5
        + df["market"].map(market_surcharge)
        + np.random.normal(0, 3, len(df))
    ).clip(lower=5).round(2)
    return df


def run_eda(df):
    num_cols = ["actual_shipping_days", "sales_capped", "transportation_cost", "order_item_quantity", "benefit_per_order"]

    desc = df[num_cols].describe().T[["mean", "50%", "std", "min", "max"]]
    desc.columns = ["mean", "median", "std", "min", "max"]
    print("=== Descriptive statistics ===")
    print(desc.round(2), "\n")

    corr = df[num_cols + ["late_delivery_risk"]].corr()
    print("=== Correlation matrix ===")
    print(corr.round(2), "\n")

    by_mode = df.groupby("shipping_mode").agg(
        avg_cost=("transportation_cost", "mean"),
        avg_delivery_days=("actual_shipping_days", "mean"),
        orders=("order_id", "count"),
    ).round(2).sort_values("avg_cost")
    print("=== By shipping mode ===")
    print(by_mode, "\n")

    by_market = df.groupby("market").agg(
        late_rate_pct=("late_delivery_risk", lambda s: round(s.mean() * 100, 1)),
        avg_delivery_days=("actual_shipping_days", "mean"),
        orders=("order_id", "count"),
    ).sort_values("late_rate_pct", ascending=False)
    print("=== By market ===")
    print(by_market)

    return desc, corr, by_mode, by_market


def make_visualizations(df, corr, by_market, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    order = ["Standard Class", "Second Class", "First Class", "Same Day"]

    # Fig 1: delivery time distribution
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.histplot(df["actual_shipping_days"], bins=range(1, df["actual_shipping_days"].max() + 2),
                 color="#2E75B6", ax=ax, edgecolor="white")
    ax.axvline(df["actual_shipping_days"].mean(), color="#C00000", linestyle="--",
               label=f"Mean = {df['actual_shipping_days'].mean():.1f} days")
    ax.axvline(df["actual_shipping_days"].median(), color="#548235", linestyle="--",
               label=f"Median = {df['actual_shipping_days'].median():.1f} days")
    ax.set_title("Distribution of Actual Delivery Times")
    ax.set_xlabel("Delivery time (days)"); ax.set_ylabel("Number of orders"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/fig1_delivery_time_distribution.png", facecolor="white"); plt.close(fig)

    # Fig 2: cost by shipping mode
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(data=df, x="shipping_mode", y="transportation_cost", order=order,
                estimator=np.mean, errorbar=("ci", 95), color="#2E75B6", ax=ax)
    ax.set_title("Average Transportation Cost by Shipping Mode")
    ax.set_xlabel("Shipping mode"); ax.set_ylabel("Avg. transportation cost (USD)")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig2_cost_by_shipping_mode.png", facecolor="white"); plt.close(fig)

    # Fig 3: correlation heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Among Key Logistics Variables")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig3_correlation_heatmap.png", facecolor="white"); plt.close(fig)

    # Fig 4: cost vs delivery time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    jitter = df["actual_shipping_days"] + np.random.uniform(-0.2, 0.2, len(df))
    sns.scatterplot(x=jitter, y=df["transportation_cost"], hue=df["shipping_mode"],
                     hue_order=order, alpha=0.5, s=22, ax=ax, palette="deep")
    ax.set_title("Transportation Cost vs. Delivery Time, by Shipping Mode")
    ax.set_xlabel("Actual delivery time (days, jittered)"); ax.set_ylabel("Transportation cost (USD)")
    ax.legend(title="Shipping mode", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig4_cost_vs_delivery_time.png", facecolor="white"); plt.close(fig)

    # Fig 5: late-delivery rate by market
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plot_data = by_market.reset_index().sort_values("late_rate_pct", ascending=False)
    sns.barplot(data=plot_data, x="market", y="late_rate_pct", color="#C00000", ax=ax)
    ax.set_title("Late-Delivery Rate by Market")
    ax.set_xlabel("Market"); ax.set_ylabel("Orders exceeding scheduled delivery (%)")
    for i, v in enumerate(plot_data["late_rate_pct"]):
        ax.text(i, v + 0.5, f"{v}%", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig5_late_rate_by_market.png", facecolor="white"); plt.close(fig)

    print(f"\n5 figures written to {outdir}/")


if __name__ == "__main__":
    raw_df = generate_raw_sample()
    clean_df = clean_pipeline(raw_df)
    clean_df = add_transportation_cost(clean_df)

    desc, corr, by_mode, by_market = run_eda(clean_df)
    make_visualizations(clean_df, corr, by_market)

    clean_df.to_csv("week3_dataset_with_cost.csv", index=False)
