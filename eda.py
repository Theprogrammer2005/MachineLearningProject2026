"""Exploratory data analysis: tables saved as CSV, figures saved as PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.load import (MONTH_ORDER, add_basic_columns, drop_duplicates,
                           load_raw, standardize_missing, text_columns, verify_schema)

sns.set_theme(style="whitegrid")


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [fig] {path}")


def _table(df: pd.DataFrame, path: Path, title: str, show: bool = True, index: bool = True) -> None:
    df.to_csv(path, index=index)
    if show:
        print(f"\n=== {title} ===")
        print(df.to_string(index=index))
    print(f"  [csv] {path}")


# ----------------------------------------------------------------------------
def overview(raw: pd.DataFrame, df: pd.DataFrame, tab: Path) -> None:
    print(f"\nShape: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    dt = pd.DataFrame({"dtype": df.dtypes.astype(str), "n_unique": df.nunique()})
    _table(dt, tab / "dtypes.csv", "Data types and unique counts")
    _table(df.describe().T, tab / "describe_numeric.csv", "describe() - numeric")
    cats = df[text_columns(df)]
    _table(cats.describe().T, tab / "describe_categorical.csv", "describe() - categorical")


def cardinality(df: pd.DataFrame, cfg: dict, tab: Path, fig: Path) -> None:
    cats = text_columns(df) + ["Agent", "Company"]
    card = pd.Series({c: df[c].nunique(dropna=True) for c in cats}, name="n_categories")
    card = card.sort_values(ascending=False).to_frame()
    _table(card, tab / "category_cardinality.csv", "Category cardinality")

    for col in cfg["eda"]["high_cardinality_cols"]:
        vc = df[col].value_counts(dropna=False)
        top10_share = vc.head(10).sum() / len(df)
        print(f"  {col}: {df[col].nunique()} unique; top-10 cover {top10_share:.1%} of rows")
        vc.head(15).to_csv(tab / f"top_values_{col}.csv")

    fig_, ax = plt.subplots(figsize=(8, 5))
    card["n_categories"].sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_xscale("log")
    ax.set_title("Number of categories per categorical column (log scale)")
    _save(fig_, fig / "category_cardinality.png")


def missing_values(raw: pd.DataFrame, df: pd.DataFrame, tab: Path, fig: Path) -> None:
    null_str = (raw == "NULL").sum()                 # literal "NULL" strings in raw file
    nan_raw = raw.isna().sum()                       # true empty cells in raw file
    nan_clean = df.isna().sum()                      # after standardization
    res = pd.DataFrame({
        "NULL_strings_in_raw": null_str,
        "empty_cells_in_raw": nan_raw,
        "missing_after_cleaning": nan_clean,
    })
    res["missing_pct"] = (res["missing_after_cleaning"] / len(df) * 100).round(2)
    res = res[(res[["NULL_strings_in_raw", "empty_cells_in_raw", "missing_after_cleaning"]] > 0).any(axis=1)]
    _table(res.sort_values("missing_after_cleaning", ascending=False),
           tab / "missing_values.csv", "Missing values (string 'NULL' vs real NaN)")
    if res.empty:
        print("  No missing values found.")
        return
    fig_, ax = plt.subplots(figsize=(7, 4))
    res["missing_pct"].sort_values().plot.barh(ax=ax, color="indianred")
    ax.set_xlabel("% missing")
    ax.set_title("Missing values per column")
    _save(fig_, fig / "missing_values.png")


def duplicates(df: pd.DataFrame, tab: Path) -> pd.DataFrame:
    n_dup = int(df.duplicated().sum())
    pct = n_dup / len(df) * 100
    print(f"\n=== Duplicates ===\n  Exact duplicate rows: {n_dup:,} ({pct:.1f}%)")
    clean, removed = drop_duplicates(df)
    print(f"  Decision: DROP them (before any split) -> {len(clean):,} rows remain")
    print("  Reason: identical rows could land in both train and test and inflate scores.")
    print("  Caveat: some duplicates may be genuinely separate identical bookings, but they")
    print("          cannot be told apart, and keeping them risks leakage.")
    pd.DataFrame({"rows_before": [len(df)], "duplicates_removed": [removed],
                  "rows_after": [len(clean)]}).to_csv(tab / "duplicates_summary.csv", index=False)
    return clean


def unusual_values(df: pd.DataFrame, cfg: dict, tab: Path, fig: Path) -> None:
    q = cfg["eda"]["adr_outlier_quantile"]
    adr_cap = df["ADR"].quantile(q)
    lead_cap = df["LeadTime"].quantile(0.999)
    checks = {
        "ADR < 0": (df["ADR"] < 0).sum(),
        "ADR == 0": (df["ADR"] == 0).sum(),
        f"ADR > {q:.1%} quantile ({adr_cap:.0f})": (df["ADR"] > adr_cap).sum(),
        "ADR > 1000": (df["ADR"] > 1000).sum(),
        "Adults == 0": (df["Adults"] == 0).sum(),
        "Adults == 0 and Children == 0 and Babies == 0": (df["TotalGuests"] == 0).sum(),
        "TotalNights == 0": (df["TotalNights"] == 0).sum(),
        "LeadTime > 365": (df["LeadTime"] > 365).sum(),
        f"LeadTime > 99.9% quantile ({lead_cap:.0f})": (df["LeadTime"] > lead_cap).sum(),
        "DaysInWaitingList > 0": (df["DaysInWaitingList"] > 0).sum(),
        "DaysInWaitingList > 100": (df["DaysInWaitingList"] > 100).sum(),
        "Children missing": df["Children"].isna().sum(),
        "Babies > 2": (df["Babies"] > 2).sum(),
        "Adults > 10": (df["Adults"] > 10).sum(),
    }
    res = pd.DataFrame({"count": pd.Series(checks)})
    res["pct_of_rows"] = (res["count"] / len(df) * 100).round(3)
    _table(res, tab / "unusual_values.csv", "Unusual / suspicious values")

    print("\n  ADR summary:")
    print(df["ADR"].describe(percentiles=[.01, .05, .5, .95, .99, .999]).round(2).to_string())
    print(f"  ADR skew = {df['ADR'].skew():.2f}, LeadTime skew = {df['LeadTime'].skew():.2f}, "
          f"DaysInWaitingList skew = {df['DaysInWaitingList'].skew():.2f}")

    fig_, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.boxplot(y=df["ADR"], ax=axes[0], color="skyblue");           axes[0].set_title("ADR")
    sns.boxplot(y=df["LeadTime"], ax=axes[1], color="lightgreen");   axes[1].set_title("LeadTime")
    sns.boxplot(y=df["DaysInWaitingList"], ax=axes[2], color="salmon"); axes[2].set_title("DaysInWaitingList")
    _save(fig_, fig / "outlier_boxplots.png")


# ----------------------------------------------------------------------------
def plots(df: pd.DataFrame, tab: Path, fig: Path) -> None:
    # 1. IsCanceled balance
    fig_, ax = plt.subplots(figsize=(5, 4))
    counts = df["IsCanceled"].value_counts().sort_index()
    ax.bar(["Not canceled (0)", "Canceled (1)"], counts.values, color=["seagreen", "indianred"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("IsCanceled class balance"); ax.set_ylim(0, counts.max() * 1.18)
    _save(fig_, fig / "iscanceled_balance.png")
    print(f"\n  Cancellation rate: {df['IsCanceled'].mean():.1%}")

    # 2. ADR distribution (raw and log) - exclude absurd values for readability
    adr = df["ADR"]
    adr_view = adr[(adr >= 0) & (adr <= adr.quantile(0.999))]
    fig_, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(adr_view, bins=60, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"ADR distribution (skew={adr.skew():.2f})")
    sns.histplot(np.log1p(adr_view), bins=60, kde=True, ax=axes[1], color="darkorange")
    axes[1].set_title("log(1 + ADR)")
    _save(fig_, fig / "adr_distribution.png")

    # 3. LeadTime histogram
    fig_, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["LeadTime"], bins=60, ax=ax, color="seagreen")
    ax.set_title(f"LeadTime distribution (skew={df['LeadTime'].skew():.2f})")
    _save(fig_, fig / "leadtime_hist.png")

    # 4. Cancellation rate by DepositType, MarketSegment, month
    def rate_plot(col, order=None, rotate=0):
        g = df.groupby(col)["IsCanceled"].agg(["mean", "count"]).rename(columns={"mean": "cancel_rate"})
        if order is not None:
            g = g.reindex([o for o in order if o in g.index])
        else:
            g = g.sort_values("cancel_rate", ascending=False)
        g.to_csv(tab / f"cancel_rate_by_{col}.csv")
        print(f"\n=== Cancellation rate by {col} ===\n{g.round(3).to_string()}")
        fig_, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=g.index, y=g["cancel_rate"], ax=ax, color="indianred")
        ax.axhline(df["IsCanceled"].mean(), ls="--", c="k", lw=1, label="overall rate")
        ax.set_ylabel("Cancellation rate"); ax.set_title(f"Cancellation rate by {col}")
        ax.legend(); plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right" if rotate else "center")
        _save(fig_, fig / f"cancel_rate_by_{col}.png")

    rate_plot("DepositType")
    rate_plot("MarketSegment", rotate=30)
    rate_plot("ArrivalDateMonth", order=MONTH_ORDER, rotate=45)

    # 5. Correlation heatmap (numeric columns)
    num = df.select_dtypes(include="number").drop(columns=["ArrivalDateYear"], errors="ignore")
    corr = num.corr()
    corr.to_csv(tab / "correlation_matrix.csv")
    fig_, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 6}, ax=ax, square=True, cbar_kws={"shrink": .7})
    ax.set_title("Correlation heatmap (numeric features)")
    _save(fig_, fig / "correlation_heatmap.png")
    for target in ["IsCanceled", "ADR"]:
        top = corr[target].drop(target).abs().sort_values(ascending=False).head(6)
        print(f"\n  Top |correlation| with {target}:\n{corr[target].loc[top.index].round(3).to_string()}")


# ----------------------------------------------------------------------------
def run_analysis(cfg: dict) -> pd.DataFrame:
    """Run the full EDA and return the cleaned (deduplicated) dataframe."""
    out = Path(cfg["paths"]["outputs"])
    tab, fig = out / "tables", out / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    print("Loading H2.csv ...")
    raw = load_raw(cfg)
    verify_schema(raw, cfg)
    print("  Schema verified: 79,330 rows x 31 columns, labels match Section 4.")

    df = add_basic_columns(standardize_missing(raw))

    overview(raw, df, tab)
    cardinality(df, cfg, tab, fig)
    missing_values(raw, df, tab, fig)
    clean = duplicates(df, tab)
    unusual_values(clean, cfg, tab, fig)
    plots(clean, tab, fig)

    print(f"\nDone. Tables -> {tab}  Figures -> {fig}")
    return clean
