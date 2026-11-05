"""Data loading and verification for the City Hotel (H2) dataset."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# The 31 exact source labels from Section 4 of the project description.
EXPECTED_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType",
    "AssignedRoomType", "BookingChanges", "DepositType", "Agent", "Company",
    "DaysInWaitingList", "CustomerType", "ADR", "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests", "ReservationStatus", "ReservationStatusDate",
]

MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def text_columns(df: pd.DataFrame) -> list[str]:
    """Names of non-numeric, non-datetime columns (robust across pandas versions)."""
    return [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])]


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_if_missing(raw_path: str | Path, url: str) -> Path:
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[load] Downloading H2.csv -> {raw_path}")
        urllib.request.urlretrieve(url, raw_path)
    return raw_path


def load_raw(cfg: dict) -> pd.DataFrame:
    """Load H2.csv as-is, except for whitespace stripping on text columns.

    Missing-value markers ("NULL") are deliberately NOT converted here so the
    EDA can report them honestly. Use `standardize_missing` afterwards.
    """
    d = cfg["data"]
    path = download_if_missing(d["raw_path"], d["url"])
    # keep_default_na=False: pandas would otherwise silently turn "NULL"/"NA" into NaN
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)

    # The raw file pads text values with spaces (e.g. "HB       ").
    for col in text_columns(df):
        df[col] = df[col].str.strip()
    return df


def verify_schema(df: pd.DataFrame, cfg: dict) -> None:
    """Check the dataframe matches the documented H2 schema."""
    d = cfg["data"]
    assert df.shape == (d["expected_rows"], d["expected_cols"]), (
        f"Unexpected shape {df.shape}, expected "
        f"({d['expected_rows']}, {d['expected_cols']})")
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    assert not missing and not extra, f"Schema mismatch. Missing={missing} Extra={extra}"


def standardize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'NULL' strings to NaN and fix numeric types. Returns a copy.

    No statistics are learned here (no imputation), so this is leakage-safe
    to apply before splitting.
    """
    out = df.copy()
    out = out.replace("NULL", np.nan)
    for col in ["IsRepeatedGuest", "Children", "Agent", "Company"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["ReservationStatusDate"] = pd.to_datetime(out["ReservationStatusDate"], errors="coerce")
    return out


def add_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Helper columns used for EDA only (derived from the row itself, no leakage)."""
    out = df.copy()
    out["TotalNights"] = out["StaysInWeekendNights"] + out["StaysInWeekNights"]
    out["TotalGuests"] = out["Adults"] + out["Children"].fillna(0) + out["Babies"]
    return out


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows. Do this BEFORE any train/test split."""
    n_before = len(df)
    out = df.drop_duplicates(keep="first").reset_index(drop=True)
    return out, n_before - len(out)
