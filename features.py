"""
Reusable feature engineering functions for the train‑delay model.
Single Source of Truth for Part 2 (Training) and Part 4 (Inference).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

STOCKHOLM_TZ = "Europe/Stockholm"

def ensure_event_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure event_time exists and is timezone-aware (Europe/Stockholm).
    This MUST be called before any rolling / weather / lag features.
    """
    df = df.copy()

    if "event_time" not in df.columns:
        for c in ["observed_time", "estimated_time", "scheduled_time", "actual_time"]:
            if c in df.columns:
                df["event_time"] = df[c]
                break

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

    if df["event_time"].dt.tz is None:
        df["event_time"] = df["event_time"].dt.tz_localize(
            STOCKHOLM_TZ, nonexistent="shift_forward", ambiguous="NaT"
        )
    else:
        df["event_time"] = df["event_time"].dt.tz_convert(STOCKHOLM_TZ)

    return df


"""
This module exposes reusable feature engineering utilities for both the training and
inference pipelines. In addition to the original functions, new helpers
have been added to normalize timestamps, derive cause-based flags from
textual reason fields, compute congestion-aware rolling statistics, and
produce a simple baseline for incident duration prediction. These
functions are designed to be composed together without data leakage when
the caller respects the order of operations (e.g. shifting prior to
rolling and restricting to information available at prediction time).

New functionality includes:

- ``ensure_event_time``: Normalize the ``event_time`` column to a timezone-aware
  datetime in ``Europe/Stockholm``. This is a prerequisite for all time-based
  operations and ensures consistent behaviour between training and
  inference.
- ``CAUSE_PATTERNS`` and ``add_cause_flags``: A dictionary of regular
  expressions capturing common categories of delay causes (e.g. weather,
  signal faults, persons on track). ``add_cause_flags`` parses
  textual fields (``reason_desc``, ``reason_text``, ``Deviation``, etc.)
  and produces binary indicator columns for each pattern. It also adds a
  ``has_reason_text`` flag.
- ``add_station_congestion_features``: Derived congestion metrics per
  station. It first shifts delay values by one row within each station
  before computing rolling mean delay and count of severe delays over
  user-specified windows. This prevents leakage from the current row
  into its own features.
- ``build_duration_baseline``: Computes a median-based baseline of
  ``additional_delay_min`` grouped by station, cause bucket and hour. This
  baseline can be used as a simple fallback for duration prediction or
  as a benchmarking reference.

See the documentation of each function for details.
"""

# --- Configuration Constants (Defaults) ---
DELAY_THRESHOLD_MIN = 10

# ---------------------------------------------------------------------------
# Cause pattern definitions used for deriving textual reason categories.
# These patterns should match Swedish and English keywords commonly found
# in the reason fields returned from Trafikverket APIs. Extend or update
# them as needed for your domain.
CAUSE_PATTERNS: Dict[str, str] = {
    "cause_weather": r"(snö|snow|storm|vind|wind|regn|rain|solkurva|heat|halk|is|ice)",
    "cause_signal": r"(signal|signalsystem|ställverk)",
    "cause_switch_track": r"(växel|spårfel|spår|rail|track)",
    "cause_power": r"(el|ström|kontaktledning|power|catenary)",
    "cause_person_track": r"(obehörig|person.*spår|people on track|trespass)",
    "cause_accident": r"(olycka|accident|påkörd|kollision)",
    "cause_vehicle": r"(fordon|tågfel|dörr|door|vehicle|brake|broms)",
    "cause_maintenance": r"(banarbete|spårarbete|maintenance|arbete)",
    "cause_capacity": r"(kapacit|trängsel|congestion|trafikledning|konflikt|prioriter)",
}

# ---------------------------------------------------------------------------
# Timezone handling for event_time.
#
# When ingesting data from multiple sources (Trafikverket events, weather
# forecasts, etc.), event timestamps can be naive or timezone-aware and in
# different timezones. To avoid subtle bugs in rolling or merging
# operations, callers should normalize event_time via ensure_event_time
# before any further feature engineering. This function localizes naive
# timestamps to Europe/Stockholm and converts already-aware timestamps to
# the same timezone.


# ---------------------------------------------------------------------------
# Cause flagging
#
# Use CAUSE_PATTERNS to produce binary flags from free‑text reason fields. The
# fields checked include ``reason_desc``, ``reason_text`` and ``Deviation``
# (if present). A generic ``has_reason_text`` flag is also emitted.
def add_cause_flags(df: pd.DataFrame, text_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Add binary flags describing inferred cause categories from text.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with reason columns (e.g. ``reason_desc``, ``reason_text``, ``Deviation``).
    text_cols : list of str, optional
        Explicit list of columns to inspect. If None, defaults to
        common reason columns present in ``df``.

    Returns
    -------
    pd.DataFrame
        DataFrame with new cause flag columns and ``has_reason_text``.
    """
    out = df.copy()
    # Determine which columns to join for the free text. Only use those that exist.
    if text_cols is None:
        candidate_cols = ["reason_desc", "reason_text", "Deviation", "ExternalDescription"]
        text_cols = [c for c in candidate_cols if c in out.columns]
    if text_cols:
        text_series = out[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
    else:
        text_series = pd.Series([""] * len(out), index=out.index)
    # Generate cause flags
    for cause_name, pattern in CAUSE_PATTERNS.items():
        out[cause_name] = text_series.str.contains(pattern, regex=True, na=False).astype("int8")
    # Generic flag: any non‑empty reason text
    out["has_reason_text"] = (text_series.str.len() > 3).astype("int8")
    return out

# ---------------------------------------------------------------------------
# Station congestion features
#
# These rolling features measure congestion and cascading delays at each
# station. They are computed after a prior shift to prevent the current
# observation from leaking into the window. ``lag1_delay_station`` captures
# the previous delay for each station. Rolling windows (in minutes)
# produce ``roll_mean_delay_station_{w}m`` (mean delay) and
# ``roll_cnt_delay_ge_{DELAY_THRESHOLD_MIN}_station_{w}m`` (count of severe
# delays >= DELAY_THRESHOLD_MIN) using only the shifted values.
def add_station_congestion_features(df: pd.DataFrame, windows_min: List[int] = None) -> pd.DataFrame:
    """Compute congestion features per station based on historical delays.

    This function should be called after ``ensure_event_time`` to ensure
    timestamps are timezone‑aware. It will sort by station and event_time,
    shift ``delay_min`` by one to avoid leakage, then compute rolling
    statistics over the specified windows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least ``station_code``, ``event_time`` and ``delay_min``.
    windows_min : list of int, optional
        Window lengths in minutes over which to compute rolling statistics. If
        omitted, defaults to ``[30, 60]``.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional congestion feature columns.
    """
    if windows_min is None:
        windows_min = [30, 60]
    out = df.copy()
    if "station_code" not in out.columns or "event_time" not in out.columns or "delay_min" not in out.columns:
        # Nothing to do if required columns missing
        return out
    # Ensure proper sorting
    out = out.sort_values(["station_code", "event_time"]).copy()
    # Shift previous delay within each station
    out["lag1_delay_station"] = (
        out.groupby("station_code")["delay_min"].shift(1)
    )
    # We'll compute rolling over the shifted delay values
    # Prepare a view with event_time as index
    out = out.set_index("event_time")
    # Precompute the shifted series aligned to the new index
    # Use .groupby on station_code to maintain independence across groups
    shifted = out.groupby("station_code")["delay_min"].shift(1)
    for w in windows_min:
        win = f"{w}min"
        # Rolling mean of delay
        out[f"roll_mean_delay_station_{w}m"] = (
            shifted.groupby(out["station_code"]).rolling(win, closed="left").mean().reset_index(level=0, drop=True)
        )
        # Rolling count of delays above threshold
        out[f"roll_cnt_delay_ge_{DELAY_THRESHOLD_MIN}_station_{w}m"] = (
            shifted.groupby(out["station_code"]).rolling(win, closed="left").apply(
                lambda x: np.sum(np.asarray(x) >= DELAY_THRESHOLD_MIN), raw=False
            ).reset_index(level=0, drop=True)
        )
    # Reset index to restore event_time as a column
    out = out.reset_index()
    return out

# ---------------------------------------------------------------------------
# Duration baseline
#
# This baseline provides a simple reference for how long delays typically
# persist given a station, the inferred cause and the time of day. It can
# be used for benchmarking or as a fallback prediction in inference.
def build_duration_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a median baseline for additional delay duration.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing ``additional_delay_min`` as well as cause
        flags (columns beginning with ``cause_``) and an ``hour`` column.

    Returns
    -------
    pd.DataFrame
        Baseline table with columns ``station_code``, ``cause_bucket``,
        ``hour`` and ``baseline_additional_delay_median``. If no
        ``additional_delay_min`` values are present, returns an empty
        DataFrame.
    """
    if "additional_delay_min" not in df.columns:
        return pd.DataFrame()
    # Work on a copy and drop rows without a label
    base = df[df["additional_delay_min"].notna()].copy()
    if base.empty:
        return pd.DataFrame()
    # Ensure cause flags exist
    cause_cols = [c for c in base.columns if c.startswith("cause_")]
    # Determine a cause bucket per row: first matching cause or unknown
    def pick_cause(row: pd.Series) -> str:
        for c in cause_cols:
            if row.get(c, 0) == 1:
                return c
        return "cause_unknown"
    base["cause_bucket"] = base.apply(pick_cause, axis=1)
    # Ensure hour column exists; if not, derive from event_time
    if "hour" not in base.columns:
        # Derive hour from event_time; ensure timezone aware
        temp = ensure_event_time(base)
        base["hour"] = pd.to_datetime(temp["event_time"]).dt.hour.astype("int16")
    # Group by station, cause_bucket and hour and compute median additional_delay
    group_keys = [c for c in ["station_code", "cause_bucket", "hour"] if c in base.columns]
    baseline = (
        base.groupby(group_keys)["additional_delay_min"].median().reset_index()
    )
    baseline = baseline.rename(columns={"additional_delay_min": "baseline_additional_delay_median"})
    return baseline

def add_calendar_features(d: pd.DataFrame) -> pd.DataFrame:
    """Add basic calendar features (Hour, Day, Month, Weekend)."""
    out = d.copy()
    et = pd.to_datetime(out["event_time"])
    out["cal_hour"] = et.dt.hour.astype("int16")
    out["cal_dow"] = et.dt.dayofweek.astype("int8")
    out["cal_month"] = et.dt.month.astype("int8")
    out["cal_is_weekend"] = (out["cal_dow"] >= 5).astype("int8")
    return out

def add_train_lag_features(d: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag features.
    SAFEGUARD: Checks if target column exists before shifting.
    """
    out = d.copy()
    # Ensure correct sorting
    if "train_run_id" in out.columns:
        out = out.sort_values(["train_run_id", "event_time"])
    else:
        out = out.sort_values(["train_id", "station_code", "event_time"])
        
    # Only calculate target lags if the target actually exists (Training mode)
    if "y_delay_within_horizon" in out.columns:
        # Example: Previous delay label for this specific train ID/Station combo
        out["lag_y_delay"] = (
            out.groupby(["train_id", "station_code"])["y_delay_within_horizon"]
            .shift(1)
            .fillna(0)
        )
    else:
        # Inference mode: We don't have the target, so we set lag to 0 or NaN
        out["lag_y_delay"] = 0
        
    return out

def add_station_network_state_features(d: pd.DataFrame, windows_min: List[int] = [30, 60]) -> pd.DataFrame:
    """Rolling stats per station_code over past time windows."""
    out = d.sort_values(["station_code", "event_time"]).copy()
    out = out.set_index("event_time")
    
    for w in windows_min:
        win = f"{w}min"
        # Rolling Mean Delay
        out[f"roll_mean_delay_station_{w}m"] = (
            out.groupby("station_code")["delay_min"]
               .rolling(win, closed="left")
               .mean()
               .reset_index(level=0, drop=True)
        )
        # Rolling Count of Delays
        out[f"roll_cnt_delay_ge_{DELAY_THRESHOLD_MIN}_station_{w}m"] = (
            out.groupby("station_code")["delay_min"]
               .rolling(win, closed="left")
               .apply(lambda x: np.sum(np.asarray(x) >= DELAY_THRESHOLD_MIN), raw=False)
               .reset_index(level=0, drop=True)
        )

    out = out.reset_index()
    return out

def detect_trigger_time(d: pd.DataFrame) -> pd.DataFrame:
    """
    Identify trigger based on CURRENT DELAY or REASON CODE.
    (Fixed to work in Inference where 'y_delay_within_horizon' does not exist).
    """
    out = d.copy()
    
    # Logic: A trigger happens if delay >= threshold OR a reason code exists
    # This relies only on input data, not prediction targets.
    
    # 1. Check for reason code presence
    has_reason = pd.Series(False, index=out.index)
    if "reason_code" in out.columns:
        has_reason = out["reason_code"].notna() & (out["reason_code"].astype(str) != "") & (out["reason_code"].astype(str) != "<NA>")

    # 2. Check for current delay threshold
    is_delayed = pd.Series(False, index=out.index)
    if "delay_min" in out.columns:
        is_delayed = out["delay_min"] >= DELAY_THRESHOLD_MIN

    # 3. Create flag
    out["_trigger_flag"] = (has_reason | is_delayed)

    # 4. Find first trigger time per train run
    triggers = (
        out[out["_trigger_flag"]]
        .groupby("train_run_id")["event_time"]
        .min()
        .rename("trigger_time")
    )
    
    # 5. Merge back
    if "trigger_time" in out.columns:
        out = out.drop(columns=["trigger_time"]) # Avoid duplicate column error
        
    out = out.merge(triggers, on="train_run_id", how="left")

    # 6. Calculate minutes since trigger
    # If no trigger (NaN), fill with a large negative number (-9999) indicating "Pre-trigger"
    out["min_since_trigger"] = (
        (out["event_time"] - out["trigger_time"]).dt.total_seconds() / 60.0
    ).fillna(-9999)

    out = out.drop(columns=["_trigger_flag"], errors="ignore")
    return out

def add_reactive_early_dynamics(d: pd.DataFrame) -> pd.DataFrame:
    """Add features capturing early reactive dynamics after the trigger."""
    out = d.copy()
    
    # Calculate Delay at the moment of trigger
    # (Safe map using train_run_id)
    if "trigger_time" in out.columns and "delay_min" in out.columns:
        delay_at_trigger = (
            out.loc[out["event_time"] == out["trigger_time"], ["train_run_id", "delay_min"]]
               .drop_duplicates("train_run_id")
               .set_index("train_run_id")["delay_min"]
        )
        out["delay_at_trigger"] = out["train_run_id"].map(delay_at_trigger)
        
        # Slope: (Current Delay - Trigger Delay) / Time elapsed
        # Avoid division by zero
        out["delay_slope_since_trigger"] = (
            (out["delay_min"] - out["delay_at_trigger"]) / 
            out["min_since_trigger"].replace(0, np.nan)
        )
    else:
        out["delay_at_trigger"] = np.nan
        out["delay_slope_since_trigger"] = np.nan

    # Indicator: Is this within 15 mins of the trigger?
    out["is_first15m_after_trigger"] = (
        (out["min_since_trigger"] >= 0) & (out["min_since_trigger"] <= 15)
    ).astype("int8")

    return out

def add_weather_rolling_features_if_present(d: pd.DataFrame, windows_h: List[int] = [3, 6]) -> pd.DataFrame:
    """
    If numeric weather columns exist (prefix weather_), add rolling means.
    """
    out = d.sort_values(["station_code", "event_time"]).copy()
    
    # Identify weather columns that are ALSO numeric
    all_weather_cols = [c for c in out.columns if c.startswith("weather_")]
    weather_cols = [c for c in all_weather_cols if pd.api.types.is_numeric_dtype(out[c])]

    if not weather_cols:
        return out

    out = out.set_index("event_time")
    
    for h in windows_h:
        win = f"{h}H"
        for c in weather_cols:
            out[f"{c}_rollmean_{h}h"] = (
                out.groupby("station_code")[c]
                  .rolling(win, closed="left")
                  .mean()
                  .reset_index(level=0, drop=True)
            )
            
    out = out.reset_index()
    return out

def add_station_delay_features(d: pd.DataFrame, windows_h: List[int] = [3, 6]) -> pd.DataFrame:
    """Add features related to station delay statistics."""
    out = d.copy()
    
    # Average delay per station
    station_avg_delay = (
        out.groupby("station_code")["delay_min"]
           .mean()
           .rename("station_avg_delay")
    )
    out = out.merge(station_avg_delay, on="station_code", how="left")

    # Rolling average delay per station over specified windows
    #TODO
    
    return out

def build_features(raw_df: pd.DataFrame, windows_h = [3,6], windows_min = [15,30]) -> pd.DataFrame:
    """Master pipeline function."""
    """
    Master pipeline for feature engineering across training and inference.

    This function orchestrates the construction of feature columns in a
    leakage‑safe manner. It now includes:

    - ``ensure_event_time`` to standardize timestamps
    - ``add_cause_flags`` to create textual cause indicators
    - ``add_calendar_features`` for temporal context
    - ``add_train_lag_features`` for historical label information
    - ``add_station_congestion_features`` for congestion proxies
    - ``add_station_network_state_features`` for legacy rolling statistics
    - ``detect_trigger_time`` for identifying reactive phases
    - ``add_weather_rolling_features_if_present`` to summarise weather history
    - ``add_station_delay_features`` as a placeholder for additional station
      statistics (currently only station average delay)

    Parameters
    ----------
    raw_df : pd.DataFrame
        The raw event dataset to transform.
    windows_h : list of int, optional
        Window lengths in hours for weather rolling features.
    windows_min : list of int, optional
        Window lengths in minutes for congestion features and network state
        rolling stats.

    Returns
    -------
    pd.DataFrame
        A dataframe with additional feature columns.
    """
    df = raw_df.copy()
    # Ensure event_time is timezone aware before any time‑dependent operations
    df = ensure_event_time(df)
    # Textual cause indicators (cheap and no leakage)
    df = add_cause_flags(df)
    # Calendar features (hour, dow, month, weekend)
    df = add_calendar_features(df)
    # Lag on target label
    df = add_train_lag_features(df)
    # Congestion proxies (shifted rolling statistics)
    df = add_station_congestion_features(df, windows_min=windows_min)
    # Legacy network state features (rolling mean and counts without shift) for backwards compatibility
    df = add_station_network_state_features(df, windows_min=windows_min)
    # Trigger detection (reactive phase start)
    df = detect_trigger_time(df)
    # (optional) early reactive dynamics can be added later when needed
    # Weather rolling features (if weather columns present)
    df = add_weather_rolling_features_if_present(df, windows_h=windows_h)
    # Station delay features placeholder
    df = add_station_delay_features(df, windows_h=windows_h)
    return df