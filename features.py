"""
Reusable feature engineering functions for the train‑delay model.
Single Source of Truth for Part 2 (Training) and Part 4 (Inference).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional

# --- Configuration Constants (Defaults) ---
DELAY_THRESHOLD_MIN = 10

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
    df = raw_df.copy()
    df = add_calendar_features(df)
    df = add_train_lag_features(df)
    df = add_station_network_state_features(df, windows_min=windows_min)
    df = detect_trigger_time(df)
    #df = add_reactive_early_dynamics(df)
    df = add_weather_rolling_features_if_present(df, windows_h=windows_h)
    df = add_station_delay_features(df, windows_h=windows_h)
    return df