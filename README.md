# Train Delay Prediction (Two‑Stage Model)

This project builds a **two‑stage (predictive + reactive)** model for train delays using Trafikverket data and weather prognos from open-meteo.com.

- Stage 1 — Predictive (risk) model: before/around an event*, predict the probability that a train’s delay will exceed a threshold within a future horizon.
- Stage 2 — Reactive (duration) model: once a “delay trigger” is reached, estimate how much worse the delay will get (point estimate + quantiles for uncertainty).

The workflow is implemented as a set of Jupyter notebooks plus a shared feature engineering module (`features.py`).

The results are presented at https://ololpol.github.io/Train_Delay_Predictor/ (updated daily)
---

## Repo contents

## Pipeline overview 

### 0) Setup

Create an environment (example):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements
```

### 1) Part 01 — Backfill and label
Open and run:
- `1_train_feature_backfill.ipynb`

- Backfills historical train-stop events from Trafikverket’s Open API, engineers labels, and writes in Hopswork.
  What it does:
- Calls Trafikverket Open API (v2) for historical TrainAnnouncement data 
- Builds/standardizes timestamps in Europe/Stockholm.
- Computes labels (`y_delay_within_horizon`, `final_delay_min`, `additional_delay_min`).
- Saves:
      in a feature store in hopswork(`train_stop_events_labeled` feature group)

Required environment variable**
- `API_KEY_TRAFIK`: Trafikverket API key used in Part 01.

### 2) Part 02 — Feature engineering + splits (no leakage)
Open and run:
- `2_train_feature_pipeline.ipynb`

Default inputs:
- loading the feature store from the Hopswork

Outputs:

- `pred_train`, `pred_val`, `pred_test`
- `react_train`, `react_val`, `react_test`
- `feature_metadata` 


### 3) Part 03 — Train models
Open and run:
- `3_train_training_pipeline.ipynb`

It loads the Part 02 splits and trains:

#### Predictive model (risk)
- Baseline classifier (for sanity checks)
- Tuned `GradientBoostingClassifier`
- Isotonic calibration via `CalibratedClassifierCV`
- Metrics/plots (Average Precision, Brier score, calibration curve)

Saved to `data/models/`:
- `predictive_model.pkl`
- `calibrator.pnk`
- `predictive_meta.json` (params and eval metrics)

#### Reactive model (additional delay)
- Baseline `DummyRegressor`
- Point model: `GradientBoostingRegressor(loss="squared_error")`
- Quantile models: `GradientBoostingRegressor(loss="quantile", alpha=0.10/0.50/0.90)`
- Metrics/plots (MAE/RMSE + coverage diagnostics)

Saved to `data/models/`:
- `calibrator.pnk` 
- `reactive_meta.json`


### 4) Part 04 — Batch inference + alerts
Open and run:
- `4_train_batch_inference.ipynb`

Inputs:
-  `train_stop_events_labeled` feature group from Hopsworks
- Part 02 metadata (`data/feature_pipeline_outputs/feature_metadata.json`)
- Part 03 models (`data/models/`)

Outputs:
- `predictions_full` feature group in Hopsworks
- `predictions_early` feature group in Hopsworks
- `assets/img/station_delay_risk_map.png`
- `assets/img/top10_station_delay_risk.png`



### Shared feature module
- `features.py` 
  Single source of truth for feature engineering used in Part 02 (training data build) and Part 04 (inference).

---

## Data and terminology

### dataset (output of Part 01) 

This file must contain (at minimum) the columns below (Part 02 validates these):
- `event_time` (timezone-aware, Europe/Stockholm)
- `station_code`, `train_id`
- `delay_min`
- Predictive label: `y_delay_within_horizon`
- Reactive labels: `final_delay_min`, `additional_delay_min`

Label definitions (as implemented in Part 01):
- `final_delay_min`: the last observed, `delay_min` for a given `train_run_id`
- `additional_delay_min`: `final_delay_min - delay_min` (how much worse it will get from “now”)
- `y_delay_within_horizon`: for each event, looks forward `HORIZON_MIN` minutes and sets `1` if the max future delay in that horizon is ≥ `DELAY_THRESHOLD_MIN`, else `0`

### Trigger (reactive phase start)
The reactive pipeline detects a “trigger time” per train-run (see `detect_trigger_time()` in `features.py`).  
Conceptually: the earliest event where delay crosses the configured threshold (used to define reactive-only dynamics).


---

## Configuration (environment variables)

Common:
- `CANONICAL_PATH` (Part 02 & 04)  
  Default: `data/train_stop_events_labeled.parquet`
- `OUT_DIR` (Part 02)  
  Default: `data/feature_pipeline_outputs`
- `HORIZON_MIN` (Part 02 & 04)  
  Default: `60`
- `DELAY_THRESHOLD_MIN` (Part 02 & 04)  
  Default: `10`

Inference window (Part 04):
- `INFER_START` / `INFER_END`  
  Optional timestamps used to filter events.
- `DEFAULT_LOOKBACK_DAYS`  
  Default: `7` (used when `INFER_START/END` aren’t supplied)

---

## Feature engineering (what’s in `features.py`)

`build_features()` orchestrates leakage-safe feature creation. Major feature groups:

- **Timestamp normalization**: `ensure_event_time()`
- **Cause flags** from deviation/reason text: `add_cause_flags()`
- **Calendar features**: hour, day-of-week, month, weekend: `add_calendar_features()`
- **Lag features** on past labels/targets: `add_train_lag_features()`
- **Station congestion proxies**: shifted rolling stats: `add_station_congestion_features()`
- **Network state at station**: rolling mean/count stats: `add_station_network_state_features()`
- **Trigger detection / reactive phase**: `detect_trigger_time()`
- **Weather rollups** (if `weather_*` columns exist): `add_weather_rolling_features_if_present()`
- **Station delay features** placeholder: `add_station_delay_features()`

---


## Typical end-to-end run

1. `1_train_feature_backfill.ipynb` → creates `train_stop_events_labeled`
2. `2_train_feature_pipeline.ipynb` → creates `Train and test splits` + `feature_metadata.json`
3. `3_train_training_pipeline.ipynb` → creates `data/models/*`
4. `4_train_batch_inference.ipynb` → creates `docs/assets/*`

---

## Notes / next ideas

- Add holiday/special-day features (Sweden) as additional calendar signals (ensure they’re *known in advance* so they don’t leak).
- Improve reactive dynamics features (`add_reactive_early_dynamics`) once you have consistent post-trigger event frequency.
- Consider uncertainty-aware alerting (use P90 for “worst-case” delay escalation).
