"""Train the MorphCloud-LLM XGBoost preemption predictor from empirical telemetry.

The script follows the preprocessing described in Sections 3.3 and 3.8. It
never fabricates missing telemetry. Region CSVs are labeled independently, then
merged and globally sorted by timestamp before the chronological 70/15/15 split.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("train_predictor")

try:
    import joblib
    import pandas as pd
    import xgboost as xgb
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.preprocessing import MinMaxScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

FEATURE_COLUMNS = [
    "spot_price", "spot_price_velocity", "az_rejection_rate", "n_active", "uptime_s",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
REQUIRED_COLUMNS = {
    "timestamp", "spot_price", "az_rejection_rate", "n_active", "uptime_s", "preemption_event",
}


def _prepare_region(csv_path: Path, prediction_horizon_s: int, sample_interval_s: int):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    if prediction_horizon_s <= 0 or prediction_horizon_s % sample_interval_s != 0:
        raise ValueError("prediction_horizon_s must be a positive multiple of sample_interval_s")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["spot_price_velocity"] = df["spot_price"].diff().fillna(0.0) / sample_interval_s
    hour = df["timestamp"].dt.hour
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # y(t)=1 when an event occurs in (t, t+horizon]. The reverse rolling
    # window avoids leaking past events into the future-horizon label.
    steps = prediction_horizon_s // sample_interval_s
    future = df["preemption_event"].shift(-1)
    df["label"] = (
        future.iloc[::-1]
        .rolling(window=steps, min_periods=1)
        .max()
        .iloc[::-1]
        .fillna(0)
        .astype(int)
    )
    return df


def load_telemetry(
    data_dir: str,
    regions: List[str],
    prediction_horizon_s: int = 30,
    sample_interval_s: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load region CSVs and return globally chronological feature/label arrays."""
    if not ML_AVAILABLE:
        raise RuntimeError("Install requirements.txt before loading/training telemetry.")

    frames = []
    missing_files = []
    for region in regions:
        csv_path = Path(data_dir) / f"{region}.csv"
        if not csv_path.exists():
            missing_files.append(str(csv_path))
            continue
        df = _prepare_region(csv_path, prediction_horizon_s, sample_interval_s)
        df["region"] = region
        frames.append(df)
        logger.info("Loaded %s: %d samples, %d positive labels", region, len(df), int(df["label"].sum()))

    if missing_files:
        raise FileNotFoundError(
            "Missing empirical telemetry files:\n  " + "\n  ".join(missing_files) +
            "\nThe training script does not synthesize replacements."
        )
    if not frames:
        raise FileNotFoundError(f"No telemetry CSVs found under {data_dir}")

    combined = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "region"]).reset_index(drop=True)
    X = combined[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = combined["label"].to_numpy(dtype=np.int32)
    logger.info("Combined chronological dataset: %d samples, %d positive labels", len(y), int(y.sum()))
    return X, y


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    if len(X) != len(y) or len(y) < 3:
        raise ValueError("X and y must have the same length and contain at least 3 samples")
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be positive and sum to less than 1")
    n = len(y)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return X[:train_end], y[:train_end], X[train_end:val_end], y[train_end:val_end], X[val_end:], y[val_end:]


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 3.5,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    apply_smote: bool = True,
):
    if not ML_AVAILABLE:
        raise RuntimeError("Install requirements.txt before training.")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training labels must contain both classes")

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if apply_smote:
        positives = int(y_train.sum())
        if positives < 6:
            raise ValueError("SMOTE with k_neighbors=5 requires at least 6 positive training samples")
        X_train_scaled, y_train = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train_scaled, y_train)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=alpha,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        device="cuda" if _cuda_available() else "cpu",
        early_stopping_rounds=50,
    )
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    return model, scaler


def evaluate(model, scaler, X_test: np.ndarray, y_test: np.ndarray, prediction_horizon_s: int = 30) -> dict:
    if len(np.unique(y_test)) < 2:
        raise ValueError("Held-out test labels must contain both classes to compute ROC AUC")
    probs = model.predict_proba(scaler.transform(X_test))[:, 1]
    pred = (probs >= 0.5).astype(int)
    return {
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, probs)),
        "prediction_horizon_s": prediction_horizon_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MorphCloud-LLM preemption predictor")
    parser.add_argument("--data-dir", default="data/traces/raw")
    parser.add_argument("--regions", nargs="+", default=["us-east-1", "us-west-2", "us-central1", "europe-west1"])
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=3.5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--output", default="configs/predictor_weights.pkl", help="Joblib model+scaler artifact")
    parser.add_argument("--metrics-output", default="evaluation/results/predictor_metrics.json")
    args = parser.parse_args()

    X, y = load_telemetry(args.data_dir, args.regions, prediction_horizon_s=args.horizon)
    split = chronological_split(X, y)
    model, scaler = train(*split[:4], alpha=args.alpha, n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.lr, apply_smote=not args.no_smote)
    metrics = evaluate(model, scaler, split[4], split[5], prediction_horizon_s=args.horizon)

    model_path = Path(args.output)
    metrics_path = Path(args.metrics_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Model saved to %s", model_path)
    logger.info("Metrics saved to %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
