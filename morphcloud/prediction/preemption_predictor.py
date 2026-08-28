"""
Gradient-Boosted Preemption Prediction Model.

Implements the preemption predictor described in Section 3.3 of MorphCloud-LLM
(Electronics 2026). The model is a gradient-boosted decision tree ensemble
(XGBoost) trained on public cloud API telemetry features with an asymmetric
recall-weighted loss function (Equation 11) to minimize missed preemption events.

Key specifications:
  - Feature vector: 7 conceptual features; cyclical hour/day encoding yields 9 numeric inputs (Eq. 9)
  - Prediction horizon: 30 seconds
  - Recall at 30-s horizon: 89% (Table 6)
  - AUC: 0.97 (Section 4.4)
  - Class imbalance ratio: ~1:538 (6,220,800 samples, 11,541 positive)
  - Asymmetric loss weight alpha = 3.5 (cross-validated)
  - Weekly retraining on the latest 30 days of telemetry
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from imblearn.over_sampling import SMOTE
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost/sklearn/imbalanced-learn not installed; training/inference is unavailable.")


# ---------------------------------------------------------------------------
# Feature vector definition (Equation 9, Section 3.3)
# ---------------------------------------------------------------------------

@dataclass
class PreemptionFeatureVector:
    """
    x(t) = [P_spot(t), delta_P_spot(t), u_az(t), n_active(t),
             T_uptime(t), h_t, d_t]

    All features are obtainable exclusively from public cloud APIs,
    instance metadata endpoints, or derived arithmetic operations.
    No proprietary or provider-internal telemetry is required.

    Attributes
    ----------
    spot_price : float
        Current spot instance price in USD/h, sampled every 10 s via
        AWS EC2 DescribeSpotPriceHistory or GCP Spot VM pricing API.
    spot_price_velocity : float
        First-order finite difference of consecutive P_spot(t) samples.
    az_utilization_proxy : float
        Estimated AZ utilization from EC2 RunInstances API launch rejection
        rate over a 60-second sliding window (a reliable public signal of
        zone capacity pressure; not from private provider telemetry).
    n_active_spot : int
        Count of active spot instances in the AZ, from CloudWatch
        EC2/InstanceCount metric filtered by spot lifecycle.
    uptime_seconds : float
        Current instance uptime, from AWS instance metadata combined with
        launch timestamp from DescribeInstances API.
    hour_sin : float
        Cyclical encoding of hour-of-day: sin(2*pi*hour/24).
    hour_cos : float
        Cyclical encoding of hour-of-day: cos(2*pi*hour/24).
    dow_sin : float
        Cyclical encoding of day-of-week: sin(2*pi*dow/7).
    dow_cos : float
        Cyclical encoding of day-of-week: cos(2*pi*dow/7).
    """
    spot_price: float = 0.0
    spot_price_velocity: float = 0.0
    az_utilization_proxy: float = 0.0
    n_active_spot: int = 0
    uptime_seconds: float = 0.0
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    dow_sin: float = 0.0
    dow_cos: float = 0.0

    # Stable public feature order used by training/inference pipelines.
    FEATURE_NAMES = (
        "spot_price", "spot_price_velocity", "az_utilization_proxy",
        "n_active_spot", "uptime_seconds", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
    )

    @classmethod
    def from_raw(cls, hour: int, dow: int, **kwargs) -> "PreemptionFeatureVector":
        """Construct with auto-computed cyclical time features."""
        obj = cls(**kwargs)
        obj.hour_sin = math.sin(2 * math.pi * hour / 24)
        obj.hour_cos = math.cos(2 * math.pi * hour / 24)
        obj.dow_sin = math.sin(2 * math.pi * dow / 7)
        obj.dow_cos = math.cos(2 * math.pi * dow / 7)
        return obj

    def to_array(self) -> np.ndarray:
        return np.array([
            self.spot_price,
            self.spot_price_velocity,
            self.az_utilization_proxy,
            float(self.n_active_spot),
            self.uptime_seconds,
            self.hour_sin,
            self.hour_cos,
            self.dow_sin,
            self.dow_cos,
        ], dtype=np.float32)


# Feature names for the PreemptionFeatureVector (kept outside dataclass to
# avoid mutable default field error in Python 3.12+)
PREEMPTION_FEATURE_NAMES: List[str] = [
    "spot_price",
    "spot_price_velocity",
    "az_utilization_proxy",
    "n_active_spot",
    "uptime_seconds",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


# ---------------------------------------------------------------------------
# Adaptive migration threshold (Equation 13, Section 3.3)
# ---------------------------------------------------------------------------

@dataclass
class MigrationThresholdConfig:
    """
    rho_thresh(t) = rho_base * (1 - lambda * load(t) / capacity(t))

    rho_base = 0.7 (nominal threshold)
    lambda   = 0.3 (load sensitivity coefficient)
    """
    rho_base: float = 0.7
    load_sensitivity: float = 0.3  # lambda

    def compute(self, current_load: float, capacity: float) -> float:
        """Equation 13: adaptive threshold based on current cluster load."""
        if capacity <= 0:
            return self.rho_base
        load_ratio = min(current_load / capacity, 1.0)
        return self.rho_base * (1.0 - self.load_sensitivity * load_ratio)


# ---------------------------------------------------------------------------
# XGBoost preemption predictor
# ---------------------------------------------------------------------------

class PreemptionPredictor:
    """
    Gradient-boosted preemption prediction model.

    Training details (Section 3.8):
      - 6 months of data (July–December 2024), 4 regions.
      - 6,220,800 total samples; 11,541 positive (class imbalance ~1:538).
      - SMOTE oversampling applied during training.
      - Asymmetric loss weight alpha = 3.5 (Equation 11).
      - 70/15/15 chronological train/validation/test split.
      - Weekly retraining on the latest 30 days of telemetry.

    Performance at 30-second horizon (Table 6):
      - Average recall:    0.89
      - Average precision: 0.83
      - Average F1:        0.86
      - AUC:               0.97
    """

    def __init__(
        self,
        prediction_horizon_s: int = 30,
        alpha: float = 3.5,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        model_path: Optional[str] = None,
    ):
        self.prediction_horizon_s = prediction_horizon_s
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self.threshold_config = MigrationThresholdConfig()
        self.scaler = None
        self.model = None
        self._price_history: List[Tuple[float, float]] = []  # (timestamp, price)

        if model_path:
            self.load(model_path)
        else:
            self._build_model()

    def _build_model(self):
        if not XGB_AVAILABLE:
            logger.warning("XGBoost stack not available; predictor model was not created.")
            return

        # Asymmetric loss (Equation 11): scale_pos_weight approximates alpha
        # weighting; the full recall-precision trade-off is controlled by alpha.
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.alpha,       # penalizes false negatives
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42,
        )
        self.scaler = MinMaxScaler()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        apply_smote: bool = True,
    ):
        """
        Train the XGBoost predictor on historical spot market telemetry.

        The training set is partitioned chronologically (Section 3.8):
          - Months 1–4 (Jul–Oct 2024): training (70%)
          - Month 5 (Nov 2024): validation (15%)
          - Month 6 (Dec 2024): held-out test (15%)

        SMOTE is applied to address the 1:538 class imbalance.
        Continuous features are normalized with min-max scaling.
        """
        if not XGB_AVAILABLE:
            return

        # Min-max scaling for continuous features
        X_train_scaled = self.scaler.fit_transform(X_train)

        if apply_smote:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            logger.info(
                "SMOTE applied: %d positive / %d total samples",
                int(y_train.sum()), len(y_train),
            )

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]

        # XGBoost >=2.1 removed early_stopping_rounds from fit(); set it as
        # an estimator parameter so the code remains compatible with 2.0.3+ APIs.
        self.model.set_params(early_stopping_rounds=50 if eval_set else None)
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            verbose=100,
        )
        logger.info("Preemption predictor training complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: PreemptionFeatureVector) -> float:
        """
        Equation 12: rho(t) = f_theta(x(t)) in [0, 1].

        Returns the preemption confidence score for the current instance state.
        """
        if self.model is None or not XGB_AVAILABLE:
            return 0.0

        x = features.to_array().reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        prob = self.model.predict_proba(x_scaled)[0, 1]
        return float(prob)

    def should_migrate(
        self,
        features: PreemptionFeatureVector,
        current_load: float,
        capacity: float,
    ) -> Tuple[bool, float, float]:
        """
        Equation 15 migration trigger:
          Mode(t) = Continuity if rho(t) > rho_thresh(t), else Normal.

        Returns
        -------
        trigger : bool
            True if migration should be initiated.
        rho : float
            Preemption confidence score.
        threshold : float
            Adaptive threshold used for this decision.
        """
        rho = self.predict_proba(features)
        threshold = self.threshold_config.compute(current_load, capacity)
        return rho > threshold, rho, threshold

    def update_price_history(self, timestamp: float, price: float):
        """
        Maintain a rolling price history for computing spot_price_velocity.
        Velocity = first-order finite difference of consecutive price samples.
        """
        self._price_history.append((timestamp, price))
        if len(self._price_history) > 100:
            self._price_history.pop(0)

    def compute_price_velocity(self) -> float:
        """Compute delta_P_spot(t) from the two most recent price samples."""
        if len(self._price_history) < 2:
            return 0.0
        t1, p1 = self._price_history[-2]
        t2, p2 = self._price_history[-1]
        dt = t2 - t1
        if dt <= 0:
            return 0.0
        return (p2 - p1) / dt

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        if self.model is None:
            return
        import joblib
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        logger.info("Predictor saved to %s", path)

    def load(self, path: str):
        import joblib
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        logger.info("Predictor loaded from %s", path)


# ---------------------------------------------------------------------------
# Request-state classifier (Section 3.3 / Section 4.3)
# ---------------------------------------------------------------------------

class RequestStateClassifier:
    """
    Lightweight three-class request-state classifier.

    Classes: Normal | Migrating | Recovered
    Operates at orchestrator level to track per-request operational state.
    Distinct from the instance-level PreemptionPredictor.

    Model: XGBoost, 50 trees, max depth 4, standard cross-entropy loss.
    Overall accuracy: 99.8% (all rounds), 98.5% on 521 preemption events.

    Per-class accuracy (Section 4.4):
      Normal:    99.7%
      Migrating: 96.2% (101/105)
      Recovered: 94.0% (47/50)
    """

    CLASSES = ["Normal", "Migrating", "Recovered"]
    CLASS_NORMAL = 0
    CLASS_MIGRATING = 1
    CLASS_RECOVERED = 2

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            self.load(model_path)
        else:
            self._build_model()

    def _build_model(self):
        if not XGB_AVAILABLE:
            return
        self.model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            eval_metric="mlogloss",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42,
        )

    def classify(
        self,
        queue_position: int,
        instance_health: float,
        version_vector_available: bool,
        elapsed_since_last_token_ms: float,
    ) -> int:
        """
        Classify the operational state of a single request.

        Features: request queue position, instance health signal,
        KV-cache checkpoint version vector availability, and elapsed
        time since the last verified token.

        Returns index into CLASSES list.
        """
        if self.model is None:
            # Heuristic fallback
            if elapsed_since_last_token_ms > 500:
                return self.CLASS_MIGRATING
            if version_vector_available and instance_health < 0.5:
                return self.CLASS_RECOVERED
            return self.CLASS_NORMAL

        x = np.array([
            float(queue_position),
            instance_health,
            float(version_vector_available),
            elapsed_since_last_token_ms,
        ], dtype=np.float32).reshape(1, -1)
        return int(self.model.predict(x)[0])

    def save(self, path: str):
        if self.model is None:
            return
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str):
        import joblib
        self.model = joblib.load(path)
