from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
MODEL_PATH = MODELS_DIR / "logistic_regression_pipeline.joblib"

THRESHOLD_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
MIN_PRECISION = 0.4

NUMERIC_FEATURES = [
    "tenure_days",
    "avg_order_value",
    "order_frequency",
    "total_orders",
    "recency_days",
    "repeat_purchase_propensity",
    "prev_response_rate",
    "prev_comm_count_channel",
    "prev_comm_response_rate_channel",
    "last_comm_days_channel",
    "prev_comm_count_scenario",
    "prev_comm_response_rate_scenario",
    "promo_sensitivity",
    "b2c_age",
    "time_trigger",
    "comm_hour",
]

CATEGORICAL_FEATURES = [
    "client_type",
    "industry_category",
    "channel_type",
    "scenario_type",
]