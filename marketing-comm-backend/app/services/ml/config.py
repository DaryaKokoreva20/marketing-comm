from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_pipeline.joblib"
CATBOOST_MODEL_PATH = MODELS_DIR / "catboost_model.cbm"

THRESHOLD_CANDIDATES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
MIN_PRECISION = 0.4
MIN_RECALL = 0.4

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
    "comm_time_sin",
    "comm_time_cos",
    "orders_per_30_days",
]

CATEGORICAL_FEATURES = [
    "client_type",
    "industry_category",
    "channel_type",
    "scenario_type",
]

CATBOOST_FEATURES = [
    "client_type",
    "tenure_days",
    "avg_order_value",
    "order_frequency",
    "total_orders",
    "recency_days",
    "repeat_purchase_propensity",
    "industry_category",
    "channel_type",
    "scenario_type",
    "prev_response_rate",
    "time_trigger",
    "prev_comm_count_channel",
    "prev_comm_response_rate_channel",
    "last_comm_days_channel",
    "prev_comm_count_scenario",
    "prev_comm_response_rate_scenario",
    "promo_sensitivity",
    "b2c_age",
    "comm_time_sin",
    "comm_time_cos",
    "orders_per_30_days",
]