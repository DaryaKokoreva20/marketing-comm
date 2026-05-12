from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BACKEND_DIR / "storage"

UPLOADS_DIR = STORAGE_DIR / "uploads"
MODELS_DIR = STORAGE_DIR / "models"
PREDICTIONS_DIR = STORAGE_DIR / "predictions"
METADATA_PATH = STORAGE_DIR / "metadata.json"

LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_pipeline.joblib"
BOOSTING_MODEL_PATH = MODELS_DIR / "gradient_boosting_pipeline.joblib"
