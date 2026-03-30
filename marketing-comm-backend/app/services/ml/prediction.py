import joblib
import pandas as pd

from .config import MODEL_PATH
from sklearn.pipeline import Pipeline


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Файл обученной модели не найден.")

    return joblib.load(MODEL_PATH)


def predict_probabilities(X: pd.DataFrame) -> pd.Series:
    model = load_model()
    probabilities = model.predict_proba(X)[:, 1]
    return pd.Series(probabilities, index=X.index)