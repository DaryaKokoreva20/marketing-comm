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


def get_feature_coefficients() -> pd.DataFrame:
    model = load_model()

    preprocessor = model.named_steps["preprocessor"]
    logistic_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = logistic_model.coef_[0]

    coefficients_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })

    coefficients_df["abs_coefficient"] = coefficients_df["coefficient"].abs()
    coefficients_df = coefficients_df.sort_values(
        by="abs_coefficient",
        ascending=False
    ).reset_index(drop=True)

    return coefficients_df


def get_feature_coefficients() -> pd.DataFrame:
    model = load_model()

    preprocessor = model.named_steps["preprocessor"]
    logistic_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = logistic_model.coef_[0]

    coefficients_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })

    coefficients_df["abs_coefficient"] = coefficients_df["coefficient"].abs()
    coefficients_df = coefficients_df.sort_values(
        by="abs_coefficient",
        ascending=False
    ).reset_index(drop=True)