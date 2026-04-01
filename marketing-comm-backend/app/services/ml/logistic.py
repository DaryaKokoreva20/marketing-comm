import joblib
import pandas as pd

from .config import MODEL_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES, THRESHOLD_CANDIDATES, MIN_PRECISION
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_logistic_regression_pipeline(class_weight=None, use_scaler=False) -> Pipeline:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if use_scaler:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight=class_weight,)),
        ]
    )
    return pipeline


def calculate_metrics_for_threshold(y_true: pd.Series, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "rocAuc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def select_best_threshold(y_true: pd.Series, y_proba) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = None
    best_f1 = -1.0
    best_precision = -1.0

    valid_candidates = []

    for threshold in THRESHOLD_CANDIDATES:
        metrics = calculate_metrics_for_threshold(y_true, y_proba, threshold)

        if metrics["precision"] >= MIN_PRECISION:
            valid_candidates.append((threshold, metrics))

    candidates_to_check = valid_candidates if valid_candidates else [
        (threshold, calculate_metrics_for_threshold(y_true, y_proba, threshold))
        for threshold in THRESHOLD_CANDIDATES
    ]

    for threshold, metrics in candidates_to_check:
        if (
            metrics["f1"] > best_f1 or
            (metrics["f1"] == best_f1 and metrics["precision"] > best_precision)
        ):
            best_f1 = metrics["f1"]
            best_precision = metrics["precision"]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def train_single_logistic_regression_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    class_weight=None,
    use_scaler=False,
) -> dict:
    pipeline = build_logistic_regression_pipeline(
        class_weight=class_weight,
        use_scaler=use_scaler,
    )

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    best_threshold, best_metrics = select_best_threshold(y_test, y_proba)

    return {
        "pipeline": pipeline,
        "algorithm": "Logistic Regression",
        "classWeight": class_weight,
        "useScaler": use_scaler,
        "threshold": best_threshold,
        "metrics": best_metrics,
    }


def is_better_result(candidate: dict, current_best: dict | None) -> bool:
    if current_best is None:
        return True

    candidate_f1 = candidate["metrics"]["f1"]
    current_f1 = current_best["metrics"]["f1"]

    candidate_precision = candidate["metrics"]["precision"]
    current_precision = current_best["metrics"]["precision"]

    if candidate_f1 > current_f1:
        return True

    if candidate_f1 == current_f1 and candidate_precision > current_precision:
        return True

    return False


def train_logistic_regression_model(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidate_results = [
        train_single_logistic_regression_model(
            X_train, X_test, y_train, y_test,
            class_weight=None,
            use_scaler=False,
        ),
        train_single_logistic_regression_model(
            X_train, X_test, y_train, y_test,
            class_weight=None,
            use_scaler=True,
        ),
        train_single_logistic_regression_model(
            X_train, X_test, y_train, y_test,
            class_weight="balanced",
            use_scaler=False,
        ),
        train_single_logistic_regression_model(
            X_train, X_test, y_train, y_test,
            class_weight="balanced",
            use_scaler=True,
        ),
    ]

    best_result = None

    for candidate in candidate_results:
        if is_better_result(candidate, best_result):
            best_result = candidate

    save_model(best_result["pipeline"])

    return {
        "algorithm": best_result["algorithm"],
        "metrics": best_result["metrics"],
        "threshold": best_result["threshold"],
        "classWeight": best_result["classWeight"],
        "useScaler": best_result["useScaler"],
    }


def save_model(model: Pipeline) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


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