from catboost import CatBoostClassifier
import pandas as pd

from .config import (
    CATBOOST_MODEL_PATH,
    CATBOOST_CATEGORICAL_FEATURES,
    THRESHOLD_CANDIDATES,
    MIN_PRECISION,
)


def build_catboost_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=42,
    )


def calculate_metrics_for_threshold(y_true: pd.Series, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

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


def save_catboost_model(model: CatBoostClassifier) -> None:
    CATBOOST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(CATBOOST_MODEL_PATH)


def load_catboost_model() -> CatBoostClassifier:
    if not CATBOOST_MODEL_PATH.exists():
        raise FileNotFoundError("Файл обученной модели CatBoost не найден.")

    model = build_catboost_model()
    model.load_model(CATBOOST_MODEL_PATH)
    return model


def train_catboost_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    model = build_catboost_model()

    model.fit(
        X_train,
        y_train,
        cat_features=CATBOOST_CATEGORICAL_FEATURES,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    best_threshold, best_metrics = select_best_threshold(y_test, y_proba)

    save_catboost_model(model)

    return {
        "algorithm": "CatBoost",
        "threshold": best_threshold,
        "metrics": best_metrics,
    }


def predict_catboost_probabilities(X: pd.DataFrame) -> pd.Series:
    model = load_catboost_model()
    probabilities = model.predict_proba(X)[:, 1]
    return pd.Series(probabilities, index=X.index)