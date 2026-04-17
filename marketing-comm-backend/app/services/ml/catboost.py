import itertools

import pandas as pd
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from .config import (
    CATBOOST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    CATBOOST_FEATURES,
    THRESHOLD_CANDIDATES,
    MODEL_SELECTION_CONFIG,
    CATBOOST_SEARCH_CONFIG,
)


def build_catboost_model(
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    class_weights=None,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=42,
        class_weights=class_weights,
    )


def calculate_metrics_for_threshold(
    y_true: pd.Series,
    y_proba,
    threshold: float,
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "rocAuc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def select_best_threshold(y_true: pd.Series, y_proba) -> tuple[float, dict]:
    min_precision = MODEL_SELECTION_CONFIG["min_precision"]
    min_recall = MODEL_SELECTION_CONFIG["min_recall"]
    primary_metric = MODEL_SELECTION_CONFIG["primary_metric"]
    secondary_metric = MODEL_SELECTION_CONFIG["secondary_metric"]

    threshold_results = []

    for threshold in THRESHOLD_CANDIDATES:
        metrics = calculate_metrics_for_threshold(y_true, y_proba, threshold)
        threshold_results.append((threshold, metrics))

    valid_candidates = [
        (threshold, metrics)
        for threshold, metrics in threshold_results
        if metrics["precision"] >= min_precision and metrics["recall"] >= min_recall
    ]

    candidates_to_check = valid_candidates if valid_candidates else threshold_results

    best_threshold = None
    best_metrics = None

    for threshold, metrics in candidates_to_check:
        if best_metrics is None:
            best_threshold = threshold
            best_metrics = metrics
            continue

        if metrics[primary_metric] > best_metrics[primary_metric]:
            best_threshold = threshold
            best_metrics = metrics
            continue

        if (
            metrics[primary_metric] == best_metrics[primary_metric]
            and metrics[secondary_metric] > best_metrics[secondary_metric]
        ):
            best_threshold = threshold
            best_metrics = metrics
            continue

        if (
            metrics[primary_metric] == best_metrics[primary_metric]
            and metrics[secondary_metric] == best_metrics[secondary_metric]
            and metrics["rocAuc"] > best_metrics["rocAuc"]
        ):
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def generate_catboost_param_combinations() -> list[dict]:
    params_grid = CATBOOST_SEARCH_CONFIG["params"]
    class_weights_options = CATBOOST_SEARCH_CONFIG["class_weights"]

    keys = list(params_grid.keys())
    values_product = itertools.product(*(params_grid[key] for key in keys))

    combinations = []

    for values in values_product:
        base_params = dict(zip(keys, values))

        for class_weights in class_weights_options:
            combinations.append(
                {
                    "iterations": base_params["iterations"],
                    "learning_rate": base_params["learning_rate"],
                    "depth": base_params["depth"],
                    "l2_leaf_reg": base_params["l2_leaf_reg"],
                    "class_weights": class_weights,
                }
            )

    if not combinations:
        raise ValueError("Не найдено ни одной конфигурации CatBoost.")

    return combinations


def save_catboost_model(model: CatBoostClassifier) -> None:
    CATBOOST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(CATBOOST_MODEL_PATH)


def load_catboost_model() -> CatBoostClassifier:
    if not CATBOOST_MODEL_PATH.exists():
        raise FileNotFoundError("Файл обученной модели CatBoost не найден.")

    model = CatBoostClassifier()
    model.load_model(CATBOOST_MODEL_PATH)
    return model


def serialize_class_weights(class_weights) -> list[int] | None:
    if class_weights is None:
        return None
    return list(class_weights)


def train_single_catboost_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict,
) -> dict:
    model = build_catboost_model(
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        class_weights=params["class_weights"],
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    for column in CATEGORICAL_FEATURES:
        if column in X_train.columns:
            X_train[column] = X_train[column].fillna("").astype(str)
        if column in X_test.columns:
            X_test[column] = X_test[column].fillna("").astype(str)

    X_train = X_train[CATBOOST_FEATURES]
    X_test = X_test[CATBOOST_FEATURES]

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    best_threshold, best_metrics = select_best_threshold(y_test, y_proba)

    return {
        "model": model,
        "algorithm": "CatBoost",
        "threshold": best_threshold,
        "classWeights": serialize_class_weights(params["class_weights"]),
        "iterations": params["iterations"],
        "learningRate": params["learning_rate"],
        "depth": params["depth"],
        "l2LeafReg": params["l2_leaf_reg"],
        "metrics": best_metrics,
    }


def is_better_result(candidate: dict, current_best: dict | None) -> bool:
    if current_best is None:
        return True

    primary_metric = MODEL_SELECTION_CONFIG["primary_metric"]
    secondary_metric = MODEL_SELECTION_CONFIG["secondary_metric"]

    candidate_primary = candidate["metrics"][primary_metric]
    current_primary = current_best["metrics"][primary_metric]

    if candidate_primary > current_primary:
        return True

    if candidate_primary < current_primary:
        return False

    candidate_secondary = candidate["metrics"][secondary_metric]
    current_secondary = current_best["metrics"][secondary_metric]

    if candidate_secondary > current_secondary:
        return True

    if candidate_secondary < current_secondary:
        return False

    return candidate["metrics"]["rocAuc"] > current_best["metrics"]["rocAuc"]


def train_catboost_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    param_combinations = generate_catboost_param_combinations()

    candidate_results = []

    for params in param_combinations:
        result = train_single_catboost_model(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=params,
        )

        print(
            "CatBoost candidate:",
            {
                "classWeights": result["classWeights"],
                "iterations": result["iterations"],
                "learningRate": result["learningRate"],
                "depth": result["depth"],
                "l2LeafReg": result["l2LeafReg"],
                "threshold": result["threshold"],
                "metrics": result["metrics"],
            }
        )

        candidate_results.append(result)

    best_result = None

    for candidate in candidate_results:
        if is_better_result(candidate, best_result):
            best_result = candidate

    save_catboost_model(best_result["model"])

    return {
        "algorithm": best_result["algorithm"],
        "threshold": best_result["threshold"],
        "classWeights": best_result["classWeights"],
        "iterations": best_result["iterations"],
        "learningRate": best_result["learningRate"],
        "depth": best_result["depth"],
        "l2LeafReg": best_result["l2LeafReg"],
        "metrics": best_result["metrics"],
    }


def predict_catboost_probabilities(X: pd.DataFrame) -> pd.Series:
    model = load_catboost_model()

    X = X.copy()

    for column in CATEGORICAL_FEATURES:
        if column in X.columns:
            X[column] = X[column].fillna("").astype(str)

    X = X[CATBOOST_FEATURES]

    probabilities = model.predict_proba(X)[:, 1]
    return pd.Series(probabilities, index=X.index)