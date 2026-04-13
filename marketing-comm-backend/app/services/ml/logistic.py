import itertools
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    LOGISTIC_MODEL_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    THRESHOLD_CANDIDATES,
    MODEL_SELECTION_CONFIG,
    TRAIN_TEST_SPLIT_CONFIG,
    LOGISTIC_SEARCH_CONFIG,
)


def build_logistic_regression_pipeline(params: dict) -> Pipeline:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if params["use_scaler"]:
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

    model_params = {
        "max_iter": params["max_iter"],
        "class_weight": params["class_weight"],
        "solver": params["solver"],
        "C": params["C"],
    }

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(**model_params)),
        ]
    )

    return pipeline


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

    return best_threshold, best_metrics


def normalize_logistic_params(raw_params: dict) -> dict:
    return {
        "class_weight": raw_params["class_weight"],
        "use_scaler": raw_params["use_scaler"],
        "solver": raw_params["solver"],
        "max_iter": raw_params["max_iter"],
        "C": raw_params["C"],
    }


def generate_logistic_param_combinations() -> list[dict]:
    params_grid = LOGISTIC_SEARCH_CONFIG["params"]

    keys = list(params_grid.keys())
    values_product = itertools.product(*(params_grid[key] for key in keys))

    combinations = []

    for values in values_product:
        raw_params = dict(zip(keys, values))
        params = normalize_logistic_params(raw_params)
        combinations.append(params)

    if not combinations:
        raise ValueError("Не найдено ни одной конфигурации логистической регрессии.")

    return combinations


def train_single_logistic_regression_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict,
) -> dict:
    pipeline = build_logistic_regression_pipeline(params)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    best_threshold, best_metrics = select_best_threshold(y_test, y_proba)

    return {
        "pipeline": pipeline,
        "algorithm": "Logistic Regression",
        "classWeight": serialize_class_weight(params["class_weight"]),
        "useScaler": params["use_scaler"],
        "solver": serialize_optional_value(params["solver"]),
        "maxIter": params["max_iter"],
        "C": params["C"],
        "threshold": best_threshold,
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


def train_logistic_regression_model(X: pd.DataFrame, y: pd.Series) -> dict:
    split_config = TRAIN_TEST_SPLIT_CONFIG.copy()
    stratify_value = y if split_config.get("stratify", False) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
        stratify=stratify_value,
    )

    param_combinations = generate_logistic_param_combinations()

    candidate_results = []

    for params in param_combinations:
        result = train_single_logistic_regression_model(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=params,
        )
        candidate_results.append(result)

    best_result = None

    for candidate in candidate_results:
        if is_better_result(candidate, best_result):
            best_result = candidate

    save_logistic_model(best_result["pipeline"])

    return {
        "algorithm": best_result["algorithm"],
        "metrics": best_result["metrics"],
        "threshold": best_result["threshold"],
        "classWeight": best_result["classWeight"],
        "useScaler": best_result["useScaler"],
        "solver": best_result["solver"],
        "maxIter": best_result["maxIter"],
        "C": best_result["C"],
    }


def save_logistic_model(model: Pipeline) -> None:
    LOGISTIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LOGISTIC_MODEL_PATH)


def load_logistic_model() -> Pipeline:
    if not LOGISTIC_MODEL_PATH.exists():
        raise FileNotFoundError("Файл обученной модели логистической регрессии не найден.")

    return joblib.load(LOGISTIC_MODEL_PATH)


def predict_logistic_probabilities(X: pd.DataFrame) -> pd.Series:
    model = load_logistic_model()
    probabilities = model.predict_proba(X)[:, 1]
    return pd.Series(probabilities, index=X.index)


def get_logistic_feature_coefficients() -> pd.DataFrame:
    model = load_logistic_model()

    preprocessor = model.named_steps["preprocessor"]
    logistic_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = logistic_model.coef_[0]

    coefficients_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    coefficients_df["abs_coefficient"] = coefficients_df["coefficient"].abs()
    coefficients_df = coefficients_df.sort_values(
        by="abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    return coefficients_df


def serialize_class_weight(class_weight) -> str:
    if class_weight is None:
        return "—"

    if isinstance(class_weight, str):
        return class_weight

    if isinstance(class_weight, dict):
        return ", ".join(f"{key}:{value}" for key, value in sorted(class_weight.items()))

    return str(class_weight)


def serialize_optional_value(value) -> str:
    if value is None:
        return "—"
    return str(value)