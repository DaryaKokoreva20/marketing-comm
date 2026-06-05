import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.paths import LOGISTIC_MODEL_PATH
from ..config import (
    DATA_SPLIT_CONFIG,
    LOGISTIC_SEARCH_CONFIG,
)
from .custom_logistic import CustomLogisticRegression
from .custom_logistic_pipeline import CustomLogisticPipeline
from ..evaluation import (
    select_best_threshold,
    is_better_result,
    calculate_metrics_for_threshold,
)
from ..search import generate_param_combinations
from ..serialization import serialize_class_weight


def build_logistic_regression_pipeline(params: dict) -> CustomLogisticPipeline:
    model = CustomLogisticRegression(
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        l2_lambda=params["l2_lambda"],
        class_weight=params["class_weight"],
        tolerance=params["tolerance"],
    )

    return CustomLogisticPipeline(
        model=model,
        use_scaler=params["use_scaler"],
        merge_rare_categories=params["merge_rare_categories"],
    )


def build_logistic_training_result(
    pipeline: CustomLogisticPipeline,
    params: dict,
    threshold: float,
    metrics: dict,
) -> dict:
    return {
        "pipeline": pipeline,
        "algorithm": "Custom Logistic Regression",
        "classWeight": serialize_class_weight(params["class_weight"]),
        "useScaler": params["use_scaler"],
        "learningRate": params["learning_rate"],
        "maxIter": params["max_iter"],
        "l2Lambda": params["l2_lambda"],
        "tolerance": params["tolerance"],
        "mergeRareCategories": params["merge_rare_categories"],
        "threshold": threshold,
        "metrics": metrics,
    }


def extract_logistic_model_summary(result: dict) -> dict:
    return {
        "algorithm": result["algorithm"],
        "metrics": result["metrics"],
        "threshold": result["threshold"],
        "classWeight": result["classWeight"],
        "useScaler": result["useScaler"],
        "learningRate": result["learningRate"],
        "maxIter": result["maxIter"],
        "l2Lambda": result["l2Lambda"],
        "tolerance": result["tolerance"],
        "mergeRareCategories": result["mergeRareCategories"],
    }


def train_single_logistic_regression_model(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    params: dict,
) -> dict:
    pipeline = build_logistic_regression_pipeline(params)
    pipeline.fit(X_train, y_train)

    y_valid_proba = pipeline.predict_proba(X_valid)[:, 1]

    best_threshold, validation_metrics = select_best_threshold(
        y_true=y_valid,
        y_proba=y_valid_proba,
    )

    return build_logistic_training_result(
        pipeline=pipeline,
        params=params,
        threshold=best_threshold,
        metrics=validation_metrics,
    )


def train_logistic_regression_model(X: pd.DataFrame, y: pd.Series) -> dict:
    split_config = DATA_SPLIT_CONFIG.copy()
    use_stratify = split_config.get("stratify", False)

    stratify_value = y if use_stratify else None

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
        stratify=stratify_value,
    )

    validation_size = split_config["validation_size"]
    test_size = split_config["test_size"]

    validation_size_from_train_valid = validation_size / (1 - test_size)

    train_valid_stratify_value = y_train_valid if use_stratify else None

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=validation_size_from_train_valid,
        random_state=split_config["random_state"],
        stratify=train_valid_stratify_value,
    )

    param_combinations = generate_logistic_param_combinations()

    best_result = None

    for params in param_combinations:
        candidate = train_single_logistic_regression_model(
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            params=params,
        )

        if is_better_result(candidate, best_result):
            best_result = candidate

    print("Best logistic regression validation metrics:")
    print(best_result["metrics"])
    print("Best logistic regression threshold:")
    print(best_result["threshold"])

    y_test_proba = best_result["pipeline"].predict_proba(X_test)[:, 1]

    test_metrics = calculate_metrics_for_threshold(
        y_true=y_test,
        y_proba=y_test_proba,
        threshold=best_result["threshold"],
    )

    best_result["metrics"] = test_metrics

    save_logistic_model(best_result["pipeline"])

    return extract_logistic_model_summary(best_result)


def generate_logistic_param_combinations() -> list[dict]:
    raw_combinations = generate_param_combinations(LOGISTIC_SEARCH_CONFIG["params"])

    combinations = [
        normalize_logistic_params(raw_params)
        for raw_params in raw_combinations
    ]

    if not combinations:
        raise ValueError("Не найдено ни одной конфигурации логистической регрессии.")

    return combinations


def save_logistic_model(model: CustomLogisticPipeline) -> None:
    LOGISTIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LOGISTIC_MODEL_PATH)


def load_logistic_model() -> CustomLogisticPipeline:
    if not LOGISTIC_MODEL_PATH.exists():
        raise FileNotFoundError("Файл обученной логистической регрессии не найден.")

    return joblib.load(LOGISTIC_MODEL_PATH)


def predict_logistic_probabilities(X: pd.DataFrame) -> pd.Series:
    pipeline = load_logistic_model()
    probabilities = pipeline.predict_proba(X)[:, 1]
    return pd.Series(probabilities, index=X.index)


def normalize_logistic_params(raw_params: dict) -> dict:
    return {
        "class_weight": raw_params["class_weight"],
        "use_scaler": raw_params["use_scaler"],
        "learning_rate": raw_params["learning_rate"],
        "max_iter": raw_params["max_iter"],
        "l2_lambda": raw_params["l2_lambda"],
        "tolerance": raw_params["tolerance"],
        "merge_rare_categories": raw_params["merge_rare_categories"],
    }


def get_logistic_feature_coefficients() -> pd.DataFrame:
    pipeline = load_logistic_model()

    if pipeline.feature_columns is None:
        raise ValueError("У pipeline отсутствует список признаков.")

    coefficients = pipeline.model.coef_[0]

    coefficients_df = pd.DataFrame(
        {
            "feature": pipeline.feature_columns,
            "coefficient": coefficients,
        }
    )

    coefficients_df["abs_coefficient"] = coefficients_df["coefficient"].abs()
    coefficients_df = coefficients_df.sort_values(
        by="abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    return coefficients_df
