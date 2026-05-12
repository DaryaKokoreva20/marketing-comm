import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.paths import BOOSTING_MODEL_PATH
from ..config import (
    BOOSTING_SEARCH_CONFIG,
    TRAIN_TEST_SPLIT_CONFIG,
)
from ..evaluation import select_best_threshold, is_better_result
from ..search import generate_param_combinations
from ..serialization import serialize_class_weights
from .custom_gradient_boosting import CustomGradientBoostingClassifier
from .custom_boosting_pipeline import CustomBoostingPipeline


def train_boosting_model(X: pd.DataFrame, y: pd.Series) -> dict:
    split_config = TRAIN_TEST_SPLIT_CONFIG.copy()
    stratify_value = y if split_config.get("stratify", False) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
        stratify=stratify_value,
    )

    param_combinations = generate_boosting_param_combinations()

    best_result = None

    for params in param_combinations:
        candidate = train_single_boosting_model(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=params,
        )

        if is_better_result(candidate, best_result):
            best_result = candidate

    save_boosting_model(best_result["model"])

    return {
        "algorithm": best_result["algorithm"],
        "threshold": best_result["threshold"],
        "classWeights": best_result["classWeights"],
        "iterations": best_result["iterations"],
        "learningRate": best_result["learningRate"],
        "depth": best_result["depth"],
        "l2LeafReg": best_result["l2LeafReg"],
        "mergeRareCategories": best_result["mergeRareCategories"],
        "metrics": best_result["metrics"],
    }


def generate_boosting_param_combinations() -> list[dict]:
    base_combinations = generate_param_combinations(BOOSTING_SEARCH_CONFIG["params"])
    class_weights_options = BOOSTING_SEARCH_CONFIG["class_weights"]

    combinations = []

    for base_params in base_combinations:
        for class_weights in class_weights_options:
            combinations.append(
                {
                    "iterations": base_params["iterations"],
                    "learning_rate": base_params["learning_rate"],
                    "depth": base_params["depth"],
                    "l2_leaf_reg": base_params["l2_leaf_reg"],
                    "merge_rare_categories": base_params["merge_rare_categories"],
                    "class_weights": class_weights,
                }
            )

    if not combinations:
        raise ValueError("Не найдено ни одной конфигурации градиентного бустинга.")

    return combinations


def train_single_boosting_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict,
) -> dict:
    pipeline = build_gradient_boosting_pipeline(params)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    best_threshold, best_metrics = select_best_threshold(y_test, y_proba)

    return {
        "model": pipeline,
        "algorithm": "Custom Gradient Boosting",
        "threshold": best_threshold,
        "classWeights": serialize_class_weights(params["class_weights"]),
        "iterations": params["iterations"],
        "learningRate": params["learning_rate"],
        "depth": params["depth"],
        "l2LeafReg": params["l2_leaf_reg"],
        "mergeRareCategories": params["merge_rare_categories"],
        "metrics": best_metrics,
    }


def build_gradient_boosting_pipeline(params: dict) -> CustomBoostingPipeline:
    model = CustomGradientBoostingClassifier(
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        class_weights=params["class_weights"],
    )

    return CustomBoostingPipeline(
        model=model,
        merge_rare_categories=params["merge_rare_categories"],
    )


def save_boosting_model(model: CustomBoostingPipeline) -> None:
    BOOSTING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BOOSTING_MODEL_PATH)


def load_boosting_model() -> CustomBoostingPipeline:
    if not BOOSTING_MODEL_PATH.exists():
        raise FileNotFoundError("Файл обученной модели градиентного бустинга не найден.")

    return joblib.load(BOOSTING_MODEL_PATH)


def predict_boosting_probabilities(X: pd.DataFrame) -> pd.Series:
    pipeline = load_boosting_model()
    probabilities = pipeline.predict_proba(X)[:, 1]
    return pd.Series(probabilities, index=X.index)
