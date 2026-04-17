import pandas as pd
from catboost import CatBoostClassifier

from .config import (
    CATBOOST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    CATBOOST_FEATURES,
    CATBOOST_SEARCH_CONFIG,
)
from .evaluation import select_best_threshold, is_better_result
from .search import generate_param_combinations
from .serialization import serialize_class_weights


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


def save_catboost_model(model: CatBoostClassifier) -> None:
    CATBOOST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(CATBOOST_MODEL_PATH)


def load_catboost_model() -> CatBoostClassifier:
    if not CATBOOST_MODEL_PATH.exists():
        raise FileNotFoundError("Файл обученной модели CatBoost не найден.")

    model = CatBoostClassifier()
    model.load_model(CATBOOST_MODEL_PATH)
    return model


def predict_catboost_probabilities(X: pd.DataFrame) -> pd.Series:
    model = load_catboost_model()

    X = X.copy()

    for column in CATEGORICAL_FEATURES:
        if column in X.columns:
            X[column] = X[column].fillna("").astype(str)

    X = X[CATBOOST_FEATURES]

    probabilities = model.predict_proba(X)[:, 1]
    return pd.Series(probabilities, index=X.index)


def generate_catboost_param_combinations() -> list[dict]:
    base_combinations = generate_param_combinations(CATBOOST_SEARCH_CONFIG["params"])
    class_weights_options = CATBOOST_SEARCH_CONFIG["class_weights"]

    combinations = []

    for base_params in base_combinations:
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
