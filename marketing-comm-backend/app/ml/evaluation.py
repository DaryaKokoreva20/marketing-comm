import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from .config import THRESHOLD_CANDIDATES, MODEL_SELECTION_CONFIG


def calculate_metrics_for_threshold(
    y_true: pd.Series,
    y_proba: pd.Series,
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


def is_better_metrics(candidate_metrics: dict, current_metrics: dict | None) -> bool:
    if current_metrics is None:
        return True

    primary_metric = MODEL_SELECTION_CONFIG["primary_metric"]
    secondary_metric = MODEL_SELECTION_CONFIG["secondary_metric"]

    candidate_primary = candidate_metrics[primary_metric]
    current_primary = current_metrics[primary_metric]

    if candidate_primary > current_primary:
        return True
    if candidate_primary < current_primary:
        return False

    candidate_secondary = candidate_metrics[secondary_metric]
    current_secondary = current_metrics[secondary_metric]

    if candidate_secondary > current_secondary:
        return True
    if candidate_secondary < current_secondary:
        return False

    return candidate_metrics["rocAuc"] > current_metrics["rocAuc"]


def filter_valid_threshold_candidates(
    threshold_results: list[tuple[float, dict]],
    min_precision: float,
    min_recall: float,
) -> list[tuple[float, dict]]:
    return [
        (threshold, metrics)
        for threshold, metrics in threshold_results
        if metrics["precision"] >= min_precision and metrics["recall"] >= min_recall
    ]


def select_best_threshold(y_true: pd.Series, y_proba: pd.Series) -> tuple[float, dict]:
    min_precision = MODEL_SELECTION_CONFIG["min_precision"]
    min_recall = MODEL_SELECTION_CONFIG["min_recall"]

    threshold_results = []

    for threshold in THRESHOLD_CANDIDATES:
        metrics = calculate_metrics_for_threshold(y_true, y_proba, threshold)
        threshold_results.append((threshold, metrics))

    valid_candidates = filter_valid_threshold_candidates(
        threshold_results=threshold_results,
        min_precision=min_precision,
        min_recall=min_recall,
    )

    candidates_to_check = valid_candidates if valid_candidates else threshold_results

    best_threshold = None
    best_metrics = None

    for threshold, metrics in candidates_to_check:
        if is_better_metrics(metrics, best_metrics):
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def is_better_result(candidate: dict, current_best: dict | None) -> bool:
    if current_best is None:
        return True

    return is_better_metrics(candidate["metrics"], current_best["metrics"])