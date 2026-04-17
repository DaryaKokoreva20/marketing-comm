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
