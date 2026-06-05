from __future__ import annotations
from typing import Any
import pandas as pd


OTHER_CATEGORY = "other"


def fit_transform_one_hot(
    df: pd.DataFrame,
    categorical_columns: list[str],
    merge_rare_categories: bool = False,
    rare_category_columns: list[str] | None = None,
    min_frequency: int = 10,
    other_category: str = OTHER_CATEGORY,
    drop_original_columns: bool = True,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    encoder_mapping = fit_one_hot_encoder(
        df=df,
        categorical_columns=categorical_columns,
        merge_rare_categories=merge_rare_categories,
        rare_category_columns=rare_category_columns,
        min_frequency=min_frequency,
        other_category=other_category,
        drop_first=drop_first,
    )

    transformed_df = transform_one_hot(
        df=df,
        encoder_mapping=encoder_mapping,
        drop_original_columns=drop_original_columns,
    )

    return transformed_df, encoder_mapping


def fit_one_hot_encoder(
    df: pd.DataFrame,
    categorical_columns: list[str],
    merge_rare_categories: bool = False,
    rare_category_columns: list[str] | None = None,
    min_frequency: int = 10,
    other_category: str = OTHER_CATEGORY,
    drop_first: bool = True,
) -> dict[str, Any]:
    if rare_category_columns is None:
        rare_category_columns = []

    categories: dict[str, list[str]] = {}
    dropped_categories: dict[str, str | None] = {}

    for column_name in categorical_columns:
        if column_name not in df.columns:
            raise ValueError(
                f"В DataFrame отсутствует категориальный столбец {column_name}."
            )

        prepared_series = _prepare_series_for_fit(
            series=df[column_name],
            column_name=column_name,
            merge_rare_categories=merge_rare_categories,
            rare_category_columns=rare_category_columns,
            min_frequency=min_frequency,
            other_category=other_category,
        )

        unique_categories = sorted(prepared_series.unique().tolist())
        categories[column_name] = unique_categories

        if drop_first and len(unique_categories) > 1:
            dropped_categories[column_name] = unique_categories[0]
        else:
            dropped_categories[column_name] = None

    return {
        "categorical_columns": list(categorical_columns),
        "categories": categories,
        "dropped_categories": dropped_categories,
        "drop_first": drop_first,
        "merge_rare_categories": merge_rare_categories,
        "rare_category_columns": list(rare_category_columns),
        "min_frequency": int(min_frequency),
        "other_category": other_category,
    }


def _prepare_series_for_fit(
    series: pd.Series,
    column_name: str,
    merge_rare_categories: bool,
    rare_category_columns: list[str],
    min_frequency: int,
    other_category: str,
) -> pd.Series:
    normalized = normalize_categorical_series(series)

    if _should_merge_rare_categories(
        column_name=column_name,
        merge_rare_categories=merge_rare_categories,
        rare_category_columns=rare_category_columns,
    ):
        return replace_rare_categories(
            series=normalized,
            min_frequency=min_frequency,
            other_category=other_category,
        )

    return normalized


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _should_merge_rare_categories(
    column_name: str,
    merge_rare_categories: bool,
    rare_category_columns: list[str],
) -> bool:
    return merge_rare_categories and column_name in rare_category_columns


def replace_rare_categories(
    series: pd.Series,
    min_frequency: int,
    other_category: str = OTHER_CATEGORY,
) -> pd.Series:
    normalized = normalize_categorical_series(series)
    value_counts = normalized.value_counts()

    frequent_categories = set(
        value_counts[value_counts >= min_frequency].index.tolist()
    )

    return normalized.apply(
        lambda value: value if value in frequent_categories else other_category
    )


def transform_one_hot(
    df: pd.DataFrame,
    encoder_mapping: dict[str, Any],
    drop_original_columns: bool = True,
) -> pd.DataFrame:
    result_df = df.copy()
    categorical_columns = encoder_mapping["categorical_columns"]
    dropped_categories = encoder_mapping.get("dropped_categories", {})

    for column_name in categorical_columns:
        if column_name not in result_df.columns:
            raise ValueError(
                f"В DataFrame отсутствует категориальный столбец {column_name}."
            )

        prepared_series = _prepare_series_for_transform(
            series=result_df[column_name],
            column_name=column_name,
            encoder_mapping=encoder_mapping,
        )

        dropped_category = dropped_categories.get(column_name)

        for category in encoder_mapping["categories"][column_name]:
            if category == dropped_category:
                continue

            encoded_column_name = f"{column_name}__{category}"
            result_df[encoded_column_name] = (prepared_series == category).astype(int)

    if drop_original_columns:
        result_df = result_df.drop(columns=categorical_columns)

    return result_df


def _prepare_series_for_transform(
    series: pd.Series,
    column_name: str,
    encoder_mapping: dict[str, Any],
) -> pd.Series:
    normalized = normalize_categorical_series(series)

    categories = encoder_mapping["categories"][column_name]
    category_set = set(categories)

    merge_rare_categories = encoder_mapping["merge_rare_categories"]
    rare_category_columns = encoder_mapping["rare_category_columns"]
    other_category = encoder_mapping["other_category"]

    if (
        _should_merge_rare_categories(
            column_name=column_name,
            merge_rare_categories=merge_rare_categories,
            rare_category_columns=rare_category_columns,
        )
        and other_category in category_set
    ):
        return normalized.apply(
            lambda value: value if value in category_set else other_category
        )

    return normalized


def align_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Приводит DataFrame к фиксированному набору признаков и их порядку.
    - если какого-то столбца нет, он добавляется и заполняется нулями
    - лишние столбцы удаляются
    - итоговый порядок столбцов совпадает с feature_columns
    """
    result_df = df.copy()

    for column_name in feature_columns:
        if column_name not in result_df.columns:
            result_df[column_name] = 0

    result_df = result_df[feature_columns]
    return result_df


def build_model_artifact(
    model: Any,
    encoder_mapping: dict[str, Any] | None = None,
    feature_columns: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = {
        "model": model,
        "encoder_mapping": encoder_mapping,
        "feature_columns": feature_columns,
    }

    if extra_metadata:
        artifact["metadata"] = extra_metadata

    return artifact
