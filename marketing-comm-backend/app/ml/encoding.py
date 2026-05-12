from __future__ import annotations
from typing import Any
import pandas as pd


OTHER_CATEGORY = "other"


def normalize_categorical_series(series: pd.Series) -> pd.Series:
    """
    Приводит категориальный столбец к единому виду:
    - заменяет пропуски на пустую строку
    - приводит значения к строковому типу
    - убирает пробелы по краям
    """
    return series.fillna("").astype(str).str.strip()


def replace_rare_categories(
    series: pd.Series,
    min_frequency: int,
    other_category: str = OTHER_CATEGORY,
) -> pd.Series:
    """
    Объединяет редкие категории в общее значение other_category.
    Категория считается редкой, если встречается реже min_frequency раз.
    """
    normalized = normalize_categorical_series(series)
    value_counts = normalized.value_counts()

    frequent_categories = set(
        value_counts[value_counts >= min_frequency].index.tolist()
    )

    return normalized.apply(
        lambda value: value if value in frequent_categories else other_category
    )


def _should_merge_rare_categories(
    column_name: str,
    merge_rare_categories: bool,
    rare_category_columns: list[str],
) -> bool:
    """
    Проверяет, нужно ли для данного столбца объединять редкие категории.
    """
    return merge_rare_categories and column_name in rare_category_columns


def _prepare_series_for_fit(
    series: pd.Series,
    column_name: str,
    merge_rare_categories: bool,
    rare_category_columns: list[str],
    min_frequency: int,
    other_category: str,
) -> pd.Series:
    """
    Подготавливает столбец на этапе обучения: 
    нормализует и, если нужно, заменяет редкие категории на "other".
    """
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


def _prepare_series_for_transform(
    series: pd.Series,
    column_name: str,
    encoder_mapping: dict[str, Any],
) -> pd.Series:
    """
    Подготавливает столбец на этапе прогнозирования 
    с использованием сохраненной схемы.
    """
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


def fit_one_hot_encoder(
    df: pd.DataFrame,
    categorical_columns: list[str],
    merge_rare_categories: bool = False,
    rare_category_columns: list[str] | None = None,
    min_frequency: int = 10,
    other_category: str = OTHER_CATEGORY,
) -> dict[str, Any]:
    """
    Формирует mapping для собственного one-hot кодировщика.

    Возвращаемый словарь содержит:
    - список категориальных столбцов
    - список категорий для каждого столбца
    - настройки режима объединения редких категорий
    """
    if rare_category_columns is None:
        rare_category_columns = []

    categories: dict[str, list[str]] = {}

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

    return {
        "categorical_columns": list(categorical_columns),
        "categories": categories,
        "merge_rare_categories": merge_rare_categories,
        "rare_category_columns": list(rare_category_columns),
        "min_frequency": int(min_frequency),
        "other_category": other_category,
    }


def transform_one_hot(
    df: pd.DataFrame,
    encoder_mapping: dict[str, Any],
    drop_original_columns: bool = True,
) -> pd.DataFrame:
    """
    Применяет ранее обученный one-hot encoder к DataFrame.

    Поведение для новых категорий:
    - если для столбца включён режим объединения редких категорий
      и в mapping присутствует 'other', неизвестные категории
      переводятся в 'other'
    - иначе для них все one-hot признаки этого столбца будут равны 0
    """
    result_df = df.copy()
    categorical_columns = encoder_mapping["categorical_columns"]

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

        for category in encoder_mapping["categories"][column_name]:
            encoded_column_name = f"{column_name}__{category}"
            result_df[encoded_column_name] = (prepared_series == category).astype(int)

    if drop_original_columns:
        result_df = result_df.drop(columns=categorical_columns)

    return result_df


def fit_transform_one_hot(
    df: pd.DataFrame,
    categorical_columns: list[str],
    merge_rare_categories: bool = False,
    rare_category_columns: list[str] | None = None,
    min_frequency: int = 10,
    other_category: str = OTHER_CATEGORY,
    drop_original_columns: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Функция для обучения кодировщика и применения к обучающему DataFrame.
    """
    encoder_mapping = fit_one_hot_encoder(
        df=df,
        categorical_columns=categorical_columns,
        merge_rare_categories=merge_rare_categories,
        rare_category_columns=rare_category_columns,
        min_frequency=min_frequency,
        other_category=other_category,
    )

    transformed_df = transform_one_hot(
        df=df,
        encoder_mapping=encoder_mapping,
        drop_original_columns=drop_original_columns,
    )

    return transformed_df, encoder_mapping


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
    """
    Формирует единый объект для сохранения модели вместе с артефактами
    предобработки.
    """
    artifact = {
        "model": model,
        "encoder_mapping": encoder_mapping,
        "feature_columns": feature_columns,
    }

    if extra_metadata:
        artifact["metadata"] = extra_metadata

    return artifact
