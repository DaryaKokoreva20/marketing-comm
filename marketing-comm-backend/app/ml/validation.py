import pandas as pd

from app.constants.model import (
    AVAILABLE_CHANNELS,
    AVAILABLE_SCENARIOS,
    REQUIRED_TRAIN_COLUMNS,
    REQUIRED_PREDICT_COLUMNS,
    TRAIN_TARGET_COLUMN,
)
from app.ml.config import (
    NON_NEGATIVE_NUMERIC_COLUMNS, 
    RATIO_COLUMNS,
    TRAIN_BINARY_COLUMNS,
    PREDICT_BINARY_COLUMNS
)


def validate_train_dataframe(df: pd.DataFrame) -> None:
    validate_required_columns(df, REQUIRED_TRAIN_COLUMNS, "обучающем")
    validate_not_empty(df, "Обучающий файл пуст.")

    validate_common_dataframe(df)
    validate_channel_type_column(df)
    validate_scenario_type_column(df)
    validate_binary_columns(df, TRAIN_BINARY_COLUMNS)


def validate_predict_dataframe(df: pd.DataFrame) -> None:
    validate_required_columns(df, REQUIRED_PREDICT_COLUMNS, "файле для прогнозирования")
    validate_forbidden_columns(
        df=df,
        forbidden_columns={TRAIN_TARGET_COLUMN, "channel_type", "scenario_type"},
        file_label="файле для прогнозирования",
    )
    validate_not_empty(df, "Файл для прогнозирования пуст.")

    validate_common_dataframe(df)
    validate_binary_columns(df, PREDICT_BINARY_COLUMNS)


def validate_common_dataframe(df: pd.DataFrame) -> None:
    validate_client_type_column(df)
    validate_industry_column(df)
    validate_non_negative_numeric_columns(df, NON_NEGATIVE_NUMERIC_COLUMNS)
    validate_ratio_columns(df, RATIO_COLUMNS)
    validate_b2c_age_column(df)
    validate_comm_time_column(df)


def validate_required_columns(df: pd.DataFrame, required_columns: set, file_label: str) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"В {file_label} отсутствуют обязательные колонки: {', '.join(sorted(missing_columns))}"
        )


def validate_forbidden_columns(
    df: pd.DataFrame,
    forbidden_columns: set[str],
    file_label: str,
) -> None:
    existing_forbidden_columns = forbidden_columns & set(df.columns)

    if existing_forbidden_columns:
        raise ValueError(
            f"В {file_label} не должно быть колонок: "
            f"{', '.join(sorted(existing_forbidden_columns))}. "
            "Похоже, загружен файл для обучения, а не файл для прогнозирования."
        )
    

def validate_not_empty(df: pd.DataFrame, error_message: str) -> None:
    if df.empty:
        raise ValueError(error_message)


def validate_numeric_column(
    df: pd.DataFrame,
    column_name: str,
    allow_null: bool = False,
) -> pd.Series:
    series = df[column_name]

    if not allow_null and series.isnull().any():
        raise ValueError(f"Колонка {column_name} содержит пустые значения.")

    converted = pd.to_numeric(series, errors="coerce")

    if allow_null:
        invalid_mask = series.notnull() & converted.isnull()
    else:
        invalid_mask = converted.isnull()

    if invalid_mask.any():
        raise ValueError(f"Колонка {column_name} должна содержать только числовые значения.")

    return converted


def validate_non_negative_numeric_column(df: pd.DataFrame, column_name: str) -> None:
    converted = validate_numeric_column(df, column_name)

    if (converted < 0).any():
        raise ValueError(f"Колонка {column_name} не должна содержать отрицательные значения.")


def validate_non_negative_numeric_columns(df: pd.DataFrame, column_names: list[str]) -> None:
    for column_name in column_names:
        validate_non_negative_numeric_column(df, column_name)


def validate_ratio_column(df: pd.DataFrame, column_name: str) -> None:
    converted = validate_numeric_column(df, column_name)

    if ((converted < 0) | (converted > 1)).any():
        raise ValueError(f"Колонка {column_name} должна содержать значения в диапазоне от 0 до 1.")


def validate_ratio_columns(df: pd.DataFrame, column_names: list[str]) -> None:
    for column_name in column_names:
        validate_ratio_column(df, column_name)


def validate_binary_column(df: pd.DataFrame, column_name: str) -> None:
    converted = validate_numeric_column(df, column_name)

    if not converted.isin([0, 1]).all():
        raise ValueError(f"Колонка {column_name} должна содержать только значения 0 и 1.")


def validate_binary_columns(df: pd.DataFrame, column_names: list[str]) -> None:
    for column_name in column_names:
        validate_binary_column(df, column_name)


def validate_allowed_values(df: pd.DataFrame, column_name: str, allowed_values: list[str]) -> None:
    series = df[column_name]

    if series.isnull().any():
        raise ValueError(f"Колонка {column_name} содержит пустые значения.")

    normalized_values = series.astype(str).str.strip()
    invalid_values = set(normalized_values) - set(allowed_values)

    if invalid_values:
        raise ValueError(
            f"Колонка {column_name} содержит недопустимые значения: {', '.join(sorted(invalid_values))}"
        )


def validate_client_type_column(df: pd.DataFrame) -> None:
    validate_allowed_values(df, "client_type", ["B2B", "B2C"])


def validate_channel_type_column(df: pd.DataFrame) -> None:
    validate_allowed_values(df, "channel_type", AVAILABLE_CHANNELS)


def validate_scenario_type_column(df: pd.DataFrame) -> None:
    validate_allowed_values(df, "scenario_type", AVAILABLE_SCENARIOS)


def validate_industry_column(df: pd.DataFrame) -> None:
    series = df["industry_category"]

    if series.isnull().any():
        raise ValueError("Колонка industry_category содержит пустые значения.")

    if (series.astype(str).str.strip() == "").any():
        raise ValueError("Колонка industry_category содержит пустые строки.")


def validate_b2c_age_column(df: pd.DataFrame) -> None:
    age_series = df["b2c_age"]
    client_type_series = df["client_type"]

    converted_age = pd.to_numeric(age_series, errors="coerce")

    invalid_numeric_mask = age_series.notnull() & converted_age.isnull()
    if invalid_numeric_mask.any():
        raise ValueError("Колонка b2c_age должна содержать числовые значения или быть пустой для B2B.")

    b2c_missing_age_mask = (client_type_series == "B2C") & age_series.isnull()
    if b2c_missing_age_mask.any():
        raise ValueError("Для клиентов типа B2C колонка b2c_age должна быть заполнена.")

    non_null_age_mask = converted_age.notnull()
    if ((converted_age[non_null_age_mask] < 0) | (converted_age[non_null_age_mask] > 120)).any():
        raise ValueError("Колонка b2c_age должна содержать значения в диапазоне от 0 до 120.")


def validate_comm_time_column(df: pd.DataFrame) -> None:
    series = df["comm_time"]

    if series.isnull().any():
        raise ValueError("Колонка comm_time содержит пустые значения.")

    stripped_series = series.astype(str).str.strip()
    if (stripped_series == "").any():
        raise ValueError("Колонка comm_time содержит пустые строки.")

    parsed_series = pd.to_datetime(stripped_series, errors="coerce")
    if parsed_series.isnull().any():
        raise ValueError("Колонка comm_time должна содержать корректные значения времени или даты-времени.")
