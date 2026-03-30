import pandas as pd

from app.constants.model import (
    AVAILABLE_CHANNELS,
    AVAILABLE_SCENARIOS,
    REQUIRED_TRAIN_COLUMNS,
    REQUIRED_PREDICT_COLUMNS,
)


def validate_train_dataframe(df: pd.DataFrame) -> None:
    validate_required_columns(df, REQUIRED_TRAIN_COLUMNS, "обучающем")
    validate_not_empty(df, "Обучающий файл пуст.")

    validate_client_type_column(df)
    validate_channel_type_column(df)
    validate_scenario_type_column(df)
    validate_industry_column(df)

    validate_non_negative_numeric_column(df, "tenure_days")
    validate_non_negative_numeric_column(df, "avg_order_value")
    validate_non_negative_numeric_column(df, "order_frequency")
    validate_non_negative_numeric_column(df, "total_orders")
    validate_non_negative_numeric_column(df, "recency_days")
    validate_non_negative_numeric_column(df, "prev_comm_count_channel")
    validate_non_negative_numeric_column(df, "last_comm_days_channel")
    validate_non_negative_numeric_column(df, "prev_comm_count_scenario")

    validate_ratio_column(df, "repeat_purchase_propensity")
    validate_ratio_column(df, "prev_response_rate")
    validate_ratio_column(df, "prev_comm_response_rate_channel")
    validate_ratio_column(df, "prev_comm_response_rate_scenario")
    validate_ratio_column(df, "promo_sensitivity")

    validate_binary_column(df, "time_trigger")
    validate_binary_column(df, "target")

    validate_b2c_age_column(df)
    validate_comm_time_column(df)


def validate_predict_dataframe(df: pd.DataFrame) -> None:
    validate_required_columns(df, REQUIRED_PREDICT_COLUMNS, "файле для прогнозирования")
    validate_not_empty(df, "Файл для прогнозирования пуст.")

    validate_client_type_column(df)
    validate_industry_column(df)

    validate_non_negative_numeric_column(df, "tenure_days")
    validate_non_negative_numeric_column(df, "avg_order_value")
    validate_non_negative_numeric_column(df, "order_frequency")
    validate_non_negative_numeric_column(df, "total_orders")
    validate_non_negative_numeric_column(df, "recency_days")
    validate_non_negative_numeric_column(df, "prev_comm_count_channel")
    validate_non_negative_numeric_column(df, "last_comm_days_channel")
    validate_non_negative_numeric_column(df, "prev_comm_count_scenario")

    validate_ratio_column(df, "repeat_purchase_propensity")
    validate_ratio_column(df, "prev_response_rate")
    validate_ratio_column(df, "prev_comm_response_rate_channel")
    validate_ratio_column(df, "prev_comm_response_rate_scenario")
    validate_ratio_column(df, "promo_sensitivity")

    validate_binary_column(df, "time_trigger")

    validate_b2c_age_column(df)
    validate_comm_time_column(df)


def validate_required_columns(df: pd.DataFrame, required_columns: set, file_label: str) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"В {file_label} отсутствуют обязательные колонки: {', '.join(sorted(missing_columns))}"
        )


def validate_not_empty(df: pd.DataFrame, error_message: str) -> None:
    if df.empty:
        raise ValueError(error_message)


def validate_numeric_column(df: pd.DataFrame, column_name: str, allow_null: bool = False) -> pd.Series:
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


def validate_ratio_column(df: pd.DataFrame, column_name: str) -> None:
    converted = validate_numeric_column(df, column_name)

    if ((converted < 0) | (converted > 1)).any():
        raise ValueError(f"Колонка {column_name} должна содержать значения в диапазоне от 0 до 1.")


def validate_binary_column(df: pd.DataFrame, column_name: str) -> None:
    converted = validate_numeric_column(df, column_name)

    if not converted.isin([0, 1]).all():
        raise ValueError(f"Колонка {column_name} должна содержать только значения 0 и 1.")


def validate_allowed_values(df: pd.DataFrame, column_name: str, allowed_values: list[str]) -> None:
    series = df[column_name]

    if series.isnull().any():
        raise ValueError(f"Колонка {column_name} содержит пустые значения.")

    invalid_values = set(series.astype(str)) - set(allowed_values)
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

    if (series.astype(str).str.strip() == "").any():
        raise ValueError("Колонка comm_time содержит пустые строки.")
