from app.constants.model import NUMERIC_COLUMNS, TRAIN_TARGET_COLUMN
from app.services.ml.config import CATEGORICAL_FEATURES, CATBOOST_FEATURES

import numpy as np
import pandas as pd


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in result_df.columns:
            result_df[column] = pd.to_numeric(result_df[column], errors="coerce")

    if TRAIN_TARGET_COLUMN in result_df.columns:
        result_df[TRAIN_TARGET_COLUMN] = pd.to_numeric(
            result_df[TRAIN_TARGET_COLUMN],
            errors="coerce",
        )

    return result_df


def extract_comm_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()

    comm_time_parsed = pd.to_datetime(result_df["comm_time"], errors="coerce")

    seconds_from_midnight = (
        comm_time_parsed.dt.hour * 3600
        + comm_time_parsed.dt.minute * 60
        + comm_time_parsed.dt.second
    )

    seconds_in_day = 24 * 60 * 60

    angle = 2 * np.pi * seconds_from_midnight / seconds_in_day

    result_df["comm_time_sin"] = np.sin(angle)
    result_df["comm_time_cos"] = np.cos(angle)

    return result_df


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()
    result_df = convert_numeric_columns(result_df)
    result_df = extract_comm_time_features(result_df)
    result_df = add_new_features(result_df)
    result_df = add_polynomial_features(result_df)

    if "comm_time" in result_df.columns:
        result_df = result_df.drop(columns=["comm_time"])

    return result_df


def prepare_training_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    result_df = prepare_base_dataframe(df)

    X = result_df.drop(columns=[TRAIN_TARGET_COLUMN])
    y = result_df[TRAIN_TARGET_COLUMN]

    return X, y


def prepare_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result_df = prepare_base_dataframe(df)
    return result_df


def prepare_catboost_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result_df = prepare_base_dataframe(df)

    for column in CATEGORICAL_FEATURES:
        if column in result_df.columns:
            result_df[column] = result_df[column].fillna("").astype(str)

    result_df = result_df[CATBOOST_FEATURES]

    return result_df


import numpy as np
import pandas as pd


def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()

    safe_total_orders = result_df["total_orders"].fillna(0)
    safe_tenure_days = result_df["tenure_days"].fillna(0)
    safe_recency_days = result_df["recency_days"].fillna(0)
    safe_prev_comm_count_channel = result_df["prev_comm_count_channel"].fillna(0)
    safe_last_comm_days_channel = result_df["last_comm_days_channel"].fillna(0)

    comm_time = pd.to_datetime(result_df["comm_time"], errors="coerce")

    result_df["orders_per_30_days"] = (
        safe_total_orders / safe_tenure_days.clip(lower=1)
    ) * 30

    result_df["recency_to_tenure_ratio"] = (
        safe_recency_days / (safe_tenure_days + 1)
    )

    result_df["channel_fatigue_ratio"] = (
        safe_prev_comm_count_channel / (safe_last_comm_days_channel + 1)
    )

    # result_df["comm_weekday"] = comm_time.dt.weekday
    # result_df["is_weekend"] = (result_df["comm_weekday"] >= 5).astype(int)

    # result_df["avg_order_value_squared"] = (
    #     result_df["avg_order_value"].fillna(0) ** 2
    # )

    # result_df["random_noise_feature"] = np.random.normal(0, 1, len(result_df))

    return result_df


def add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()

    base_features = [
        "recency_days",
        "order_frequency",
        "total_orders",
        "last_comm_days_channel",
        "promo_sensitivity",
    ]

    for col in base_features:
        safe_col = result_df[col].fillna(0)

        normalized = (safe_col - safe_col.mean()) / (safe_col.std() + 1e-6)

        result_df[f"{col}_squared"] = normalized ** 2
        result_df[f"{col}_cubed"] = normalized ** 3

    return result_df
