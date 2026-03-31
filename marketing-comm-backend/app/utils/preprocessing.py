from app.constants.model import NUMERIC_COLUMNS, TRAIN_TARGET_COLUMN
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