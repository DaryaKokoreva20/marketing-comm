from app.constants.model import (
    REQUIRED_TRAIN_COLUMNS,
    REQUIRED_PREDICT_COLUMNS,
)


def validate_train_dataframe(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_TRAIN_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"В обучающем файле отсутствуют обязательные колонки: {', '.join(sorted(missing_columns))}"
        )

    if df.empty:
        raise ValueError("Обучающий файл пуст.")

    if not df["target"].isin([0, 1]).all():
        raise ValueError("Колонка target должна содержать только 0 и 1.")


def validate_predict_dataframe(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_PREDICT_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"В файле для прогнозирования отсутствуют обязательные колонки: {', '.join(sorted(missing_columns))}"
        )

    if df.empty:
        raise ValueError("Файл для прогнозирования пуст.")