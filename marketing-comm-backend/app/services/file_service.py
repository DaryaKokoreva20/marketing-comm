import pandas as pd
from pathlib import Path

from app.core.paths import PREDICTIONS_DIR


def validate_file_extension(filename: str) -> None:
    if not filename:
        raise ValueError("Имя файла отсутствует.")

    allowed_extensions = (".csv", ".xlsx")
    if not filename.lower().endswith(allowed_extensions):
        raise ValueError("Допустимы только файлы .csv и .xlsx.")
    

def load_dataframe(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    if file_path.suffix.lower() == ".xlsx":
        return pd.read_excel(file_path)

    raise ValueError("Неподдерживаемый формат файла.")


def save_uploaded_file(upload_file, destination: Path) -> None:
    with open(destination, "wb") as f:
        f.write(upload_file.file.read())


def get_prediction_file_path(prediction_id: str) -> Path:
    matching_files = list(PREDICTIONS_DIR.glob(f"*{prediction_id}.xlsx"))

    if not matching_files:
        raise FileNotFoundError("Файл результата не найден.")

    return matching_files[0]
