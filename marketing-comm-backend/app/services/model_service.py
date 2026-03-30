import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from app.constants.model import (
    AVAILABLE_CHANNELS,
    AVAILABLE_SCENARIOS
)
from app.services.ml import (
    train_logistic_regression_model,
    predict_probabilities,
)
from app.utils.validation import (
    validate_train_dataframe,
    validate_predict_dataframe,
)
from app.utils.preprocessing import (
    prepare_training_dataframe,
    prepare_prediction_dataframe,
)

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
METADATA_PATH = STORAGE_DIR / "metadata.json"
UPLOADS_DIR = STORAGE_DIR / "uploads"
PREDICTIONS_DIR = STORAGE_DIR / "predictions"


def read_metadata() -> dict:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_metadata(data: dict) -> None:
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def train_model_from_file(upload_file) -> dict:
    validate_file_extension(upload_file.filename)

    file_path = UPLOADS_DIR / upload_file.filename
    save_uploaded_file(upload_file, file_path)

    df = load_dataframe(file_path)
    validate_train_dataframe(df)

    X, y = prepare_training_dataframe(df)
    training_result = train_logistic_regression_model(X, y)

    metadata = {
        "trained": True,
        "algorithm": training_result["algorithm"],
        "trainedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rowsCount": int(len(df)),
        "featuresCount": int(len(X.columns)),
        "metrics": training_result["metrics"]
    }

    write_metadata(metadata)
    return metadata


def get_model_status() -> dict:
    return read_metadata()


def predict_from_file(upload_file) -> dict:
    metadata = read_metadata()
    if not metadata.get("trained"):
        raise ValueError("Модель еще не обучена.")

    validate_file_extension(upload_file.filename)

    file_path = UPLOADS_DIR / upload_file.filename
    save_uploaded_file(upload_file, file_path)

    df = load_dataframe(file_path)
    validate_predict_dataframe(df)

    prediction_id = str(uuid4())
    output_file_name = f"prediction_results_{prediction_id}.xlsx"
    output_path = PREDICTIONS_DIR / output_file_name

    base_df = df.copy()
    base_df = base_df.reset_index(drop=True)
    base_df["client_row_id"] = base_df.index + 1

    expanded_rows = []

    for _, row in base_df.iterrows():
        row_dict = row.to_dict()

        for channel in AVAILABLE_CHANNELS:
            for scenario in AVAILABLE_SCENARIOS:
                new_row = row_dict.copy()
                new_row["channel_type"] = channel
                new_row["scenario_type"] = scenario
                expanded_rows.append(new_row)

    result_df = pd.DataFrame(expanded_rows)

    prediction_features_df = prepare_prediction_dataframe(result_df)

    result_df["predicted_probability"] = predict_probabilities(prediction_features_df)
    result_df["predicted_class"] = (result_df["predicted_probability"] >= 0.5).astype(int)

    result_df = result_df.sort_values(
        by=["client_row_id", "predicted_probability"],
        ascending=[True, False]
    ).reset_index(drop=True)

    result_df["rank_within_client"] = (
        result_df.groupby("client_row_id").cumcount() + 1
    )

    top_recommendations_df = result_df[result_df["rank_within_client"] == 1].copy()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="all_predictions")
        top_recommendations_df.to_excel(writer, index=False, sheet_name="top_recommendations")

    return {
        "predictionId": prediction_id,
        "fileName": output_file_name,
        "rowsProcessed": int(len(df)),
        "predictionsGenerated": int(len(result_df)),
        "downloadUrl": f"/api/model/download-prediction/{prediction_id}"
    }


def get_prediction_file_path(prediction_id: str) -> Path:
    matching_files = list(PREDICTIONS_DIR.glob(f"prediction_results_{prediction_id}.xlsx"))
    if not matching_files:
        raise FileNotFoundError("Файл результата не найден.")

    return matching_files[0]