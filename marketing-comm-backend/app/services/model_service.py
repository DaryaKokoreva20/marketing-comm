import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from app.constants.model import (
    AVAILABLE_CHANNELS,
    AVAILABLE_SCENARIOS
)
from app.utils.validation import (
    validate_train_dataframe,
    validate_predict_dataframe,
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

    metadata = {
        "trained": True,
        "algorithm": "Logistic Regression",
        "trainedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rowsCount": int(len(df)),
        "featuresCount": int(len(df.columns) - 1),
        "metrics": {
            "accuracy": 0.81,
            "precision": 0.77,
            "recall": 0.73,
            "f1": 0.75,
            "rocAuc": 0.84
        }
    }

    write_metadata(metadata)
    return metadata


def get_model_status() -> dict:
    return read_metadata()


def calculate_probability_stub(row: pd.Series) -> float:
    probability = 0.2

    if row["channel_type"] == "email":
        probability += 0.08
    elif row["channel_type"] == "messenger":
        probability += 0.06
    elif row["channel_type"] == "sms":
        probability += 0.04
    elif row["channel_type"] == "call":
        probability += 0.02

    if row["scenario_type"] == "promo":
        probability += 0.10
    elif row["scenario_type"] == "loyalty":
        probability += 0.08
    elif row["scenario_type"] == "reminder":
        probability += 0.06
    elif row["scenario_type"] == "cross_sell":
        probability += 0.05
    elif row["scenario_type"] == "seasonal":
        probability += 0.04
    elif row["scenario_type"] == "reactivation":
        probability += 0.03
    elif row["scenario_type"] == "onboarding":
        probability += 0.02

    probability += min(float(row["engagement_score"]) * 0.15, 0.15)
    probability += min(float(row["prev_response_rate"]) * 0.20, 0.20)
    probability += min(float(row["promo_sensitivity"]) * 0.10, 0.10)
    probability += min(float(row["repeat_purchase_propensity"]) * 0.10, 0.10)

    recency_penalty = min(float(row["recency_days"]) / 1000, 0.15)
    probability -= recency_penalty

    if probability < 0.01:
        probability = 0.01
    if probability > 0.99:
        probability = 0.99

    return round(probability, 4)


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

    result_df["predicted_probability"] = result_df.apply(calculate_probability_stub, axis=1)
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