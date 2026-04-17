from datetime import datetime
from pathlib import Path

import pandas as pd

from app.services.ml import (
    train_logistic_regression_model,
    predict_logistic_probabilities,
    train_catboost_model,
    predict_catboost_probabilities,
)
from app.services.prediction_service import (
    build_prediction_output_info,
    expand_clients_with_all_channel_scenario_pairs,
    apply_prediction_results,
    save_prediction_results_to_excel,
    build_prediction_response,
)
from app.utils.validation import (
    validate_train_dataframe,
    validate_predict_dataframe,
)
from app.services.ml.preprocessing import (
    prepare_training_dataframe,
    prepare_prediction_dataframe,
    prepare_catboost_dataframe,
)


BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
METADATA_PATH = STORAGE_DIR / "metadata.json"
UPLOADS_DIR = STORAGE_DIR / "uploads"
PREDICTIONS_DIR = STORAGE_DIR / "predictions"


def train_logistic_model_from_file(upload_file) -> dict:
    validate_file_extension(upload_file.filename)

    file_path = UPLOADS_DIR / upload_file.filename
    save_uploaded_file(upload_file, file_path)

    df = load_dataframe(file_path)
    validate_train_dataframe(df)

    X, y = prepare_training_dataframe(df)
    training_result = train_logistic_regression_model(X, y)

    metadata = read_metadata()
    metadata["logistic"] = {
        "trained": True,
        "algorithm": training_result["algorithm"],
        "trainedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rowsCount": int(len(df)),
        "featuresCount": int(len(X.columns)),
        "threshold": training_result["threshold"],
        "classWeight": training_result["classWeight"],
        "useScaler": training_result["useScaler"],
        "metrics": training_result["metrics"],
    }

    write_metadata(metadata)
    return metadata["logistic"]


def train_catboost_model_from_file(upload_file) -> dict:
    validate_file_extension(upload_file.filename)

    file_path = UPLOADS_DIR / upload_file.filename
    save_uploaded_file(upload_file, file_path)

    df = load_dataframe(file_path)
    validate_train_dataframe(df)

    catboost_df = prepare_catboost_dataframe(df)
    X_train = catboost_df
    y_train = pd.to_numeric(df["target"], errors="coerce")

    from sklearn.model_selection import train_test_split

    X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    training_result = train_catboost_model(
        X_train_part,
        X_test_part,
        y_train_part,
        y_test_part,
    )

    metadata = read_metadata()
    metadata["catboost"] = {
        "trained": True,
        "algorithm": training_result["algorithm"],
        "trainedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rowsCount": int(len(df)),
        "featuresCount": int(len(X_train.columns)),
        "threshold": training_result["threshold"],
        "classWeights": training_result["classWeights"],
        "iterations": training_result.get("iterations"),
        "learningRate": training_result.get("learningRate"),
        "depth": training_result.get("depth"),
        "l2LeafReg": training_result["l2LeafReg"],
        "metrics": training_result["metrics"],
    }

    write_metadata(metadata)
    return metadata["catboost"]


def get_logistic_model_status() -> dict:
    metadata = read_metadata()
    return metadata["logistic"]


def get_catboost_model_status() -> dict:
    metadata = read_metadata()
    return metadata["catboost"]


def predict_logistic_from_file(upload_file) -> dict:
    metadata = read_metadata()
    logistic_metadata = metadata["logistic"]

    if not logistic_metadata.get("trained"):
        raise ValueError("Модель логистической регрессии еще не обучена.")

    validate_file_extension(upload_file.filename)

    file_path = UPLOADS_DIR / upload_file.filename
    save_uploaded_file(upload_file, file_path)

    df = load_dataframe(file_path)
    validate_predict_dataframe(df)

    prediction_id, output_file_name, output_path = build_prediction_output_info(
        model_name="logistic",
        predictions_dir=PREDICTIONS_DIR,
    )

    result_df = expand_clients_with_all_channel_scenario_pairs(df)
    prediction_features_df = prepare_prediction_dataframe(result_df)

    predicted_probabilities = predict_logistic_probabilities(prediction_features_df)

    all_predictions_df, top_recommendations_df = apply_prediction_results(
        result_df=result_df,
        predicted_probabilities=predicted_probabilities,
        threshold=logistic_metadata.get("threshold"),
    )

    save_prediction_results_to_excel(
        all_predictions_df=all_predictions_df,
        top_recommendations_df=top_recommendations_df,
        output_path=output_path,
    )

    return build_prediction_response(
        prediction_id=prediction_id,
        file_name=output_file_name,
        rows_processed=int(len(df)),
        predictions_generated=int(len(all_predictions_df)),
    )


def predict_catboost_from_file(upload_file) -> dict:
    metadata = read_metadata()
    catboost_metadata = metadata["catboost"]

    if not catboost_metadata.get("trained"):
        raise ValueError("Модель CatBoost еще не обучена.")

    validate_file_extension(upload_file.filename)

    file_path = UPLOADS_DIR / upload_file.filename
    save_uploaded_file(upload_file, file_path)

    df = load_dataframe(file_path)
    validate_predict_dataframe(df)

    prediction_id, output_file_name, output_path = build_prediction_output_info(
        model_name="catboost",
        predictions_dir=PREDICTIONS_DIR,
    )

    result_df = expand_clients_with_all_channel_scenario_pairs(df)
    prediction_features_df = prepare_catboost_dataframe(result_df)

    predicted_probabilities = predict_catboost_probabilities(prediction_features_df)

    all_predictions_df, top_recommendations_df = apply_prediction_results(
        result_df=result_df,
        predicted_probabilities=predicted_probabilities,
        threshold=catboost_metadata.get("threshold"),
    )

    save_prediction_results_to_excel(
        all_predictions_df=all_predictions_df,
        top_recommendations_df=top_recommendations_df,
        output_path=output_path,
    )

    return build_prediction_response(
        prediction_id=prediction_id,
        file_name=output_file_name,
        rows_processed=int(len(df)),
        predictions_generated=int(len(all_predictions_df)),
    )
