from datetime import datetime
from typing import Callable

import pandas as pd

from app.core.paths import PREDICTIONS_DIR, UPLOADS_DIR
from app.ml.logistic.logistic import (
    train_logistic_regression_model,
    predict_logistic_probabilities
)
from app.ml.boosting.boosting import (
    train_boosting_model,
    predict_boosting_probabilities,
)
from app.ml.preprocessing import (
    prepare_training_dataframe,
    prepare_prediction_dataframe,
    prepare_boosting_dataframe,
)
from app.ml.validation import (
    validate_train_dataframe,
    validate_predict_dataframe,
)
from app.services.file_service import (
    validate_file_extension,
    save_uploaded_file,
    load_dataframe,
)
from app.services.metadata_service import (
    read_metadata,
    write_metadata,
)
from app.services.recommendation_service import (
    build_prediction_output_info,
    expand_clients_with_all_channel_scenario_pairs,
    apply_prediction_results,
    save_prediction_results_to_excel,
    build_prediction_response,
)


def load_uploaded_dataframe(upload_file) -> pd.DataFrame:
    validate_file_extension(upload_file.filename)

    file_path = UPLOADS_DIR / upload_file.filename
    save_uploaded_file(upload_file, file_path)

    return load_dataframe(file_path)


def save_model_metadata(model_key: str, model_metadata: dict) -> dict:
    metadata = read_metadata()
    metadata[model_key] = model_metadata
    write_metadata(metadata)
    return metadata[model_key]


def get_model_status(model_key: str) -> dict:
    metadata = read_metadata()
    return metadata[model_key]


def get_trained_model_metadata(model_key: str, error_message: str) -> dict:
    model_metadata = get_model_status(model_key)

    if not model_metadata.get("trained"):
        raise ValueError(error_message)

    return model_metadata


def train_logistic_model_from_file(upload_file) -> dict:
    df = load_uploaded_dataframe(upload_file)
    validate_train_dataframe(df)

    X, y = prepare_training_dataframe(df)
    training_result = train_logistic_regression_model(X, y)

    logistic_metadata = {
        "trained": True,
        "algorithm": training_result["algorithm"],
        "trainedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rowsCount": int(len(df)),
        "featuresCount": int(len(X.columns)),
        "threshold": training_result["threshold"],
        "classWeight": training_result["classWeight"],
        "useScaler": training_result["useScaler"],
        "learningRate": training_result["learningRate"],
        "maxIter": training_result["maxIter"],
        "l2Lambda": training_result["l2Lambda"],
        "tolerance": training_result["tolerance"],
        "mergeRareCategories": training_result["mergeRareCategories"],
        "metrics": training_result["metrics"],
    }

    return save_model_metadata("logistic", logistic_metadata)


def train_boosting_model_from_file(upload_file) -> dict:
    df = load_uploaded_dataframe(upload_file)
    validate_train_dataframe(df)

    X = prepare_boosting_dataframe(df)
    y = pd.to_numeric(df["target"], errors="coerce")

    training_result = train_boosting_model(X, y)

    boosting_metadata = {
        "trained": True,
        "algorithm": training_result["algorithm"],
        "trainedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rowsCount": int(len(df)),
        "featuresCount": int(len(X.columns)),
        "threshold": training_result["threshold"],
        "classWeights": training_result["classWeights"],
        "iterations": training_result.get("iterations"),
        "learningRate": training_result.get("learningRate"),
        "depth": training_result.get("depth"),
        "l2LeafReg": training_result["l2LeafReg"],
        "metrics": training_result["metrics"],
    }

    return save_model_metadata("boosting", boosting_metadata)


def get_logistic_model_status() -> dict:
    return get_model_status("logistic")


def get_boosting_model_status() -> dict:
    return get_model_status("boosting")


def predict_from_file(
    upload_file,
    model_key: str,
    model_name: str,
    prepare_features_func: Callable[[pd.DataFrame], pd.DataFrame],
    predict_probabilities_func: Callable[[pd.DataFrame], pd.Series],
    model_not_trained_message: str,
) -> dict:
    model_metadata = get_trained_model_metadata(model_key, model_not_trained_message)

    df = load_uploaded_dataframe(upload_file)
    validate_predict_dataframe(df)

    prediction_id, output_file_name, output_path = build_prediction_output_info(
        model_name=model_name,
        predictions_dir=PREDICTIONS_DIR,
    )

    result_df = expand_clients_with_all_channel_scenario_pairs(df)
    prediction_features_df = prepare_features_func(result_df)
    predicted_probabilities = predict_probabilities_func(prediction_features_df)

    all_predictions_df, top_recommendations_df = apply_prediction_results(
        result_df=result_df,
        predicted_probabilities=predicted_probabilities,
        threshold=model_metadata.get("threshold"),
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


def predict_logistic_from_file(upload_file) -> dict:
    return predict_from_file(
        upload_file=upload_file,
        model_key="logistic",
        model_name="logistic",
        prepare_features_func=prepare_prediction_dataframe,
        predict_probabilities_func=predict_logistic_probabilities,
        model_not_trained_message="Модель логистической регрессии еще не обучена.",
    )


def predict_boosting_from_file(upload_file) -> dict:
    return predict_from_file(
        upload_file=upload_file,
        model_key="boosting",
        model_name="boosting",
        prepare_features_func=prepare_boosting_dataframe,
        predict_probabilities_func=predict_boosting_probabilities,
        model_not_trained_message="Модель бустинга еще не обучена.",
    )
