from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.constants.model import (
    AVAILABLE_CHANNELS,
    AVAILABLE_SCENARIOS,
)


def build_prediction_output_info(
    model_name: str,
    predictions_dir: Path,
) -> tuple[str, str, Path]:
    prediction_id = str(uuid4())
    output_file_name = f"prediction_results_{model_name}_{prediction_id}.xlsx"
    output_path = predictions_dir / output_file_name

    return prediction_id, output_file_name, output_path


def expand_clients_with_all_channel_scenario_pairs(df: pd.DataFrame) -> pd.DataFrame:
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

    return pd.DataFrame(expanded_rows)


def apply_prediction_results(
    result_df: pd.DataFrame,
    predicted_probabilities: pd.Series,
    threshold: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_threshold = threshold if threshold is not None else 0.5

    result_df = result_df.copy()
    result_df["predicted_probability"] = predicted_probabilities
    result_df["predicted_class"] = (
        result_df["predicted_probability"] >= final_threshold
    ).astype(int)

    result_df = result_df.sort_values(
        by=["client_row_id", "predicted_probability"],
        ascending=[True, False],
    ).reset_index(drop=True)

    result_df["rank_within_client"] = (
        result_df.groupby("client_row_id").cumcount() + 1
    )

    top_recommendations_df = result_df[
        result_df["rank_within_client"] == 1
    ].copy()

    return result_df, top_recommendations_df


def save_prediction_results_to_excel(
    all_predictions_df: pd.DataFrame,
    top_recommendations_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        all_predictions_df.to_excel(
            writer,
            index=False,
            sheet_name="all_predictions",
        )
        top_recommendations_df.to_excel(
            writer,
            index=False,
            sheet_name="top_recommendations",
        )


def build_prediction_response(
    prediction_id: str,
    file_name: str,
    rows_processed: int,
    predictions_generated: int,
) -> dict:
    return {
        "predictionId": prediction_id,
        "fileName": file_name,
        "rowsProcessed": rows_processed,
        "predictionsGenerated": predictions_generated,
        "downloadUrl": f"/api/model/download-prediction/{prediction_id}",
    }
