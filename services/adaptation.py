from __future__ import annotations

from typing import Any

import pandas as pd

from config import ORDER_COLUMN, STOPLIGHT_ID_COLUMN, SURVEY_ID_COLUMN
from survey_adaptation_engine import ContextInput, SurveyAdaptationEngine


def run_adaptation(
    engine: SurveyAdaptationEngine,
    base_survey_id: int,
    country: str,
    language: str,
    population_types: list[str],
    focuses: list[str],
    organization_type: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    context = ContextInput(
        country=country,
        language=language,
        population_type="; ".join(population_types),
        focus="; ".join(focuses),
        organization_type=organization_type,
    )
    return engine.adapt_survey(
        base_survey_id=base_survey_id,
        context=context,
        **parameters,
    )


def merge_adaptation_with_upload(
    uploaded: pd.DataFrame,
    adapted_records: list[dict[str, Any]],
    base_survey_id: int,
) -> pd.DataFrame:
    """Preserve uploaded columns, attach recommendations, and append Add rows."""
    recommendations = pd.DataFrame(adapted_records)
    existing = recommendations[recommendations["action"] != "Add"].copy()
    additions = recommendations[recommendations["action"] == "Add"].copy()

    uploaded_copy = uploaded.copy()
    uploaded_copy[STOPLIGHT_ID_COLUMN] = uploaded_copy[STOPLIGHT_ID_COLUMN].astype(str)
    existing["survey_stoplight_id"] = existing["survey_stoplight_id"].astype(str)

    model_columns = [column for column in recommendations.columns if column not in {
        "survey_stoplight_id",
        "base_survey_definition_id",
    }]
    merge_columns = ["survey_stoplight_id", *model_columns]

    merged = uploaded_copy.merge(
        existing[merge_columns],
        how="left",
        left_on=STOPLIGHT_ID_COLUMN,
        right_on="survey_stoplight_id",
        suffixes=("", "_model"),
    )
    if "survey_stoplight_id_model" in merged.columns:
        merged.drop(columns=["survey_stoplight_id_model"], inplace=True)

    # Fallback for uploads whose stoplight IDs were read with numeric formatting differences.
    if "action" in merged.columns and merged["action"].isna().any():
        engine_lookup = {
            str(record.get("indicator_key", "")): record
            for record in adapted_records
            if record.get("action") != "Add"
        }
        from survey_adaptation_engine import SurveyAdaptationEngine
        for idx in merged.index[merged["action"].isna()]:
            key = SurveyAdaptationEngine.indicator_key(merged.loc[idx].to_dict())
            record = engine_lookup.get(key)
            if record:
                for column in model_columns:
                    merged.at[idx, column] = record.get(column, "")

    if not additions.empty:
        addition_rows: list[dict[str, Any]] = []
        for _, recommendation in additions.iterrows():
            row = {column: "" for column in uploaded.columns}
            row[SURVEY_ID_COLUMN] = base_survey_id
            if ORDER_COLUMN in row:
                row[ORDER_COLUMN] = recommendation.get("order_number", "")
            mappings = {
                "survey_stoplight_question_text": "adapted_question_text",
                "survey_stoplight_description": "adapted_description",
                "red_description": "adapted_red_description",
                "yellow_description": "adapted_yellow_description",
                "green_description": "adapted_green_description",
                "survey_stoplight_dimension": "dimension",
                "survey_stoplight_short_name": "short_name",
                "survey_stoplight_code_name": "code_name",
                "survey_stoplight_survey_indicator_id": "survey_indicator_id",
            }
            for upload_column, model_column in mappings.items():
                if upload_column in row:
                    row[upload_column] = recommendation.get(model_column, "")
            for column in model_columns:
                row[column] = recommendation.get(column, "")
            addition_rows.append(row)
        merged = pd.concat([merged, pd.DataFrame(addition_rows)], ignore_index=True, sort=False)

    sort_column = "order_number" if "order_number" in merged.columns else ORDER_COLUMN
    if sort_column in merged.columns:
        merged["_sort_order"] = pd.to_numeric(merged[sort_column], errors="coerce")
        merged.sort_values("_sort_order", inplace=True, na_position="last")
        merged.drop(columns="_sort_order", inplace=True)
        merged.reset_index(drop=True, inplace=True)

    return merged
