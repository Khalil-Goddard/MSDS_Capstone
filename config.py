from pathlib import Path

APP_TITLE = "Custom Survey Builder"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MASTER_DATA_PATH = DATA_DIR / "combined_survey_data.csv"
REFERENCE_CONTEXTS_PATH = DATA_DIR / "reference_survey_contexts.csv"

SURVEY_ID_COLUMN = "survey_definition_id"
STOPLIGHT_ID_COLUMN = "survey_stoplight_id"
ORDER_COLUMN = "survey_stoplight_order_number"

REQUIRED_UPLOAD_COLUMNS = {
    SURVEY_ID_COLUMN,
    STOPLIGHT_ID_COLUMN,
    "survey_stoplight_question_text",
    "survey_stoplight_description",
    "red_description",
    "yellow_description",
    "green_description",
}

EDITABLE_ADAPTED_COLUMNS = [
    "adapted_question_text",
    "adapted_red_description",
    "adapted_yellow_description",
    "adapted_green_description",
]

MODEL_DEFAULTS = {
    "top_k_similar": 5,
    "max_modify_fraction": 0.15,
    "max_additions": 5,
    "min_similar_context_score": 0.30,
    "min_question_similarity": 0.30,
    "min_add_score": 0.20,
}
