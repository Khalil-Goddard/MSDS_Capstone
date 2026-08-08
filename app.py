from __future__ import annotations
import hashlib
import pandas as pd
import streamlit as st
from config import (
    APP_TITLE,
    EDITABLE_ADAPTED_COLUMNS,
    MASTER_DATA_PATH,
    MODEL_DEFAULTS,
    REFERENCE_CONTEXTS_PATH,
)
from services.adaptation import merge_adaptation_with_upload, run_adaptation
from services.contexts import distinct_context_values, load_reference_contexts
from services.files import (
    dataframe_to_csv,
    dataframe_to_excel,
    load_uploaded_survey,
    validate_uploaded_survey,
)
from survey_adaptation_engine import SurveyAdaptationEngine

st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")

# Load and cache survey adaptation engine
@st.cache_resource(show_spinner="Loading survey adaptation model...")
def load_engine() -> SurveyAdaptationEngine:
    return SurveyAdaptationEngine(MASTER_DATA_PATH, REFERENCE_CONTEXTS_PATH)

# Load and cache the reference context
@st.cache_data
def load_context_data() -> pd.DataFrame:
    return load_reference_contexts(REFERENCE_CONTEXTS_PATH)

# create uniqe file signature of the uploaded survey
def file_signature(uploaded_file) -> str:
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    uploaded_file.seek(0)
    return digest

# Create progress visual in streamlit UI 
def render_progress(active_step: int) -> None:
    labels = ["Upload", "Configure", "Adapt", "Review", "Download"]
    columns = st.columns(len(labels))
    for index, (column, label) in enumerate(zip(columns, labels), start=1):
        icon = "✅" if index < active_step else ("🔵" if index == active_step else "⚪")
        column.markdown(f"**{icon} {index}. {label}**")


for key, default in {
    "adapted_survey": None,
    "adaptation_result": None,
    "upload_signature": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title(APP_TITLE)
st.write(
    "Upload an existing survey, configure the target context, generate model-supported "
    "Keep/Modify/Add recommendations, edit the adapted wording, and download the final survey."
)
render_progress(1 if st.session_state.adapted_survey is None else 4)

st.header("1. Upload an Existing Survey")
uploaded_file = st.file_uploader(
    "Drag and drop an Excel or CSV survey file",
    type=["csv", "xlsx", "xls"],
    help="The file must contain exactly one survey_definition_id.",
)
if uploaded_file is None:
    st.info("Upload a survey file to begin.")
    st.stop()

try:
    signature = file_signature(uploaded_file)
    survey = load_uploaded_survey(uploaded_file)
except Exception as error:
    st.error(f"The survey could not be opened: {error}")
    st.stop()

errors, base_survey_id = validate_uploaded_survey(survey)
if errors:
    for error in errors:
        st.error(error)
    st.stop()
assert base_survey_id is not None

if st.session_state.upload_signature != signature:
    st.session_state.upload_signature = signature
    st.session_state.adapted_survey = None
    st.session_state.adaptation_result = None

try:
    engine = load_engine()
    contexts = load_context_data()
except Exception as error:
    st.error(f"The adaptation engine could not be loaded: {error}")
    st.stop()

if base_survey_id not in engine.surveys:
    st.error(f"Survey ID {base_survey_id} is not present in the master survey dataset.")
    st.stop()
if base_survey_id not in engine.reference_surveys:
    st.error(f"Survey ID {base_survey_id} is not present in reference_survey_contexts.csv.")
    st.stop()

st.success("Survey uploaded and validated successfully.")
summary_columns = st.columns(4)
summary_columns[0].metric("Survey ID", str(base_survey_id))
summary_columns[1].metric("Uploaded Rows", f"{len(survey):,}")
summary_columns[2].metric("Columns", f"{len(survey.columns):,}")
summary_columns[3].metric("Reference Surveys", f"{len(engine.reference_surveys):,}")
with st.expander("Preview uploaded survey"):
    st.dataframe(survey, use_container_width=True, hide_index=True)

st.header("2. Configure the Target Context")
country_options = distinct_context_values(contexts, "country")
language_options = distinct_context_values(contexts, "language")
population_options = distinct_context_values(contexts, "population_type")
focus_options = distinct_context_values(contexts, "focus")
organization_options = distinct_context_values(contexts, "organization_type")

with st.form("adaptation_form"):
    left, middle = st.columns(2)
    with left:
        country = st.selectbox("Target country", country_options, index=None, placeholder="Select a country")
        language = st.selectbox("Target language", language_options, index=None, placeholder="Select a language")
        population_types = st.multiselect("Population type", population_options)
    with middle:
        focuses = st.multiselect("Focus", focus_options)
        organization_type = st.selectbox(
            "Organization type", organization_options, index=None, placeholder="Select an organization type"
        )

    with st.expander("Advanced settings"):
        top_k_similar = st.number_input("Similar surveys to use", min_value=1, max_value=20, value=MODEL_DEFAULTS["top_k_similar"])
        max_modify_fraction = st.slider("Maximum modification fraction", 0.0, 1.0, MODEL_DEFAULTS["max_modify_fraction"], 0.05)
        max_additions = st.number_input("Maximum additions", min_value=0, max_value=50, value=MODEL_DEFAULTS["max_additions"])
        min_similar_context_score = st.slider("Minimum similar-context score", 0.0, 1.0, MODEL_DEFAULTS["min_similar_context_score"], 0.05)
        min_question_similarity = st.slider("Minimum question similarity", 0.0, 1.0, MODEL_DEFAULTS["min_question_similarity"], 0.05)
        min_add_score = st.slider("Minimum addition score", 0.0, 1.0, MODEL_DEFAULTS["min_add_score"], 0.05)

    submitted = st.form_submit_button("Generate Adapted Survey", type="primary", use_container_width=True)

if submitted:
    missing_inputs = []
    if not country:
        missing_inputs.append("country")
    if not language:
        missing_inputs.append("language")
    if not population_types:
        missing_inputs.append("population type")
    if not focuses:
        missing_inputs.append("focus")
    if not organization_type:
        missing_inputs.append("organization type")

    if missing_inputs:
        st.warning("Complete these inputs: " + ", ".join(missing_inputs))
    else:
        parameters = {
            "top_k_similar": int(top_k_similar),
            "max_modify_fraction": float(max_modify_fraction),
            "max_additions": int(max_additions),
            "min_similar_context_score": float(min_similar_context_score),
            "min_question_similarity": float(min_question_similarity),
            "min_add_score": float(min_add_score),
        }
        try:
            with st.spinner("Comparing surveys and generating recommendations..."):
                result = run_adaptation(
                    engine=engine,
                    base_survey_id=base_survey_id,
                    country=country,
                    language=language,
                    population_types=population_types,
                    focuses=focuses,
                    organization_type=organization_type,
                    parameters=parameters,
                )
                merged = merge_adaptation_with_upload(
                    uploaded=survey,
                    adapted_records=result["adapted_questions"],
                    base_survey_id=base_survey_id,
                )
                st.session_state.adaptation_result = result
                st.session_state.adapted_survey = merged
        except Exception as error:
            st.error(f"The model could not adapt the survey: {error}")

if st.session_state.adapted_survey is not None:
    result = st.session_state.adaptation_result
    st.header("3. Adaptation Summary")
    summary = result["summary"]
    cards = st.columns(4)
    cards[0].metric("Kept", summary["Keep"])
    cards[1].metric("Modified", summary["Modify"])
    cards[2].metric("Added", summary["Add"])
    cards[3].metric("Final Questions", summary["final_question_count"])

    with st.expander("Similar surveys used"):
        st.dataframe(pd.DataFrame(result["similar_surveys"]), use_container_width=True, hide_index=True)

    st.header("4. Review and Edit")
    st.caption(
        "All original uploaded columns are preserved. Only the four adapted wording fields below are editable. "
        "Add recommendations are included automatically."
    )
    disabled_columns = [
        column for column in st.session_state.adapted_survey.columns
        if column not in EDITABLE_ADAPTED_COLUMNS
    ]
    column_config = {
        column: st.column_config.TextColumn(column.replace("_", " ").title(), width="large")
        for column in EDITABLE_ADAPTED_COLUMNS
        if column in st.session_state.adapted_survey.columns
    }
    edited = st.data_editor(
        st.session_state.adapted_survey,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=disabled_columns,
        column_config=column_config,
        key="adapted_survey_editor",
    )
    st.session_state.adapted_survey = edited

    st.header("5. Download")
    csv_data = dataframe_to_csv(edited)
    excel_data = dataframe_to_excel(edited)
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download CSV",
        data=csv_data,
        file_name=f"adapted_survey_{base_survey_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_columns[1].download_button(
        "Download Excel",
        data=excel_data,
        file_name=f"adapted_survey_{base_survey_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
