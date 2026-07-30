from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Protocol

import pandas as pd
import streamlit as st


# =========================================================
# Application configuration
# =========================================================

st.set_page_config(
    page_title="Custom Survey Builder",
    page_icon="📝",
    layout="wide",
)

DESCRIPTION_COLUMN = "survey_stoplight_description"
COUNTRY_COLUMN = "survey_definition_country_code"
LANGUAGE_COLUMN = "survey_definition_lang"
TAILORED_COLUMN = "tailored_survey_stoplight_description"

REQUIRED_COLUMNS = {
    DESCRIPTION_COLUMN,
    COUNTRY_COLUMN,
}

POPULATION_OPTIONS = [
    "Women",
    "Children",
    "Elderly people",
    "Entrepreneurs",
    "Students",
    "Adolescents",
    "Men",
    "Parents and caregivers",
    "People with disabilities",
    "Refugees",
    "Rural populations",
    "Urban populations",
    "General population",
]


# =========================================================
# File-processing functions
# =========================================================

def load_csv_with_encoding_fallback(uploaded_file) -> pd.DataFrame:
    """
    Read a CSV file while accounting for common encoding differences.

    The example survey dataset required a non-UTF-8 fallback.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            uploaded_file.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding=encoding,
                low_memory=False,
            )

        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        "The CSV encoding could not be identified."
    ) from last_error


def load_survey(uploaded_file) -> pd.DataFrame:
    """
    Load an uploaded CSV or Excel survey.
    """
    extension = Path(uploaded_file.name).suffix.lower()

    if extension == ".csv":
        return load_csv_with_encoding_fallback(uploaded_file)

    if extension in {".xlsx", ".xls"}:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    raise ValueError(
        "Unsupported file type. Upload a CSV, XLSX, or XLS file."
    )


def validate_survey(dataframe: pd.DataFrame) -> list[str]:
    """
    Return a list of validation errors.
    """
    errors: list[str] = []

    if dataframe.empty:
        errors.append("The uploaded survey does not contain any rows.")

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        errors.append(
            "The following required columns are missing: "
            + ", ".join(sorted(missing_columns))
        )

    if (
        DESCRIPTION_COLUMN in dataframe.columns
        and dataframe[DESCRIPTION_COLUMN].isna().all()
    ):
        errors.append(
            f"The '{DESCRIPTION_COLUMN}' column contains no descriptions."
        )

    return errors


# =========================================================
# Tailoring interface
# =========================================================

class DescriptionTailor(Protocol):
    """
    Interface for any rule-based or model-based tailoring system.

    Your teammate's model should eventually implement this method.
    """

    def tailor_dataframe(
        self,
        dataframe: pd.DataFrame,
        populations: list[str],
    ) -> pd.DataFrame:
        ...


class RuleBasedDescriptionTailor:
    """
    Temporary rule-based tailoring engine.

    Replace this class with a model-backed implementation later.
    """

    @staticmethod
    def format_populations(populations: list[str]) -> str:
        cleaned = [
            population.strip().lower()
            for population in populations
            if population.strip()
        ]

        if not cleaned:
            return ""

        if len(cleaned) == 1:
            return cleaned[0]

        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"

        return (
            ", ".join(cleaned[:-1])
            + f", and {cleaned[-1]}"
        )

    @staticmethod
    def normalize_language(language_value: object) -> str:
        """
        Convert language values such as en_US or es_PY
        into a simple language code.
        """
        if pd.isna(language_value):
            return ""

        language = str(language_value).strip().lower()

        return language.split("_")[0].split("-")[0]

    def tailor_description(
        self,
        description: object,
        country_code: object,
        populations: list[str],
        language: object = None,
    ) -> object:
        """
        Create a basic tailored description.

        This MVP preserves the original description and adds context.
        It does not attempt advanced semantic rewriting.
        """
        if pd.isna(description):
            return description

        description_text = str(description).strip()

        if not description_text:
            return description_text

        population_text = self.format_populations(populations)

        if pd.isna(country_code) or not str(country_code).strip():
            country_text = "the selected country"
        else:
            country_text = str(country_code).strip().upper()

        language_code = self.normalize_language(language)

        # Basic language-aware templates.
        if language_code == "es":
            return (
                f"{description_text} — considerando las experiencias "
                f"de {population_text} en {country_text}"
            )

        if language_code == "pt":
            return (
                f"{description_text} — considerando as experiências "
                f"de {population_text} em {country_text}"
            )

        if language_code == "fr":
            return (
                f"{description_text} — en tenant compte des expériences "
                f"de {population_text} dans le pays {country_text}"
            )

        # Default English template.
        return (
            f"{description_text} — considering the experiences "
            f"of {population_text} in {country_text}"
        )

    def tailor_dataframe(
        self,
        dataframe: pd.DataFrame,
        populations: list[str],
    ) -> pd.DataFrame:
        """
        Apply the rule-based rewriting to every survey row.
        """
        result = dataframe.copy()

        def tailor_row(row: pd.Series) -> object:
            language = (
                row.get(LANGUAGE_COLUMN)
                if LANGUAGE_COLUMN in result.columns
                else None
            )

            return self.tailor_description(
                description=row.get(DESCRIPTION_COLUMN),
                country_code=row.get(COUNTRY_COLUMN),
                populations=populations,
                language=language,
            )

        result[TAILORED_COLUMN] = result.apply(
            tailor_row,
            axis=1,
        )

        return result


# =========================================================
# Future model integration
# =========================================================

class ModelDescriptionTailor:
    """
    Placeholder for your teammate's future model.

    The model can be local, hosted behind an API, or loaded from
    another Python module. It only needs to return a DataFrame
    containing the tailored-description column.
    """

    def __init__(self, model=None):
        self.model = model

    def tailor_dataframe(
        self,
        dataframe: pd.DataFrame,
        populations: list[str],
    ) -> pd.DataFrame:
        result = dataframe.copy()

        # -------------------------------------------------
        # Replace this section with the model integration.
        # -------------------------------------------------
        #
        # Example batch-model pattern:
        #
        # model_inputs = result.apply(
        #     lambda row: {
        #         "description": row[DESCRIPTION_COLUMN],
        #         "country_code": row[COUNTRY_COLUMN],
        #         "language": row.get(LANGUAGE_COLUMN),
        #         "populations": populations,
        #     },
        #     axis=1,
        # ).tolist()
        #
        # tailored_descriptions = self.model.predict(model_inputs)
        #
        # result[TAILORED_COLUMN] = tailored_descriptions
        #
        # return result
        # -------------------------------------------------

        raise NotImplementedError(
            "The model-based tailoring function has not been connected."
        )


def get_tailoring_engine() -> DescriptionTailor:
    """
    Select the active tailoring engine.

    For now, the application uses the rule-based engine.
    Later, change this one line to return ModelDescriptionTailor(...).
    """
    return RuleBasedDescriptionTailor()

    # Future example:
    # trained_model = load_your_team_model()
    # return ModelDescriptionTailor(model=trained_model)


# =========================================================
# Export functions
# =========================================================

def dataframe_to_csv(dataframe: pd.DataFrame) -> bytes:
    """
    Export the survey as a UTF-8 CSV.
    """
    return dataframe.to_csv(
        index=False,
    ).encode("utf-8-sig")


def dataframe_to_excel(dataframe: pd.DataFrame) -> bytes:
    """
    Export the survey as a formatted Excel workbook.
    """
    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Tailored Survey",
        )

        worksheet = writer.sheets["Tailored Survey"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column_cells in worksheet.columns:
            column_letter = column_cells[0].column_letter

            maximum_length = max(
                len(str(cell.value))
                if cell.value is not None
                else 0
                for cell in column_cells
            )

            worksheet.column_dimensions[column_letter].width = min(
                maximum_length + 2,
                70,
            )

    output.seek(0)
    return output.getvalue()


# =========================================================
# Session-state initialization
# =========================================================

if "tailored_survey" not in st.session_state:
    st.session_state.tailored_survey = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# =========================================================
# User interface
# =========================================================

st.title("Custom Survey Builder")

st.markdown(
    """
    Upload an existing survey, select one or more target populations,
    review the newly tailored descriptions, and download the completed
    survey.
    """
)

st.info(
    "This prototype uses temporary rule-based tailoring. "
    "The application is structured so a trained model can replace "
    "the rule-based component later."
)


# ---------------------------------------------------------
# Step 1: Upload
# ---------------------------------------------------------

st.header("1. Upload an Existing Survey")

uploaded_file = st.file_uploader(
    "Drag and drop an Excel or CSV survey file",
    type=["csv", "xlsx", "xls"],
    help=(
        "The survey must contain the columns "
        f"'{DESCRIPTION_COLUMN}' and '{COUNTRY_COLUMN}'."
    ),
)

if uploaded_file is None:
    st.caption("Upload a survey file to begin.")
    st.stop()


try:
    survey = load_survey(uploaded_file)

except Exception as error:
    st.error(f"The survey could not be opened: {error}")
    st.stop()


validation_errors = validate_survey(survey)

if validation_errors:
    for error in validation_errors:
        st.error(error)

    st.stop()


# Reset prior tailored data when a different file is uploaded.
if st.session_state.uploaded_file_name != uploaded_file.name:
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.tailored_survey = None


st.success(
    f"Survey uploaded successfully: "
    f"{len(survey):,} rows and "
    f"{len(survey.columns):,} columns."
)


# ---------------------------------------------------------
# Uploaded-file summary
# ---------------------------------------------------------

summary_column_1, summary_column_2, summary_column_3 = st.columns(3)

with summary_column_1:
    st.metric(
        "Survey Rows",
        f"{len(survey):,}",
    )

with summary_column_2:
    country_count = survey[COUNTRY_COLUMN].nunique(
        dropna=True
    )

    st.metric(
        "Country Codes",
        f"{country_count:,}",
    )

with summary_column_3:
    description_count = survey[DESCRIPTION_COLUMN].notna().sum()

    st.metric(
        "Descriptions Found",
        f"{description_count:,}",
    )


with st.expander(
    "Preview uploaded survey",
    expanded=False,
):
    st.dataframe(
        survey,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Step 2: Filters
# ---------------------------------------------------------

st.header("2. Select Tailoring Filters")

selected_populations = st.multiselect(
    "Target population",
    options=POPULATION_OPTIONS,
    placeholder="Select one or more populations",
    help=(
        "The selected populations will be applied to every description. "
        "Each row's country code will also be passed into the tailoring logic."
    ),
)

custom_population = st.text_input(
    "Optional custom population",
    placeholder="Enter another population not listed above",
)

populations_to_apply = selected_populations.copy()

if custom_population.strip():
    populations_to_apply.append(custom_population.strip())


st.caption(
    f"Country context will come automatically from each row's "
    f"`{COUNTRY_COLUMN}` value."
)


generate_button = st.button(
    "Generate Tailored Survey",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# Generate tailored descriptions
# ---------------------------------------------------------

if generate_button:
    if not populations_to_apply:
        st.warning(
            "Select at least one target population or enter "
            "a custom population."
        )

    else:
        tailoring_engine = get_tailoring_engine()

        try:
            with st.spinner("Tailoring survey descriptions..."):
                st.session_state.tailored_survey = (
                    tailoring_engine.tailor_dataframe(
                        dataframe=survey,
                        populations=populations_to_apply,
                    )
                )

        except Exception as error:
            st.error(
                f"The survey could not be tailored: {error}"
            )


# ---------------------------------------------------------
# Step 3: Review and edit
# ---------------------------------------------------------

if st.session_state.tailored_survey is not None:
    st.header("3. Review and Edit the Tailored Survey")

    st.markdown(
        f"""
        The original `{DESCRIPTION_COLUMN}` field has been preserved.
        Edit the new `{TAILORED_COLUMN}` values below before downloading.
        """
    )

    column_configuration = {
        TAILORED_COLUMN: st.column_config.TextColumn(
            "Tailored Survey Description",
            help="Edit the generated description as needed.",
            width="large",
        ),
        DESCRIPTION_COLUMN: st.column_config.TextColumn(
            "Original Survey Description",
            disabled=True,
            width="large",
        ),
        COUNTRY_COLUMN: st.column_config.TextColumn(
            "Country Code",
            disabled=True,
            width="small",
        ),
    }

    edited_survey = st.data_editor(
        st.session_state.tailored_survey,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config=column_configuration,
        disabled=[
            column
            for column in st.session_state.tailored_survey.columns
            if column != TAILORED_COLUMN
        ],
        key="tailored_survey_editor",
    )

    # Store the latest edits for downloading.
    st.session_state.tailored_survey = edited_survey


    # -----------------------------------------------------
    # Step 4: Download
    # -----------------------------------------------------

    st.header("4. Download the Tailored Survey")

    excel_data = dataframe_to_excel(edited_survey)
    csv_data = dataframe_to_csv(edited_survey)

    download_column_1, download_column_2 = st.columns(2)

    with download_column_1:
        st.download_button(
            label="Download Excel Survey",
            data=excel_data,
            file_name="tailored_survey.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    with download_column_2:
        st.download_button(
            label="Download CSV Survey",
            data=csv_data,
            file_name="tailored_survey.csv",
            mime="text/csv",
            use_container_width=True,
        )
