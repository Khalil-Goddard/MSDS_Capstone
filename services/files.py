from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from config import REQUIRED_UPLOAD_COLUMNS, SURVEY_ID_COLUMN


def load_uploaded_survey(uploaded_file) -> pd.DataFrame:
    extension = Path(uploaded_file.name).suffix.lower()
    if extension == ".csv":
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=encoding, low_memory=False)
            except UnicodeDecodeError as error:
                last_error = error
        raise ValueError("The CSV encoding could not be identified.") from last_error

    if extension in {".xlsx", ".xls"}:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Upload CSV, XLSX, or XLS.")


def validate_uploaded_survey(dataframe: pd.DataFrame) -> tuple[list[str], int | None]:
    errors: list[str] = []
    if dataframe.empty:
        errors.append("The uploaded survey does not contain any rows.")
        return errors, None

    missing = REQUIRED_UPLOAD_COLUMNS - set(dataframe.columns)
    if missing:
        errors.append("Missing required columns: " + ", ".join(sorted(missing)))

    if SURVEY_ID_COLUMN not in dataframe.columns:
        return errors, None

    ids = pd.to_numeric(dataframe[SURVEY_ID_COLUMN], errors="coerce").dropna().astype(int).unique()
    if len(ids) != 1:
        errors.append(
            f"The uploaded survey must contain exactly one nonblank {SURVEY_ID_COLUMN}; found {len(ids)}."
        )
        return errors, None

    return errors, int(ids[0])


def dataframe_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel(dataframe: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Adapted Survey")
        worksheet = writer.sheets["Adapted Survey"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cells in worksheet.columns:
            letter = cells[0].column_letter
            max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in cells)
            worksheet.column_dimensions[letter].width = min(max_length + 2, 60)
    output.seek(0)
    return output.getvalue()
