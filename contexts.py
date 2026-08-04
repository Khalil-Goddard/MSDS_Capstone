from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CONTEXT_COLUMNS = [
    "country",
    "language",
    "population_type",
    "focus",
    "organization_type",
]


def load_reference_contexts(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(CONTEXT_COLUMNS + ["survey_definition_id"]) - set(dataframe.columns)
    if missing:
        raise ValueError("Reference contexts file is missing: " + ", ".join(sorted(missing)))
    return dataframe


def distinct_context_values(dataframe: pd.DataFrame, column: str) -> list[str]:
    """Return individual normalized UI options, including values stored as tag lists."""
    values: set[str] = set()
    for raw in dataframe[column].dropna().astype(str):
        for piece in re.split(r"[;,|/]", raw):
            value = piece.strip()
            if value:
                values.add(value)
    return sorted(values, key=str.casefold)


def join_multiselect(values: list[str]) -> str:
    return "; ".join(values)
