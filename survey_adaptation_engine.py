"""Template-preserving survey adaptation engine.

The engine uses a selected survey_definition_id as the base survey and five
context inputs:
    country, language, population_type, focus, organization_type

It returns Keep, Modify, and Add recommendations by comparing the base survey
with the curated reference surveys.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_WEIGHTS = {
    "country": 0.15,
    "language": 0.15,
    "population_type": 0.25,
    "focus": 0.25,
    "organization_type": 0.20,
}

ALIASES = {
    "usa": "united states",
    "us": "united states",
    "u s": "united states",
    "uk": "united kingdom",
    "gb": "united kingdom",
    "za": "south africa",
    "py": "paraguay",
    "mg": "madagascar",
    "uz": "uzbekistan",
    "ar": "argentina",
    "in": "india",
    "gt": "guatemala",
    "ec": "ecuador",
    "sg": "singapore",
    "hn": "honduras",
    "br": "brazil",
    "ni": "nicaragua",
    "tz": "tanzania",
    "co": "colombia",
    "en": "english",
    "en us": "english",
    "english language": "english",
    "es": "spanish",
    "es py": "spanish",
    "spanish language": "spanish",
    "pt": "portuguese",
    "pt br": "portuguese",
    "ru": "russian",
    "ru ru": "russian",
    "mg mg": "malagasy",
    "sw": "swahili",
    "young people": "youth",
    "children": "youth",
    "elderly": "older adults",
    "senior citizens": "older adults",
    "migrants": "migrant workers",
    "farmers": "small scale agricultural producers",
    "small scale producers": "small scale agricultural producers",
    "small agricultural producers": "small scale agricultural producers",
    "school directors": "directors",
    "businesses": "company",
    "companies": "company",
    "corporation": "company",
    "nonprofit": "ngo",
    "non profit": "ngo",
    "micro finance": "microfinance",
    "environmental": "environment",
    "educational": "education",
    "gender equity": "gender",
}


def normalize_text(value: Any) -> str:
    """Lowercase, remove accents/punctuation, and apply simple aliases."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)




def normalize_question_text(value: Any) -> str:
    """Normalize multilingual question text without discarding non-Latin scripts."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def split_tags(value: Any) -> set[str]:
    if value is None:
        return set()
    pieces = re.split(r"[;,|/]", str(value))
    tags = set()
    for piece in pieces:
        normalized = normalize_text(piece)
        if normalized:
            tags.add(ALIASES.get(normalized, normalized))
    return tags


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def field_similarity(user_value: Any, reference_value: Any) -> float:
    """Compare categorical fields that may contain one or several tags."""
    user_tags = split_tags(user_value)
    reference_tags = split_tags(reference_value)
    if not user_tags or not reference_tags:
        return 0.0
    if user_tags & reference_tags:
        # Exact overlap should be stronger than generic Jaccard.
        return max(0.8, jaccard(user_tags, reference_tags))

    # Token overlap allows inputs such as "women entrepreneurs" to partially
    # match a reference containing separate women and entrepreneurs tags.
    user_tokens = set(" ".join(user_tags).split())
    reference_tokens = set(" ".join(reference_tags).split())
    return 0.5 * jaccard(user_tokens, reference_tokens)


@dataclass(frozen=True)
class ContextInput:
    country: str
    language: str
    population_type: str
    focus: str
    organization_type: str

    def as_text(self) -> str:
        return (
            f"country {self.country}; language {self.language}; "
            f"population {self.population_type}; focus {self.focus}; "
            f"organization {self.organization_type}"
        )


@dataclass(frozen=True)
class ReferenceSurvey:
    survey_definition_id: int
    title: str
    country: str
    language: str
    population_type: str
    focus: str
    organization_type: str
    comparison_group: str
    notes: str

    def context_text(self) -> str:
        return (
            f"{self.title}; country {self.country}; language {self.language}; "
            f"population {self.population_type}; focus {self.focus}; "
            f"organization {self.organization_type}; {self.notes}"
        )


class SurveyAdaptationEngine:
    """Retrieve similar surveys and generate Keep/Modify/Add recommendations."""

    def __init__(
        self,
        combined_data_path: str | Path,
        reference_contexts_path: str | Path,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.combined_data_path = Path(combined_data_path)
        self.reference_contexts_path = Path(reference_contexts_path)
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)

        self.rows = self._read_csv(self.combined_data_path)
        self.surveys = self._group_rows_by_survey(self.rows)
        self.reference_surveys = self._load_reference_contexts(
            self.reference_contexts_path
        )

        missing_ids = sorted(set(self.reference_surveys) - set(self.surveys))
        if missing_ids:
            raise ValueError(
                "Reference survey IDs are absent from the combined dataset: "
                + ", ".join(map(str, missing_ids))
            )

        self._question_vectorizer, self._question_vectors, self._question_index = (
            self._build_question_similarity_index()
        )
        self._indicator_vectorizer = self._build_indicator_vectorizer()

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        """Read CSV data using common encodings used by the project files."""
        if not path.exists():
            raise FileNotFoundError(path)

        last_error: UnicodeDecodeError | None = None
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                with path.open("r", encoding=encoding, newline="") as handle:
                    return list(csv.DictReader(handle))
            except UnicodeDecodeError as error:
                last_error = error

        raise ValueError(f"Unable to determine CSV encoding for {path}") from last_error

    @staticmethod
    def _group_rows_by_survey(
        rows: Sequence[dict[str, str]],
    ) -> dict[int, list[dict[str, str]]]:
        grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[safe_int(row.get("survey_definition_id"))].append(row)
        for survey_rows in grouped.values():
            survey_rows.sort(
                key=lambda row: safe_int(
                    row.get("survey_stoplight_order_number"), 999999
                )
            )
        return dict(grouped)

    def _load_reference_contexts(
        self, path: Path
    ) -> dict[int, ReferenceSurvey]:
        contexts = self._read_csv(path)
        loaded: dict[int, ReferenceSurvey] = {}
        for row in contexts:
            survey_id = safe_int(row["survey_definition_id"])
            title = ""
            if survey_id in self.surveys and self.surveys[survey_id]:
                title = self.surveys[survey_id][0].get(
                    "survey_definition_title", ""
                )
            loaded[survey_id] = ReferenceSurvey(
                survey_definition_id=survey_id,
                title=title,
                country=row.get("country", ""),
                language=row.get("language", ""),
                population_type=row.get("population_type", ""),
                focus=row.get("focus", ""),
                organization_type=row.get("organization_type", ""),
                comparison_group=row.get("comparison_group", ""),
                notes=row.get("notes", ""),
            )
        return loaded

    @staticmethod
    def indicator_key(row: Mapping[str, str]) -> str:
        """Prefer normalized code_name; use survey_indicator_id as fallback."""
        code_name = normalize_text(row.get("survey_stoplight_code_name", ""))
        if code_name:
            return f"code:{code_name.replace(' ', '')}"
        indicator_id = normalize_text(
            row.get("survey_stoplight_survey_indicator_id", "")
        )
        if indicator_id:
            return f"indicator:{indicator_id}"
        question = normalize_question_text(
            row.get("survey_stoplight_question_text", "")
        )
        return f"question:{question}"

    @staticmethod
    def indicator_text(row: Mapping[str, str]) -> str:
        columns = (
            "survey_stoplight_short_name",
            "survey_stoplight_question_text",
            "survey_stoplight_description",
            "survey_stoplight_definition",
            "survey_stoplight_dimension",
            "red_description",
            "yellow_description",
            "green_description",
        )
        return " ".join(str(row.get(column, "") or "") for column in columns)

    def _build_question_similarity_index(self):
        texts: list[str] = []
        text_index: dict[str, int] = {}
        for survey_id in self.reference_surveys:
            for row in self.surveys[survey_id]:
                text = str(row.get("survey_stoplight_question_text", "") or "")
                if text not in text_index:
                    text_index[text] = len(texts)
                    texts.append(text)

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            strip_accents="unicode",
            lowercase=True,
            min_df=1,
        )
        vectors = vectorizer.fit_transform(texts)
        return vectorizer, vectors, text_index

    def _build_indicator_vectorizer(self) -> TfidfVectorizer:
        corpus = []
        for survey_id in self.reference_surveys:
            corpus.append(self.reference_surveys[survey_id].context_text())
            corpus.extend(self.indicator_text(row) for row in self.surveys[survey_id])
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            strip_accents="unicode",
            lowercase=True,
            min_df=1,
            sublinear_tf=True,
        )
        vectorizer.fit(corpus)
        return vectorizer

    def question_similarity(self, left: str, right: str) -> float:
        if normalize_question_text(left) == normalize_question_text(right):
            return 1.0
        left_index = self._question_index.get(left)
        right_index = self._question_index.get(right)
        if left_index is not None and right_index is not None:
            return float(
                cosine_similarity(
                    self._question_vectors[left_index],
                    self._question_vectors[right_index],
                )[0, 0]
            )
        vectors = self._question_vectorizer.transform([left, right])
        return float(cosine_similarity(vectors[0], vectors[1])[0, 0])

    def indicator_context_similarity(
        self, context: ContextInput, indicator_row: Mapping[str, str]
    ) -> float:
        vectors = self._indicator_vectorizer.transform(
            [context.as_text(), self.indicator_text(indicator_row)]
        )
        return float(cosine_similarity(vectors[0], vectors[1])[0, 0])

    def context_score(
        self, context: ContextInput, reference: ReferenceSurvey
    ) -> tuple[float, dict[str, float]]:
        values = {
            "country": field_similarity(context.country, reference.country),
            "language": field_similarity(context.language, reference.language),
            "population_type": field_similarity(
                context.population_type, reference.population_type
            ),
            "focus": field_similarity(context.focus, reference.focus),
            "organization_type": field_similarity(
                context.organization_type, reference.organization_type
            ),
        }

        active_weight = 0.0
        weighted_sum = 0.0
        for field, weight in self.weights.items():
            user_value = getattr(context, field)
            if normalize_text(user_value):
                active_weight += weight
                weighted_sum += weight * values[field]

        score = weighted_sum / active_weight if active_weight else 0.0
        return score, values

    def list_base_surveys(self) -> list[dict[str, Any]]:
        result = []
        for survey_id, reference in sorted(
            self.reference_surveys.items(),
            key=lambda item: item[1].title.lower(),
        ):
            result.append(
                {
                    "survey_definition_id": survey_id,
                    "title": reference.title,
                    "country": reference.country,
                    "language": reference.language,
                    "population_type": reference.population_type,
                    "focus": reference.focus,
                    "organization_type": reference.organization_type,
                    "question_count": len(self.surveys[survey_id]),
                }
            )
        return result

    def find_similar_surveys(
        self,
        base_survey_id: int,
        context: ContextInput,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if base_survey_id not in self.reference_surveys:
            raise ValueError(
                f"Base survey {base_survey_id} is not in the curated reference list."
            )

        scored = []
        for survey_id, reference in self.reference_surveys.items():
            if survey_id == base_survey_id:
                continue
            score, field_scores = self.context_score(context, reference)
            scored.append(
                {
                    "survey_definition_id": survey_id,
                    "title": reference.title,
                    "country": reference.country,
                    "language": reference.language,
                    "population_type": reference.population_type,
                    "focus": reference.focus,
                    "organization_type": reference.organization_type,
                    "comparison_group": reference.comparison_group,
                    "context_score": round(score, 4),
                    **{f"{key}_score": round(value, 4) for key, value in field_scores.items()},
                    "question_count": len(self.surveys[survey_id]),
                }
            )

        scored.sort(
            key=lambda row: (
                row["context_score"],
                row["question_count"],
            ),
            reverse=True,
        )
        return scored[: max(1, top_k)]

    def _survey_question_index(
        self, survey_id: int
    ) -> dict[str, list[dict[str, str]]]:
        index: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.surveys[survey_id]:
            index[self.indicator_key(row)].append(row)
        return dict(index)

    @staticmethod
    def _language_name(value: str) -> str:
        normalized = normalize_text(value)
        if normalized in ALIASES:
            normalized = ALIASES[normalized]
        # Locale-like values from raw data.
        prefixes = {
            "en": "english",
            "es": "spanish",
            "pt": "portuguese",
            "ru": "russian",
            "mg": "malagasy",
            "sw": "swahili",
            "uz": "uzbek",
        }
        first = normalized.split(" ")[0] if normalized else ""
        return prefixes.get(first, normalized)

    def _base_language(self, base_survey_id: int) -> str:
        reference = self.reference_surveys.get(base_survey_id)
        if reference is not None and normalize_text(reference.language):
            return self._language_name(reference.language)
        row = self.surveys[base_survey_id][0]
        return self._language_name(row.get("survey_definition_lang", ""))

    @staticmethod
    def _copy_base_output_fields(
        row: Mapping[str, str],
        base_survey_id: int,
        base_title: str,
    ) -> dict[str, Any]:
        return {
            "base_survey_definition_id": base_survey_id,
            "base_survey_title": base_title,
            "action": "Keep",
            "survey_stoplight_id": row.get("survey_stoplight_id", ""),
            "indicator_key": SurveyAdaptationEngine.indicator_key(row),
            "survey_indicator_id": row.get(
                "survey_stoplight_survey_indicator_id", ""
            ),
            "code_name": row.get("survey_stoplight_code_name", ""),
            "order_number": safe_int(row.get("survey_stoplight_order_number")),
            "short_name": row.get("survey_stoplight_short_name", ""),
            "dimension": row.get("survey_stoplight_dimension", ""),
            "original_question_text": row.get(
                "survey_stoplight_question_text", ""
            ),
            "adapted_question_text": row.get(
                "survey_stoplight_question_text", ""
            ),
            "original_description": row.get(
                "survey_stoplight_description", ""
            ),
            "adapted_description": row.get(
                "survey_stoplight_description", ""
            ),
            "original_red_description": row.get("red_description", ""),
            "adapted_red_description": row.get("red_description", ""),
            "original_yellow_description": row.get(
                "yellow_description", ""
            ),
            "adapted_yellow_description": row.get(
                "yellow_description", ""
            ),
            "original_green_description": row.get("green_description", ""),
            "adapted_green_description": row.get("green_description", ""),
            "source_survey_definition_id": base_survey_id,
            "source_survey_title": base_title,
            "source_context_score": 1.0,
            "question_similarity": 1.0,
            "recommendation_score": 0.0,
            "reason": "Question is retained from the selected base survey.",
        }

    def adapt_survey(
        self,
        base_survey_id: int,
        context: ContextInput,
        top_k_similar: int = 5,
        max_modify_fraction: float = 0.15,
        max_additions: int = 5,
        min_similar_context_score: float = 0.30,
        min_question_similarity: float = 0.30,
        min_add_score: float = 0.20,
    ) -> dict[str, Any]:
        """Generate a template-preserving Keep/Modify/Add recommendation.

        Existing questions default to Keep. Modify recommendations use an
        existing wording variant of the same indicator from a high-scoring
        reference survey. Add recommendations are indicators absent from the
        base but supported by the top similar surveys.
        """
        if base_survey_id not in self.surveys:
            raise ValueError(f"Unknown survey_definition_id: {base_survey_id}")
        if base_survey_id not in self.reference_surveys:
            raise ValueError(
                "The requested base survey is not one of the curated reference surveys."
            )

        base_rows = self.surveys[base_survey_id]
        base_reference = self.reference_surveys[base_survey_id]
        base_title = base_reference.title
        similar_surveys = self.find_similar_surveys(
            base_survey_id, context, top_k=top_k_similar
        )
        similar_surveys = [
            survey
            for survey in similar_surveys
            if survey["context_score"] >= min_similar_context_score
        ] or self.find_similar_surveys(base_survey_id, context, top_k=1)

        survey_score = {
            row["survey_definition_id"]: float(row["context_score"])
            for row in similar_surveys
        }
        survey_title = {
            row["survey_definition_id"]: row["title"] for row in similar_surveys
        }
        question_indexes = {
            survey_id: self._survey_question_index(survey_id)
            for survey_id in survey_score
        }

        base_keys = {self.indicator_key(row) for row in base_rows}
        output_rows = [
            self._copy_base_output_fields(row, base_survey_id, base_title)
            for row in base_rows
        ]

        target_language = self._language_name(context.language)
        base_language = self._base_language(base_survey_id)
        language_change_requested = bool(
            target_language
            and base_language
            and target_language != base_language
        )

        modification_candidates: list[dict[str, Any]] = []
        for output_row, base_row in zip(output_rows, base_rows):
            key = output_row["indicator_key"]
            base_question = output_row["original_question_text"]

            best_candidate = None
            best_rank = None
            for similar in similar_surveys:
                survey_id = similar["survey_definition_id"]
                candidate_rows = question_indexes[survey_id].get(key, [])
                for candidate in candidate_rows:
                    candidate_question = candidate.get(
                        "survey_stoplight_question_text", ""
                    )
                    if not normalize_question_text(candidate_question):
                        continue
                    question_score = self.question_similarity(
                        base_question, candidate_question
                    )
                    candidate_language = self._language_name(
                        self.reference_surveys[survey_id].language
                    )
                    target_language_match = (
                        language_change_requested
                        and bool(target_language)
                        and candidate_language == target_language
                    )
                    text_changed = (
                        normalize_question_text(base_question)
                        != normalize_question_text(candidate_question)
                    )
                    if not text_changed:
                        continue

                    # Same indicator is strong evidence. Require moderate text
                    # similarity unless a target-language variant is available.
                    if (
                        not target_language_match
                        and question_score < min_question_similarity
                    ):
                        continue

                    context_score = float(similar["context_score"])
                    rank = (
                        1 if target_language_match else 0,
                        context_score,
                        question_score,
                    )
                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        best_candidate = {
                            "candidate": candidate,
                            "survey_id": survey_id,
                            "survey_title": similar["title"],
                            "context_score": context_score,
                            "question_similarity": question_score,
                            "target_language_match": target_language_match,
                        }

            if best_candidate is None:
                continue

            # Prefer changes that are clearly supported by context and that use
            # a target-language wording when language adaptation is requested.
            evidence = (
                best_candidate["context_score"]
                * (
                    1.0
                    if best_candidate["target_language_match"]
                    else max(0.10, 1.0 - best_candidate["question_similarity"])
                )
            )
            modification_candidates.append(
                {
                    "output_row": output_row,
                    **best_candidate,
                    "evidence": evidence,
                }
            )

        modification_candidates.sort(
            key=lambda item: (
                item["target_language_match"],
                item["evidence"],
                item["context_score"],
            ),
            reverse=True,
        )

        if language_change_requested:
            max_modifications = len(base_rows)
        else:
            max_modifications = max(
                0, math.ceil(len(base_rows) * max_modify_fraction)
            )

        for item in modification_candidates[:max_modifications]:
            output_row = item["output_row"]
            candidate = item["candidate"]
            output_row.update(
                {
                    "action": "Modify",
                    "adapted_question_text": candidate.get(
                        "survey_stoplight_question_text", ""
                    ),
                    "adapted_description": candidate.get(
                        "survey_stoplight_description", ""
                    ),
                    "adapted_red_description": candidate.get(
                        "red_description", ""
                    ),
                    "adapted_yellow_description": candidate.get(
                        "yellow_description", ""
                    ),
                    "adapted_green_description": candidate.get(
                        "green_description", ""
                    ),
                    "source_survey_definition_id": item["survey_id"],
                    "source_survey_title": item["survey_title"],
                    "source_context_score": round(
                        item["context_score"], 4
                    ),
                    "question_similarity": round(
                        item["question_similarity"], 4
                    ),
                    "recommendation_score": round(item["evidence"], 4),
                    "reason": (
                        "Existing wording from a similar survey matches the "
                        "requested language and preserves the same indicator."
                        if item["target_language_match"]
                        else "A similar survey uses context-specific wording for the same indicator."
                    ),
                }
            )

        # Build Add candidates from indicators that occur in similar surveys but
        # are absent from the selected base survey.
        add_groups: dict[str, list[tuple[int, dict[str, str], float]]] = defaultdict(list)
        for similar in similar_surveys:
            survey_id = similar["survey_definition_id"]
            context_score = float(similar["context_score"])
            seen_in_survey = set()
            for candidate in self.surveys[survey_id]:
                key = self.indicator_key(candidate)
                if key in base_keys or key in seen_in_survey:
                    continue
                seen_in_survey.add(key)
                add_groups[key].append((survey_id, candidate, context_score))

        total_context_weight = sum(survey_score.values()) or 1.0
        addition_candidates = []
        for key, occurrences in add_groups.items():
            weighted_support = sum(item[2] for item in occurrences) / total_context_weight
            survey_support = len({item[0] for item in occurrences}) / len(similar_surveys)
            representative_survey_id, representative, representative_score = max(
                occurrences, key=lambda item: item[2]
            )
            text_relevance = self.indicator_context_similarity(
                context, representative
            )
            add_score = (
                0.65 * weighted_support
                + 0.20 * survey_support
                + 0.15 * text_relevance
            )
            if add_score < min_add_score:
                continue
            addition_candidates.append(
                {
                    "key": key,
                    "representative": representative,
                    "source_survey_id": representative_survey_id,
                    "source_title": survey_title[representative_survey_id],
                    "source_context_score": representative_score,
                    "weighted_support": weighted_support,
                    "survey_support": survey_support,
                    "text_relevance": text_relevance,
                    "add_score": add_score,
                }
            )

        addition_candidates.sort(
            key=lambda item: (
                item["add_score"],
                item["survey_support"],
                item["source_context_score"],
            ),
            reverse=True,
        )

        next_order = max(
            (safe_int(row.get("survey_stoplight_order_number")) for row in base_rows),
            default=0,
        )
        for offset, item in enumerate(addition_candidates[:max_additions], start=1):
            candidate = item["representative"]
            output_rows.append(
                {
                    "base_survey_definition_id": base_survey_id,
                    "base_survey_title": base_title,
                    "action": "Add",
                    "survey_stoplight_id": "",
                    "indicator_key": item["key"],
                    "survey_indicator_id": candidate.get(
                        "survey_stoplight_survey_indicator_id", ""
                    ),
                    "code_name": candidate.get(
                        "survey_stoplight_code_name", ""
                    ),
                    "order_number": next_order + offset,
                    "short_name": candidate.get(
                        "survey_stoplight_short_name", ""
                    ),
                    "dimension": candidate.get(
                        "survey_stoplight_dimension", ""
                    ),
                    "original_question_text": "",
                    "adapted_question_text": candidate.get(
                        "survey_stoplight_question_text", ""
                    ),
                    "original_description": "",
                    "adapted_description": candidate.get(
                        "survey_stoplight_description", ""
                    ),
                    "original_red_description": "",
                    "adapted_red_description": candidate.get(
                        "red_description", ""
                    ),
                    "original_yellow_description": "",
                    "adapted_yellow_description": candidate.get(
                        "yellow_description", ""
                    ),
                    "original_green_description": "",
                    "adapted_green_description": candidate.get(
                        "green_description", ""
                    ),
                    "source_survey_definition_id": item[
                        "source_survey_id"
                    ],
                    "source_survey_title": item["source_title"],
                    "source_context_score": round(
                        item["source_context_score"], 4
                    ),
                    "question_similarity": "",
                    "recommendation_score": round(item["add_score"], 4),
                    "reason": (
                        f"Indicator is absent from the base survey and appears in "
                        f"{len({occurrence[0] for occurrence in add_groups[item['key']]})} "
                        "of the selected similar surveys."
                    ),
                }
            )

        output_rows.sort(key=lambda row: safe_int(row["order_number"]))
        counts = Counter(row["action"] for row in output_rows)

        return {
            "base_survey": {
                "survey_definition_id": base_survey_id,
                "title": base_title,
                "original_question_count": len(base_rows),
            },
            "context": asdict(context),
            "parameters": {
                "top_k_similar": top_k_similar,
                "max_modify_fraction": max_modify_fraction,
                "max_additions": max_additions,
                "min_similar_context_score": min_similar_context_score,
                "min_question_similarity": min_question_similarity,
                "min_add_score": min_add_score,
            },
            "summary": {
                "Keep": counts.get("Keep", 0),
                "Modify": counts.get("Modify", 0),
                "Add": counts.get("Add", 0),
                "final_question_count": len(output_rows),
                "unchanged_rate": round(
                    counts.get("Keep", 0) / len(base_rows), 4
                    if base_rows
                    else 0.0
                ),
                "modification_rate": round(
                    counts.get("Modify", 0) / len(base_rows), 4
                    if base_rows
                    else 0.0
                ),
                "base_indicator_preservation_rate": round(
                    (counts.get("Keep", 0) + counts.get("Modify", 0))
                    / len(base_rows),
                    4,
                )
                if base_rows
                else 0.0,
            },
            "similar_surveys": similar_surveys,
            "adapted_questions": output_rows,
        }


def write_records_csv(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    if not records:
        raise ValueError("No records to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(records[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def records_to_csv_text(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        return ""
    import io

    buffer = io.StringIO()
    headers = list(records[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="combined_survey_data.csv")
    parser.add_argument("--contexts", default="reference_survey_contexts.csv")
    parser.add_argument("--base-id", type=int, required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--focus", required=True)
    parser.add_argument("--organization-type", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-modify-fraction", type=float, default=0.15)
    parser.add_argument("--max-additions", type=int, default=5)
    parser.add_argument("--output", default="adapted_survey.csv")
    parser.add_argument("--similar-output", default="similar_surveys.csv")
    parser.add_argument("--json-output", default="adaptation_result.json")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    engine = SurveyAdaptationEngine(args.data, args.contexts)
    context = ContextInput(
        country=args.country,
        language=args.language,
        population_type=args.population,
        focus=args.focus,
        organization_type=args.organization_type,
    )
    result = engine.adapt_survey(
        base_survey_id=args.base_id,
        context=context,
        top_k_similar=args.top_k,
        max_modify_fraction=args.max_modify_fraction,
        max_additions=args.max_additions,
    )
    write_records_csv(args.output, result["adapted_questions"])
    write_records_csv(args.similar_output, result["similar_surveys"])
    Path(args.json_output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Adapted survey: {args.output}")
    print(f"Similar surveys: {args.similar_output}")
    print(f"Full result: {args.json_output}")


if __name__ == "__main__":
    main()
