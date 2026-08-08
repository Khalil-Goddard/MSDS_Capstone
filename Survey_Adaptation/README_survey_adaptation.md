# Survey Adaptation Prototype

## What the prototype does

The user selects one of the curated `survey_definition_id` values as the base survey and enters five context variables:

- Country
- Language
- Population type
- Focus
- Type of organization

The engine then:

1. Scores the other curated surveys for contextual similarity.
2. Keeps every base-survey question by default.
3. Recommends **Modify** only when a similar survey contains a different wording for the same indicator.
4. Recommends **Add** when an indicator is absent from the base survey but is supported by the most similar surveys.
5. Preserves the red, yellow, and green descriptions from the selected source record.

This is a baseline retrieval model, not a trained supervised classifier. The raw data does not contain historical Keep/Modify/Add labels.

## Files

- `survey_adaptation_engine.py`: model and command-line interface
- `streamlit_app.py`: Streamlit-ready user interface
- `reference_survey_contexts.csv`: the curated survey IDs and adaptation categories
- `combined_survey_data.csv`: input data created from the three raw tables
- `requirements_survey_adaptation.txt`: Python dependencies

## Install

```bash
python -m pip install -r requirements_survey_adaptation.txt
```

## Run from the command line

```bash
python survey_adaptation_engine.py \
  --data combined_survey_data.csv \
  --contexts reference_survey_contexts.csv \
  --base-id 31 \
  --country Paraguay \
  --language Spanish \
  --population students \
  --focus education \
  --organization-type "educational organization" \
  --output adapted_survey_31.csv
```

The command produces:

- `adapted_survey_31.csv`
- `similar_surveys.csv`
- `adaptation_result.json`

## Run Streamlit

```bash
streamlit run streamlit_app.py
```

## Keep, Modify, and Add logic

### Keep

Every question in the selected base survey starts as **Keep**.

### Modify

A question becomes **Modify** when:

- the same normalized `code_name` appears in a similar survey;
- the wording is different;
- the reference survey has a sufficient context score; and
- the wording is sufficiently similar, unless it is a matching target-language version.

When the requested language is unchanged, the default maximum is 15% of the base survey. When the requested language differs, the engine can use all available target-language variants because the indicator set is still preserved.

### Add

An indicator becomes **Add** when:

- it does not exist in the base survey;
- it appears in one or more of the top similar surveys; and
- its weighted support score exceeds the minimum threshold.

The default maximum is five additions.

## Important limitations

- The context categories are manually curated from the sponsor's examples.
- `code_name` is used as the main indicator-matching key because `survey_indicator_id` is missing or inconsistent in some rows.
- TF-IDF is used as a practical baseline. A multilingual sentence-transformer can replace it later.
- The output must be reviewed by the Methodology Team before use.
- The model should be evaluated using expert precision for Modify/Add recommendations, the unchanged rate, and the base-indicator preservation rate.
