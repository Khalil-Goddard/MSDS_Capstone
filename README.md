# Custom Survey Builder

## Project files

- `app.py`: Streamlit interface and workflow orchestration.
- `survey_adaptation_engine.py`: survey adaptation model supplied by the project team, with the missing `ALIASES` declaration and CSV encoding fallback corrected.
- `services/`: upload validation, context-option handling, model invocation, merge, and export helpers.
- `data/combined_survey_data.csv`: master survey dataset.
- `data/reference_survey_contexts.csv`: curated reference survey contexts and UI option source.
- `Survey_Adaptation/`: upload original Survey_Adaptation_Engine development
- `services/`: upload Masterfile and data merger

## To Run the Streamlit App locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Expected uploaded survey

The uploaded CSV or Excel file must contain exactly one nonblank `survey_definition_id`. Under the confirmed architecture, that ID must exist in both the master dataset and `reference_survey_contexts.csv`.

## Model inputs

- Country: single select
- Language: single select
- Population type: multiselect
- Focus: multiselect
- Organization type: single select

Advanced model parameters are available under an expandable section with the engine defaults preloaded.
