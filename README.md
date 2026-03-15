# A Sustainability-Aware AutoML nowledge Base

A lightweight Streamlit UI for exploring historical AutoML runs and getting intent-aware model recommendations from the AutoML knowledge base.

## Features

- Explore datasets already stored in `runs.db`
- Compare candidates across frameworks such as H2O, AutoSklearn, and TPOT
- Visualize trade-offs between:
  - accuracy vs energy
  - accuracy vs latency
- Get conversational recommendations based on:
  - accuracy
  - energy
  - latency
- Apply optional constraints such as:
  - maximum inference latency
  - minimum accuracy
- Inspect the winning configuration and export it

## Project structure

```text
experiment-factory/
├── automl_kb/
│   ├── config.py
│   └── data/
│       └── runs.db
└── apps/
    └── GUI/
        └── streamlit_app/
            └── app.py
```

## Installation

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

Run the app from the repository root so imports such as `automl_kb.config` resolve correctly:

```bash
streamlit run apps/GUI/streamlit_app/app.py
```

Then open the local Streamlit URL shown in the terminal.

## Notes

- The app expects the SQLite database at `automl_kb/data/runs.db`
- Database paths are read from `automl_kb/config.py`
- If the DB is missing or the path is wrong, the app will show a connection error in the sidebar

## Recommendation flow

1. Enter an OpenML task ID
2. Load the dataset if it exists in the knowledge base
3. Optionally view the framework leaderboard
4. Choose your optimization goal
5. Add constraints if needed
6. Review the recommended winner and alternatives
7. Inspect the configuration and export the result

## Minimal dependencies

- `streamlit`
- `pandas`
- `plotly`

## License

Add your preferred license here.
