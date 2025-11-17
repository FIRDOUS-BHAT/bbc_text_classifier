## BBC Text Classifier – Project Documentation

### 1. Purpose

- Build a lightweight NLP system that classifies BBC news articles into topical categories using classical ML.
- Provide both an API (FastAPI) and a browser UI for real-time predictions.
- Maintain a reproducible pipeline from EDA through training to deployment-ready artifacts.

### 2. High-Level Workflow

1. **Data prep & EDA:** explore or refresh datasets inside `notebooks/eda_bbc.ipynb`, exporting a cleaned CSV to `notebooks/data/`.
2. **Training:** run `python train.py` to create/update the TF-IDF + Logistic Regression pipeline and label metadata.
3. **Serving:** start the FastAPI service (`uvicorn api.main:app --reload`) which auto-loads the latest artifacts.
4. **Prediction:** interact through the HTML UI at `http://localhost:8000` or call the `/predict` API endpoint.

### 3. Repository Map

- `README.md` – quickstart checklist.
- `notebooks/eda_bbc.ipynb` – exploratory analysis, optional full-dataset preparation.
- `notebooks/data/` – input CSVs (e.g., `bbc_sample.csv`).
- `notebooks/model_artifacts/` – default output folder for `model.pkl` and `labels.json`.
- `train.py` – standalone training pipeline.
- `api/main.py` – FastAPI server with artifact auto-discovery and prediction endpoints.
- `ui/index.html` – Jinja2 template rendered by the API.
- `requirements.txt` / `pyproject.toml` – dependency specs.
- `model_artifacts/` – optional staging area for serving-only deployments.

### 4. Data & EDA (`notebooks/eda_bbc.ipynb`)

- Notebook walks through loading BBC news CSVs, cleaning text, label distribution checks, and optional train/test splits.
- Can be extended to ingest the original folder-structured BBC dataset, convert it to CSV, and write to `notebooks/data/bbc_sample.csv`.
- Use `%run` cells or exports within the notebook to ensure column names include keywords (`text`, `content`, `label`, `category`) so `train.py` auto-detects them.

### 5. Training Pipeline (`train.py`)

- Loads `notebooks/data/bbc_sample.csv` (customize `DATA_PATH` as needed).
- Column auto-rename logic:
  - Scans headers for text-like keywords and coerces to `text`.
  - Scans for label-like keywords and coerces to `category`.
- Drops rows missing required columns, prints summaries of columns and class counts.
- Performs an 80/20 split with stratification whenever every class has at least two samples; otherwise falls back to a random split.
- Model definition: `Pipeline(TfidfVectorizer(max_features=5000, stop_words='english'), LogisticRegression(max_iter=1000, random_state=42))`.
- Evaluation outputs accuracy and classification report to stdout.
- Persists artifacts under `notebooks/model_artifacts/`:
  - `model.pkl` – pickled scikit-learn Pipeline handling raw text input.
  - `labels.json` – sorted list of category names (`{"categories": [...]}`).
- Artifacts are self-contained: the API loads the pipeline directly without needing extra preprocessing code.

### 6. Model Artifacts & Discovery

- Primary location: `notebooks/model_artifacts/`. Copy or symlink to `model_artifacts/` at project root for deployment if desired.
- `api/main.py` searches the following directories (in order) for expected files:
  1. `model_artifacts/`
  2. `notebooks/model_artifacts/`
  3. `notebooks/`
- Expected files:
  - `model.pkl`, `models.pkl`, or `pipeline.pkl` (pipeline or estimator).
  - Optional `vectorizer.pkl` if the pipeline is split.
  - `labels.json` or `labels.pkl`.
- The loader gracefully handles JSON lists, `{"categories": ...}` dicts, tuples, or pickled objects for labels.

### 7. API Service (`api/main.py`)

- FastAPI application configured with logging and Jinja2 templates (pointing to `ui/`).
- Startup routine:
  - Discover artifact directory via `find_artifact_dir`.
  - Load labels via `load_labels`.
  - Attempt to load a single pipeline; if unavailable, try separate `vectorizer.pkl` + `model.pkl`.
  - Abort startup if no model can be loaded to prevent silent failures.
- Routes:
  - `GET /` – renders `ui/index.html`, injecting `categories` for display.
  - `POST /predict` – accepts form data (`text`), validates non-empty input, runs `predict` and `predict_proba` when available, and returns JSON `{category, confidence, all_probs}`. Falls back to vectorizer+model path if direct pipeline inference fails.
- Error handling:
  - `HTTPException 400` for empty text.
  - `HTTPException 500` with logged stack traces for inference failures.

### 8. Web UI (`ui/index.html`)

- Minimal HTML/CSS/JS page rendered via Jinja2.
- Textarea for article input, submit button tied to Fetch request to `/predict`.
- Displays predicted category and probability bars for all classes returned by the API.
- Includes a sample sentence pre-filled in the textarea for quick testing.
- Can be customized (styles, branding, additional metadata) without changing backend logic.

### 9. Running the Project

1. **Install dependencies**
   - `poetry install` (preferred) or `pip install -r requirements.txt`.
2. **(Optional) Launch Jupyter for EDA**
   - `jupyter notebook notebooks/eda_bbc.ipynb`.
   - Export/update `notebooks/data/bbc_sample.csv`.
3. **Train/refresh the model**
   - `python train.py`.
   - Verify console output for accuracy and artifact save paths.
4. **Start the API**
   - `uvicorn api.main:app --reload`.
   - Visit `http://localhost:8000` for the UI.
5. **Programmatic access**
   - `POST http://localhost:8000/predict` with form data `text=...`.

### 10. Extending the System

- Swap classifiers (e.g., `LinearSVC`, `RandomForest`) inside `train.py`; ensure the final estimator exposes `predict` (and optionally `predict_proba`).
- Adjust TF-IDF settings (n-grams, vocabulary size) or add preprocessing steps inside the Pipeline.
- Add CLI arguments/env variables to `train.py` for data path, test split size, hyperparameters, or artifact output directory.
- Enhance the API with extra endpoints (`/health`, `/labels`) or add a JSON-based request/response format alongside form-based submissions.
- Containerize with Docker by wrapping training/inference commands; include artifact volume mounts.

### 11. Key Considerations & Best Practices

- Keep dataset CSVs small enough for local experimentation; for full BBC datasets, ensure column names match auto-detect patterns or manually rename before training.
- When deploying, copy the `notebooks/model_artifacts/` directory to a predictable location (or set up symlinks) so the API can discover artifacts without modifying code.
- Re-run `train.py` whenever data changes; the API does not retrain automatically.
- Monitor class balance via the notebook or training logs; logistic regression assumes moderately balanced classes for best performance.

### 12. Troubleshooting

- **Model not found on API start:** ensure `train.py` has been run and artifacts exist under one of the candidate directories.
- **Mismatched column names:** the auto-rename logic depends on keywords; rename columns manually in the CSV or notebook if detection fails.
- **No probability output:** some estimators lack `predict_proba`; either switch to one that supports it or handle the API response without `confidence/all_probs`.
- **Dependency issues:** sync the environment using `poetry lock` / `uv.lock`, or reinstall via `pip install -r requirements.txt`.

### 13. Reference Commands

```
# install
poetry install

# run notebook
jupyter notebook notebooks/eda_bbc.ipynb

# train model
python train.py

# serve API
uvicorn api.main:app --reload

# sample prediction via curl
curl -X POST -F "text=Markets rallied today..." http://localhost:8000/predict
```

---

Maintainer tip: keep this document in sync whenever you add new scripts, change artifact paths, or extend the API/UI so newcomers can ramp up quickly.
