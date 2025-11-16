"""
api/main.py

A clearer, commented version of your FastAPI app that:
- auto-discovers model artifact directories
- loads labels saved as .json or pickled formats (list, dict with "categories", tuple, etc.)
- loads either a Pipeline (that handles raw text) or a separate vectorizer + model
- exposes a simple "/" page (renders templates/index.html) and a POST /predict endpoint

Notes:
- Uses pathlib.Path for safer path handling
- Uses logging instead of bare prints
- Keeps backwards-compatible fallback behavior (joblib -> pickle)
"""

from pathlib import Path
import json
import logging
import joblib
import pickle
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# -----------------------------
# Basic configuration
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bbc-text-classifier")

app = FastAPI(title="BBC Text Classifier API")

# -----------------------------
# Directories & candidate paths
# -----------------------------
# BASE is the project root (one level up from this file)
BASE = Path(__file__).resolve().parent.parent

# Candidate locations where model artifacts might live.
# Ordered by preference; keeps the original intent of supporting different layouts.
CANDIDATE_DIRS = [
    BASE / "model_artifacts",
    BASE / "notebooks" / "model_artifacts",
    BASE / "notebooks",
]

# Template directory: ../ui relative to this file (adjust if your UI is elsewhere)
TEMPLATES_DIR = BASE / "ui"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Filenames we will search for inside the discovered model directory
MODEL_FILENAMES = ["model.pkl", "models.pkl", "pipeline.pkl"]
VECT_FILENAME = "vectorizer.pkl"
LABEL_FILENAMES = ["labels.json", "labels.pkl"]

# -----------------------------
# Helper functions
# -----------------------------


def find_artifact_dir(candidates: List[Path]) -> Path:
    """
    Return the first candidate directory that exists and contains at least one
    expected artifact (model, pipeline, vectorizer, or labels). If none found,
    return BASE / "model_artifacts" as the fallback location.
    """
    for d in candidates:
        try:
            if not d.exists() or not d.is_dir():
                continue
            # If this directory contains any expected artifact file, accept it
            for name in MODEL_FILENAMES + [VECT_FILENAME] + LABEL_FILENAMES:
                if (d / name).exists():
                    logger.info("Artifact directory found: %s", d)
                    return d
        except Exception:
            # Best effort: ignore permission errors or weird FS issues and keep checking
            continue

    fallback = BASE / "model_artifacts"
    logger.info(
        "No candidate artifact dir matched. Falling back to: %s", fallback)
    return fallback


def safe_joblib_load(p: Path) -> Any:
    """
    Try joblib.load, fall back to pickle.load if joblib fails.
    Raise the original exception if both fail.
    """
    try:
        return joblib.load(p)
    except Exception as jb_err:
        logger.debug("joblib.load failed for %s: %s", p, jb_err)
        # fallback to pickle
        try:
            with p.open("rb") as fh:
                return pickle.load(fh)
        except Exception as pk_err:
            logger.debug("pickle.load also failed for %s: %s", p, pk_err)
            # Re-raise the first failure for clarity
            raise jb_err


def load_labels(label_paths: List[Path]) -> Optional[List[str]]:
    """
    Attempt to load labels from given paths. Handles:
      - labels.json containing a list or {"categories": [...]}
      - pickled list/tuple
      - single object coerced to a list of str
    Returns a list of label strings or None if none found/loaded.
    """
    for lp in label_paths:
        if not lp.exists():
            continue
        try:
            if lp.suffix.lower() == ".json":
                with lp.open("r", encoding="utf-8") as fh:
                    obj = json.load(fh)
            else:
                obj = safe_joblib_load(lp)

            # Normalise to a list of strings
            if isinstance(obj, list):
                labels = [str(x) for x in obj]
                logger.info("Loaded labels (list) from %s", lp)
                return labels
            if isinstance(obj, dict) and "categories" in obj:
                labels = [str(x) for x in obj["categories"]]
                logger.info("Loaded labels (dict['categories']) from %s", lp)
                return labels
            if isinstance(obj, (tuple, set)):
                labels = [str(x) for x in obj]
                logger.info("Loaded labels (tuple/set) from %s", lp)
                return labels

            # Last-resort: coerce single object into a one-element list
            labels = [str(obj)]
            logger.info("Loaded labels (coerced single object) from %s", lp)
            return labels

        except Exception as e:
            # If one label file fails to load, try the next candidate
            logger.warning("Failed to load labels from %s: %s", lp, e)
            continue

    logger.info("No labels loaded from any candidate paths.")
    return None


def load_model_from_candidates(model_paths: List[Path]) -> Optional[Any]:
    """
    Attempt to load a model/pipeline from the provided model_paths list.
    Returns the loaded model or None if none loaded.
    """
    for mp in model_paths:
        if not mp.exists():
            continue
        try:
            m = safe_joblib_load(mp)
            logger.info("Loaded model/pipeline from %s", mp)
            return m
        except Exception as e:
            logger.warning("Failed to load model from %s: %s", mp, e)
            continue
    return None


# -----------------------------
# Discover and load artifacts
# -----------------------------
MODEL_DIR = find_artifact_dir(CANDIDATE_DIRS)

# Paths to try
VECT_PATH = MODEL_DIR / VECT_FILENAME
MODEL_PATHS = [MODEL_DIR / name for name in MODEL_FILENAMES]
LABEL_PATHS = [MODEL_DIR / name for name in LABEL_FILENAMES]

# Load labels (if any)
labels: Optional[List[str]] = load_labels(LABEL_PATHS)

# Load model/pipeline
model = load_model_from_candidates(MODEL_PATHS)
vectorizer = None  # will be set if we find one and need it

# If a single model didn't load, try vectorizer + model.pkl combination
if model is None and VECT_PATH.exists() and (MODEL_DIR / "model.pkl").exists():
    try:
        vectorizer = safe_joblib_load(VECT_PATH)
        model = safe_joblib_load(MODEL_DIR / "model.pkl")
        logger.info("Loaded separate vectorizer and model from %s", MODEL_DIR)
    except Exception as e:
        logger.exception("Failed to load vectorizer+model fallback: %s", e)
        model = None

# If still no model, error out loudly (so the app doesn't start silently broken)
if model is None:
    msg = (
        f"Could not locate or load a model in {MODEL_DIR}. "
        f"Expected one of: {', '.join(str(p.name) for p in MODEL_PATHS)} "
        f"or vectorizer+model.pkl."
    )
    logger.error(msg)
    # Raise here so uvicorn exits with an error rather than running a broken API.
    raise FileNotFoundError(msg)

logger.info("Model directory used: %s", MODEL_DIR)
logger.info("Labels loaded: %s", labels)


# -----------------------------
# FastAPI routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the index.html template. The template can use `categories` to show
    available labels (if labels were loaded).
    """
    return templates.TemplateResponse(
        "index.html", {"request": request, "categories": labels or []}
    )


def build_scores_dict(probs: List[float], labels: Optional[List[str]]) -> Dict[str, float]:
    """
    Convert a list/array of probabilities to a dictionary mapping label->prob.
    If labels is None or length mismatch, use string indices as keys.
    """
    if labels and len(labels) == len(probs):
        return {labels[i]: float(probs[i]) for i in range(len(probs))}
    else:
        return {str(i): float(probs[i]) for i in range(len(probs))}


@app.post("/predict")
async def predict(text: str = Form(...)):
    """
    Accept `text` via an HTML form and return the predicted category.
    Behavior:
    - If the loaded model/pipeline supports predict_proba, return category, confidence, and all_probs.
    - If not, return only the category.
    - If pipeline-like prediction fails and we have a vectorizer, try vectorizer.transform + model.predict(_proba).
    """
    # Basic validation: non-empty text required
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    # Try to use the model directly (works when model is a Pipeline that handles raw text)
    try:
        if hasattr(model, "predict_proba"):
            # returns array-like of probabilities
            probs = model.predict_proba([text])[0]
            pred = model.predict([text])[0]
            scores = build_scores_dict(probs, labels)
            confidence = float(max(probs))
            return {"category": str(pred), "confidence": confidence, "all_probs": scores}
        else:
            # Model can't compute probabilities but can predict
            pred = model.predict([text])[0]
            return {"category": str(pred)}
    except Exception as primary_err:
        # If model usage fails, but we have a separate vectorizer available, try vectorizer->model flow
        logger.warning("Direct model prediction failed: %s", primary_err)
        if vectorizer is not None:
            try:
                X = vectorizer.transform([text])
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    pred = model.predict(X)[0]
                    scores = build_scores_dict(probs, labels)
                    return {"category": str(pred), "all_probs": scores}
                else:
                    pred = model.predict(X)[0]
                    return {"category": str(pred)}
            except Exception as secondary_err:
                logger.exception(
                    "Vectorizer+model prediction also failed: %s", secondary_err)
                raise HTTPException(status_code=500, detail=str(secondary_err))

        # If no vectorizer fallback available, return a 500 with the original error message.
        logger.exception(
            "Prediction failed and no fallback vectorizer available.")
        raise HTTPException(status_code=500, detail=str(primary_err))


# -----------------------------
# Optional CLI runner for dev
# -----------------------------
if __name__ == "__main__":
    # Run with: python api/main.py (dev only). For production use uvicorn/gunicorn.
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
