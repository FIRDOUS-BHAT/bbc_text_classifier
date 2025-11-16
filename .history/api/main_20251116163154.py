# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
import json

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")
MODEL_DIR = os.path.abspath(MODEL_DIR)

# Request/response models


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str
    scores: dict = None  # optionally include class probabilities or decision function


app = FastAPI(title="BBC Text Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local testing. Lock this down in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts on startup
vectorizer = None
model = None
labels = None


@app.on_event("startup")
def load_models():
    global vectorizer, model, labels
    vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    labels_path = os.path.join(MODEL_DIR, "labels.json")
    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        raise RuntimeError(
            "Model artifacts not found. Run train.py first and ensure artifacts are in model_artifacts/")
    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    if os.path.exists(labels_path):
        with open(labels_path, "r") as fh:
            labels = json.load(fh)
    else:
        labels = None
    print("Models loaded.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    X = vectorizer.transform([text])
    # If classifier has predict_proba, include top probabilities
    try:
        probs = None
        if hasattr(model, "predict_proba"):
            prob_arr = model.predict_proba(X)[0]
            if labels:
                scores = {str(lbl): float(prob_arr[i])
                          for i, lbl in enumerate(labels)}
            else:
                scores = {str(i): float(prob_arr[i])
                          for i in range(len(prob_arr))}
            pred = model.predict(X)[0]
            return {"prediction": str(pred), "scores": scores}
        else:
            # For LinearSVC or Logistic without prob, return decision function as scores
            if hasattr(model, "decision_function"):
                df = model.decision_function(X)
                # df can be 1d or 2d
                if df.ndim == 1:
                    # binary
                    scores = {"score": float(df[0])}
                else:
                    if labels:
                        scores = {str(lbl): float(df[0][i])
                                  for i, lbl in enumerate(labels)}
                    else:
                        scores = {str(i): float(df[0][i])
                                  for i in range(df.shape[1])}
            else:
                scores = None
            pred = model.predict(X)[0]
            return {"prediction": str(pred), "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
