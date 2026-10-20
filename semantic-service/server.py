"""
Phase 2 serving layer: exposes the trained calibrated conflict-probability
model over HTTP, matching the contract cc-engine/predictor.go's
HTTPPredictor already expects (POST /classify -> {conflict_probability,
confidence}).

This is the "LLM semantic intent classifier" referenced throughout the
project's design: right now `extract` (in train.py) reads the transaction's
natural-language description with TF-IDF, as a locally-substituted stand-in
for an actual LLM call (see train.py's module docstring and the README).
Swapping in a real LLM means replacing feature extraction; this server and
everything in cc-engine stays the same, since the contract is just
"text in, calibrated probability out."
"""
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Required even though unused directly here: the pickled Pipeline's
# FunctionTransformer references "features.select_numeric" by qualified
# name, so that module must be imported into this process before
# pickle.load can resolve it (see features.py's docstring).
from features import select_numeric  # noqa: F401

MODEL_PATH = Path(__file__).parent / "model.pkl"

app = FastAPI(title="SAPCC semantic-service")
_model = None


def get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


class ClassifyRequest(BaseModel):
    txn_id: str
    type: str
    text: str
    keys: list[str]
    writes: list[bool]


class ClassifyResponse(BaseModel):
    conflict_probability: float
    confidence: float


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    model = get_model()
    row = pd.DataFrame([{
        "text": req.text,
        "num_keys": len(req.keys),
        "num_writes": sum(req.writes),
    }])
    p = float(model.predict_proba(row)[0, 1])
    # Confidence: distance from 0.5, rescaled to [0,1] -- how decisively the
    # calibrated model is leaning either way, not the conflict probability itself.
    confidence = abs(p - 0.5) * 2
    return ClassifyResponse(conflict_probability=p, confidence=confidence)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}
