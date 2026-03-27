from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from api.model import get_classifier
import time

app = FastAPI(
    title="Sarcasm Detector API",
    description="Detects sarcasm in English text. Fine-tuned DistilBERT trained on 120k+ samples from Reddit, Twitter & news headlines.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LABEL_MAP = {"LABEL_0": "Not Sarcastic", "LABEL_1": "Sarcastic"}


# ── Schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(
        ..., min_length=3, max_length=512,
        example="Oh great, another Monday. Just what I needed."
    )
    threshold: Optional[float] = Field(
        default=0.65, ge=0.5, le=1.0,
        description="Confidence below this returns 'Uncertain'"
    )

class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    is_uncertain: bool
    latency_ms: float
    model_version: str = "distilbert-sarcasm-v1"

class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=100)
    threshold: Optional[float] = Field(default=0.65, ge=0.5, le=1.0)


# ── Routes ─────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "name": "Sarcasm Detector API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/predict", "/batch", "/health"],
    }


@app.get("/health", tags=["General"])
def health():
    clf = get_classifier()
    return {"status": "ok", "model_loaded": clf is not None}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    clf = get_classifier()
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0  = time.perf_counter()
    out = clf(req.text[:512])[0]
    ms  = round((time.perf_counter() - t0) * 1000, 2)

    label = LABEL_MAP.get(out["label"], out["label"])
    conf  = round(float(out["score"]), 4)

    return PredictResponse(
        text         = req.text,
        label        = label if conf >= req.threshold else "Uncertain",
        confidence   = conf,
        is_uncertain = conf < req.threshold,
        latency_ms   = ms,
    )


@app.post("/batch", tags=["Inference"])
def batch_predict(req: BatchRequest):
    clf = get_classifier()
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0      = time.perf_counter()
    results = []

    for text in req.texts:
        out   = clf(text[:512])[0]
        label = LABEL_MAP.get(out["label"], out["label"])
        conf  = round(float(out["score"]), 4)
        results.append({
            "text":         text[:100] + "..." if len(text) > 100 else text,
            "label":        label if conf >= req.threshold else "Uncertain",
            "confidence":   conf,
            "is_uncertain": conf < req.threshold,
        })

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {"results": results, "count": len(results), "total_latency_ms": total_ms}
