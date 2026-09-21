"""FastAPI prediction API for Hierarchical COICOP Classifier."""

from __future__ import annotations

import argparse
import logging
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_STATIC_DIR = Path(__file__).parent / "static"

from .predict import HierarchicalCOICOPPredictor, MultiHeadCOICOPPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "checkpoints/hierarchical/hierarchical_model"


class PredictRequest(BaseModel):
    text: str
    return_all_levels: bool = True
    top_k: int = Field(default=1, ge=1, le=10)


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(..., max_length=1024)
    return_all_levels: bool = True
    batch_size: int = 64
    top_k: int = Field(default=1, ge=1, le=10)


class LevelAlternative(BaseModel):
    code: str
    confidence: float


class LevelPrediction(BaseModel):
    code: str
    confidence: float
    alternatives: list[LevelAlternative] | None = None


class PredictionResult(BaseModel):
    text: str
    code: str
    final_level: str
    confidence: float
    levels: dict[str, LevelPrediction] | None = None


class PredictResponse(BaseModel):
    prediction: PredictionResult


class PredictBatchResponse(BaseModel):
    predictions: list[PredictionResult]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


class ModelInfoResponse(BaseModel):
    model_path: str
    trained_levels: list[str]
    level_class_counts: dict[str, int]


# ---------------------------------------------------------------------------
# App lifespan – load model at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("COICOP_MODEL_PATH", DEFAULT_MODEL_PATH)
    logger.info("Loading model from %s ...", model_path)

    # Detect model type
    model_dir = Path(model_path)
    if (model_dir / "multihead_metadata.pkl").exists():
        logger.info("Detected multi-head model")
        predictor = MultiHeadCOICOPPredictor(model_path)
    else:
        predictor = HierarchicalCOICOPPredictor(model_path)

    app.state.predictor = predictor
    logger.info("Model loaded successfully.")
    yield


app = FastAPI(
    title="COICOP Classifier API",
    description="Hierarchical COICOP product classification",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_predictor(request: Request) -> HierarchicalCOICOPPredictor:
    predictor: HierarchicalCOICOPPredictor | None = getattr(
        request.app.state, "predictor", None
    )
    if predictor is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


def _to_prediction_result(pred: dict) -> PredictionResult:
    levels = None
    if "levels" in pred:
        levels = {}
        for name, data in pred["levels"].items():
            alts = None
            if "alternatives" in data:
                alts = [LevelAlternative(code=a["code"], confidence=a["confidence"]) for a in data["alternatives"]]
            levels[name] = LevelPrediction(code=data["code"], confidence=data["confidence"], alternatives=alts)
    return PredictionResult(
        text=pred["text"],
        code=pred["code"],
        final_level=pred["final_level"],
        confidence=pred["confidence"],
        levels=levels,
    )


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception during request:\n%s", traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    model_path = os.environ.get("COICOP_MODEL_PATH", DEFAULT_MODEL_PATH)
    return HealthResponse(
        status="ok" if predictor is not None else "unavailable",
        model_loaded=predictor is not None,
        model_path=model_path,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request):
    predictor = _get_predictor(request)
    results = predictor.predict(
        [body.text], return_all_levels=body.return_all_levels, top_k=body.top_k
    )
    return PredictResponse(prediction=_to_prediction_result(results[0]))


@app.post("/predict/batch", response_model=PredictBatchResponse)
async def predict_batch(body: PredictBatchRequest, request: Request):
    predictor = _get_predictor(request)
    results = predictor.predict_batch(
        body.texts,
        batch_size=body.batch_size,
        return_all_levels=body.return_all_levels,
        top_k=body.top_k,
    )
    predictions = [_to_prediction_result(r) for r in results]
    return PredictBatchResponse(predictions=predictions, count=len(predictions))


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request):
    predictor = _get_predictor(request)
    classifier = predictor.classifier

    if isinstance(predictor, MultiHeadCOICOPPredictor):
        trained_levels = list(classifier.level_names)
    else:
        trained_levels = list(classifier.level_classifiers.keys())

    level_class_counts = {
        level: len(labels) for level, labels in classifier.level_label_names.items()
    }
    return ModelInfoResponse(
        model_path=str(predictor.model_path),
        trained_levels=trained_levels,
        level_class_counts=level_class_counts,
    )


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="COICOP Classifier API server")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to hierarchical model",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    args = parser.parse_args()

    os.environ["COICOP_MODEL_PATH"] = args.model
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)
