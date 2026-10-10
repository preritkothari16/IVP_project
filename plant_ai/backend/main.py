"""
Strawberry disease classifier - FastAPI backend.

Serves a YOLOv5 classification model (ultralytics/yolov5, classify mode) trained on
9 strawberry classes. Load the weights once at startup and fail loudly (503) if the
weights file is missing, so a placeholder can never silently masquerade as a model.
"""

import io
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Cap BLAS/OpenMP threads before torch is imported; see plant_ai/threadcaps.py. Without this the
# server can fail to start with WinError 1455 on a busy machine, which is a memory-commit
# failure rather than a torch problem.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import threadcaps  # noqa: F401
except Exception:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "2")

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
YOLOV5_DIR = PROJECT_DIR / "yolov5"

WEIGHTS = Path(os.environ.get("STRAWBERRY_WEIGHTS", BACKEND_DIR / "best.pt"))
DISEASE_INFO_PATH = BACKEND_DIR / "disease_info.json"
LOW_SUPPORT_PATH = BACKEND_DIR / "low_support.json"

IMG_SIZE = 224
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
LOW_CONFIDENCE_THRESHOLD = 0.6

sys.path.append(str(YOLOV5_DIR))
from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import classify_transforms  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


class ModelUnavailable(RuntimeError):
    """Raised when the trained weights are absent or unreadable."""


class Predictor:
    """Holds the YOLOv5 classification model, its class names, and the transform."""

    def __init__(self) -> None:
        self._model = None
        self._names: List[str] = []
        self._transform = None
        self._device = None

    def load(self) -> None:
        if not YOLOV5_DIR.is_dir():
            raise ModelUnavailable(f"YOLOv5 repo not found at {YOLOV5_DIR}")
        if not WEIGHTS.is_file():
            raise ModelUnavailable(
                f"Trained weights not found at {WEIGHTS}. Train the classifier first, "
                "then point STRAWBERRY_WEIGHTS at the resulting best.pt."
            )
        self._device = select_device("")
        self._model = DetectMultiBackend(str(WEIGHTS), device=self._device, fuse=False)
        # yolov5 stores ClassificationModel.names as {index: name}; normalise to an ordered list.
        raw = self._model.names
        self._names = [str(raw[i]) for i in sorted(raw)] if isinstance(raw, dict) else [str(n) for n in raw]
        self._transform = classify_transforms(IMG_SIZE)
        self._model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))
        print(f"[predictor] loaded {WEIGHTS} on {self._device} with classes={self._names}")

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def names(self) -> List[str]:
        return self._names

    def predict(self, image: Image.Image) -> Dict[str, float]:
        # classify_transforms expects an HWC numpy array (it resizes with cv2 internally).
        tensor = self._transform(np.asarray(image.convert("RGB"))).unsqueeze(0).to(self._device)
        tensor = tensor.float() if not self._model.fp16 else tensor.half()
        with torch.inference_mode():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        return {name: float(p) for name, p in zip(self._names, probs.tolist())}


PREDICTOR = Predictor()


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


DISEASE_INFO: Dict[str, Any] = load_json(DISEASE_INFO_PATH, {})
LOW_SUPPORT: set = set(load_json(LOW_SUPPORT_PATH, []))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        PREDICTOR.load()
    except ModelUnavailable as exc:
        print(f"[predictor] NOT READY: {exc}")
    yield


app = FastAPI(title="Strawberry Disease Detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiseaseInfo(BaseModel):
    display_name: str
    summary: str
    cause: str
    symptoms: List[str]
    treatment: List[str]
    prevention: List[str]
    severity: str
    spread: str
    uncertainty: Optional[str] = None
    verified: bool = False
    # UI grouping: powdery mildew on leaf and on fruit are one disease (Podosphaera aphanis),
    # so the frontend renders them as "Powdery Mildew - detected on <organ>".
    disease_group: Optional[str] = None
    organ: Optional[str] = None
    group_display_name: Optional[str] = None
    group_note: Optional[str] = None


class Alternative(BaseModel):
    class_name: str
    confidence: float


class PredictResponse(BaseModel):
    top_class: str
    confidence: float
    top3: List[Alternative]
    disease_info: Optional[DiseaseInfo]
    low_confidence: bool
    low_support: bool
    message: Optional[str] = None


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": PREDICTOR.ready,
        "weights": str(WEIGHTS),
        "weights_present": WEIGHTS.is_file(),
        "num_classes": len(PREDICTOR.names),
        "classes": PREDICTOR.names,
        "disease_info_entries": sum(1 for k in DISEASE_INFO if not k.startswith("_")),
        "low_support_classes": sorted(LOW_SUPPORT),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
    }


@app.get("/classes")
async def classes() -> Dict[str, Any]:
    names = PREDICTOR.names or sorted(k for k in DISEASE_INFO if not k.startswith("_"))
    return {
        "count": len(names),
        "classes": names,
        "low_support": sorted(LOW_SUPPORT),
        "detects": names,
        "does_not_detect": [
            "spider mites",
            "nutrient deficiencies",
            "non-strawberry plants",
        ],
        "limitations": [
            "Strawberry only. The model was trained on 9 strawberry classes and has never seen "
            "another species, so it will confidently label a tomato or houseplant as one of them.",
            "Leaf and fruit powdery mildew are the same fungus (Podosphaera aphanis). The two "
            "classes are only 'which organ', and the model is not reliable at telling them apart.",
            "Grey mould and blossom blight are the same fungus (Botrytis cinerea) at different "
            "stages, and their photos overlap.",
            "Powdery mildew on fruit and grey mould are frequently confused, and the source dataset "
            "contains at least one photograph labelled as both.",
            "A photo of an out-of-distribution subject often produces a confident-looking answer. "
            "Always check that the confidence is high before acting on a diagnosis.",
        ],
    }


def build_disease_info(class_name: str) -> Optional[DiseaseInfo]:
    entry = DISEASE_INFO.get(class_name)
    if not entry:
        return None
    return DiseaseInfo(
        display_name=entry.get("display_name", class_name),
        summary=entry.get("summary", ""),
        cause=entry.get("cause", ""),
        symptoms=entry.get("symptoms", []),
        treatment=entry.get("treatment", []),
        prevention=entry.get("prevention", []),
        severity=entry.get("severity", "unknown"),
        spread=entry.get("spread", ""),
        uncertainty=entry.get("uncertainty"),
        verified=entry.get("verified", False),
        disease_group=entry.get("disease_group"),
        organ=entry.get("organ"),
        group_display_name=entry.get("group_display_name"),
        group_note=entry.get("group_note"),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if not PREDICTOR.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model weights not available at {WEIGHTS}. "
                "Train the classifier and set STRAWBERRY_WEIGHTS to the resulting best.pt."
            ),
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large ({len(content) / 1048576:.1f} MB). Maximum is 10 MB.",
        )

    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a readable image.")

    fmt = (image.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image format '{fmt or 'unknown'}'. Allowed: jpg, png, webp.",
        )
    if min(image.size) < 32:
        raise HTTPException(
            status_code=400,
            detail=f"Image is too small ({image.size[0]}x{image.size[1]}). Minimum is 32x32.",
        )

    scores = PREDICTOR.predict(image.convert("RGB"))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_class, top_conf = ranked[0]

    low_confidence = top_conf < LOW_CONFIDENCE_THRESHOLD
    low_support = top_class in LOW_SUPPORT

    message = None
    if low_confidence:
        message = (
            "Not sure about this one. Try a clearer, well-lit photo of a single "
            "strawberry leaf or fruit, filling the frame."
        )
    if low_support:
        message = (
            (message + " ") if message else ""
        ) + f"Note: the model is less reliable for '{top_class}' - it has few training images."

    return PredictResponse(
        top_class=top_class,
        confidence=round(top_conf, 4),
        top3=[Alternative(class_name=n, confidence=round(c, 4)) for n, c in ranked[:3]],
        disease_info=build_disease_info(top_class),
        low_confidence=low_confidence,
        low_support=low_support,
        message=message,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
