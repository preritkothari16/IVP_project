from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .routes import preprocess, classify, dataset

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="IVP Plant Disease Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(preprocess.router)
app.include_router(classify.router)
app.include_router(dataset.router)

app.mount("/static/preprocessed", StaticFiles(directory=str(ROOT / "preprocessed")), name="preprocessed")
app.mount("/static/union_dataset", StaticFiles(directory=str(ROOT / "union_dataset")), name="union_dataset")
app.mount("/static/frontend", StaticFiles(directory=str(ROOT / "frontend")), name="frontend")

@app.get("/")
def root():
    return FileResponse(str(ROOT / "frontend" / "index.html"))

@app.get("/style.css")
def serve_css():
    return FileResponse(str(ROOT / "frontend" / "style.css"), media_type="text/css")

@app.get("/app.js")
def serve_js():
    return FileResponse(str(ROOT / "frontend" / "app.js"), media_type="application/javascript")

@app.get("/health")
def health():
    return {"status": "ok"}
