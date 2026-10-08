import io
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .config import CLASS_NAMES, CONFIDENCE_THRESHOLD, PROJECT_ROOT
from .model_loader import TRANSFORM, get_model

app = FastAPI(
    title="FruitVision",
    description=(
        "Image classification API for Apple, Durian, and Banana images "
        "using a custom MiniResNet model."
    ),
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "app" / "static"), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(PROJECT_ROOT / "app" / "static" / "index.html")


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except (FileNotFoundError, RuntimeError) as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    try:
        model = get_model()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    tensor = TRANSFORM(image).unsqueeze(0)

    start = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        confidence, predicted_index = probabilities.max(dim=0)

    confidence = float(confidence)
    predicted_index = int(predicted_index)
    label = CLASS_NAMES[predicted_index] if confidence >= CONFIDENCE_THRESHOLD else "unknown"
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 1),
        "latency_ms": round(latency_ms, 1),
        "threshold": CONFIDENCE_THRESHOLD,
    }
