
import io
import time
from pathlib import Path
from collections import Counter, defaultdict

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    Query
)

from PIL import Image, UnidentifiedImageError
import numpy as np

from detector import VisionPortDetector


# -----------------------------------------
# 1. Initialize FastAPI
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "yolov8n.xml"

app = FastAPI(
    title="VisionPort Gateway",
    description="YOLOv8n OpenVINO Traffic Detection API",
    version="0.2"
)


# -----------------------------------------
# 2. Load OpenVINO detector
# -----------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"OpenVINO model not found: {MODEL_PATH}"
    )

detector = VisionPortDetector(str(MODEL_PATH))


# -----------------------------------------
# 3. Home endpoint
# -----------------------------------------

@app.get("/")
def home():

    return {
        "project": "VisionPort",
        "message": "VisionPort OpenVINO Gateway is running",
        "status": "success",
        "device": detector.device,
        "available_devices": detector.core.available_devices,
        "model": "YOLOv8n",
        "inference_engine": "OpenVINO",
        "available_endpoints": [
            "/detect",
            "/trace",
            "/docs"
        ]
    }


# -----------------------------------------
# 4. Object detection endpoint
# -----------------------------------------

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.35, ge=0.01, le=1.0),
    iou: float = Query(0.50, ge=0.01, le=1.0)
):

    # Validate file format
    allowed_types = (
        "image/jpeg",
        "image/png"
    )

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=415,
            detail="Send a JPEG or PNG image."
        )

    # Read and decode image
    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        image_array = np.array(image)

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        raise HTTPException(
            status_code=400,
            detail="Could not decode image."
        )

    finally:
        await file.close()

    # Run OpenVINO detection
    try:

        result = detector.predict(
            image_array,
            conf=conf,
            iou=iou
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(error)}"
        )

    detections = result["detections"]

    # Object counts
    object_counts = dict(
        Counter(
            item["class"] for item in detections
        )
    )

    # Average confidence by object class
    confidence_values = defaultdict(list)

    for item in detections:

        confidence_values[item["class"]].append(
            item["confidence"]
        )

    average_confidence = {
        name: round(
            sum(values) / len(values), 4
        )
        for name, values in confidence_values.items()
    }

    # API response
    return {

        "project": "VisionPort",

        "model": "YOLOv8n",

        "inference_engine": "OpenVINO",

        "device": result["device"],

        "image_size": {
            "width": image.width,
            "height": image.height
        },

        "num_detections": result["num_detections"],

        "object_counts": object_counts,

        "average_confidence": average_confidence,

        "detections": detections,

        "timing_ms": result["timing_ms"]

    }


@app.post("/trace")
async def trace(
    file: UploadFile = File(...),
    conf: float = Query(0.35, ge=0.01, le=1.0),
    iou: float = Query(0.50, ge=0.01, le=1.0)
):
    if file.content_type not in ("image/jpeg", "image/png"):
        await file.close()
        raise HTTPException(
            status_code=415,
            detail="Send a JPEG or PNG image."
        )

    stages = []

    def add_stage(name, where, data_format, shape, start, note=""):
        stages.append({
            "stage": len(stages) + 1,
            "name": name,
            "where": where,
            "format": data_format,
            "shape": shape,
            "note": note,
            "ms": round((time.perf_counter() - start) * 1000, 2)
        })

    total_start = time.perf_counter()

    try:
        start = time.perf_counter()
        contents = await file.read()
        add_stage(
            "receive upload",
            "CPU / gateway",
            "bytes",
            f"{len(contents)} bytes",
            start
        )

        start = time.perf_counter()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)
        original_size = list(image_array.shape[:2])
        add_stage(
            "decode image",
            "CPU / gateway",
            "numpy uint8 (H,W,C)",
            list(image_array.shape),
            start
        )
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Could not decode image."
        )
    finally:
        await file.close()

    try:
        result = detector.predict(
            image_array,
            conf=conf,
            iou=iou,
            trace_stages=stages
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Trace inference failed: {str(error)}"
        )

    start = time.perf_counter()
    add_stage(
        "prepare JSON response",
        "CPU / gateway",
        "JSON",
        f"{result['num_detections']} detections",
        start,
        "FastAPI encodes the response after this handler returns"
    )

    return {
        "device": result["device"],
        "original_size_hw": original_size,
        "stages": stages,
        "total_ms": round((time.perf_counter() - total_start) * 1000, 2),
        "detections": result["detections"]
    }
