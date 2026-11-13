# VisionPort

VisionPort detects objects in traffic images using YOLOv8n and OpenVINO.

## Features

- FastAPI backend for image detection
- `/detect` returns detections as JSON
- `/trace` returns detection results and processing-stage details
- Optional Streamlit app displays images with detection boxes

## Project structure

- `app.py` — Streamlit user interface
- `detector.py` — image preprocessing and YOLO inference using OpenVINO
- `serve/gateway.py` — FastAPI endpoints
- `models/` — OpenVINO model files
- `env_check.py` — checks the environment and creates `env_report.md`
- `env_report.md` — environment check results

## Requirements

Python and the packages listed in `requirements.txt` are required. The OpenVINO model files must be available in `models/`.

Install packages with:

```powershell
python -m pip install -r requirements.txt
