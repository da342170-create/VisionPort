# VisionPort

VisionPort is a prototype traffic-object detection application. It combines a
Streamlit image-upload interface, a FastAPI backend, and a YOLOv8n model
executed with OpenVINO. The detector prefers an available Intel GPU and falls
back to CPU.

> **Project status:** this repository contains the application, model/dataset
> inspection tools, and a small synthetic dataset for pipeline checks. The
> synthetic data is not the official IDD dataset, and the current YOLOv8n
> checkpoint is the standard COCO model, not a model fine-tuned on IDD.

## Features

- Upload JPEG or PNG traffic images through the Streamlit app.
- Adjust confidence and IoU thresholds.
- View detections, object counts, confidence scores, inference timing, and a
  downloadable image with bounding boxes.
- Use the FastAPI `/detect` endpoint for inference and `/trace` to inspect
  processing stages.
- Audit COCO-format annotations and visualize bounding boxes.
- Inspect PyTorch and ONNX model artefacts.

## Architecture

```text
Browser
  -> Streamlit frontend (app.py)
       -> FastAPI gateway (serve/gateway.py)
            -> VisionPortDetector (detector.py)
                 -> OpenVINO YOLOv8n model
```

The detector resizes and pads images to the model's 640 x 640 input, runs
inference, maps detections back to the original image, and applies class-wise
non-maximum suppression.

## Requirements

- Python (the current development environment uses Python 3.13)
- An OpenVINO-supported CPU or device
- OpenVINO IR model files at `models/yolov8n.xml` and `models/yolov8n.bin`

The model files are intentionally excluded from Git, as are files under
`data/`. A fresh clone therefore needs a model before starting the backend.

## Setup

From the project directory, create and activate a virtual environment, then
install dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Prepare the OpenVINO model

Export the YOLOv8n checkpoint to OpenVINO format:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='openvino', imgsz=640)"
```

Ultralytics creates an OpenVINO model directory. Copy the generated XML and BIN
files into `models` using the filenames expected by the backend:

```powershell
Copy-Item .\yolov8n_openvino_model\yolov8n.xml .\models\yolov8n.xml
Copy-Item .\yolov8n_openvino_model\yolov8n.bin .\models\yolov8n.bin
```

If the generated directory has a different name, use that directory in the
copy commands. The backend checks for `models/yolov8n.xml` when it starts.

## Run the application

Open two terminals in the project directory. Start the API in the first:

```powershell
python -m uvicorn serve.gateway:app --reload
```

Start the Streamlit frontend in the second:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The frontend defaults to the API at
`http://127.0.0.1:8000`.

FastAPI's interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Backend status, selected device, and available endpoints |
| `POST` | `/detect` | Detect objects in an uploaded JPEG or PNG image |
| `POST` | `/trace` | Return detections and inference pipeline stage timings |

`/detect` and `/trace` accept an image as multipart form field `file`, with
optional `conf` and `iou` query parameters. Both thresholds default to `0.35`
and `0.50`, respectively.

## Dataset tools

### Create the small synthetic sample

```powershell
python make_sample_idd.py
```

This creates three simple generated images and nine COCO-format annotations in
`data/`. It is only for checking the data pipeline; it is not real traffic data
or the official IDD dataset.

### Audit annotations and images

```powershell
python audit_dataset.py
```

The audit checks annotation references and bounding boxes, image availability
and dimensions, basic image quality, class distribution, and object sizes. It
writes `docs/audit_report.md` and `docs/audit_charts.png`. Thresholds in the
script are heuristics; review flagged images manually.

### Explore COCO annotations

```powershell
python explore_coco.py
```

This prints dataset summaries, checks basic annotation consistency, and saves a
bounding-box preview under `docs/`.

### Inspect model artefacts

```powershell
python inspect_artefacts.py
```

The inspector looks for `.pt`, `.onnx`, `.engine`, and `.plan` files. It
inspects the default `yolov8n.pt` checkpoint and an ONNX file if found. To
specify an ONNX file or export one from the checkpoint:

```powershell
python inspect_artefacts.py --onnx models/yolov8n.onnx
python inspect_artefacts.py --export-onnx
```

ONNX export requires the checkpoint and may download `yolov8n.pt` if it is not
already present. This script's ONNX export is separate from the OpenVINO model
preparation described above.

## Repository layout

```text
app.py                 Streamlit frontend
detector.py            OpenVINO inference and post-processing
serve/gateway.py       FastAPI endpoints
audit_dataset.py       COCO dataset audit and chart generation
explore_coco.py        Annotation summary and bounding-box preview
make_sample_idd.py     Synthetic sample data generator
inspect_artefacts.py   PyTorch, ONNX, and TensorRT artefact inspector
data/                  Local datasets (not committed)
models/                Local model artefacts (not committed)
docs/                  Audit report and generated visualizations
```

## Current sample audit

The report currently in `docs/audit_report.md` describes the synthetic sample:
3 images, 9 annotations, and 5 categories. It reports no annotation issues;
all 3 images are flagged by the script's blur heuristic. These results should
not be interpreted as an audit of the full IDD dataset.
