# VisionPort Environment Report

Generated: 2026-11-13T10:25:55+05:30

## System
- Operating system: Windows-11-10.0.26300-SP0
- Architecture: AMD64
- Python: 3.13.15
- Interpreter: `C:\Y25-PBL-DL\Visionport\.venv\Scripts\python.exe`

## Required packages
| Package | Version | Status |
|---|---:|---|
| fastapi | 0.162.4 | OK |
| uvicorn | 0.54.0 | OK |
| python-multipart | 0.0.33 | OK |
| Pillow | 13.0.0 | OK |
| numpy | 2.5.4 | OK |
| opencv-python | 5.0.0.93 | OK |
| openvino | 2026.4.1 | OK |

## Model
- Path: `C:\Y25-PBL-DL\Visionport\models\yolov8n.xml`
- Status: Available

## OpenVINO
- Version: 2026.4.1-22982-e213a147257-releases/2026/4
- Available devices: CPU, GPU
- Note: Available devices are detected by OpenVINO; the API reports the selected inference device when it starts.

## Notes
- This VisionPort setup uses OpenVINO. PyTorch, CUDA, and `nvidia-smi` are not required.

## Check result
PASSED

Start the API with `python -m uvicorn serve.gateway:app --reload`.
