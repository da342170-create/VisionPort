"""Check VisionPort's environment and save the results to env_report.md."""

import importlib
from importlib.metadata import PackageNotFoundError, version
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.xml"
REPORT_PATH = PROJECT_DIR / "env_report.md"
PACKAGES = (
    "torch",
    "torchvision",
    "ultralytics",
    "numpy",
    "Pillow",
    "fastapi",
    "uvicorn",
    "streamlit",
    "pandas",
    "openvino",
)

rows = []


def log(key, value):
    rows.append((key, str(value)))
    print(f"{key:30}{value}")


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def main():
    log("Date", datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"))
    log("Operating system", platform.platform())
    log("Python version", sys.version.split()[0])
    log("Python interpreter", sys.executable)

    for package in PACKAGES:
        log(f"Package: {package}", package_version(package))

    if MODEL_PATH.is_file():
        log("OpenVINO model", f"available: {MODEL_PATH}")
    else:
        log("OpenVINO model", f"MISSING: {MODEL_PATH}")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            driver = result.stdout.strip().splitlines()
            log("NVIDIA driver", ", ".join(driver) if driver else "unavailable")
        except (OSError, subprocess.SubprocessError) as error:
            log("NVIDIA driver", f"check failed: {error}")
    else:
        log("NVIDIA driver", "nvidia-smi not found (NVIDIA GPU not detected)")

    try:
        torch = importlib.import_module("torch")
        log("PyTorch CUDA build", torch.version.cuda or "None (CPU-only build)")
        log("torch.cuda.is_available()", torch.cuda.is_available())
        if torch.backends.cudnn.is_available():
            log("cuDNN version", torch.backends.cudnn.version())
        else:
            log("cuDNN version", "not available")

        if torch.cuda.is_available():
            log("GPU count", torch.cuda.device_count())
            for index in range(torch.cuda.device_count()):
                gpu = torch.cuda.get_device_properties(index)
                log(
                    f"GPU {index}",
                    f"{gpu.name}, {gpu.total_memory / 1e9:.1f} GB",
                )
            tensor = torch.randn(1000, 1000, device="cuda")
            result = tensor @ tensor
            torch.cuda.synchronize()
            log("GPU matrix test", f"PASS (result on {result.device})")
        else:
            log("GPU matrix test", "SKIPPED (CUDA unavailable)")
    except Exception as error:
        log("PyTorch / CUDA", f"FAILED to import or run: {type(error).__name__}: {error}")

    try:
        openvino = importlib.import_module("openvino")
        devices = openvino.Core().available_devices
        log("OpenVINO devices", ", ".join(devices))
    except Exception as error:
        log("OpenVINO devices", f"FAILED: {type(error).__name__}: {error}")

    lines = [
        "# VisionPort Environment Report",
        "",
        "| Item | Value |",
        "|---|---|",
    ]
    lines.extend(
        f"| {key} | {value.replace('|', '\\|').replace(chr(10), '<br>')} |"
        for key, value in rows
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
