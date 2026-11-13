import importlib
from importlib.metadata import PackageNotFoundError, version
import platform
import sys
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.xml"
REPORT_PATH = PROJECT_DIR / "env_report.md"
REQUIRED_MODULES = (
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("python-multipart", "multipart"),
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("openvino", "openvino"),
)


def main():
    lines = [
        "# VisionPort Environment Report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## System",
        f"- Operating system: {platform.platform()}",
        f"- Architecture: {platform.machine()}",
        f"- Python: {sys.version.split()[0]}",
        f"- Interpreter: `{sys.executable}`",
        "",
        "## Required packages",
        "| Package | Version | Status |",
        "|---|---:|---|",
    ]
    missing = []

    for package, module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            package_version = version(package)
        except (ImportError, OSError, PackageNotFoundError) as error:
            missing.append(package)
            lines.append(f"| {package} | unavailable | FAIL: {error} |")
        else:
            lines.append(f"| {package} | {package_version} | OK |")

    if MODEL_PATH.is_file():
        model_status = "Available"
    else:
        missing.append("models/yolov8n.xml")
        model_status = "MISSING"

    lines.extend(["", "## Model", f"- Path: `{MODEL_PATH}`", f"- Status: {model_status}"])

    devices = []
    if "openvino" not in missing:
        import openvino as ov

        core = ov.Core()
        devices = core.available_devices
        lines.extend([
            "",
            "## OpenVINO",
            f"- Version: {ov.__version__}",
            f"- Available devices: {', '.join(devices)}",
            "- Note: Available devices are detected by OpenVINO; the API reports "
            "the selected inference device when it starts.",
        ])

    lines.extend([
        "",
        "## Notes",
        "- This VisionPort setup uses OpenVINO. PyTorch, CUDA, and `nvidia-smi` "
        "are not required.",
        "",
        "## Check result",
    ])

    if missing:
        lines.append("FAILED: " + ", ".join(missing))
        next_step = "Resolve the missing package or model before starting the API."
    else:
        lines.append("PASSED")
        next_step = "Start the API with `python -m uvicorn serve.gateway:app --reload`."

    lines.extend(["", next_step, ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Environment report saved to: {REPORT_PATH}")
    print(f"OpenVINO devices: {', '.join(devices) if devices else 'unavailable'}")
    if missing:
        print("Environment check failed: " + ", ".join(missing))
        return 1

    print("Environment check passed.")
    print("Start the API with: python -m uvicorn serve.gateway:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
