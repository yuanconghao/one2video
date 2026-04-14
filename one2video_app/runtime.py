from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
WAN_ROOT = ROOT_DIR / "wan"
FACEFUSION_ROOT = ROOT_DIR / "facefusion"
OUTPUT_ROOT = ROOT_DIR / "outputs"
MOCK_OUTPUT_ROOT = OUTPUT_ROOT / "mock"
FACEFUSION_JOBS_ROOT = OUTPUT_ROOT / "facefusion_jobs"


@dataclass(frozen=True)
class ServiceStatus:
    name: str
    code_available: bool
    real_available: bool
    interpreter: str
    note: str


def resolve_python(project_root: Path) -> str:
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def detect_wan_status() -> ServiceStatus:
    interpreter = resolve_python(WAN_ROOT)
    code_available = (WAN_ROOT / "generate.py").exists()
    real_available = code_available and Path(interpreter).exists()
    note = "Uses wan/generate.py for real runs and falls back to mock output."
    return ServiceStatus(
        name="Wan",
        code_available=code_available,
        real_available=real_available,
        interpreter=interpreter,
        note=note,
    )


def detect_facefusion_status() -> ServiceStatus:
    interpreter = resolve_python(FACEFUSION_ROOT)
    code_available = (FACEFUSION_ROOT / "facefusion.py").exists()
    real_available = code_available and Path(interpreter).exists()
    note = "Uses facefusion.py headless-run for real runs and falls back to mock output."
    return ServiceStatus(
        name="FaceFusion",
        code_available=code_available,
        real_available=real_available,
        interpreter=interpreter,
        note=note,
    )


def build_environment_report(prefer_mock: bool) -> str:
    wan_status = detect_wan_status()
    face_status = detect_facefusion_status()
    mode = "Mock First" if prefer_mock else "Prefer Real Backend"
    return "\n".join(
        [
            "### Runtime Overview",
            f"- Mode: `{mode}`",
            f"- Wan code: `{'yes' if wan_status.code_available else 'no'}` | real runner: `{'yes' if wan_status.real_available else 'no'}` | python: `{wan_status.interpreter}`",
            f"- FaceFusion code: `{'yes' if face_status.code_available else 'no'}` | real runner: `{'yes' if face_status.real_available else 'no'}` | python: `{face_status.interpreter}`",
            "- macOS development is supported through mock outputs even when model runtimes are unavailable.",
        ]
    )
