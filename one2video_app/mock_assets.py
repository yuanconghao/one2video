from __future__ import annotations

from pathlib import Path
from typing import Iterable
import shutil
import textwrap
from datetime import datetime

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - optional dev dependency
    Image = None
    ImageDraw = None

from .runtime import MOCK_OUTPUT_ROOT, WAN_ROOT


def make_output_path(prefix: str, suffix: str) -> Path:
    MOCK_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = prefix.replace(" ", "_").replace("/", "_")
    return MOCK_OUTPUT_ROOT / f"{safe_prefix}_{timestamp}{suffix}"


def _normalize_lines(lines: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for line in lines:
        if not line:
            continue
        normalized.extend(textwrap.wrap(str(line), width=48) or [""])
    return normalized


def _create_svg_placeholder(prefix: str, title: str, lines: Iterable[str]) -> str:
    output_path = make_output_path(prefix, ".svg")
    text_lines = [title, *list(_normalize_lines(lines)), "Mock output generated without Pillow."]
    tspans = []
    y = 80
    for line in text_lines:
        safe_line = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        tspans.append(f'<tspan x="80" y="{y}">{safe_line}</tspan>')
        y += 34
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
<rect width="1280" height="720" fill="#141826" />
<rect x="48" y="48" width="1184" height="624" fill="none" stroke="#47A3FF" stroke-width="4" />
<text fill="#C4CCDC" font-size="24" font-family="monospace">
{''.join(tspans)}
</text>
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")
    return str(output_path)


def create_mock_image(prefix: str, title: str, lines: Iterable[str]) -> str:
    if Image is None or ImageDraw is None:
        return _create_svg_placeholder(prefix, title, lines)

    output_path = make_output_path(prefix, ".png")
    image = Image.new("RGB", (1280, 720), color=(20, 24, 38))
    draw = ImageDraw.Draw(image)
    accent = (71, 163, 255)
    muted = (196, 204, 220)
    draw.rectangle((48, 48, 1232, 672), outline=accent, width=4)
    draw.text((80, 80), title, fill=accent)

    y = 150
    for line in _normalize_lines(lines):
        draw.text((80, y), line, fill=muted)
        y += 34

    draw.text(
        (80, 620),
        "Mock output generated for local development without loading heavy models.",
        fill=(140, 149, 167),
    )
    image.save(output_path)
    return str(output_path)


def copy_mock_video(prefix: str, source_video: str | None = None) -> str | None:
    if source_video and Path(source_video).exists():
        input_path = Path(source_video)
    else:
        fallback_video = WAN_ROOT / "examples" / "pose.mp4"
        if not fallback_video.exists():
            return None
        input_path = fallback_video

    output_path = make_output_path(prefix, input_path.suffix or ".mp4")
    shutil.copy2(input_path, output_path)
    return str(output_path)


def copy_mock_image(prefix: str, source_image: str | None = None) -> str | None:
    if source_image and Path(source_image).exists():
        input_path = Path(source_image)
        output_path = make_output_path(prefix, input_path.suffix or ".png")
        shutil.copy2(input_path, output_path)
        return str(output_path)
    return None
