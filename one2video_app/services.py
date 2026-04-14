from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
from typing import Sequence

from .mock_assets import copy_mock_image, copy_mock_video, create_mock_image, make_output_path
from .runtime import FACEFUSION_JOBS_ROOT, FACEFUSION_ROOT, WAN_ROOT, detect_facefusion_status, detect_wan_status


WAN_TEXT_VIDEO_SIZES = ["1280*720", "720*1280", "480*832", "832*480"]
WAN_TI2V_SIZES = ["1280*704", "704*1280"]
WAN_AUDIO_VIDEO_SIZES = ["1280*720", "720*1280", "480*832", "832*480", "1024*704"]
WAN_ANIMATE_SIZES = ["1280*720", "720*1280"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass
class TaskResult:
    status: str
    image_path: str | None = None
    video_path: str | None = None
    file_path: str | None = None
    used_mock: bool = False


def _quoted_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _run_command(command: Sequence[str], cwd: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return True, completed.stdout.strip() or "Completed successfully."
    stderr = completed.stderr.strip() or completed.stdout.strip() or "Unknown error."
    return False, stderr


def _mock_video_result(prefix: str, title: str, lines: list[str], source_video: str | None = None) -> TaskResult:
    video_path = copy_mock_video(prefix, source_video=source_video)
    if video_path:
        return TaskResult(
            status=f"{title}\n\nMock mode is active, so a preview video was copied into `{video_path}`.\n" + "\n".join(f"- {line}" for line in lines),
            video_path=video_path,
            file_path=video_path,
            used_mock=True,
        )
    image_path = create_mock_image(prefix, title, lines)
    return TaskResult(
        status=f"{title}\n\nMock mode is active and no fallback video was found, so an image placeholder was created instead.",
        image_path=image_path,
        file_path=image_path,
        used_mock=True,
    )


def _mock_image_result(prefix: str, title: str, lines: list[str], source_image: str | None = None) -> TaskResult:
    image_path = copy_mock_image(prefix, source_image=source_image)
    if not image_path:
        image_path = create_mock_image(prefix, title, lines)
    return TaskResult(
        status=f"{title}\n\nMock mode is active.\n" + "\n".join(f"- {line}" for line in lines),
        image_path=image_path,
        file_path=image_path,
        used_mock=True,
    )


def run_text_to_image(prompt: str, prefer_mock: bool = True) -> TaskResult:
    if not prompt.strip():
        return TaskResult(status="Please enter a prompt for Text to Image.")
    return _mock_image_result(
        prefix="text_to_image",
        title="Text to Image",
        lines=[
            f"Prompt: {prompt}",
            "Backend adapter is intentionally left in mock mode for now.",
        ],
    )


def run_image_to_image(image_path: str | None, prompt: str, prefer_mock: bool = True) -> TaskResult:
    if not image_path:
        return TaskResult(status="Please upload an input image for Image to Image.")
    return _mock_image_result(
        prefix="image_to_image",
        title="Image to Image",
        lines=[
            f"Prompt: {prompt or 'No prompt provided.'}",
            f"Input image: {image_path}",
            "Backend adapter is intentionally left in mock mode for now.",
        ],
        source_image=image_path,
    )


def run_text_to_video(task: str, prompt: str, size: str, frame_num: int, ckpt_dir: str, prefer_mock: bool = True) -> TaskResult:
    if not prompt.strip():
        return TaskResult(status="Please enter a prompt for Text to Video.")

    wan_status = detect_wan_status()
    if prefer_mock or not wan_status.real_available or not Path(ckpt_dir).exists():
        return _mock_video_result(
            prefix="text_to_video",
            title="Text to Video",
            lines=[
                f"Task: {task}",
                f"Prompt: {prompt}",
                f"Size: {size}",
                f"Frames: {frame_num}",
                f"Checkpoint dir: {ckpt_dir or 'not provided'}",
            ],
        )

    output_path = make_output_path("text_to_video", ".mp4")
    command = [
        wan_status.interpreter,
        str(WAN_ROOT / "generate.py"),
        "--task",
        task,
        "--size",
        size,
        "--frame_num",
        str(int(frame_num)),
        "--ckpt_dir",
        ckpt_dir,
        "--prompt",
        prompt,
        "--offload_model",
        "True",
        "--save_file",
        str(output_path),
        "--offload_model",
        "True",
    ]
    ok, message = _run_command(command, WAN_ROOT)
    if ok and output_path.exists():
        return TaskResult(
            status=f"Real Wan backend finished successfully.\n\nCommand: `{_quoted_command(command)}`",
            video_path=str(output_path),
            file_path=str(output_path),
        )
    return _mock_video_result(
        prefix="text_to_video_fallback",
        title="Text to Video",
        lines=[
            f"Real Wan command failed: {message}",
            f"Command: {_quoted_command(command)}",
        ],
    )


def run_image_to_video(
    task: str,
    image_path: str | None,
    prompt: str,
    size: str,
    frame_num: int,
    ckpt_dir: str,
    prefer_mock: bool = True,
) -> TaskResult:
    if not image_path:
        return TaskResult(status="Please upload an input image for Image to Video.")

    wan_status = detect_wan_status()
    if prefer_mock or not wan_status.real_available or not Path(ckpt_dir).exists():
        return _mock_video_result(
            prefix="image_to_video",
            title="Image to Video",
            lines=[
                f"Task: {task}",
                f"Prompt: {prompt or 'No prompt provided.'}",
                f"Input image: {image_path}",
                f"Size: {size}",
                f"Frames: {frame_num}",
            ],
        )

    output_path = make_output_path("image_to_video", ".mp4")
    command = [
        wan_status.interpreter,
        str(WAN_ROOT / "generate.py"),
        "--task",
        task,
        "--size",
        size,
        "--frame_num",
        str(int(frame_num)),
        "--ckpt_dir",
        ckpt_dir,
        "--image",
        image_path,
        "--prompt",
        prompt or "",
        "--offload_model",
        "True",
        "--save_file",
        str(output_path),
        "--offload_model",
        "True",
    ]
    ok, message = _run_command(command, WAN_ROOT)
    if ok and output_path.exists():
        return TaskResult(
            status=f"Real Wan backend finished successfully.\n\nCommand: `{_quoted_command(command)}`",
            video_path=str(output_path),
            file_path=str(output_path),
        )
    return _mock_video_result(
        prefix="image_to_video_fallback",
        title="Image to Video",
        lines=[
            f"Real Wan command failed: {message}",
            f"Command: {_quoted_command(command)}",
        ],
    )


def run_audio_to_video(
    image_path: str | None,
    audio_path: str | None,
    prompt: str,
    size: str,
    ckpt_dir: str,
    prefer_mock: bool = True,
    pose_video: str | None = None,
    enable_tts: bool = False,
    tts_prompt_audio: str | None = None,
    tts_prompt_text: str | None = None,
    tts_text: str | None = None,
    start_from_ref: bool = False,
    infer_frames: int = 80,
    num_clip: int = 1,
) -> TaskResult:
    if not image_path:
        return TaskResult(status="Please upload a character image for Audio to Video.")
    if not enable_tts and not audio_path:
        return TaskResult(status="Please upload an audio file or enable TTS.")
    if enable_tts and (not tts_prompt_audio or not tts_prompt_text or not tts_text):
        return TaskResult(status="TTS mode requires reference audio, reference text, and synthesis text.")

    wan_status = detect_wan_status()
    if prefer_mock or not wan_status.real_available or not Path(ckpt_dir).exists():
        return _mock_video_result(
            prefix="audio_to_video",
            title="Audio to Video",
            lines=[
                f"Prompt: {prompt or 'No prompt provided.'}",
                f"Image: {image_path}",
                f"Audio: {audio_path or 'TTS mode'}",
                f"Pose video: {pose_video or 'None'}",
                f"Size: {size}",
            ],
            source_video=pose_video,
        )

    output_path = make_output_path("audio_to_video", ".mp4")
    command = [
        wan_status.interpreter,
        str(WAN_ROOT / "generate.py"),
        "--task",
        "s2v-14B",
        "--size",
        size,
        "--ckpt_dir",
        ckpt_dir,
        "--image",
        image_path,
        "--prompt",
        prompt or "",
        "--infer_frames",
        str(int(infer_frames)),
        "--num_clip",
        str(int(num_clip)),
        "--save_file",
        str(output_path),
        "--offload_model",
        "True",
    ]
    if audio_path and not enable_tts:
        command.extend(["--audio", audio_path])
    if pose_video:
        command.extend(["--pose_video", pose_video])
    if start_from_ref:
        command.append("--start_from_ref")
    if enable_tts:
        command.append("--enable_tts")
        command.extend(["--tts_prompt_audio", tts_prompt_audio, "--tts_prompt_text", tts_prompt_text, "--tts_text", tts_text])

    ok, message = _run_command(command, WAN_ROOT)
    if ok and output_path.exists():
        return TaskResult(
            status=f"Real Wan backend finished successfully.\n\nCommand: `{_quoted_command(command)}`",
            video_path=str(output_path),
            file_path=str(output_path),
        )
    return _mock_video_result(
        prefix="audio_to_video_fallback",
        title="Audio to Video",
        lines=[
            f"Real Wan command failed: {message}",
            f"Command: {_quoted_command(command)}",
        ],
        source_video=pose_video,
    )


def run_animate(
    src_root_path: str,
    size: str,
    frame_num: int,
    ckpt_dir: str,
    prefer_mock: bool = True,
    replace_flag: bool = False,
    use_relighting_lora: bool = False,
    refert_num: int = 1,
) -> TaskResult:
    if not src_root_path:
        return TaskResult(status="Please provide the preprocess results directory for Animate.")

    wan_status = detect_wan_status()
    if prefer_mock or not wan_status.real_available or not Path(ckpt_dir).exists():
        mock_source = None
        pose_video = Path(src_root_path) / "src_pose.mp4"
        if pose_video.exists():
            mock_source = str(pose_video)
        return _mock_video_result(
            prefix="animate",
            title="Animate",
            lines=[
                f"Preprocess directory: {src_root_path}",
                f"Size: {size}",
                f"Frames: {frame_num}",
                f"Replacement mode: {replace_flag}",
                f"Relighting LoRA: {use_relighting_lora}",
            ],
            source_video=mock_source,
        )

    output_path = make_output_path("animate", ".mp4")
    command = [
        wan_status.interpreter,
        str(WAN_ROOT / "generate.py"),
        "--task",
        "animate-14B",
        "--size",
        size,
        "--frame_num",
        str(int(frame_num)),
        "--ckpt_dir",
        ckpt_dir,
        "--src_root_path",
        src_root_path,
        "--refert_num",
        str(int(refert_num)),
        "--save_file",
        str(output_path),
        "--offload_model",
        "True",
    ]
    if replace_flag:
        command.append("--replace_flag")
    if use_relighting_lora:
        command.append("--use_relighting_lora")

    ok, message = _run_command(command, WAN_ROOT)
    if ok and output_path.exists():
        return TaskResult(
            status=f"Real Wan backend finished successfully.\n\nCommand: `{_quoted_command(command)}`",
            video_path=str(output_path),
            file_path=str(output_path),
        )
    return _mock_video_result(
        prefix="animate_fallback",
        title="Animate",
        lines=[
            f"Real Wan command failed: {message}",
            f"Command: {_quoted_command(command)}",
        ],
    )


def run_face_swap(
    source_path: str | None,
    target_path: str | None,
    prefer_mock: bool = True,
    use_enhancer: bool = True,
    enhancer_model: str = "gfpgan_1.4",
    swapper_model: str = "inswapper_128_fp16",
) -> TaskResult:
    if not target_path:
        return TaskResult(status="Please provide a target image or video for Face Swap.")
    if not source_path and not use_enhancer:
        return TaskResult(status="Please provide a source face, or enable enhancer-only mode.")

    target_suffix = Path(target_path).suffix.lower()
    is_target_video = target_suffix in VIDEO_EXTENSIONS
    is_target_image = target_suffix in IMAGE_EXTENSIONS
    if not is_target_video and not is_target_image:
        return TaskResult(status="Target must be an image or a video file.")

    facefusion_status = detect_facefusion_status()
    processors = []
    if source_path:
        processors.append("face_swapper")
    if use_enhancer:
        processors.append("face_enhancer")
    if not processors:
        return TaskResult(status="No FaceFusion processors selected.")

    if prefer_mock or not facefusion_status.real_available:
        lines = [
            f"Source: {source_path or 'None'}",
            f"Target: {target_path}",
            f"Processors: {', '.join(processors)}",
            f"Swapper model: {swapper_model}",
            f"Enhancer model: {enhancer_model}",
        ]
        if is_target_video:
            return _mock_video_result(
                prefix="face_swap",
                title="Face Swap",
                lines=lines,
                source_video=target_path,
            )
        return _mock_image_result(
            prefix="face_swap",
            title="Face Swap",
            lines=lines,
            source_image=target_path,
        )

    FACEFUSION_JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    output_suffix = ".mp4" if is_target_video else ".png"
    output_path = make_output_path("face_swap", output_suffix)
    command = [
        facefusion_status.interpreter,
        str(FACEFUSION_ROOT / "facefusion.py"),
        "headless-run",
        "--jobs-path",
        str(FACEFUSION_JOBS_ROOT),
        "--processors",
        *processors,
        "-t",
        target_path,
        "-o",
        str(output_path),
    ]
    if source_path:
        command.extend(["-s", source_path])
    if source_path and swapper_model:
        command.extend(["--face-swapper-model", swapper_model])
    if use_enhancer and enhancer_model:
        command.extend(["--face-enhancer-model", enhancer_model])

    ok, message = _run_command(command, FACEFUSION_ROOT)
    if ok and output_path.exists():
        result = TaskResult(
            status=f"Real FaceFusion backend finished successfully.\n\nCommand: `{_quoted_command(command)}`",
            file_path=str(output_path),
        )
        if is_target_video:
            result.video_path = str(output_path)
        else:
            result.image_path = str(output_path)
        return result

    lines = [
        f"Real FaceFusion command failed: {message}",
        f"Command: {_quoted_command(command)}",
    ]
    if is_target_video:
        return _mock_video_result(
            prefix="face_swap_fallback",
            title="Face Swap",
            lines=lines,
            source_video=target_path,
        )
    return _mock_image_result(
        prefix="face_swap_fallback",
        title="Face Swap",
        lines=lines,
        source_image=target_path,
    )
