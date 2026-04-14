import os

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit(
        "Gradio is not installed in the current interpreter. Use an environment with gradio available, "
        "for example `wan/.venv/bin/python gradio_app.py`."
    ) from exc

from one2video_app.runtime import WAN_ROOT, build_environment_report
from one2video_app.services import (
    WAN_ANIMATE_SIZES,
    WAN_AUDIO_VIDEO_SIZES,
    WAN_TEXT_VIDEO_SIZES,
    WAN_TI2V_SIZES,
    run_animate,
    run_audio_to_video,
    run_face_swap,
    run_image_to_image,
    run_image_to_video,
    run_text_to_image,
    run_text_to_video,
)


T2V_MODEL_CONFIG = {
    "Wan T2V-A14B": {
        "task": "t2v-A14B",
        "sizes": WAN_TEXT_VIDEO_SIZES,
        "ckpt": str(WAN_ROOT / "Wan2.2-T2V-A14B"),
        "frames": 81,
    },
    "Wan TI2V-5B": {
        "task": "ti2v-5B",
        "sizes": WAN_TI2V_SIZES,
        "ckpt": str(WAN_ROOT / "Wan2.2-TI2V-5B"),
        "frames": 121,
    },
}

I2V_MODEL_CONFIG = {
    "Wan I2V-A14B": {
        "task": "i2v-A14B",
        "sizes": WAN_TEXT_VIDEO_SIZES,
        "ckpt": str(WAN_ROOT / "Wan2.2-I2V-A14B"),
        "frames": 81,
    },
    "Wan TI2V-5B": {
        "task": "ti2v-5B",
        "sizes": WAN_TI2V_SIZES,
        "ckpt": str(WAN_ROOT / "Wan2.2-TI2V-5B"),
        "frames": 121,
    },
}

S2V_DEFAULT_CKPT = str(WAN_ROOT / "Wan2.2-S2V-14B")
ANIMATE_DEFAULT_CKPT = str(WAN_ROOT / "Wan2.2-Animate-14B")


def refresh_environment(prefer_mock: bool) -> str:
    return build_environment_report(bool(prefer_mock))


def update_t2v_model(choice: str):
    config = T2V_MODEL_CONFIG[choice]
    return (
        gr.update(choices=config["sizes"], value=config["sizes"][0]),
        gr.update(value=config["ckpt"]),
        gr.update(value=config["frames"]),
    )


def update_i2v_model(choice: str):
    config = I2V_MODEL_CONFIG[choice]
    return (
        gr.update(choices=config["sizes"], value=config["sizes"][0]),
        gr.update(value=config["ckpt"]),
        gr.update(value=config["frames"]),
    )


def handle_text_to_image(prompt: str, prefer_mock: bool):
    result = run_text_to_image(prompt, prefer_mock=prefer_mock)
    return result.image_path, result.status


def handle_image_to_image(image_path: str, prompt: str, prefer_mock: bool):
    result = run_image_to_image(image_path, prompt, prefer_mock=prefer_mock)
    return result.image_path, result.status


def handle_text_to_video(model_choice: str, prompt: str, size: str, frame_num: int, ckpt_dir: str, prefer_mock: bool):
    task = T2V_MODEL_CONFIG[model_choice]["task"]
    result = run_text_to_video(task, prompt, size, frame_num, ckpt_dir, prefer_mock=prefer_mock)
    return result.video_path, result.status


def handle_image_to_video(model_choice: str, image_path: str, prompt: str, size: str, frame_num: int, ckpt_dir: str, prefer_mock: bool):
    task = I2V_MODEL_CONFIG[model_choice]["task"]
    result = run_image_to_video(task, image_path, prompt, size, frame_num, ckpt_dir, prefer_mock=prefer_mock)
    return result.video_path, result.status


def handle_audio_to_video(
    image_path: str,
    audio_path: str,
    prompt: str,
    pose_video: str,
    size: str,
    ckpt_dir: str,
    prefer_mock: bool,
    enable_tts: bool,
    tts_prompt_audio: str,
    tts_prompt_text: str,
    tts_text: str,
    start_from_ref: bool,
    infer_frames: int,
    num_clip: int,
):
    result = run_audio_to_video(
        image_path=image_path,
        audio_path=audio_path,
        prompt=prompt,
        pose_video=pose_video,
        size=size,
        ckpt_dir=ckpt_dir,
        prefer_mock=prefer_mock,
        enable_tts=enable_tts,
        tts_prompt_audio=tts_prompt_audio,
        tts_prompt_text=tts_prompt_text,
        tts_text=tts_text,
        start_from_ref=start_from_ref,
        infer_frames=infer_frames,
        num_clip=num_clip,
    )
    return result.video_path, result.status


def handle_face_swap(
    source_path: str,
    target_path: str,
    prefer_mock: bool,
    use_enhancer: bool,
    enhancer_model: str,
    swapper_model: str,
):
    result = run_face_swap(
        source_path=source_path,
        target_path=target_path,
        prefer_mock=prefer_mock,
        use_enhancer=use_enhancer,
        enhancer_model=enhancer_model,
        swapper_model=swapper_model,
    )
    return result.image_path, result.video_path, result.file_path, result.status


def handle_animate(
    src_root_path: str,
    size: str,
    frame_num: int,
    ckpt_dir: str,
    prefer_mock: bool,
    replace_flag: bool,
    use_relighting_lora: bool,
    refert_num: int,
):
    result = run_animate(
        src_root_path=src_root_path,
        size=size,
        frame_num=frame_num,
        ckpt_dir=ckpt_dir,
        prefer_mock=prefer_mock,
        replace_flag=replace_flag,
        use_relighting_lora=use_relighting_lora,
        refert_num=refert_num,
    )
    return result.video_path, result.status


with gr.Blocks(title="One2Video Studio") as demo:
    gr.Markdown("# One2Video Studio")
    gr.Markdown(
        "Unify image generation, video generation, face swap, and animation workflows in one workspace. "
        "This build is designed to stay useful on macOS by preferring mock output unless you explicitly switch to real backends."
    )

    with gr.Row():
        prefer_mock = gr.Checkbox(
            value=True,
            label="Prefer Mock Mode (Recommended on macOS)",
        )
        refresh_btn = gr.Button("Refresh Runtime Overview")

    env_report = gr.Markdown(value=build_environment_report(True))
    prefer_mock.change(refresh_environment, inputs=[prefer_mock], outputs=[env_report])
    refresh_btn.click(refresh_environment, inputs=[prefer_mock], outputs=[env_report])

    with gr.Tabs():
        with gr.Tab("Text to Image"):
            t2i_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the image you want to create...")
            t2i_run = gr.Button("Generate Image", variant="primary")
            t2i_output = gr.Image(label="Output Image")
            t2i_status = gr.Textbox(label="Status", lines=8, interactive=False)
            t2i_run.click(handle_text_to_image, inputs=[t2i_prompt, prefer_mock], outputs=[t2i_output, t2i_status])

        with gr.Tab("Image to Image"):
            i2i_image = gr.Image(label="Input Image", type="filepath")
            i2i_prompt = gr.Textbox(label="Prompt / Edit Instruction", lines=4)
            i2i_run = gr.Button("Transform Image", variant="primary")
            i2i_output = gr.Image(label="Output Image")
            i2i_status = gr.Textbox(label="Status", lines=8, interactive=False)
            i2i_run.click(handle_image_to_image, inputs=[i2i_image, i2i_prompt, prefer_mock], outputs=[i2i_output, i2i_status])

        with gr.Tab("Text to Video"):
            t2v_model = gr.Radio(label="Model", choices=list(T2V_MODEL_CONFIG.keys()), value="Wan T2V-A14B")
            t2v_prompt = gr.Textbox(label="Prompt", lines=4)
            t2v_size = gr.Dropdown(label="Resolution", choices=T2V_MODEL_CONFIG["Wan T2V-A14B"]["sizes"], value=T2V_MODEL_CONFIG["Wan T2V-A14B"]["sizes"][0])
            t2v_frames = gr.Slider(label="Frames", minimum=5, maximum=201, step=4, value=T2V_MODEL_CONFIG["Wan T2V-A14B"]["frames"])
            t2v_ckpt = gr.Textbox(label="Checkpoint Directory", value=T2V_MODEL_CONFIG["Wan T2V-A14B"]["ckpt"])
            t2v_model.change(update_t2v_model, inputs=[t2v_model], outputs=[t2v_size, t2v_ckpt, t2v_frames])
            t2v_run = gr.Button("Generate Video", variant="primary")
            t2v_output = gr.Video(label="Output Video")
            t2v_status = gr.Textbox(label="Status", lines=10, interactive=False)
            t2v_run.click(handle_text_to_video, inputs=[t2v_model, t2v_prompt, t2v_size, t2v_frames, t2v_ckpt, prefer_mock], outputs=[t2v_output, t2v_status])

        with gr.Tab("Image to Video"):
            i2v_model = gr.Radio(label="Model", choices=list(I2V_MODEL_CONFIG.keys()), value="Wan I2V-A14B")
            i2v_image = gr.Image(label="Input Image", type="filepath")
            i2v_prompt = gr.Textbox(label="Prompt", lines=4)
            i2v_size = gr.Dropdown(label="Resolution", choices=I2V_MODEL_CONFIG["Wan I2V-A14B"]["sizes"], value=I2V_MODEL_CONFIG["Wan I2V-A14B"]["sizes"][0])
            i2v_frames = gr.Slider(label="Frames", minimum=5, maximum=201, step=4, value=I2V_MODEL_CONFIG["Wan I2V-A14B"]["frames"])
            i2v_ckpt = gr.Textbox(label="Checkpoint Directory", value=I2V_MODEL_CONFIG["Wan I2V-A14B"]["ckpt"])
            i2v_model.change(update_i2v_model, inputs=[i2v_model], outputs=[i2v_size, i2v_ckpt, i2v_frames])
            i2v_run = gr.Button("Generate Video", variant="primary")
            i2v_output = gr.Video(label="Output Video")
            i2v_status = gr.Textbox(label="Status", lines=10, interactive=False)
            i2v_run.click(handle_image_to_video, inputs=[i2v_model, i2v_image, i2v_prompt, i2v_size, i2v_frames, i2v_ckpt, prefer_mock], outputs=[i2v_output, i2v_status])

        with gr.Tab("Audio to Video"):
            s2v_image = gr.Image(label="Character Image", type="filepath")
            s2v_audio = gr.Audio(label="Input Audio", type="filepath")
            s2v_prompt = gr.Textbox(label="Prompt", lines=3)
            s2v_pose = gr.Video(label="Pose Video (Optional)")
            s2v_size = gr.Dropdown(label="Resolution", choices=WAN_AUDIO_VIDEO_SIZES, value=WAN_AUDIO_VIDEO_SIZES[0])
            s2v_ckpt = gr.Textbox(label="Checkpoint Directory", value=S2V_DEFAULT_CKPT)
            with gr.Accordion("TTS Settings", open=False):
                s2v_enable_tts = gr.Checkbox(label="Enable TTS")
                s2v_tts_prompt_audio = gr.Audio(label="TTS Reference Audio", type="filepath")
                s2v_tts_prompt_text = gr.Textbox(label="TTS Reference Text")
                s2v_tts_text = gr.Textbox(label="TTS Content", lines=3)
            with gr.Accordion("Advanced Settings", open=False):
                s2v_start_from_ref = gr.Checkbox(label="Start From Reference Frame")
                s2v_infer_frames = gr.Dropdown(label="Infer Frames Per Clip", choices=[48, 80, 96, 120], value=80)
                s2v_num_clip = gr.Number(label="Num Clips", value=1, precision=0)
            s2v_run = gr.Button("Generate Video", variant="primary")
            s2v_output = gr.Video(label="Output Video")
            s2v_status = gr.Textbox(label="Status", lines=10, interactive=False)
            s2v_run.click(
                handle_audio_to_video,
                inputs=[
                    s2v_image,
                    s2v_audio,
                    s2v_prompt,
                    s2v_pose,
                    s2v_size,
                    s2v_ckpt,
                    prefer_mock,
                    s2v_enable_tts,
                    s2v_tts_prompt_audio,
                    s2v_tts_prompt_text,
                    s2v_tts_text,
                    s2v_start_from_ref,
                    s2v_infer_frames,
                    s2v_num_clip,
                ],
                outputs=[s2v_output, s2v_status],
            )

        with gr.Tab("Face Swap"):
            gr.Markdown("Source and target both accept image or video files. When the target is a video, FaceFusion will process the video directly in real mode. In mock mode, the target is copied as preview output.")
            ff_source = gr.File(label="Source Face (Image or Video)", type="filepath")
            ff_target = gr.File(label="Target (Image or Video)", type="filepath")
            ff_swapper_model = gr.Dropdown(label="Swapper Model", choices=["inswapper_128_fp16", "simswap_unofficial_512", "ghost_1_256"], value="inswapper_128_fp16")
            ff_use_enhancer = gr.Checkbox(label="Enable Face Enhancer", value=True)
            ff_enhancer_model = gr.Dropdown(label="Enhancer Model", choices=["gfpgan_1.4", "codeformer"], value="gfpgan_1.4")
            ff_run = gr.Button("Run Face Swap", variant="primary")
            with gr.Row():
                ff_output_image = gr.Image(label="Output Image")
                ff_output_video = gr.Video(label="Output Video")
            ff_output_file = gr.File(label="Output File")
            ff_status = gr.Textbox(label="Status", lines=10, interactive=False)
            ff_run.click(
                handle_face_swap,
                inputs=[ff_source, ff_target, prefer_mock, ff_use_enhancer, ff_enhancer_model, ff_swapper_model],
                outputs=[ff_output_image, ff_output_video, ff_output_file, ff_status],
            )

        with gr.Tab("Animate"):
            gr.Markdown("Point this tab at the preprocess output directory produced by `wan/wan/modules/animate/preprocess/preprocess_data.py`.")
            anim_src = gr.Textbox(label="Preprocess Results Directory", value=str(WAN_ROOT / "examples" / "wan_animate" / "animate" / "process_results"))
            anim_size = gr.Dropdown(label="Resolution", choices=WAN_ANIMATE_SIZES, value=WAN_ANIMATE_SIZES[0])
            anim_frames = gr.Slider(label="Frames", minimum=5, maximum=201, step=4, value=77)
            anim_ckpt = gr.Textbox(label="Checkpoint Directory", value=ANIMATE_DEFAULT_CKPT)
            with gr.Row():
                anim_replace = gr.Checkbox(label="Replacement Mode")
                anim_lora = gr.Checkbox(label="Use Relighting LoRA")
            anim_refert = gr.Slider(label="Temporal Guidance Frames (refert_num)", minimum=1, maximum=15, step=1, value=1)
            anim_run = gr.Button("Run Animate", variant="primary")
            anim_output = gr.Video(label="Output Video")
            anim_status = gr.Textbox(label="Status", lines=10, interactive=False)
            anim_run.click(
                handle_animate,
                inputs=[anim_src, anim_size, anim_frames, anim_ckpt, prefer_mock, anim_replace, anim_lora, anim_refert],
                outputs=[anim_output, anim_status],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7682, share=False)
