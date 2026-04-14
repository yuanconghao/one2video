# One2Video Studio: Wan2.2 & FaceFusion Unified Workspace

This project unifies **Wan2.2** (Video Generation) and **FaceFusion** (Face Swap/Enhancement) into a single Gradio interface. It is optimized for **NVIDIA RTX 4090 (24GB)** deployment.

## 🚀 Deployment Guide (NVIDIA 4090)

### 1. Prerequisites
- **OS**: Linux (Ubuntu 22.04 recommended) or Windows with WSL2.
- **GPU**: NVIDIA RTX 4090 (24GB VRAM).
- **Driver**: NVIDIA Driver 535+ & CUDA 12.1+.
- **Python**: 3.12.

### 2. Environment Setup

We recommend using `conda` to manage the environment:

```bash
# Create and activate environment
conda create -n one2video python=3.12 -y
conda activate one2video

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gradio modelscope huggingface_hub modelscope
```

### 3. Install Component Dependencies

#### Wan2.2 Dependencies
```bash
cd wan
# Adjust numpy for compatibility if needed
pip install -r requirements.txt
# Install flash-attention for 4090 performance boost
pip install flash-attn --no-build-isolation
cd ..
```

#### FaceFusion Dependencies
```bash
cd facefusion
# Note: FaceFusion might try to install numpy 2.x, which is fine for Python 3.12
pip install -r requirements.txt
# Ensure onnxruntime-gpu is used for 4090
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu
cd ..
```

### 4. Download Model Weights

Use the provided downloader script. We recommend **ModelScope** for faster downloads in China.

```bash
# Download all models (Wan2.2 & FaceFusion assets)
python download_models.py --source modelscope --group all
```

*Note: This will download several GBs of data. Ensure you have enough disk space.*

### 5. Launch the App

The app is configured to run on **Port 7682**.

```bash
python gradio_app.py
```

Access the UI at: `http://<your-server-ip>:7682`

---

## 🛠 Features

- **Wan2.2 Video Generation**:
  - **T2V**: Text to Video (A14B & 5B).
  - **I2V**: Image to Video (A14B & 5B).
  - **S2V**: Speech to Video (Talking Head).
  - **Animate**: Character Animation.
- **FaceFusion**:
  - High-quality Face Swap (InSwapper, SimSwap, Ghost).
  - Face Enhancement (GFPGAN, CodeFormer).
- **Unified Workflow**: Generate a video with Wan2.2 and immediately refine it with FaceFusion.
- **Mock Mode**: Toggle "Prefer Mock Mode" to test UI/Workflow on non-GPU machines (like macOS) without loading models.

## 💡 4090 Optimization Notes

- **VRAM Usage**: 
  - Wan2.2 **5B** models run comfortably on 4090.
  - Wan2.2 **A14B (MoE)** models are optimized for 24GB VRAM. The app is pre-configured to use `--offload_model True` to ensure stable inference on a single 4090.
- **Flash Attention**: Installing `flash-attn` significantly reduces memory usage and speeds up generation.
- **ONNX Execution**: FaceFusion will automatically use the `CUDAExecutionProvider` on your 4090 for real-time swapping.

## 📁 Project Structure

- `wan/`: Core Wan2.2 inference code and checkpoints.
- `facefusion/`: Core FaceFusion engine.
- `one2video_app/`: Integration logic and mock asset management.
- `gradio_app.py`: Main UI entry point.
- `download_models.py`: One-click weight downloader.
