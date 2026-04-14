import os
import argparse
import sys

# Wan2.2 Video Generation Models
WAN_MODELS = {
    "T2V-A14B": "Wan-AI/Wan2.2-T2V-A14B",
    "I2V-A14B": "Wan-AI/Wan2.2-I2V-A14B",
    "TI2V-5B":  "Wan-AI/Wan2.2-TI2V-5B",
    "S2V-14B":  "Wan-AI/Wan2.2-S2V-14B",
    "Animate-14B": "Wan-AI/Wan2.2-Animate-14B"
}

# FaceFusion / FaceSwap Models (Aesthetics & Enhancers)
FACE_MODELS = {
    "FF-Assets": "facefusion/facefusion-assets", # Contains inswapper_128, gfpgan, codeformer, etc.
    "SimSwap-512": "shareAI/simswap_512",
    "Ghost-256": "shareAI/ghost_256"
}

def download_via_modelscope(repo_id, local_dir):
    try:
        from modelscope import snapshot_download
        print(f"\n🚀 [ModelScope] Starting download: {repo_id}")
        print(f"📂 Target directory: {os.path.abspath(local_dir)}")
        snapshot_download(repo_id, local_dir=local_dir)
        print(f"✅ Finished: {repo_id}")
    except ImportError:
        print("\n❌ Error: 'modelscope' is not installed.")
        print("💡 Fix: pip install modelscope")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def download_via_hf(repo_id, local_dir):
    try:
        from huggingface_hub import snapshot_download
        print(f"\n🚀 [HuggingFace] Starting download: {repo_id}")
        print(f"📂 Target directory: {os.path.abspath(local_dir)}")
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
        print(f"✅ Finished: {repo_id}")
    except ImportError:
        print("\n❌ Error: 'huggingface_hub' is not installed.")
        print("💡 Fix: pip install huggingface_hub")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 One-click Model Downloader")
    parser.add_argument("--source", type=str, default="modelscope", choices=["modelscope", "hf"], 
                        help="Download source: 'modelscope' (recommended for China) or 'hf' (HuggingFace)")
    parser.add_argument("--group", type=str, default="wan", choices=["wan", "face", "all"],
                        help="Model group to download: 'wan' (Wan2.2), 'face' (FaceFusion), or 'all'")
    parser.add_argument("--model", type=str, default="all", help="Specific model name or 'all' within the group")
    
    args = parser.parse_args()

    # Combine models based on group
    selected_models = {}
    if args.group in ["wan", "all"]:
        selected_models.update(WAN_MODELS)
    if args.group in ["face", "all"]:
        selected_models.update(FACE_MODELS)
    
    # Filter by specific model if requested
    if args.model != "all":
        if args.model in selected_models:
            to_download = {args.model: selected_models[args.model]}
        else:
            print(f"❌ Error: Model '{args.model}' not found in group '{args.group}'.")
            sys.exit(1)
    else:
        to_download = selected_models
    
    print("="*50)
    print(f"Wan2.2 & FaceFusion Downloader | Source: {args.source.upper()}")
    print(f"Target Group: {args.group.upper()}")
    print("="*50)

    for name, repo in to_download.items():
        repo_name = repo.split('/')[-1]
        if name in WAN_MODELS:
            local_path = os.path.join("wan", repo_name)
        else:
            # FaceFusion assets go into its specific structure
            local_path = os.path.join("facefusion", ".assets", "models")
            if not os.path.exists(local_path):
                os.makedirs(local_path, exist_ok=True)
            local_path = os.path.join(local_path, repo_name)
        
        if args.source == "modelscope":
            download_via_modelscope(repo, local_path)
        else:
            download_via_hf(repo, local_path)
    
    print("\n🎉 Process finished. Happy generating!")
