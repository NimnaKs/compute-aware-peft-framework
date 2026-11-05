"""Preflight diagnostic before paying for GPU time."""
import os
import sys


def main():
    print("Python:", sys.version.split()[0])
    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("VRAM (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
            print("BF16 supported:", torch.cuda.is_bf16_supported())
        else:
            print("GPU experiment requires NVIDIA CUDA. CPU-only env can run unit tests.")
    except ImportError:
        print("PyTorch not installed")
    try:
        from huggingface_hub import HfApi, get_token
        print("Hugging Face token present:", bool(get_token()))
        if not get_token():
            print("Run: hf auth login   (accept gated model terms first)")
    except ImportError:
        print("huggingface_hub not installed")
    from .utils import dependency_versions
    print("Versions:")
    for k, v in dependency_versions().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
