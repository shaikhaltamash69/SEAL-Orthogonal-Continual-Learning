"""
Step 0 Verification Script
Run this FIRST after reinstalling PyTorch with CUDA support.
Expected output: RTX 4500 Ada Gen, CUDA available, 24GB VRAM
"""
import sys

print("=" * 60)
print("SEAL Project — GPU Environment Verification")
print("=" * 60)

# Python
print(f"\nPython: {sys.version}")

# PyTorch
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
        
        # Functional test
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"\nGPU matmul test: {z.shape} on {z.device} ✅")
        
        if vram >= 20:
            print("\n✅ EXCELLENT: 24GB VRAM — can run Qwen2.5-7B in bf16 or 1.5B at full precision")
            print("   No need for Colab. All 5 configs can run locally.")
        else:
            print(f"\n⚠️  VRAM = {vram:.1f}GB — use 4-bit quantization (bitsandbytes)")
    else:
        print("\n❌ CUDA NOT AVAILABLE — PyTorch is CPU-only!")
        print("   Run this to fix:")
        print("   pip uninstall torch torchvision torchaudio -y")
        print("   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126")
except ImportError:
    print("❌ PyTorch not installed")

# Transformers
try:
    import transformers
    print(f"\nTransformers: {transformers.__version__} ✅")
except ImportError:
    print("\n❌ transformers not installed")

# PEFT
try:
    import peft
    print(f"PEFT: {peft.__version__} ✅")
except ImportError:
    print("❌ peft not installed")

# bitsandbytes
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes: {bnb.__version__} ✅")
except ImportError:
    print("❌ bitsandbytes not installed (needed for 4-bit quantization)")

print("\n" + "=" * 60)
print("If all checks pass → proceed to Phase 1 (fact pool curation)")
print("=" * 60)
