# SEAL Research Project — Setup Guide

## Your Machine
- **GPU**: NVIDIA RTX 4500 Ada Generation (24GB VRAM) ← BEAST. No Colab needed.
- **CPU**: Intel Core Ultra 9 285
- **RAM**: 64 GB
- **CUDA Driver**: 595.95 (supports CUDA 13.2)
- **OS**: Windows 11 Education

## Problem: PyTorch is currently CPU-only
Run `python verify_gpu.py` — if it says `CUDA available: False`, follow steps below.

---

## Step 1: Create Virtual Environment

Open PowerShell in `C:\Users\Admin\Desktop\seal\`:

```powershell
python -m venv seal-env
.\seal-env\Scripts\activate
```

## Step 2: Uninstall CPU PyTorch

```powershell
pip uninstall torch torchvision torchaudio -y
```

## Step 3: Install CUDA-Enabled PyTorch

Try Option A first. If it fails, try Option B.

**Option A — PyTorch Nightly (CUDA 12.6, compatible with driver 595.95)**
```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

**Option B — PyTorch Stable (CUDA 12.6)**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Step 4: Install Research Dependencies

```powershell
pip install -r requirements.txt
```

## Step 5: Verify Everything Works

```powershell
python verify_gpu.py
```

Expected output:
```
GPU: NVIDIA RTX 4500 Ada Generation
VRAM: 24.0 GB
GPU matmul test: torch.Size([1000, 1000]) on cuda:0 ✅
EXCELLENT: 24GB VRAM — can run Qwen2.5-7B in bf16 or 1.5B at full precision
```

## Step 6: Clone the Official SEAL Repo

```powershell
git clone https://github.com/jyopari/seal
```

## Step 7: Run Your First Smoke Test

Try running 1 self-edit cycle from SEAL's own examples (check their README).

---

## Model Choice

**Primary**: `Qwen/Qwen2.5-1.5B-Instruct` (1.5B params, permissive license, excellent instruction following)
**Backup**: `meta-llama/Llama-3.2-1B-Instruct` (need Hugging Face token for gated access)

With 24GB VRAM you can also run `Qwen/Qwen2.5-7B-Instruct` in bf16 for stronger results.

---

## Project Structure

```
seal/
├── src/                    # Your research code
├── configs/                # Config files for each experimental config
├── data/                   # fact_pool.json + baseline_accuracy.json
├── results/                # All experiment outputs (auto-created)
├── figures/                # Paper-ready figures
├── notebooks/              # Analysis notebooks
├── scripts/                # Run scripts
├── seal/                   # Official SEAL repo (cloned)
├── verify_gpu.py           # Run first!
├── requirements.txt
└── README_SETUP.md         # This file
```
