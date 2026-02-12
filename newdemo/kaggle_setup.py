"""
NCERT Physics RAG – Kaggle Setup
This replaces the bash setup script for Kaggle environment.
"""

import os
import sys
import subprocess
from pathlib import Path

print("=" * 60)
print("NCERT Physics RAG - Kaggle Setup")
print("=" * 60)

# -----------------------------------
# 1. Install dependencies
# -----------------------------------
print("\n[1/4] Installing dependencies...")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
    check=True
)

print("✓ Dependencies installed")


# -----------------------------------
# 2. Create directory structure (Kaggle paths)
# -----------------------------------
print("\n[2/4] Creating directories...")

BASE_DIR = Path("/kaggle/working")

dirs = [
    BASE_DIR / "data/pdfs",
    BASE_DIR / "data/extracted_images",
    BASE_DIR / "data/metadata",
    BASE_DIR / "data/vector_db",
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

print("✓ Directory structure ready")


# -----------------------------------
# 3. GPU check (Kaggle usually has GPU)
# -----------------------------------
print("\n[3/4] Checking GPU...")

try:
    import torch

    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA: {torch.version.cuda}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ Memory: {mem:.2f} GB")
    else:
        print("⚠ No GPU detected (CPU mode)")
except Exception as e:
    print("⚠ Torch not available:", e)


# -----------------------------------
# 4. Installation test
# -----------------------------------
print("\n[4/4] Testing core libraries...")

tests = {
    "pdfplumber": "import pdfplumber",
    "PyMuPDF": "import fitz",
    "chromadb": "import chromadb",
    "sentence-transformers": "from sentence_transformers import SentenceTransformer",
    "streamlit": "import streamlit",
    "transformers": "from transformers import AutoTokenizer",
}

for name, code in tests.items():
    try:
        exec(code)
        print(f"✓ {name}")
    except Exception as e:
        print(f"✗ {name}: {e}")

print("\n✅ Kaggle setup complete!")
print("\nNext step:")
print("Run index building:")
print("python build_index.py --pdf /kaggle/input/YOUR_DATASET/physics_class11.pdf --class 11")
