#!/bin/bash

# NCERT Physics RAG System - Setup Script
# This script helps set up the complete RAG system

set -e  # Exit on error

echo "=================================="
echo "NCERT Physics RAG - Setup Script"
echo "=================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 9 ]; then
    print_error "Python 3.9+ is required. Found: $PYTHON_VERSION"
    exit 1
else
    print_success "Python version: $PYTHON_VERSION"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    print_info "Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "Pip upgraded"

# Install requirements
echo ""
echo "Installing dependencies (this may take 10-15 minutes)..."
print_info "Installing core packages..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "All dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data/pdfs
mkdir -p data/extracted_images
mkdir -p data/metadata
mkdir -p data/vector_db

print_success "Directory structure created"

# Download models (optional)
echo ""
read -p "Do you want to pre-download the embedding model? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Downloading embedding model (this may take a few minutes)..."
    python3 << 'EOF'
from sentence_transformers import SentenceTransformer
print("Downloading BAAI/bge-large-en-v1.5...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
print("Model downloaded successfully!")
EOF
    print_success "Embedding model downloaded"
fi

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("ℹ No GPU detected. Will use CPU (slower but functional)")
EOF

# Summary
echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
print_success "Installation successful!"
echo ""
echo "Next steps:"
echo "1. Place your NCERT Physics PDF files in: data/pdfs/"
echo "2. Build the index:"
echo "   python build_index.py --pdf data/pdfs/physics_class11.pdf --class 11"
echo "3. Run the application:"
echo "   streamlit run app.py"
echo ""
print_info "See README.md for detailed usage instructions"
echo ""

# Create a simple test script
cat > test_installation.py << 'EOF'
"""Quick test to verify installation"""
print("Testing installation...")

try:
    import pdfplumber
    print("✓ pdfplumber")
except ImportError as e:
    print(f"✗ pdfplumber: {e}")

try:
    import fitz
    print("✓ PyMuPDF")
except ImportError as e:
    print(f"✗ PyMuPDF: {e}")

try:
    import chromadb
    print("✓ chromadb")
except ImportError as e:
    print(f"✗ chromadb: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers")
except ImportError as e:
    print(f"✗ sentence-transformers: {e}")

try:
    import streamlit
    print("✓ streamlit")
except ImportError as e:
    print(f"✗ streamlit: {e}")

try:
    from transformers import AutoTokenizer
    print("✓ transformers")
except ImportError as e:
    print(f"✗ transformers: {e}")

print("\nInstallation test complete!")
EOF

print_info "Running installation test..."
python3 test_installation.py
rm test_installation.py

echo ""
print_success "All systems ready!"