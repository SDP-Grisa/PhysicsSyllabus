#!/usr/bin/env python3
"""
Setup and Initialization Script for NCERT Physics RAG System
"""

import sys
import os
from pathlib import Path
import subprocess

def check_system_dependencies():
    print("🔍 Checking system dependencies...")
    dependencies = {
        'tesseract': 'tesseract --version',
        'poppler': 'pdftotext -v'
    }
    missing = []
    for name, command in dependencies.items():
        try:
            subprocess.run(command.split(), capture_output=True, check=True)
            print(f"  ✅ {name} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  ❌ {name} is NOT installed")
            missing.append(name)
    
    if missing:
        print("\n⚠️  Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstallation instructions:")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr poppler-utils")
        print("  macOS: brew install tesseract poppler")
        print("  Windows: See README.md for download links")
        return False
    return True

def check_python_packages():
    print("\n🔍 Checking Python packages...")
    try:
        import fastapi, uvicorn, pdfplumber, pytesseract, chromadb
        from sentence_transformers import SentenceTransformer
        print("  ✅ All Python packages are installed")
        return True
    except ImportError as e:
        print(f"  ❌ Missing package: {e}")
        print("\nPlease run: pip install -r requirements.txt")
        return False

def check_pdf_exists():
    print("\n🔍 Checking for PDF file...")
    pdf_candidates = ["../pdf/keph201.pdf"]
    for pdf in pdf_candidates:
        if Path(pdf).exists():
            print(f"  ✅ Found PDF: {pdf}")
            return pdf
    print("  ❌ No PDF file found")
    print("\nPlease place your NCERT Physics 11th textbook PDF in the project directory.")
    return None

def setup_directories():
    print("\n📁 Setting up directories...")
    directories = [
        "extracted_content",
        "extracted_content/images",
        "extracted_content/tables",
        "chroma_db"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")

def extract_content(pdf_path):
    print("\n📚 Extracting content from PDF...")
    try:
        from pdf_extractor import ImprovedPDFExtractor
        extractor = ImprovedPDFExtractor(pdf_path)
        extractor.extract_all_content()
        print("\n✅ Content extraction completed!")
        return True
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        return False

def build_vector_database():
    print("\n🗄️  Building vector database...")
    try:
        from vector_db import build_vector_db
        build_vector_db()
        print("\n✅ Vector database built successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Database building failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🔬 NCERT Physics 11th RAG System - Setup")
    print("=" * 60)
    
    if not check_system_dependencies() or not check_python_packages() or not check_pdf_exists():
        sys.exit(1)
    
    setup_directories()
    
    pdf_path = check_pdf_exists()
    if not pdf_path:
        sys.exit(1)
    
    response = input("\nProceed with content extraction and DB build? (y/n): ").lower()
    if response != 'y':
        print("Setup paused. Run again when ready.")
        sys.exit(0)
    
    if not extract_content(pdf_path) or not build_vector_database():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("Run the app:   python app.py   or   uvicorn app:app --reload")
    print("Open: http://localhost:8000")
    print("=" * 60)

if __name__ == "__main__":
    main()