#!/bin/bash

echo "🚀 Setting up Lightweight World Model Environment"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || venv\Scripts\activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install base requirements
echo "Installing base requirements..."
pip install -e .

# Install training requirements
echo "Installing training requirements..."
pip install -e ".[training]"

# Install development requirements
echo "Installing development requirements..."
pip install -e ".[dev]"

# Create necessary directories
echo "Creating project directories..."
mkdir -p checkpoints
mkdir -p datasets/raw
mkdir -p datasets/processed
mkdir -p logs
mkdir -p outputs

echo "✅ Setup complete! Activate the environment with:"
echo "    source venv/bin/activate  # On Linux/Mac"
echo "    venv\\Scripts\\activate     # On Windows"
echo ""
echo "🎮 To test the installation, run:"
echo "    python -m src.tests.test_vqvae"