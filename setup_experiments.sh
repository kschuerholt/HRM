#!/bin/bash

# HRM Experiments Quick Setup Script
# Run this after cloning the repository on a fresh VM

set -e  # Exit on any error

echo "🚀 HRM Experiments Setup Script"
echo "================================"

# Check if we're in the right directory
if [ ! -f "pretrain.py" ]; then
    echo "❌ Error: Run this script from the HRM root directory"
    exit 1
fi

# Check for GPUs
echo "🔍 Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
echo "✅ Found $GPU_COUNT GPUs"

# Setup virtual environment
echo "🐍 Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -e .

# Note: We use standard AdamW optimizer instead of adam-atan2 for better compatibility

# Verify PyTorch CUDA
echo "🔧 Verifying PyTorch CUDA support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if not torch.cuda.is_available():
    print('❌ CUDA not available - check PyTorch installation')
    exit(1)
if torch.cuda.device_count() != $GPU_COUNT:
    print(f'⚠️  PyTorch sees {torch.cuda.device_count()} GPUs, nvidia-smi shows $GPU_COUNT')
"

# Check wandb
echo "📊 Checking Weights & Biases..."
if python3 -c "import wandb" 2>/dev/null; then
    echo "✅ wandb installed"
    if wandb status | grep -q "Logged in"; then
        echo "✅ wandb logged in"
    else
        echo "⚠️  You need to login to wandb:"
        echo "   Run: wandb login"
        echo "   Get your API key from: https://wandb.ai/authorize"
    fi
else
    echo "❌ wandb not installed"
    pip install wandb
fi

# Check data structure
echo "📁 Checking data structure..."
required_dirs=("data/arc_agi_1" "data/arc_agi_2" "data/concept_arc")
missing_dirs=()

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ Found $dir"
    else
        echo "❌ Missing $dir"
        missing_dirs+=("$dir")
    fi
done

if [ ${#missing_dirs[@]} -gt 0 ]; then
    echo "❌ Missing required data directories:"
    for dir in "${missing_dirs[@]}"; do
        echo "   - $dir"
    done
    echo ""
    echo "Please ensure the ARC data is in the data/ directory with structure:"
    echo "  data/arc_agi_1/training/*.json"
    echo "  data/arc_agi_1/evaluation/*.json"  
    echo "  data/arc_agi_2/training/*.json"
    echo "  data/arc_agi_2/evaluation/*.json"
    echo "  data/concept_arc/*/*.json"
    exit 1
fi

# Check experiment scripts
echo "🧪 Checking experiment scripts..."
if [ -f "experiments/prepare_data.py" ] && [ -f "experiments/run_experiments.py" ]; then
    echo "✅ Experiment scripts found"
else
    echo "❌ Experiment scripts missing"
    exit 1
fi

echo ""
echo "✅ Setup complete! Next steps:"
echo ""
echo "1. Login to wandb (if not already done):"
echo "   wandb login"
echo ""
echo "2. Prepare datasets:"
echo "   cd experiments"
echo "   python prepare_data.py"
echo ""
echo "3. Run experiments:"
echo "   python run_experiments.py"
echo ""
echo "The experiments will:"
echo "- Start with step_0 pipeline test (~5-10 min)"
echo "- Use all $GPU_COUNT GPUs with distributed training"
echo "- Log all results to wandb project 'hrm_experiments'"
echo "- Save checkpoints and logs locally"
echo ""
echo "Monitor progress at: https://wandb.ai/your-username/hrm_experiments"