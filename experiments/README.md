# HRM Systematic Experiments

This directory contains systematic experiments to rigorously test the claims made in the HRM paper. All experiments use the existing HRM training infrastructure (`pretrain.py`) without modifying their core code.

## Directory Structure

```
experiments/
├── README.md                    # This file
├── EXPERIMENT_PLAN.md          # Detailed experiment plan and methodology
├── MULTI_GPU_SETUP.md          # Multi-GPU distributed training setup
├── prepare_data.py             # Data preparation script
├── run_experiments.py          # Main experiment runner
├── evaluate_checkpoints.py     # Checkpoint evaluation script
└── configs/                    # Experiment configurations
    ├── step_0_pipeline_test.yaml      # Fast pipeline test
    ├── baseline_replication.yaml      # Paper replication
    ├── ablation_augmentation.yaml     # Data augmentation ablation
    └── arch/                          # Architecture configurations
        ├── hrm_small.yaml             # Small model for testing
        ├── hrm_1x1.yaml               # Standard transformer
        └── hrm_no_puzzle_emb.yaml     # No puzzle embeddings
```

## Quick Start

### 1. Prepare Data
```bash
cd experiments
python prepare_data.py
```
This creates ARC datasets with 0, 300, and 1000 augmentations needed for experiments.

**Note**: The script now uses the ARC data located in `HRM/data/` (arc_agi_1, arc_agi_2, concept_arc) and builds datasets using HRM's existing infrastructure.

### 2. Run Experiments
```bash
python run_experiments.py
```
This runs all experiments using HRM's existing training infrastructure.

### 3. Evaluate Results
```bash
python evaluate_checkpoints.py
```
This evaluates trained checkpoints using their existing evaluation logic (from `arc_eval.ipynb`).

## Experiment Design

### Experiments Included

1. **Baseline Replication**: Attempt to reproduce paper results with their exact configuration
2. **Augmentation Ablation**: Test 0, 300, 1000 augmentations per puzzle
3. **Architecture Ablations**:
   - No hierarchical cycles (standard transformer)
   - No puzzle embeddings

### Using Their Infrastructure

The experiments are designed to use HRM's existing code:
- **Training**: Uses `pretrain.py` with custom config files
- **Data**: Uses their `build_arc_dataset.py` to create datasets from local ARC JSON files
- **Evaluation**: Adapts logic from their `arc_eval.ipynb`
- **Multi-GPU**: Uses their existing `torchrun` support

### Data Sources

The experiments use ARC data provided in `HRM/data/`:
- **arc_agi_1**: ARC-AGI dataset v1 (400 training + 400 evaluation puzzles)
- **arc_agi_2**: ARC-AGI dataset v2 (1000 training + 120 evaluation puzzles)  
- **concept_arc**: ConceptARC dataset (16 concept categories)

The `prepare_data.py` script processes these raw JSON files using HRM's existing `build_arc_dataset.py` to create the augmented training datasets needed for experiments.

## Configuration Details

### Baseline Replication
```yaml
# Matches paper configuration exactly
global_batch_size: 768
lr: 1e-4
puzzle_emb_lr: 1e-2  # Key component from paper
H_cycles: 2
L_cycles: 2
halt_max_steps: 16
```

### Augmentation Ablation
- **0 augmentations**: Original ARC data only
- **300 augmentations**: Moderate augmentation
- **1000 augmentations**: Paper's full setting

### Architecture Ablations
- **hrm_1x1**: H_cycles=1, L_cycles=1 (standard transformer)
- **hrm_no_puzzle_emb**: puzzle_emb_ndim=0 (no puzzle-specific parameters)

## Multi-GPU Support

The experiments automatically detect and use available GPUs:

```bash
# Single GPU
python run_experiments.py

# Multi-GPU (automatic detection)
# Uses torchrun if >1 GPU available
```
