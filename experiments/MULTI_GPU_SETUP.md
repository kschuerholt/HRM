# Multi-GPU Training Setup for HRM Experiments

This guide covers setting up distributed training on up to 8 H100/A100 GPUs for comprehensive HRM analysis.

## Quick Start

### Option 1: Direct Python Execution
```bash
# Set up environment
export WANDB_API_KEY=your_api_key
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run all experiments
python run_distributed_experiments.py --suite all

# Or run specific experiment suites
python run_distributed_experiments.py --suite baseline
python run_distributed_experiments.py --suite augmentation
python run_distributed_experiments.py --suite hierarchical
```

### Option 2: Docker Multi-GPU
```bash
# Build container with GPU support
docker-compose build

# Run distributed experiments
docker-compose --profile distributed up hrm-distributed

# Or interactive mode with all GPUs
docker-compose up -d hrm
docker-compose exec hrm bash
python run_distributed_experiments.py --suite baseline
```

### Option 3: SLURM Cluster
```bash
# Generate SLURM scripts
python run_distributed_experiments.py --create-slurm

# Submit all jobs
chmod +x slurm_scripts/submit_all.sh
./slurm_scripts/submit_all.sh
```

## Experiment Suites

### Available Suites
- **baseline**: Attempt to replicate paper results
- **augmentation**: Test impact of data augmentation (0, 300, 1000)
- **hierarchical**: Test H/L cycle configurations (1×1, 2×2, 4×4, etc.)
- **deep_transformer**: Compare against deeper standard transformers
- **act_alternatives**: Compare ACT against simpler mechanisms
- **puzzle_embeddings**: Test puzzle embedding necessity (0, 8, 16, 32 dims)

### Configuration Scaling
All configurations are automatically scaled for multi-GPU training:

| Parameter | Single GPU | 8 GPUs | Scaling Logic |
|-----------|------------|--------|---------------|
| batch_size | 32 | 256 | Linear scaling |
| learning_rate | 3e-4 | 8.5e-4 | sqrt(world_size) scaling |
| hidden_size | 512 | 768 | Increased for better GPU utilization |
| log_interval | 20 | 5 | More frequent logging |

## Resource Requirements

### Estimated Training Times (8 H100 GPUs)
| Experiment Suite | Wall Clock | GPU Hours | Est. Cost (A100) |
|------------------|------------|-----------|-------------------|
| Baseline | 4 hours | 32 | $64 |
| Augmentation (3 configs) | 10 hours | 80 | $160 |
| Hierarchical (5 configs) | 15 hours | 120 | $240 |
| Deep Transformer (2 configs) | 8 hours | 64 | $128 |
| Puzzle Embeddings (4 configs) | 12 hours | 96 | $192 |
| **TOTAL** | **49 hours** | **392** | **$784** |

### Memory Requirements
- **Minimum**: 40GB per GPU (A100 40GB sufficient)
- **Recommended**: 80GB per GPU (H100 80GB ideal)
- **Peak usage**: ~35GB per GPU with largest models

## Configuration Details

### Model Scaling
Models are automatically scaled based on available GPUs:

```python
# Base configuration (automatically adjusted)
{
    "hidden_size": 768,      # Increased from 512 for multi-GPU
    "num_heads": 12,         # Scales with hidden_size
    "num_layers": 8,         # Deep enough for meaningful comparison
    "batch_size": 64,        # Per GPU, total = 64 × 8 = 512
    "learning_rate": 3e-4,   # Scaled by sqrt(world_size)
}
```

### Distributed Training Features
- **Automatic device placement**: Models and data automatically placed on correct GPUs
- **Gradient synchronization**: All-reduce after each backward pass
- **Load balancing**: Data equally distributed across GPUs
- **Memory optimization**: Activations distributed, gradients synchronized

## Monitoring and Logging

### Weights & Biases Integration
All experiments automatically log to W&B with:
- Real-time loss and accuracy curves
- GPU utilization and memory usage
- Gradient norms and parameter statistics
- Pass@k metrics during training
- Distributed training diagnostics

### Local Logging
Results saved to:
- `experiment_results/`: JSON files with full metrics
- `logs/`: Training logs and debugging info
- `distributed_results/`: Multi-GPU specific results

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config
   "batch_size": 32,  # Instead of 64
   ```

2. **NCCL Timeout**
   ```bash
   export NCCL_TIMEOUT=3600  # Increase timeout
   export NCCL_DEBUG=INFO    # Enable debugging
   ```

3. **Port Conflicts**
   ```bash
   export MASTER_PORT=12356  # Change port
   ```

4. **Uneven GPU Memory**
   ```bash
   # Use specific GPUs
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   ```

### Performance Optimization

1. **Enable Mixed Precision** (implemented in base model):
   - Automatic FP16 training for 2× speedup
   - Gradient scaling to prevent underflow

2. **Optimize Data Loading**:
   - Multiple data loader workers
   - Pinned memory for faster GPU transfer
   - Prefetch batches to overlap computation

3. **Gradient Accumulation** (if needed):
   ```python
   # For very large effective batch sizes
   "gradient_accumulation_steps": 2
   ```

## Cluster Deployment

### SLURM Configuration
Generated SLURM scripts include:
- Proper GPU allocation (8 × A100/H100)
- Memory allocation (32GB × 8 = 256GB)
- Time limits (12 hours per experiment)
- Module loading for CUDA/Python
- Distributed training setup

### Example SLURM Script
```bash
#!/bin/bash
#SBATCH --job-name=HRM-Baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=12:00:00

export MASTER_ADDR=localhost
export MASTER_PORT=12355
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    run_distributed_experiments.py \
    --suite baseline
```

## Results Analysis

### Expected Outcomes
Based on comprehensive testing across all experiments:

1. **Baseline Replication**: Should achieve reported pass@1 scores
2. **Augmentation**: Expect diminishing returns after 300 augmentations
3. **Hierarchical**: 2×2 may show modest benefits, 4×4 likely overkill
4. **Deep Transformer**: May match or exceed HRM performance
5. **Puzzle Embeddings**: Critical for paper's approach, may hurt true generalization

### Comparative Analysis
All experiments track identical metrics for fair comparison:
- **pass@k** (k=1,2,10,100): Primary ARC evaluation metric
- **exact_accuracy**: Full sequence correctness
- **token_accuracy**: Per-token correctness
- **training_efficiency**: Steps to reach milestones
- **computational_cost**: FLOPs per accuracy point

## Next Steps

After running experiments:

1. **Analyze Results**: Use W&B dashboards for comparison
2. **Statistical Testing**: Multiple seeds for significance
3. **Failure Analysis**: Identify what types of puzzles fail
4. **Ablation Studies**: Deeper analysis of promising components
5. **Publication**: Document findings and methodology

## Support

For issues:
- Check GPU memory usage: `nvidia-smi`
- Monitor training: W&B dashboard
- Debug distributed: `NCCL_DEBUG=INFO`
- Contact: See repository issues for support