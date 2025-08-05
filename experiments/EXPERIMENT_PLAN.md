# HRM Comprehensive Experiment Plan

## Overview

This document outlines the systematic experiments to analyze HRM's claims and architecture choices. Each experiment isolates specific components to understand their contribution.

## Experiment List

### Experiment 0: Baseline Replication ✅
**Goal**: Reproduce paper's reported results with their full configuration

**Configuration**:
```python
{
    "model": "HRM-ACTV1",
    "hidden_size": 512,
    "num_heads": 8,
    "num_layers": 6,
    "H_cycles": 2,
    "L_cycles": 2,
    "halt_max_steps": 8,
    "puzzle_emb_ndim": 32,
    "num_augmentations": 1000,
    "learning_rate": 3e-4
}
```

**Metrics**: pass@1, pass@2, pass@10, exact_accuracy

---

### Experiment 1: Data Augmentation Ablation ✅
**Goal**: Test if massive augmentation (1000x) is necessary

**Configurations**:
| Name | Augmentations | Expected Result |
|------|---------------|-----------------|
| ARC-Aug-0 | 0 | Baseline performance |
| ARC-Aug-300 | 300 | Moderate improvement |
| ARC-Aug-1000 | 1000 | Best performance (if claim true) |

**Key Question**: Is there a point of diminishing returns?

---

### Experiment 2: H/L Cycles Ablation
**Goal**: Test if hierarchical H/L cycling provides benefits over standard transformers

**Configurations**:
| Name | H_cycles | L_cycles | Description |
|------|----------|----------|-------------|
| HRM-1x1 | 1 | 1 | Standard transformer |
| HRM-2x2 | 2 | 2 | Paper's setting |
| HRM-4x4 | 4 | 4 | Deeper hierarchical |
| HRM-1x4 | 1 | 4 | L-heavy (fast thinking) |
| HRM-4x1 | 4 | 1 | H-heavy (slow thinking) |

**Key Question**: Does H/L cycling help or is it just expensive redundancy?

---

### Experiment 3: Deep Transformer Baseline
**Goal**: Compare HRM against deeper standard transformers with similar parameters

**Configurations**:
| Name | Architecture | Parameters | Description |
|------|--------------|------------|-------------|
| HRM-2x2 | 6 layers, H=2, L=2 | ~50M | Hierarchical |
| Transformer-12L | 12 layers | ~50M | Deep standard |
| Transformer-6L | 6 layers | ~25M | Shallow standard |

**Key Question**: Is HRM better than a parameter-matched standard transformer?

---

### Experiment 4: ACT vs Alternative Mechanisms ✅
**Goal**: Compare ACT halting against simpler alternatives

**Configurations**:
| Name | Mechanism | Description |
|------|-----------|-------------|
| ACT-Standard | ACT with Q-learning | Paper's approach |
| Equilibrium | Iterate until convergence | Stop when residual < threshold |
| Fixed-Adaptive | Fixed steps based on difficulty | Easy=2, Hard=8 steps |
| Random-Steps | Random 1-8 steps | Baseline |

**Key Question**: Is learned halting better than simpler heuristics?

---

### Experiment 5: Puzzle Embedding Purpose
**Goal**: Test if puzzle embeddings enable memorization vs generalization

**Configurations**:
| Name | puzzle_emb_ndim | Description |
|------|----------------|-------------|
| No-Embedding | 0 | No puzzle-specific parameters |
| Small-Embedding | 8 | Minimal capacity |
| Medium-Embedding | 16 | Moderate capacity |
| Full-Embedding | 32 | Paper's setting |

**Analysis**: 
- Track performance on seen vs unseen puzzle examples
- Analyze embedding similarity between related puzzles

---

### Experiment 6: Train/Test Split Analysis
**Goal**: Test generalization under different data splitting strategies

**Configurations**:
| Name | Split Strategy | Description |
|------|---------------|-------------|
| Original-Split | Train examples from eval puzzles seen | Paper's approach |
| Clean-Split | No eval puzzle examples during training | True held-out test |
| Leave-One-Out | Rotate which puzzles are held out | Cross-validation |

**Key Question**: How much does seeing training examples from test puzzles help?

---

### Experiment 7: Confidence Calibration
**Goal**: Analyze if Q-values (confidence scores) are well-calibrated

**Analysis**:
1. Plot Q-values vs actual accuracy
2. Compute Expected Calibration Error (ECE)
3. Analyze by puzzle difficulty
4. Check if Q-values improve with training

**Key Question**: Do Q-values provide reliable confidence estimates?

---

## Metrics to Track

### Primary Metrics (ARC-specific)
- **pass@k** (k=1,2,10,100): Fraction of puzzles with all examples correct in top-k
- **pass@k_examples**: Fraction of individual examples correct in top-k
- **exact_accuracy**: Full sequence match accuracy
- **token_accuracy**: Per-token accuracy

### Secondary Metrics
- **Training efficiency**: Steps to reach performance milestones
- **Computational cost**: FLOPs, memory, runtime
- **Gradient norms**: Training stability
- **Attention patterns**: What the model focuses on

### ACT-specific Metrics
- **avg_halt_steps**: Average computation steps used
- **halt_distribution**: Distribution of steps by difficulty
- **q_halt_accuracy**: How well Q-values predict success
- **computation_efficiency**: Performance per FLOP

## Implementation Notes

### Parameter Matching
When comparing architectures, match total parameters:
- HRM-2x2 (6 layers): ~50M params
- Transformer-12L: ~50M params (adjust hidden_size)
- Account for puzzle embeddings in parameter count

### Fair Comparison Guidelines
1. Same training data and augmentation
2. Same optimization settings (LR, schedule, etc.)
3. Same number of training steps (not epochs)
4. Report confidence intervals over multiple seeds

### Computational Considerations
- Start with smaller models for rapid iteration
- Use CPU for compatibility, GPU for final runs
- Log all experiments to W&B for easy comparison
- Save checkpoints for post-hoc analysis

## Expected Outcomes

### If HRM's claims are valid:
1. Augmentation: Monotonic improvement with more augmentation
2. H/L Cycles: Better performance than parameter-matched transformer
3. ACT: Adaptive computation outperforms fixed/random
4. Puzzle Embeddings: Improve within-puzzle generalization

### If HRM's claims are invalid:
1. Augmentation: Diminishing returns after ~100 augmentations
2. H/L Cycles: No benefit over standard transformer
3. ACT: Similar to simpler heuristics
4. Puzzle Embeddings: Just memorization, hurt true generalization

## Analysis Plan

1. **Performance Analysis**: Compare pass@k across configurations
2. **Efficiency Analysis**: Performance per parameter/FLOP
3. **Ablation Impact**: Rank components by importance
4. **Generalization Study**: Seen vs unseen puzzle performance
5. **Failure Analysis**: What types of puzzles fail?

## Timeline

1. **Week 1**: Baseline replication + core ablations (Experiments 0-3)
2. **Week 2**: Alternative mechanisms + embedding analysis (Experiments 4-5)
3. **Week 3**: Generalization studies + calibration (Experiments 6-7)
4. **Week 4**: Analysis, writeup, and additional experiments as needed