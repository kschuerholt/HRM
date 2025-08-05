#!/usr/bin/env python3
"""
HRM Experiment Runner

Uses the existing HRM training infrastructure (pretrain.py) to run systematic experiments.
This script creates appropriate config files and calls their training script.

EXPERIMENT SUITE OVERVIEW:
==========================

Total Experiments: 7

1. step_0_pipeline_test (⚡ FAST TEST - 5-10 minutes)
   └── Purpose: Validate entire pipeline before long experiments
   └── Model: Small (128 hidden, 2 layers, 4 ACT steps)
   └── Data: Minimal dataset (arc_agi_1 only, 0 augmentations)
   └── Training: 100 epochs × 10 steps = 1000 steps
   └── Multi-GPU: Yes (tests distributed training)

2. baseline_replication
   └── Purpose: Attempt to replicate paper results with exact config
   └── Model: Full HRM (512 hidden, 4 layers, 16 ACT steps)
   └── Data: arc-aug-1000 (1000 augmentations per puzzle)
   └── Training: Full paper configuration

3. aug_ablation_0
   └── Purpose: Test performance without data augmentation
   └── Model: Full HRM
   └── Data: arc-aug-0 (no augmentations)
   └── Training: Standard configuration

4. aug_ablation_300
   └── Purpose: Test performance with moderate augmentation
   └── Model: Full HRM  
   └── Data: arc-aug-300 (300 augmentations per puzzle)
   └── Training: Standard configuration

5. aug_ablation_1000
   └── Purpose: Test performance with full augmentation (paper level)
   └── Model: Full HRM
   └── Data: arc-aug-1000 (1000 augmentations per puzzle)
   └── Training: Standard configuration

6. arch_no_hierarchy
   └── Purpose: Test HRM vs standard transformer
   └── Model: Standard transformer (H_cycles=1, L_cycles=1)
   └── Data: arc-aug-1000
   └── Training: Shorter (20000 epochs for ablation)

7. arch_no_puzzle_emb
   └── Purpose: Test importance of puzzle-specific embeddings
   └── Model: HRM without puzzle embeddings (puzzle_emb_ndim=0)
   └── Data: arc-aug-1000
   └── Training: Shorter (20000 epochs for ablation)

All experiments:
- Use 8-GPU distributed training (automatic detection)
- Log to Weights & Biases (project: 'hrm_experiments')
- Use HRM's existing pretrain.py infrastructure
- Save checkpoints and comprehensive logs
"""

import os
import sys
import subprocess
import yaml
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

# Add HRM root to path
HRM_ROOT = Path(__file__).parent.parent
sys.path.append(str(HRM_ROOT))


class HRMExperimentRunner:
    """Runner that uses HRM's existing training infrastructure."""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.hrm_root = HRM_ROOT
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.experiments_dir = self.hrm_root / "experiments"
        
        print(f"HRM Root: {self.hrm_root}")
        print(f"Output Dir: {self.output_dir}")
    
    def check_data_availability(self) -> Dict[str, bool]:
        """Check which ARC datasets are available."""
        data_paths = {
            "arc-test-minimal": self.hrm_root / "data" / "arc-test-minimal",
            "arc-aug-0": self.hrm_root / "data" / "arc-aug-0",
            "arc-aug-300": self.hrm_root / "data" / "arc-aug-300", 
            "arc-aug-1000": self.hrm_root / "data" / "arc-aug-1000"
        }
        
        availability = {}
        for name, path in data_paths.items():
            availability[name] = path.exists() and (path / "train").exists()
            print(f"Data {name}: {'✓' if availability[name] else '✗'} ({path})")
        
        return availability
    
    def create_config_file(self, base_config: str, overrides: Dict[str, Any], 
                          output_name: str) -> Path:
        """Create a config file with overrides."""
        
        # Load base config
        base_config_path = self.experiments_dir / "configs" / f"{base_config}.yaml"
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply overrides
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys like arch.halt_max_steps
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        
        # Save config
        output_config_path = self.output_dir / f"{output_name}.yaml"
        with open(output_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return output_config_path
    
    def run_training(self, config_path: Path, experiment_name: str, 
                    use_distributed: bool = False, num_gpus: int = 1) -> Dict[str, Any]:
        """Run training using HRM's pretrain.py script."""
        
        print(f"\n{'='*60}")
        print(f"Running Experiment: {experiment_name}")
        print(f"Config: {config_path}")
        print(f"GPUs: {num_gpus} ({'distributed' if use_distributed and num_gpus > 1 else 'single'})")
        print(f"{'='*60}")
        
        # Change to HRM root directory
        original_cwd = os.getcwd()
        os.chdir(self.hrm_root)
        
        try:
            start_time = time.time()
            
            if use_distributed and num_gpus > 1:
                # Use torchrun for distributed training
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={num_gpus}",
                    "--standalone",
                    "pretrain.py",
                    f"--config-path={self.output_dir.absolute()}",
                    f"--config-name={config_path.stem}"
                ]
            else:
                # Single GPU training
                cmd = [
                    "python", "pretrain.py",
                    f"--config-path={self.output_dir.absolute()}",
                    f"--config-name={config_path.stem}"
                ]
            
            print(f"Command: {' '.join(cmd)}")
            
            # Run the training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 12  # 12 hour timeout
            )
            
            runtime = time.time() - start_time
            
            # Parse results
            experiment_result = {
                "experiment_name": experiment_name,
                "config_path": str(config_path),
                "runtime_seconds": runtime,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            if result.returncode == 0:
                print(f"✅ Experiment {experiment_name} completed successfully in {runtime/3600:.2f} hours")
                
                # Try to extract final metrics from W&B logs or checkpoints
                try:
                    metrics = self._extract_metrics_from_logs(result.stdout)
                    experiment_result["metrics"] = metrics
                except Exception as e:
                    print(f"Could not extract metrics: {e}")
                    
            else:
                print(f"❌ Experiment {experiment_name} failed (code {result.returncode})")
                print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
            
            # Save result
            result_file = self.output_dir / f"{experiment_name}_result.json"
            with open(result_file, 'w') as f:
                json.dump(experiment_result, f, indent=2)
            
            return experiment_result
            
        except subprocess.TimeoutExpired:
            print(f"❌ Experiment {experiment_name} timed out after 12 hours")
            experiment_result = {
                "experiment_name": experiment_name,
                "error": "timeout",
                "runtime_seconds": 3600 * 12
            }
            return experiment_result
            
        except Exception as e:
            print(f"❌ Experiment {experiment_name} failed with exception: {e}")
            experiment_result = {
                "experiment_name": experiment_name,
                "error": str(e)
            }
            return experiment_result
            
        finally:
            os.chdir(original_cwd)
    
    def _extract_metrics_from_logs(self, stdout: str) -> Dict[str, Any]:
        """Extract metrics from training logs."""
        # This is a simple parser - could be enhanced
        metrics = {}
        
        lines = stdout.split('\n')
        for line in lines:
            if 'final accuracy' in line.lower() or 'test accuracy' in line.lower():
                # Try to extract accuracy numbers
                import re
                numbers = re.findall(r'(\d+\.?\d*)', line)
                if numbers:
                    metrics['final_accuracy'] = float(numbers[-1])
        
        return metrics


def create_experiment_definitions() -> List[Dict[str, Any]]:
    """Define all experiments to run."""
    
    experiments = []
    
    # Experiment 0: Step 0 - Fast pipeline test (run first!)
    experiments.append({
        "name": "step_0_pipeline_test",
        "base_config": "step_0_pipeline_test",
        "overrides": {
            "data_path": "data/arc-test-minimal"
        },
        "description": "Fast pipeline test - small model, minimal data, few epochs"
    })
    
    # Experiment 1: Baseline replication
    experiments.append({
        "name": "baseline_replication",
        "base_config": "baseline_replication",
        "overrides": {},
        "description": "Attempt to replicate paper results with their exact config"
    })
    
    # Experiment 2: Data augmentation ablation
    aug_levels = [0, 300, 1000]
    for aug in aug_levels:
        experiments.append({
            "name": f"aug_ablation_{aug}",
            "base_config": "ablation_augmentation", 
            "overrides": {
                "data_path": f"data/arc-aug-{aug}",
                "run_name": f"augmentation_{aug}"
            },
            "description": f"Augmentation ablation with {aug} augmentations per puzzle"
        })
    
    # Experiment 3: Architecture ablations
    arch_configs = [
        ("no_hierarchy", "hrm_1x1", "Standard transformer (no H/L cycles)"),
        ("no_puzzle_emb", "hrm_no_puzzle_emb", "HRM without puzzle embeddings"),
    ]
    
    for name, arch, desc in arch_configs:
        experiments.append({
            "name": f"arch_{name}",
            "base_config": "baseline_replication",
            "overrides": {
                "defaults": [f"/arch: {arch}", "_self_"],
                "run_name": f"architecture_{name}",
                "epochs": 20000  # Shorter for ablation
            },
            "description": desc
        })
    
    return experiments


def check_prerequisites():
    """Check that all required components are available."""
    hrm_root = HRM_ROOT
    
    required_files = [
        "pretrain.py",
        "config/cfg_pretrain.yaml", 
        "config/arch/hrm_v1.yaml",
        "models/hrm/hrm_act_v1.py"
    ]
    
    missing = []
    for file_path in required_files:
        if not (hrm_root / file_path).exists():
            missing.append(file_path)
    
    if missing:
        print("❌ Missing required files:")
        for f in missing:
            print(f"  {f}")
        return False
    
    print("✅ All required HRM files found")
    return True


def main():
    """Run all experiments."""
    
    print("HRM Systematic Experiments")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Setup runner
    runner = HRMExperimentRunner()
    
    # Check data availability
    data_available = runner.check_data_availability()
    
    # If no data is available, provide instructions
    if not any(data_available.values()):
        print("\n❌ No ARC data found!")
        print("You need to create the datasets first using:")
        print("  python experiments/prepare_data.py")
        return
    
    # Get experiments to run
    experiments = create_experiment_definitions()
    
    # Filter experiments based on available data
    filtered_experiments = []
    for exp in experiments:
        data_path = exp["overrides"].get("data_path", "data/arc-aug-1000")
        data_key = data_path.split("/")[-1]  # e.g., "arc-aug-1000"
        
        if data_key in data_available and data_available[data_key]:
            filtered_experiments.append(exp)
        else:
            print(f"⚠️  Skipping {exp['name']} - data not available: {data_path}")
    
    if not filtered_experiments:
        print("❌ No experiments can be run with available data")
        return
    
    print(f"\n📋 Running {len(filtered_experiments)} experiments:")
    for exp in filtered_experiments:
        print(f"  - {exp['name']}: {exp['description']}")
    
    # Check for GPU availability
    import torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n🚀 Found {num_gpus} GPU(s)")
        use_distributed = num_gpus > 1
    else:
        print("\n💻 No GPUs found, using CPU")
        num_gpus = 1
        use_distributed = False
    
    # Confirm before starting
    response = input(f"\nProceed with {len(filtered_experiments)} experiments? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run experiments
    results = []
    for i, exp in enumerate(filtered_experiments):
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(filtered_experiments)}: {exp['name']}")
        print(f"{'='*80}")
        
        # Create config file
        config_path = runner.create_config_file(
            exp["base_config"],
            exp["overrides"],
            exp["name"]
        )
        
        # Run experiment
        result = runner.run_training(
            config_path,
            exp["name"],
            use_distributed=use_distributed,
            num_gpus=num_gpus
        )
        
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r.get("return_code") == 0)
    total_time = sum(r.get("runtime_seconds", 0) for r in results)
    
    print(f"Completed: {successful}/{len(results)} experiments")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {runner.output_dir}")
    
    for result in results:
        status = "✅" if result.get("return_code") == 0 else "❌"
        runtime = result.get("runtime_seconds", 0)
        print(f"  {status} {result['experiment_name']}: {runtime/3600:.2f}h")


if __name__ == "__main__":
    main()