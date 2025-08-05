#!/usr/bin/env python3
"""
Evaluate trained model checkpoints using HRM's existing evaluation logic.

This script adapts the logic from arc_eval.ipynb to evaluate our experimental checkpoints.
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

# Add HRM root to path
HRM_ROOT = Path(__file__).parent.parent
sys.path.append(str(HRM_ROOT))

# Import from their existing evaluation notebook logic
try:
    from dataset.common import inverse_dihedral_transform
except ImportError:
    print("Could not import HRM modules. Make sure you're in the HRM directory.")
    sys.exit(1)


class HRMCheckpointEvaluator:
    """Evaluates HRM checkpoints using their existing logic."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.pad_puzzle_identifier = 0
        
        # Load puzzle identifiers mapping
        identifiers_file = self.dataset_path / "identifiers.json"
        if identifiers_file.exists():
            with open(identifiers_file, 'r') as f:
                self.identifier_map = json.load(f)
        else:
            print(f"⚠️  Identifiers file not found: {identifiers_file}")
            self.identifier_map = {}
    
    def crop_arc_grid(self, grid: np.ndarray, grid_size: int = 30) -> np.ndarray:
        """
        Crop ARC grid to remove padding and EOS tokens.
        Based on their crop function from arc_eval.ipynb.
        """
        if len(grid.shape) == 1:
            grid = grid.reshape(grid_size, grid_size)
        
        # Find maximum-sized rectangle without any EOS token inside
        max_area = 0
        max_size = (0, 0)
        nr, nc = grid.shape
        
        num_c = nc
        for num_r in range(1, nr + 1):
            # Scan for maximum c
            for c in range(1, num_c + 1):
                x = grid[num_r - 1, c - 1]
                if (x < 2) or (x > 11):
                    num_c = c - 1
                    break
            
            area = num_r * num_c
            if area > max_area:
                max_area = area
                max_size = (num_r, num_c)
        
        if max_size[0] == 0 or max_size[1] == 0:
            return np.array([[0]], dtype=grid.dtype)  # Return minimal grid if nothing found
        
        cropped = grid[:max_size[0], :max_size[1]]
        return cropped - 2  # Convert from token IDs to color values
    
    def grid_hash(self, grid: np.ndarray) -> int:
        """Create hash for grid comparison."""
        return hash((grid.tobytes(), grid.shape))
    
    def inverse_aug(self, name: str, grid: np.ndarray) -> np.ndarray:
        """Inverse augmentation transformation."""
        if "_" not in name:
            return grid
        
        parts = name.split("_")
        if len(parts) < 3:
            return grid
        
        trans_id, perm = parts[-2:]
        try:
            trans_id = int(trans_id[1:])  # Remove "t" letter
            inv_perm = np.argsort(list(perm))
            return inv_perm[inverse_dihedral_transform(grid, trans_id)]
        except (ValueError, IndexError):
            return grid
    
    def load_checkpoint_predictions(self, checkpoint_path: str) -> Dict:
        """Load predictions from checkpoint files."""
        
        all_preds = {}
        
        # Look for prediction files
        pred_files = glob.glob(f"{checkpoint_path}_all_preds.*")
        
        if not pred_files:
            print(f"❌ No prediction files found for checkpoint: {checkpoint_path}")
            return {}
        
        print(f"Loading predictions from {len(pred_files)} files...")
        
        for filename in pred_files:
            try:
                preds = torch.load(filename, map_location='cpu')
                for k, v in preds.items():
                    all_preds.setdefault(k, [])
                    all_preds[k].append(v)
                del preds
            except Exception as e:
                print(f"⚠️  Could not load {filename}: {e}")
        
        if not all_preds:
            print("❌ No predictions loaded")
            return {}
        
        # Concatenate all predictions
        all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
        
        # Remove padding
        if "puzzle_identifiers" in all_preds:
            mask = all_preds["puzzle_identifiers"] != self.pad_puzzle_identifier
            all_preds = {k: v[mask] for k, v in all_preds.items()}
        
        print(f"Loaded {len(all_preds.get('inputs', []))} predictions")
        return all_preds
    
    def evaluate_checkpoint(self, checkpoint_path: str, k_values: List[int] = [1, 2, 10, 100]) -> Dict:
        """
        Evaluate checkpoint using pass@k metrics.
        Adapted from arc_eval.ipynb logic.
        """
        
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: {Path(checkpoint_path).name}")
        print(f"{'='*60}")
        
        # Load predictions
        all_preds = self.load_checkpoint_predictions(checkpoint_path)
        if not all_preds:
            return {"error": "Could not load predictions"}
        
        required_keys = ["puzzle_identifiers", "inputs", "labels", "logits"]
        missing_keys = [k for k in required_keys if k not in all_preds]
        if missing_keys:
            print(f"❌ Missing required keys: {missing_keys}")
            return {"error": f"Missing keys: {missing_keys}"}
        
        global_hmap = {}
        puzzle_labels = {}
        
        # Process ground truth labels
        print("Processing ground truth labels...")
        for identifier, input_tensor, label_tensor in zip(
            all_preds["puzzle_identifiers"], 
            all_preds["inputs"], 
            all_preds["labels"]
        ):
            
            if identifier.item() >= len(self.identifier_map):
                continue
                
            name = self.identifier_map[identifier.item()]
            if "_" not in name:  # Only use non-augmented puzzles for evaluation
                puzzle_labels.setdefault(name, {})
                
                input_grid = self.crop_arc_grid(input_tensor.numpy())
                label_grid = self.crop_arc_grid(label_tensor.numpy())
                
                input_hash = self.grid_hash(input_grid)
                label_hash = self.grid_hash(label_grid)
                
                global_hmap[input_hash] = input_grid
                global_hmap[label_hash] = label_grid
                
                puzzle_labels[name][input_hash] = label_hash
        
        print(f"Found {len(puzzle_labels)} puzzles for evaluation")
        
        if not puzzle_labels:
            return {"error": "No puzzles found for evaluation"}
        
        # Process predictions
        print("Processing predictions...")
        pred_answers = {}
        preds = all_preds["logits"].argmax(-1)
        
        for identifier, input_tensor, pred_tensor in zip(
            all_preds["puzzle_identifiers"],
            all_preds["inputs"], 
            preds
        ):
            
            if identifier.item() >= len(self.identifier_map):
                continue
                
            name = self.identifier_map[identifier.item()]
            orig_name = name.split("_")[0]
            
            if orig_name not in puzzle_labels:
                continue
            
            input_grid = input_tensor.numpy()
            input_hash = self.grid_hash(self.inverse_aug(name, self.crop_arc_grid(input_grid)))
            
            if input_hash not in puzzle_labels[orig_name]:
                continue
            
            pred_grid = self.inverse_aug(name, self.crop_arc_grid(pred_tensor.numpy()))
            pred_hash = self.grid_hash(pred_grid)
            global_hmap[pred_hash] = pred_grid
            
            pred_answers.setdefault(orig_name, {})
            pred_answers[orig_name].setdefault(input_hash, [])
            
            # For now, assume uniform confidence (could extract from q_halt_logits if available)
            confidence = 1.0
            pred_answers[orig_name][input_hash].append((pred_hash, confidence))
        
        # Compute pass@k metrics
        print("Computing pass@k metrics...")
        results = {}
        correct = {k: 0 for k in k_values}
        
        for name, tests in puzzle_labels.items():
            if name not in pred_answers:
                continue
                
            num_test_correct = {k: 0 for k in k_values}
            
            for input_hash, label_hash in tests.items():
                if input_hash not in pred_answers[name]:
                    continue
                    
                predictions = pred_answers[name][input_hash]
                
                # Group by prediction hash and aggregate confidence
                pred_map = {}
                for pred_hash, confidence in predictions:
                    pred_map.setdefault(pred_hash, [0, 0])
                    pred_map[pred_hash][0] += 1  # Count
                    pred_map[pred_hash][1] += confidence  # Sum confidence
                
                # Average confidence
                for pred_hash, stats in pred_map.items():
                    stats[1] /= stats[0]
                
                # Sort by confidence
                sorted_preds = sorted(pred_map.items(), 
                                    key=lambda x: x[1][1], reverse=True)
                
                # Check pass@k
                for k in k_values:
                    correct_in_topk = False
                    for i, (pred_hash, _) in enumerate(sorted_preds[:k]):
                        if pred_hash == label_hash:
                            correct_in_topk = True
                            break
                    
                    if correct_in_topk:
                        num_test_correct[k] += 1
            
            # Puzzle is "solved" if all test examples are correct
            for k in k_values:
                if num_test_correct[k] == len(tests):
                    correct[k] += 1
        
        # Final results
        total_puzzles = len(puzzle_labels)
        for k in k_values:
            accuracy = correct[k] / total_puzzles if total_puzzles > 0 else 0.0
            results[f"pass@{k}"] = accuracy
            print(f"pass@{k}: {accuracy:.3f} ({correct[k]}/{total_puzzles})")
        
        results["total_puzzles"] = total_puzzles
        results["checkpoint_path"] = checkpoint_path
        
        return results


def find_checkpoints(experiments_dir: str) -> List[str]:
    """Find all available checkpoints."""
    
    experiments_path = Path(experiments_dir)
    checkpoints = []
    
    # Look in common checkpoint locations
    for pattern in ["checkpoints/*/step_*", "outputs/*/checkpoints/step_*", "**/step_*"]:
        for path in experiments_path.rglob("step_*"):
            if path.is_file() and not path.name.endswith(('.json', '.yaml', '.log')):
                checkpoints.append(str(path))
    
    return checkpoints


def main():
    """Evaluate all available checkpoints."""
    
    print("HRM Checkpoint Evaluation")
    print("=" * 60)
    
    # Default paths
    dataset_path = HRM_ROOT / "data" / "arc-aug-1000"
    experiments_dir = HRM_ROOT / "experiments"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Run prepare_data.py first to create datasets")
        return
    
    # Find checkpoints
    checkpoints = find_checkpoints(experiments_dir)
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {experiments_dir}")
        print("Run experiments first with run_experiments.py")
        return
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints[:5]:  # Show first 5
        print(f"  {Path(cp).name}")
    if len(checkpoints) > 5:
        print(f"  ... and {len(checkpoints) - 5} more")
    
    # Create evaluator
    evaluator = HRMCheckpointEvaluator(str(dataset_path))
    
    # Evaluate checkpoints
    all_results = []
    
    for i, checkpoint_path in enumerate(checkpoints):
        print(f"\nEvaluating checkpoint {i+1}/{len(checkpoints)}")
        
        try:
            results = evaluator.evaluate_checkpoint(checkpoint_path)
            results["checkpoint_name"] = Path(checkpoint_path).name
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint_path}: {e}")
            all_results.append({
                "checkpoint_name": Path(checkpoint_path).name,
                "error": str(e)
            })
    
    # Save results
    results_file = experiments_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in all_results if "error" not in r]
    
    if successful:
        print(f"Successfully evaluated: {len(successful)}/{len(all_results)} checkpoints")
        
        # Show best results
        best_pass1 = max(successful, key=lambda x: x.get("pass@1", 0))
        best_pass2 = max(successful, key=lambda x: x.get("pass@2", 0))
        
        print(f"\nBest pass@1: {best_pass1.get('pass@1', 0):.3f} ({best_pass1['checkpoint_name']})")
        print(f"Best pass@2: {best_pass2.get('pass@2', 0):.3f} ({best_pass2['checkpoint_name']})")
        
    else:
        print("❌ No checkpoints successfully evaluated")
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()