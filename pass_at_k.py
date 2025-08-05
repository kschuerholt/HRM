"""
Pass@k evaluation logic for ARC puzzles, adapted from arc_eval.ipynb.

This module provides functionality to compute pass@k metrics for ARC puzzle evaluation
by handling grid cropping, augmentation reversal, and confidence-based ranking.
"""

import os
import json
from typing import Dict, List
import numpy as np
import torch

try:
    from dataset.common import inverse_dihedral_transform
except ImportError:
    print("Warning: Could not import inverse_dihedral_transform from dataset.common")

    def inverse_dihedral_transform(grid, trans_id):
        """Fallback implementation - just return original grid"""
        return grid


PAD_PUZZLE_IDENTIFIER = 0


def crop_arc_grid(grid: np.ndarray, grid_size: int = 30) -> np.ndarray:
    """
    Crop ARC grid to remove padding and EOS tokens.
    Based on the crop function from arc_eval.ipynb.

    Args:
        grid: Input grid as flat or 2D array
        grid_size: Expected grid dimension (default 30x30)

    Returns:
        Cropped grid with token IDs converted to color values
    """
    # Always reshape to ensure consistent array shape
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
            if (x < 2) | (x > 11):
                num_c = c - 1
                break

        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    if max_size[0] == 0 or max_size[1] == 0:
        return np.array([[0]], dtype=grid.dtype)

    return grid[: max_size[0], : max_size[1]] - 2


def grid_hash(grid: np.ndarray) -> int:
    """Create hash for grid comparison."""
    return hash((grid.tobytes(), grid.shape))


def inverse_aug(name: str, grid: np.ndarray) -> np.ndarray:
    """
    Apply inverse augmentation transformation to restore original grid.

    Args:
        name: Augmented puzzle name (e.g., "puzzle_t3_4120")
        grid: Grid to transform

    Returns:
        Grid with inverse augmentation applied
    """
    if "_" not in name:
        return grid

    parts = name.split("_")
    if len(parts) < 3:
        return grid

    try:
        trans_id, perm = parts[-2:]
        trans_id = int(trans_id[1:])  # Remove "t" letter
        inv_perm = np.argsort(list(perm))
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]
    except (ValueError, IndexError, TypeError):
        return grid


def compute_pass_at_k_from_preds(
    all_preds: Dict[str, torch.Tensor], identifier_map: Dict[int, str], k_values: List[int] = [1, 2, 10]
) -> Dict[str, float]:
    """
    Compute pass@k metrics based on model predictions.

    Args:
        all_preds: Dictionary containing model predictions with keys:
            - "puzzle_identifiers": Tensor of puzzle IDs
            - "inputs": Input grids
            - "labels": Ground truth grids
            - "logits": Model logits
            - "q_halt_logits": Halting confidence scores
        identifier_map: Mapping from puzzle IDs to names
        k_values: List of k values to compute pass@k for

    Returns:
        Dictionary with pass@k metrics
    """
    # Validate inputs
    required_keys = ["puzzle_identifiers", "inputs", "labels", "logits"]
    if not all_preds or not all(key in all_preds for key in required_keys):
        return {}

    # Remove padded entries
    mask = all_preds["puzzle_identifiers"] != PAD_PUZZLE_IDENTIFIER
    filtered_preds = {k: v[mask] for k, v in all_preds.items()}

    # Get argmax predictions
    preds = filtered_preds["logits"].argmax(-1)

    # Group ground truth by puzzle and input
    puzzle_labels = {}
    global_hmap = {}

    for identifier, input_tensor, label_tensor in zip(
        filtered_preds["puzzle_identifiers"], filtered_preds["inputs"], filtered_preds["labels"]
    ):
        name = identifier_map.get(identifier.item(), f"unknown_{identifier.item()}")

        # Only process non-augmented puzzles for ground truth
        if "_" not in name:
            puzzle_labels.setdefault(name, {})

            input_grid = crop_arc_grid(input_tensor.cpu().numpy())
            label_grid = crop_arc_grid(label_tensor.cpu().numpy())

            input_hash = grid_hash(input_grid)
            label_hash = grid_hash(label_grid)

            global_hmap[input_hash] = input_grid
            global_hmap[label_hash] = label_grid

            if input_hash not in puzzle_labels[name]:
                puzzle_labels[name][input_hash] = label_hash

    if not puzzle_labels:
        return {}

    # Collect predictions grouped by puzzle and input
    pred_answers = {}
    q_halt_logits = filtered_preds.get("q_halt_logits")

    for i, (identifier, input_tensor, pred) in enumerate(
        zip(filtered_preds["puzzle_identifiers"], filtered_preds["inputs"], preds)
    ):
        name = identifier_map.get(identifier.item(), f"unknown_{identifier.item()}")
        orig_name = name.split("_")[0]

        if orig_name not in puzzle_labels:
            continue

        # Apply inverse augmentation to get original input
        input_grid = inverse_aug(name, crop_arc_grid(input_tensor.cpu().numpy()))
        input_hash = grid_hash(input_grid)

        if input_hash not in puzzle_labels[orig_name]:
            continue

        # Apply inverse augmentation to prediction
        pred_grid = inverse_aug(name, crop_arc_grid(pred.cpu().numpy()))
        pred_hash = grid_hash(pred_grid)
        global_hmap[pred_hash] = pred_grid

        # Get confidence score
        confidence = 1.0  # Default confidence
        if q_halt_logits is not None:
            confidence = torch.sigmoid(q_halt_logits[i]).item()

        pred_answers.setdefault(orig_name, {})
        pred_answers[orig_name].setdefault(input_hash, [])
        pred_answers[orig_name][input_hash].append((pred_hash, confidence))

    # Compute pass@k metrics
    results = {}
    total_puzzles = len(puzzle_labels)

    for k in k_values:
        correct_puzzles = 0

        for puzzle_name, tests in puzzle_labels.items():
            if puzzle_name not in pred_answers:
                continue

            puzzle_correct = True

            for input_hash, label_hash in tests.items():
                if input_hash not in pred_answers[puzzle_name]:
                    puzzle_correct = False
                    break

                # Get top-k predictions by confidence
                predictions = pred_answers[puzzle_name][input_hash]

                # Aggregate by prediction hash (sum confidence, count occurrences)
                pred_map = {}
                for pred_hash, conf in predictions:
                    if pred_hash not in pred_map:
                        pred_map[pred_hash] = [0, 0]  # [count, total_confidence]
                    pred_map[pred_hash][0] += 1
                    pred_map[pred_hash][1] += conf

                # Average confidence and sort
                for pred_hash, stats in pred_map.items():
                    stats[1] /= stats[0]  # Average confidence

                top_k_preds = sorted(pred_map.items(), key=lambda x: x[1][1], reverse=True)[:k]

                # Check if any top-k prediction matches ground truth
                input_correct = any(pred_hash == label_hash for pred_hash, _ in top_k_preds)

                if not input_correct:
                    puzzle_correct = False
                    break

            if puzzle_correct:
                correct_puzzles += 1

        pass_at_k = correct_puzzles / total_puzzles if total_puzzles > 0 else 0.0
        results[f"pass_{k}"] = pass_at_k

    return results


def evaluate_pass_at_k(
    model, eval_loader, data_path: str, k_values: List[int] = [1, 2, 10], rank: int = 0, world_size: int = 1
) -> dict:
    """
    Evaluate pass@k metrics by running forward passes on evaluation data.

    Args:
        model: The trained model to evaluate
        eval_loader: DataLoader for evaluation data
        data_path: Path to dataset directory (for identifier map)
        k_values: List of k values to compute pass@k for
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Dictionary with pass@k metrics (e.g., {"pass_1": 0.4, "pass_2": 0.6})
    """
    try:
        # Load identifier map
        identifier_map = load_identifier_map(data_path)
        if not identifier_map:
            if rank == 0:
                print(f"[Rank {rank}]: No identifier map found, skipping pass@k computation")
            return {}

        # Run forward pass to collect predictions
        with torch.inference_mode():
            all_preds = {}

            carry = None
            for set_name, batch, global_batch_size in eval_loader:
                # To device
                batch = {k: v.cuda() for k, v in batch.items()}
                with torch.device("cuda"):
                    carry = model.initial_carry(batch)

                # Forward pass
                while True:
                    carry, _, metrics, preds, all_finish = model(
                        carry=carry,
                        batch=batch,
                        return_keys=["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits"],
                    )

                    if all_finish:
                        break

                # Collect predictions
                for key in ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits"]:
                    if key in batch:
                        all_preds.setdefault(key, [])
                        all_preds[key].append(batch[key].cpu())
                    elif key in preds:
                        all_preds.setdefault(key, [])
                        all_preds[key].append(preds[key].cpu())

            # Concatenate all predictions
            if all_preds:
                all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

                # All-gather predictions across ranks
                if world_size > 1:
                    import torch.distributed as dist

                    gathered_preds = {}
                    for key, value in all_preds.items():
                        gathered_values = [torch.zeros_like(value) for _ in range(world_size)]
                        dist.all_gather(gathered_values, value)
                        gathered_preds[key] = torch.cat(gathered_values, dim=0)
                    all_preds = gathered_preds

                # Compute pass@k metrics (only on rank 0)
                if rank == 0:
                    pass_at_k_results = compute_pass_at_k_from_preds(all_preds, identifier_map, k_values)

                    if pass_at_k_results:
                        print(f"[Rank {rank}]: Pass@k results: {pass_at_k_results}")

                    return pass_at_k_results

        return {}

    except Exception as e:
        if rank == 0:
            print(f"[Rank {rank}]: Error computing pass@k metrics: {e}")
        return {}


# Compatibility alias for the old function name
def compute_pass_at_k(all_preds, identifier_map, dataset_path, k_values=[1, 2, 10]):
    """Compatibility wrapper for the renamed function."""
    return compute_pass_at_k_from_preds(all_preds, identifier_map, k_values)


def load_identifier_map(dataset_path: str) -> Dict[int, str]:
    """
    Load puzzle identifier mapping from dataset.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Dictionary mapping puzzle IDs to names
    """
    identifiers_file = os.path.join(dataset_path, "identifiers.json")

    if not os.path.exists(identifiers_file):
        return {}

    try:
        with open(identifiers_file, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Convert string keys to integers if needed
            if data and isinstance(next(iter(data.keys())), str):
                return {int(k): v for k, v in data.items()}
            return data
        elif isinstance(data, list):
            # Convert list format [name0, name1, name2, ...] to dict {0: name0, 1: name1, ...}
            return {i: name for i, name in enumerate(data) if isinstance(name, str)}

        return {}
    except Exception:
        return {}
