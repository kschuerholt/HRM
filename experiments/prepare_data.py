#!/usr/bin/env python3
"""
Prepare ARC datasets for experiments.

This script uses the existing build_arc_dataset.py to create datasets with different
augmentation levels needed for our experiments.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add HRM root to path
HRM_ROOT = Path(__file__).parent.parent
sys.path.append(str(HRM_ROOT))


def check_local_arc_data():
    """Check if local ARC data is available in HRM/data."""
    arc_data_path = HRM_ROOT / "data"
    
    if not arc_data_path.exists():
        print("❌ ARC data directory not found at:")
        print(f"   {arc_data_path}")
        print("\nPlease ensure ARC data is available in HRM/data/")
        return False
    
    # Check for ARC datasets
    arc_dirs = ["arc_agi_1", "arc_agi_2", "concept_arc"]
    found_dirs = []
    
    for arc_dir in arc_dirs:
        dir_path = arc_data_path / arc_dir
        if dir_path.exists():
            found_dirs.append(arc_dir)
            # Count JSON files
            if arc_dir in ["arc_agi_1", "arc_agi_2"]:
                training_files = list((dir_path / "training").glob("*.json")) if (dir_path / "training").exists() else []
                eval_files = list((dir_path / "evaluation").glob("*.json")) if (dir_path / "evaluation").exists() else []
                print(f"✅ Found {arc_dir}: {len(training_files)} training + {len(eval_files)} evaluation puzzles")
            else:
                # Concept ARC has different structure
                subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                print(f"✅ Found {arc_dir}: {len(subdirs)} concept directories")
    
    if not found_dirs:
        print("❌ No ARC data directories found in HRM/data/")
        print("Expected: arc_agi_1, arc_agi_2, or concept_arc")
        return False
    
    print(f"✅ ARC data available: {', '.join(found_dirs)}")
    return True


def build_arc_dataset(num_aug: int, output_dir: str, dataset_type: str = "full"):
    """Build ARC dataset with specified augmentation level."""
    
    print(f"\n{'='*60}")
    print(f"Building ARC dataset: {num_aug} augmentations")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Change to HRM root
    original_cwd = os.getcwd()
    os.chdir(HRM_ROOT)
    
    try:
        # Import the build functions directly to avoid CLI parsing issues
        sys.path.append('dataset')
        from build_arc_dataset import DataProcessConfig, convert_dataset
        
        # Create config with local paths
        if dataset_type == "step_0":
            # For step_0, use only a small subset of data for fast testing
            dataset_dirs = ["data/arc_agi_1"]  # Only use arc_agi_1 for minimal dataset
            print("🚀 Creating minimal dataset for step_0 pipeline testing")
        else:
            dataset_dirs = ["data/arc_agi_1", "data/concept_arc"]
        
        config = DataProcessConfig(
            dataset_dirs=dataset_dirs,
            output_dir=output_dir,
            num_aug=num_aug,
            seed=42
        )
        
        print(f"Dataset dirs: {config.dataset_dirs}")
        print(f"Augmentations: {config.num_aug}")
        
        # Verify paths exist
        for dataset_dir in config.dataset_dirs:
            path = Path(dataset_dir)
            if not path.exists():
                print(f"❌ Dataset directory not found: {path}")
                return False
            print(f"✓ Found dataset directory: {path}")
        
        # Build the dataset
        convert_dataset(config)
        
        print(f"✅ Successfully built dataset with {num_aug} augmentations")
        print(f"   Saved to: {output_dir}")
        
        # Verify the dataset was created
        output_path = Path(output_dir)
        if (output_path / "train").exists() and (output_path / "test").exists():
            # Count files to verify
            train_files = list((output_path / "train").glob("*.npy"))
            test_files = list((output_path / "test").glob("*.npy"))
            print(f"   Train files: {len(train_files)}")
            print(f"   Test files: {len(test_files)}")
            
            # Check identifiers file
            if (output_path / "identifiers.json").exists():
                with open(output_path / "identifiers.json", 'r') as f:
                    identifiers = json.load(f)
                print(f"   Puzzle identifiers: {len(identifiers)}")
        else:
            print("⚠️  Dataset created but train/test directories not found")
            return False
            
    except Exception as e:
        print(f"❌ Dataset building failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_cwd)
    
    return True


def main():
    """Prepare all required datasets."""
    
    print("HRM ARC Dataset Preparation")
    print("=" * 60)
    
    # Check if local ARC data is available
    if not check_local_arc_data():
        return
    
    # Check if build script exists
    build_script = HRM_ROOT / "dataset" / "build_arc_dataset.py"
    if not build_script.exists():
        print(f"❌ Build script not found: {build_script}")
        return
    
    # Define datasets to build
    datasets = [
        (0, "data/arc-test-minimal", "step_0"),  # Minimal dataset for pipeline testing
        (0, "data/arc-aug-0"),
        (300, "data/arc-aug-300"), 
        (1000, "data/arc-aug-1000")
    ]
    
    print(f"\n📋 Will build {len(datasets)} datasets:")
    for dataset_spec in datasets:
        if len(dataset_spec) == 3:
            num_aug, output_dir, dataset_type = dataset_spec
            type_desc = f" ({dataset_type})" if dataset_type != "full" else ""
        else:
            num_aug, output_dir = dataset_spec
            type_desc = ""
        estimated_time = 10 + (num_aug * 0.01)  # Rough estimate
        print(f"  - {num_aug} augmentations → {output_dir}{type_desc} (~{estimated_time:.1f} min)")
    
    # Check existing datasets
    existing = []
    for dataset_spec in datasets:
        if len(dataset_spec) == 3:
            num_aug, output_dir, _ = dataset_spec
        else:
            num_aug, output_dir = dataset_spec
        path = HRM_ROOT / output_dir
        if path.exists() and (path / "train").exists():
            existing.append(dataset_spec)
            print(f"✅ Dataset already exists: {output_dir}")
    
    # Filter out existing datasets
    to_build = [d for d in datasets if d not in existing]
    
    if not to_build:
        print("\n✅ All datasets already exist!")
        return
    
    total_est_time = sum(10 + (dataset_spec[0] * 0.01) for dataset_spec in to_build)
    print(f"\nEstimated total time: {total_est_time:.1f} minutes")
    
    response = input(f"\nBuild {len(to_build)} missing datasets? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Build datasets
    successful = 0
    for dataset_spec in to_build:
        if len(dataset_spec) == 3:
            num_aug, output_dir, dataset_type = dataset_spec
        else:
            num_aug, output_dir = dataset_spec
            dataset_type = "full"
        
        if build_arc_dataset(num_aug, output_dir, dataset_type):
            successful += 1
        else:
            print(f"⚠️  Continuing with remaining datasets...")
    
    print(f"\n{'='*60}")
    print("DATASET PREPARATION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully built: {successful}/{len(to_build)} datasets")
    
    if successful == len(to_build):
        print("✅ All datasets ready for experiments!")
    else:
        print(f"⚠️  {len(to_build) - successful} datasets failed to build")
    
    # Show final status
    print(f"\nDataset status:")
    for dataset_spec in datasets:
        if len(dataset_spec) == 3:
            num_aug, output_dir, dataset_type = dataset_spec
            type_desc = f" ({dataset_type})" if dataset_type != "full" else ""
        else:
            num_aug, output_dir = dataset_spec
            type_desc = ""
        path = HRM_ROOT / output_dir
        status = "✅" if path.exists() and (path / "train").exists() else "❌"
        print(f"  {status} {output_dir}{type_desc}")


if __name__ == "__main__":
    main()