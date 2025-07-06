#!/usr/bin/env python3
"""
Load and print target maps from different map types
"""

import numpy as np
import os
from pathlib import Path

def load_and_print_target_map(dataset_path, map_type, image_name=None):
    """Load and print a target map from the specified map type directory"""
    
    # Construct the path to the images directory
    images_path = Path(dataset_path) / map_type / "images"
    
    print(f"\n=== {map_type.upper()} TARGET MAP ===")
    print(f"Looking in: {images_path}")
    
    if not images_path.exists():
        print(f"ERROR: Directory {images_path} does not exist!")
        return None
    
    # Get list of .npy files
    npy_files = list(images_path.glob("*.npy"))
    
    if not npy_files:
        print(f"ERROR: No .npy files found in {images_path}")
        return None
    
    # Use specified image or pick the first one
    if image_name:
        target_file = images_path / f"{image_name}.npy"
        if not target_file.exists():
            print(f"ERROR: File {target_file} does not exist!")
            return None
    else:
        target_file = npy_files[0]
        print(f"Using first available file: {target_file.name}")
    
    # Load the numpy array
    try:
        target_map = np.load(target_file)
        print(f"Loaded target map from: {target_file}")
        print(f"Shape: {target_map.shape}")
        print(f"Data type: {target_map.dtype}")
        print(f"Min value: {target_map.min()}")
        print(f"Max value: {target_map.max()}")
        print(f"Unique values: {np.unique(target_map)}")
        
        print(f"\nTarget map preview (first 10x10):")
        print(target_map[:10, :10])
        
        return target_map
        
    except Exception as e:
        print(f"ERROR loading {target_file}: {e}")
        return None

def main():
    # Dataset path
    dataset_path = "/cluster/home/alesweber/TerraProject/terra/data/terra/train/"
    
    print(f"Dataset path: {dataset_path}")
    
    # Load target maps from both map types
    relocations_map = load_and_print_target_map(dataset_path, "relocations")
    foundation_map = load_and_print_target_map(dataset_path, "foundation")
    
    # Compare if both loaded successfully
    if relocations_map is not None and foundation_map is not None:
        print(f"\n=== COMPARISON ===")
        print(f"Relocations shape: {relocations_map.shape}")
        print(f"Foundation shape: {foundation_map.shape}")
        print(f"Same shape: {relocations_map.shape == foundation_map.shape}")
        
        if relocations_map.shape == foundation_map.shape:
            print(f"Maps are identical: {np.array_equal(relocations_map, foundation_map)}")
            print(f"Max difference: {np.abs(relocations_map - foundation_map).max()}")

if __name__ == "__main__":
    main() 