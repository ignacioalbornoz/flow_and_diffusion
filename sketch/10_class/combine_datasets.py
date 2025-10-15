#!/usr/bin/env python3
"""
Script to combine the original Sketchy dataset with the quickdraw_extended dataset
for airplane and alarm_clock classes.
"""

import os
import shutil
from pathlib import Path

def combine_datasets():
    """Combine the two datasets for airplane and alarm_clock classes"""
    
    # Source directories
    original_dir = Path("/home/shared_data/Datasets/sketchy/train/sketch")
    quickdraw_dir = Path("/home/shared_data/sketches/quickdraw_extended")
    
    # Target directory (create a local copy)
    target_dir = Path("/home/ialbornoz/tesis/flow_and_diffusion/sketch/data/sketch")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    classes = ["airplane", "alarm_clock"]
    
    print("Combining datasets...")
    print(f"Original dataset: {original_dir}")
    print(f"Quickdraw dataset: {quickdraw_dir}")
    print(f"Target directory: {target_dir}")
    
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")
        
        # Original dataset path
        original_class_dir = original_dir / class_name
        quickdraw_class_dir = quickdraw_dir / class_name
        
        # Count existing images
        original_count = len(list(original_class_dir.glob("*.png"))) if original_class_dir.exists() else 0
        quickdraw_count = len(list(quickdraw_class_dir.glob("*.png"))) if quickdraw_class_dir.exists() else 0
        
        print(f"  Original {class_name}: {original_count} images")
        print(f"  Quickdraw {class_name}: {quickdraw_count} images")
        
        # Ensure target directory exists
        target_class_dir = target_dir / class_name
        target_class_dir.mkdir(exist_ok=True)
        
        # First, copy original images
        if original_class_dir.exists():
            copied_original = 0
            for img_file in original_class_dir.glob("*.png"):
                target_path = target_class_dir / img_file.name
                shutil.copy(img_file, target_path)
                copied_original += 1
            print(f"  ✅ Copied {copied_original} original images to {class_name}")
        
        # Then, copy quickdraw images
        if quickdraw_class_dir.exists():
            copied_count = 0
            for img_file in quickdraw_class_dir.glob("*.png"):
                # Create unique filename to avoid conflicts
                new_filename = f"quickdraw_{img_file.name}"
                target_path = target_class_dir / new_filename
                
                # Copy the file
                shutil.copy(img_file, target_path)
                copied_count += 1
                
                # Progress indicator
                if copied_count % 500 == 0:
                    print(f"    Copied {copied_count}/{quickdraw_count} images...")
            
            print(f"  ✅ Copied {copied_count} quickdraw images to {class_name}")
        else:
            print(f"  ❌ Quickdraw directory not found: {quickdraw_class_dir}")
    
    # Final count
    print("\n" + "="*50)
    print("FINAL DATASET COUNTS:")
    print("="*50)
    
    total_images = 0
    for class_name in classes:
        class_dir = target_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*.png")))
            total_images += count
            print(f"{class_name}: {count} images")
        else:
            print(f"{class_name}: 0 images")
    
    print(f"\nTOTAL: {total_images} images")
    print("="*50)

if __name__ == "__main__":
    combine_datasets()
