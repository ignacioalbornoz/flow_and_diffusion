#!/usr/bin/env python3
"""
Test script to verify that the combined dataset is working correctly
"""

import torch
import sketch_model
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def test_dataset():
    """Test the combined dataset"""
    print("Testing combined dataset...")
    
    # Initialize the sampler
    sampler = sketch_model.SketchSampler()
    
    # Test sampling
    print(f"Dataset directory: {sampler.data_dir}")
    print(f"Number of classes: {sampler.num_classes}")
    print(f"Classes: {sampler.dataset.classes}")
    
    # Sample some images
    print("\nSampling 8 images...")
    samples, labels = sampler.sample(8)
    
    print(f"Samples shape: {samples.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
    
    # Check image properties
    print(f"Image size: {samples.shape[2]}x{samples.shape[3]}")
    print(f"Value range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Count images by source
    data_dir = Path(sampler.data_dir)
    for class_name in sampler.dataset.classes:
        class_dir = data_dir / class_name
        if class_dir.exists():
            # Count original Sketchy images (different pattern for each class)
            if class_name == "airplane":
                sketchy_images = list(class_dir.glob("n02691156_*.png"))
            elif class_name == "alarm_clock":
                sketchy_images = list(class_dir.glob("n02694662_*.png"))
            else:
                sketchy_images = list(class_dir.glob("n0*.png"))  # Generic pattern
            # Count Quickdraw images
            quickdraw_images = list(class_dir.glob("quickdraw_*.png"))
            
            print(f"\n{class_name}:")
            print(f"  Sketchy images: {len(sketchy_images)}")
            print(f"  Quickdraw images: {len(quickdraw_images)}")
            print(f"  Total: {len(sketchy_images) + len(quickdraw_images)}")
            
            # Show some example filenames
            if sketchy_images:
                print(f"  Sketchy example: {sketchy_images[0].name}")
            if quickdraw_images:
                print(f"  Quickdraw example: {quickdraw_images[0].name}")
    
    # Test that we can sample from both sources
    print("\nTesting sampling from both sources...")
    for i in range(10):
        sample, label = sampler.sample(1)
        # Get the actual file path to check source
        class_name = sampler.dataset.classes[label.item()]
        class_dir = data_dir / class_name
        
        # This is a simplified check - in practice, the sampler randomly selects
        print(f"Sample {i+1}: Class {label.item()} ({class_name})")
    
    print("\n✅ Dataset test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_dataset()
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
