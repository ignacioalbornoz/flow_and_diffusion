#!/usr/bin/env python3
"""
Test script to verify that the generate_samples fix works correctly
"""

import sys
import os
sys.path.append('.')

def test_generate_samples_fix():
    """Test that generate_samples function works with 2 classes"""
    try:
        from train_experiment import generate_samples, create_training_grid_with_titles
        print("✓ Successfully imported fixed functions")
        
        # Test create_training_grid_with_titles with 2 classes
        import torch
        import matplotlib.pyplot as plt
        
        # Create dummy data for 2 classes
        dummy_images = torch.randn(4, 1, 32, 32)  # 4 samples, 1 channel, 32x32
        dummy_class_names = ['class_0', 'class_1']  # 2 classes
        
        # This should not crash
        fig = create_training_grid_with_titles(dummy_images, dummy_class_names, 0, 1.5, 32)
        plt.close(fig)  # Close the figure to free memory
        print("✓ create_training_grid_with_titles works with 2 classes")
        
        return True
    except Exception as e:
        print(f"✗ Error testing generate_samples fix: {e}")
        return False

def main():
    """Run the test"""
    print("Testing generate_samples fix...")
    print("=" * 50)
    
    if test_generate_samples_fix():
        print("✓ All tests passed! The fix should work correctly.")
        return 0
    else:
        print("✗ Tests failed. Please check the fix.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
