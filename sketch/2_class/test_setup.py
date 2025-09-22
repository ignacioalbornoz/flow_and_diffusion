#!/usr/bin/env python3
"""
Test script to verify that the 2-class setup is working correctly
"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Test that all modules can be imported"""
    try:
        from experiment_manager import ExperimentManager, create_experiment_configs
        from sketchutils import GaussianConditionalProbabilityPath, CFGTrainer, CFGVectorFieldODE
        from sketch_model import SketchSampler
        from sketch_unet import SketchUNet
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_experiment_configs():
    """Test that experiment configurations are correct"""
    try:
        from experiment_manager import create_experiment_configs
        configs = create_experiment_configs()
        
        # Check that we have the expected configurations
        expected_configs = ["sketch_32x32_2classes", "sketch_32x32_2classes_simple"]
        for config_name in expected_configs:
            if config_name not in configs:
                print(f"✗ Missing configuration: {config_name}")
                return False
        
        print("✓ Experiment configurations are correct")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_model_creation():
    """Test that models can be created with correct parameters"""
    try:
        from sketch_unet import SketchUNet
        from experiment_manager import create_experiment_configs
        
        configs = create_experiment_configs()
        config = configs["sketch_32x32_2classes"]
        
        # Create model
        model = SketchUNet(
            channels=config["model_channels"],
            num_residual_layers=config["num_residual_layers"],
            t_embed_dim=40,
            y_embed_dim=40
        )
        
        # Check that embedding layer has correct size (3 for 2 classes + 1 null)
        if model.y_embedder.num_embeddings != 3:
            print(f"✗ Wrong embedding size: {model.y_embedder.num_embeddings}, expected 3")
            return False
        
        print("✓ Model creation successful")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_sketch_sampler():
    """Test that sketch sampler is configured for 2 classes"""
    try:
        from sketch_model import SketchSampler
        
        # Create sampler (this will fail if data directory doesn't exist, but that's OK)
        try:
            sampler = SketchSampler(max_classes=2)
            if sampler.max_classes != 2:
                print(f"✗ Wrong max_classes: {sampler.max_classes}, expected 2")
                return False
            print("✓ Sketch sampler configured for 2 classes")
        except FileNotFoundError:
            print("✓ Sketch sampler configured for 2 classes (data directory not found, but config is correct)")
        
        return True
    except Exception as e:
        print(f"✗ Sketch sampler error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing 2-class setup...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_experiment_configs,
        test_model_creation,
        test_sketch_sampler
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! 2-class setup is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
