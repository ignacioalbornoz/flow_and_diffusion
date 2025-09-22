import os
import json
from datetime import datetime
import shutil

class ExperimentManager:
    def __init__(self, experiment_name: str, base_dir: str = ".", experiment_id: str = None):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        
        # Generate unique experiment ID if not provided
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{experiment_name}_{timestamp}"
        
        self.experiment_id = experiment_id
        self.experiment_dir = os.path.join(base_dir, "experiments", experiment_id)
        
        # Create experiment directories
        self.ckps_dir = os.path.join(self.experiment_dir, "ckps")
        self.samples_dir = os.path.join(self.experiment_dir, "samples")
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        self.results_dir = os.path.join(self.experiment_dir, "results")
        
        os.makedirs(self.ckps_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save experiment config
        self.config = {
            "experiment_name": experiment_name,
            "experiment_id": self.experiment_id,
            "created_at": datetime.now().isoformat(),
            "base_dir": base_dir
        }
        
    def save_config(self, config_dict: dict):
        """Save experiment configuration"""
        self.config.update(config_dict)
        config_file = os.path.join(self.experiment_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Config saved to: {config_file}")
    
    def get_paths(self):
        """Get all experiment paths"""
        return {
            "experiment_dir": self.experiment_dir,
            "ckps_dir": self.ckps_dir,
            "samples_dir": self.samples_dir,
            "logs_dir": self.logs_dir,
            "results_dir": self.results_dir
        }
    
    def list_experiments(self):
        """List all experiments"""
        experiments_dir = os.path.join(self.base_dir, "experiments")
        if not os.path.exists(experiments_dir):
            print("No experiments found")
            return []
        
        experiments = []
        for exp_name in os.listdir(experiments_dir):
            exp_path = os.path.join(experiments_dir, exp_name)
            if os.path.isdir(exp_path):
                config_file = os.path.join(exp_path, "config.json")
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    experiments.append((exp_name, config))
                else:
                    experiments.append((exp_name, {"created_at": "unknown"}))
        
        return experiments

def create_experiment_configs():
    """Create different experiment configurations"""
    configs = {
        "sketch_32x32_2classes": {
            "description": "32x32 resolution, 2 classes, research-grade model",
            "image_size": 32,
            "model_channels": [64, 128, 256, 512],  # Much larger model
            "num_residual_layers": 4,  # More layers
            "learning_rate": 1e-4,  # Lower learning rate for stability
            "guidance_scale": 1.5,
            "eta": 0.1,  # Lower eta for less null class probability
            "weight_multiplier": 1.0,  # No weighting
            "batch_size": 32,  # Moderate batch size
            "num_epochs": 10000,  # Extended training for maximum convergence
            "checkpoint_interval": 50,  # More frequent checkpoints
            "sample_interval": 50  # Generate samples every 50 epochs
        },
        "sketch_32x32_2classes_simple": {
            "description": "32x32 resolution, 2 classes, simpler model for faster training",
            "image_size": 32,
            "model_channels": [32, 64, 128, 256],  # Smaller model for faster training
            "num_residual_layers": 2,  # Fewer layers
            "learning_rate": 1e-4,  # Lower learning rate for stability
            "guidance_scale": 1.5,
            "eta": 0.1,  # Lower eta for less null class probability
            "weight_multiplier": 1.0,  # No weighting
            "batch_size": 64,  # Larger batch size for faster training
            "num_epochs": 5000,  # Fewer epochs for faster training
            "checkpoint_interval": 25,  # More frequent checkpoints
            "sample_interval": 25  # Generate samples every 25 epochs
        },
        "sketch_64x64_2classes": {
            "description": "64x64 resolution, 2 classes, research-grade model (optimized for memory)",
            "image_size": 64,
            "model_channels": [64, 128, 256, 512],  # Original size model
            "num_residual_layers": 4,  # Original number of layers
            "learning_rate": 1e-4,  # Original learning rate
            "guidance_scale": 1.5,
            "eta": 0.1,  # Lower eta for less null class probability
            "weight_multiplier": 1.0,  # No weighting
            "batch_size": 32,  # Larger batch size for 64x64
            "num_epochs": 10000,  # Extended training for maximum convergence
            "checkpoint_interval": 0,  # No intermediate checkpoints, only best model
            "sample_interval": 25  # Generate samples every 25 epochs for monitoring
        }
    }
    return configs

if __name__ == "__main__":
    # List existing experiments
    manager = ExperimentManager("dummy")
    experiments = manager.list_experiments()
    
    print("=== Existing Experiments ===")
    for exp_name, config in experiments:
        print(f"  {exp_name}: {config.get('description', 'No description')}")
        print(f"    Created: {config.get('created_at', 'Unknown')}")
    
    print("\n=== Available Configurations ===")
    configs = create_experiment_configs()
    for exp_name, config in configs.items():
        print(f"  {exp_name}: {config['description']}")
        print(f"    Image size: {config['image_size']}x{config['image_size']}")
        print(f"    Model channels: {config['model_channels']}")
        print(f"    Learning rate: {config['learning_rate']}")
