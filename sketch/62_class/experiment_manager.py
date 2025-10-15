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
    """Create optimized configuration for 62 classes"""
    configs = {
        "sketch_62classes": {
            "description": "Optimized model for 62 classes - 6x larger than 10-class model",
            "image_size": 64,
            "model_channels": [128, 256, 512, 1024],  # 6x larger than 10-class [64,128,256,512]
            "num_residual_layers": 6,  # More layers for 62 classes
            "learning_rate": 3e-5,  # Lower learning rate for stability with more classes
            "guidance_scale": 1.5,
            "eta": 0.1,  # Lower eta for less null class probability
            "weight_multiplier": 1.0,  # No weighting
            "batch_size": 64,  # As requested
            "num_epochs": 20000,  # Extended training for 62 classes
            "checkpoint_interval": 200,  # Checkpoints every 200 epochs
            "sample_interval": 100  # Generate samples every 100 epochs
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
        print(f"    Batch size: {config['batch_size']}")
        print(f"    Residual layers: {config['num_residual_layers']}")
        print()
