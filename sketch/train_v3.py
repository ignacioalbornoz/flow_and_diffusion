#!/usr/bin/env python3
"""
Enhanced training script with SD v3 improvements while maintaining Flow Matching
"""

import os
import sys
import torch
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment_manager import ExperimentManager, create_experiment_configs
from sketchutils import (
    SketchDataset, GaussianConditionalProbabilityPath, 
    CFGTrainer, CFGVectorFieldODE, ODESolver
)
from sketch_unet import SketchUNet
from sketch_unet_v3 import SketchUNetV3

def setup_logging(experiment_dir: str, experiment_name: str):
    """Setup logging for the experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(experiment_dir, "logs", f"training_{experiment_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        # Try to use GPU 1 if available (as in original setup)
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda:0')
        logging.info(f"Using device: {device}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB")
        logging.info(f"Free Memory: {torch.cuda.memory_reserved(device) / 1e9:.1f}GB")
    else:
        device = torch.device('cpu')
        logging.info(f"Using device: {device}")
    return device

def create_model(config: dict, device: torch.device):
    """Create the model based on configuration"""
    use_v3_model = config.get('use_v3_model', False)
    
    if use_v3_model:
        logging.info("Creating SketchUNetV3 model (SD v3 improved architecture)")
        model = SketchUNetV3(
            channels=config['model_channels'],
            num_residual_layers=config['num_residual_layers'],
            t_embed_dim=128,
            y_embed_dim=128
        )
    else:
        logging.info("Creating original SketchUNet model")
        model = SketchUNet(
            channels=config['model_channels'],
            num_residual_layers=config['num_residual_layers'],
            t_embed_dim=128,
            y_embed_dim=128
        )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    return model

def train_experiment(experiment_name: str, device: torch.device):
    """Train a specific experiment"""
    
    # Get configuration
    configs = create_experiment_configs()
    if experiment_name not in configs:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    config = configs[experiment_name]
    
    # Create experiment manager
    manager = ExperimentManager(experiment_name)
    manager.save_config(config)
    
    # Setup logging
    logger = setup_logging(manager.experiment_dir, experiment_name)
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Config: {config}")
    
    # Initialize probability path
    logger.info("Initializing probability path...")
    dataset = SketchDataset(
        data_dir="data/sketch",
        image_size=config['image_size'],
        num_classes=50  # Use 50 classes as specified
    )
    
    path = GaussianConditionalProbabilityPath(dataset)
    logger.info("Probability path initialized successfully")
    
    # Create model
    logger.info("Initializing SketchUNet model...")
    model = create_model(config, device)
    logger.info("Model initialized successfully")
    
    # Create trainer
    trainer = CFGTrainer(
        path=path,
        model=model,
        eta=config['eta']
    )
    
    # Training configuration
    logger.info("Training configuration:")
    logger.info(f"  - Learning rate: {config['learning_rate']}")
    logger.info(f"  - Batch size: {config['batch_size']}")
    logger.info(f"  - Number of epochs: {config['num_epochs']}")
    logger.info(f"  - Guidance scale: {config['guidance_scale']}")
    logger.info(f"  - Eta: {config['eta']}")
    logger.info(f"  - Loss: Standard MSE (no weighting)")
    logger.info(f"  - Checkpoint interval: {config['checkpoint_interval']}")
    logger.info(f"  - Sample interval: {config['sample_interval']}")
    logger.info("Starting training...")
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    patience = 2500  # Early stopping patience (same as original)
    
    for epoch in range(config['num_epochs']):
        # Get training loss
        loss = trainer.get_train_loss(config['batch_size'])
        
        # Log progress
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}")
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            checkpoint_path = os.path.join(manager.ckps_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss.item(),
                'config': config
            }, checkpoint_path)
            logger.info(f"New best model saved (loss: {loss.item():.6f})")
            
            # Generate samples for new best model
            try:
                generate_samples(model, path, manager.samples_dir, epoch, config, device)
                logger.info(f"New best model samples generated: {manager.samples_dir}/samples_epoch_{epoch:04d}_classes_0-15.png")
            except Exception as e:
                logger.warning(f"Failed to generate samples for new best model: {e}")
        else:
            patience_counter += 1
        
        # Regular checkpoints
        if epoch % config['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(manager.ckps_dir, f"checkpoint_epoch_{epoch:04d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss.item(),
                'config': config
            }, checkpoint_path)
        
        # Generate samples
        if epoch % config['sample_interval'] == 0:
            try:
                generate_samples(model, path, manager.samples_dir, epoch, config, device)
                logger.info(f"Samples generated: {manager.samples_dir}/samples_epoch_{epoch:04d}_classes_0-15.png")
            except Exception as e:
                logger.warning(f"Failed to generate samples: {e}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    logger.info(f"Training completed. Best loss: {best_loss:.6f}")
    return manager.experiment_dir

def generate_samples(model, path, samples_dir, epoch, config, device):
    """Generate sample images"""
    model.eval()
    
    with torch.no_grad():
        # Create ODE solver for generation
        ode = CFGVectorFieldODE(model, guidance_scale=config['guidance_scale'])
        solver = ODESolver(ode)
        
        # Generate samples for first 16 classes
        num_classes = 16
        samples_per_class = 1
        
        all_samples = []
        for class_idx in range(num_classes):
            # Generate samples for this class
            y = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
            
            # Sample from noise
            x0 = torch.randn(samples_per_class, 1, config['image_size'], config['image_size'], device=device)
            
            # Solve ODE
            x1 = solver.solve(x0, y, t0=0.0, t1=1.0, num_steps=50)
            
            all_samples.append(x1)
        
        # Combine samples
        samples = torch.cat(all_samples, dim=0)  # (16, 1, 32, 32)
        
        # Save as grid image
        import torchvision.utils as vutils
        grid = vutils.make_grid(samples, nrow=4, normalize=True, padding=2)
        
        # Convert to PIL and save
        from torchvision.transforms import ToPILImage
        pil_image = ToPILImage()(grid)
        
        sample_path = os.path.join(samples_dir, f"samples_epoch_{epoch:04d}_classes_0-15.png")
        pil_image.save(sample_path)
    
    model.train()

def main():
    parser = argparse.ArgumentParser(description="Train SketchUNet with Flow Matching")
    parser.add_argument("experiment_name", help="Name of the experiment to run")
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    
    # Train experiment
    experiment_dir = train_experiment(args.experiment_name, device)
    
    print(f"\nExperiment completed!")
    print(f"Results saved in: {experiment_dir}")
    print(f"Best model: {experiment_dir}/ckps/best_model.pth")
    print(f"Samples: {experiment_dir}/samples/")

if __name__ == "__main__":
    main()
