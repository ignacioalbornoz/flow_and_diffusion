#!/usr/bin/env python3

import torch
import torchvision.utils as vutils
import os
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('.')

from experiment_manager import ExperimentManager, create_experiment_configs
import sketchutils as util
import sketch_model
import sketch_unet

def setup_logging(logs_dir, experiment_name):
    """Setup logging for the experiment"""
    log_file = os.path.join(logs_dir, f"training_{experiment_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def create_training_grid_with_titles(images, class_names, epoch, guidance_scale, image_size):
    """Create a grid with class titles for training samples"""
    num_samples = images.shape[0]
    num_classes = len(class_names)
    
    # Create figure with proper spacing for titles (2x2 for 2 classes)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f'Training Samples - Epoch {epoch} (Guidance: {guidance_scale})', fontsize=16, fontweight='bold')
    
    # Convert tensor to numpy and denormalize
    images_np = images.cpu().numpy()
    images_np = (images_np + 1) / 2  # Denormalize from [-1,1] to [0,1]
    
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            ax = axes[i, j]
            
            if idx < num_samples:
                # Display image
                ax.imshow(images_np[idx, 0], cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                
                # Add class title
                class_idx = idx % num_classes
                ax.set_title(f'{class_names[class_idx]}', fontsize=12, fontweight='bold', pad=5)
            else:
                ax.axis('off')
    
    plt.tight_layout()
    return fig

def generate_samples(model, path, device, epoch, samples_dir, config, num_samples=4):
    """Generate sample sketches to monitor training progress"""
    model.eval()
    with torch.no_grad():
        # Get class names from the dataset
        class_names = path.p_data.dataset.classes
        
        # Sample from available classes (2 classes for 2-class setup)
        num_classes = len(class_names)
        classes = torch.arange(num_classes, device=device).repeat(num_samples // num_classes + 1)[:num_samples]
        
        # Setup ODE and simulator with config guidance_scale
        guidance_scale = config.get("guidance_scale", 1.5)
        ode = util.CFGVectorFieldODE(model, guidance_scale=guidance_scale)
        simulator = util.EulerSimulator(ode)
        
        # Sample initial conditions
        x0, _ = path.p_simple.sample(num_samples)
        x0 = x0.to(device)
        
        # Simulate
        ts = torch.linspace(0, 1, 100).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
        x1 = simulator.simulate(x0, ts, y=classes)
        
        # Create grid with titles (2x2 for 2 classes)
        fig = create_training_grid_with_titles(x1, class_names, epoch, guidance_scale, config['image_size'])
        
        # Save samples with class information in filename
        sample_file = os.path.join(samples_dir, f"samples_epoch_{epoch:04d}_classes_0-{num_classes-1}.png")
        fig.savefig(sample_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Log detailed class information
        class_info = [f"{i}: {class_names[i]}" for i in range(num_classes)]
        class_summary = f"Classes 0-{num_classes-1}: {', '.join([class_names[i] for i in range(num_classes)])}"
        print(f"Generated samples for: {class_summary}")
        
        # Also save a text file with class mapping for this epoch
        class_mapping_file = os.path.join(samples_dir, f"class_mapping_epoch_{epoch:04d}.txt")
        with open(class_mapping_file, 'w') as f:
            f.write(f"Sample grid layout (2x2):\n")
            f.write(f"Row 1: Classes 0-1\n")
            f.write(f"Row 2: Classes 0-1 (repeated)\n\n")
            f.write(f"Detailed mapping:\n")
            for i in range(num_samples):
                class_idx = i % num_classes
                f.write(f"Position {i}: Class {class_idx} - {class_names[class_idx]}\n")
        
    model.train()
    return sample_file

def main(experiment_name: str):
    """Main training function for experiments"""
    # Setup experiment with unique ID
    exp_manager = ExperimentManager(experiment_name)
    paths = exp_manager.get_paths()
    
    print(f"Starting experiment: {experiment_name}")
    print(f"Experiment ID: {exp_manager.experiment_id}")
    print(f"Experiment directory: {exp_manager.experiment_dir}")
    
    # Get experiment config
    configs = create_experiment_configs()
    if experiment_name not in configs:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    config = configs[experiment_name]
    exp_manager.save_config(config)
    
    # Setup logging
    logger = setup_logging(paths["logs_dir"], experiment_name)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Config: {config}")
    
    # Use GPU 1 which has more free memory
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB")
        logger.info(f"Free Memory: {torch.cuda.memory_reserved(device) / 1e9:.1f}GB")
    
    # Training parameters from config
    num_epochs = config.get("num_epochs", 1000)
    lr = config["learning_rate"]
    batch_size = config.get("batch_size", 128)
    checkpoint_interval = config.get("checkpoint_interval", 100)
    sample_interval = config.get("sample_interval", 20)
    
    try:
        # Initialize probability path
        logger.info("Initializing probability path...")
        path = util.GaussianConditionalProbabilityPath(
            p_data = sketch_model.SketchSampler(),
            p_simple_shape = [1, config["image_size"], config["image_size"]],
            alpha = util.LinearAlpha(),
            beta = util.LinearBeta()
        ).to(device)
        logger.info("Probability path initialized successfully")

        # Initialize model
        logger.info("Initializing SketchUNet model...")
        model = sketch_unet.SketchUNet(
            channels = config["model_channels"],
            num_residual_layers = config["num_residual_layers"],
            t_embed_dim = 40,
            y_embed_dim = 40,
        ).to(device)
        logger.info("Model initialized successfully")
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Initialize trainer with config parameters
        logger.info("Initializing CFG trainer...")
        trainer = util.CFGTrainer(path=path, model=model, eta=config["eta"])
        logger.info("Trainer initialized successfully")
        
        # Training parameters
        logger.info(f"Starting training:")
        logger.info(f"  - Total epochs: {num_epochs}")
        logger.info(f"  - Learning rate: {lr}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Image size: {config['image_size']}x{config['image_size']}")
        logger.info(f"  - Model channels: {config['model_channels']}")
        logger.info(f"  - Eta (null class probability): {config['eta']}")
        logger.info(f"  - Loss: Standard MSE (no weighting)")
        logger.info(f"  - Checkpoint interval: {checkpoint_interval}")
        logger.info(f"  - Sample interval: {sample_interval}")

        # Train!
        logger.info("Starting training...")
        losses = []
        best_loss = float('inf')
        patience = 2500  # Early stopping patience (much less aggressive)
        no_improve_count = 0  # Counter for epochs without improvement
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            # Get training loss
            loss = trainer.get_train_loss(batch_size=batch_size)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            losses.append(loss.item())
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}")
            
            # Save checkpoint (only if checkpoint_interval > 0)
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0 and epoch > 0:
                checkpoint_file = os.path.join(paths["ckps_dir"], f"checkpoint_epoch_{epoch:04d}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'losses': losses,
                    'config': config
                }, checkpoint_file)
                logger.info(f"Checkpoint saved: {checkpoint_file}")
            
            # Save best model and check early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improve_count = 0  # Reset counter
                best_model_file = os.path.join(paths["ckps_dir"], "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'losses': losses,
                    'config': config
                }, best_model_file)
                logger.info(f"New best model saved (loss: {best_loss:.6f})")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info(f"Early stopping triggered! No improvement for {patience} epochs.")
                    logger.info(f"Best loss achieved: {best_loss:.6f} at epoch {epoch - patience}")
                    break
            
            # Generate samples
            if epoch % sample_interval == 0:
                sample_file = generate_samples(model, path, device, epoch, paths["samples_dir"], config)
                logger.info(f"Samples generated: {sample_file}")
        
        # Save final model
        final_model_file = os.path.join(paths["ckps_dir"], "final_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'losses': losses,
            'config': config
        }, final_model_file)
        
        # Save training history
        np.save(os.path.join(paths["logs_dir"], f"losses_{experiment_name}.npy"), losses)
        
        logger.info(f"Training completed!")
        logger.info(f"Final loss: {loss.item():.6f}")
        logger.info(f"Best loss: {best_loss:.6f}")
        logger.info(f"Final model saved: {final_model_file}")
        logger.info(f"Best model saved: {best_model_file}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train_experiment.py <experiment_name>")
        print("Available experiments:")
        configs = create_experiment_configs()
        for exp_name, config in configs.items():
            print(f"  {exp_name}: {config['description']}")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    main(experiment_name)
