import torch
import sketchutils as util
import sketch_model
import sketch_unet
from torchvision.utils import save_image
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_experiment_model(experiment_name: str):
    """Load model from experiment checkpoint"""
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load experiment config
    config_file = f"experiments/{experiment_name}/config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded config: {config['description']}")
    print(f"Image size: {config['image_size']}x{config['image_size']}")
    
    # Initialize probability path
    path = util.GaussianConditionalProbabilityPath(
        p_data = sketch_model.SketchSampler(),
        p_simple_shape = [1, config["image_size"], config["image_size"]],
        alpha = util.LinearAlpha(),
        beta = util.LinearBeta()
    ).to(device)

    # Initialize model
    model = sketch_unet.SketchUNet(
        channels = config["model_channels"],
        num_residual_layers = config["num_residual_layers"],
        t_embed_dim = 40,
        y_embed_dim = 40,
    ).to(device)

    # Load the best model
    checkpoint_path = f"experiments/{experiment_name}/ckps/best_model.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
    
    return model, path, device, config

def create_grid_with_titles(images, class_names, samples_per_class, guidance_scale, image_size):
    """Create a grid with class titles for 2-class setup"""
    num_classes = len(class_names)
    
    # Create figure with proper spacing for titles (2x2 for 2 classes)
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 2, num_classes * 2.5))
    fig.suptitle(f'Generated Sketches (Guidance Scale: {guidance_scale})', fontsize=16, fontweight='bold')
    
    # Convert tensor to numpy and denormalize
    images_np = images.cpu().numpy()
    images_np = (images_np + 1) / 2  # Denormalize from [-1,1] to [0,1]
    
    for i in range(num_classes):
        for j in range(samples_per_class):
            idx = i * samples_per_class + j
            ax = axes[i, j] if num_classes > 1 else axes[j]
            
            # Display image
            ax.imshow(images_np[idx, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            # Add class title to first column
            if j == 0:
                ax.set_title(f'{class_names[i]}', fontsize=10, fontweight='bold', pad=10)
    
    plt.tight_layout()
    return fig

def generate_samples(model, path, device, config, experiment_name, num_classes=2, samples_per_class=4):
    """Generate samples with different guidance scales for 2-class setup"""
    class_names = path.p_data.dataset.classes
    print(f"Available classes: {len(class_names)}")
    
    # Create results directory
    results_dir = f"experiments/{experiment_name}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate samples for different guidance scales
    guidance_scales = [1.0, 1.5, 2.0, 3.0, 5.0]
    
    for guidance_scale in guidance_scales:
        print(f"\nGenerating samples with guidance_scale = {guidance_scale}")
        
        # Setup ODE and simulator
        ode = util.CFGVectorFieldODE(model, guidance_scale=guidance_scale)
        simulator = util.EulerSimulator(ode)
        
        # Sample from first classes (2 classes for 2-class setup)
        classes = torch.tensor([i for i in range(num_classes)], device=device)
        classes = classes.repeat_interleave(samples_per_class)
        num_samples = classes.shape[0]
        
        # Sample initial conditions
        x0, _ = path.p_simple.sample(num_samples)
        x0 = x0.to(device)
        
        # Simulate
        ts = torch.linspace(0, 1, 100).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
        x1 = simulator.simulate(x0, ts, y=classes)
        
        # Create grid with titles
        fig = create_grid_with_titles(x1, class_names[:num_classes], samples_per_class, guidance_scale, config['image_size'])
        
        # Save samples with titles
        output_file = f"{results_dir}/samples_guidance_{guidance_scale}.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_file}")
        
        # Print class information
        for i in range(num_classes):
            start_idx = i * samples_per_class
            end_idx = start_idx + samples_per_class
            print(f"  Class {i}: {class_names[i]} (samples {start_idx}-{end_idx-1})")

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python generate_samples.py <experiment_name>")
        print("Example: python generate_samples.py sketch_32x32_2classes_20250912_131644")
        sys.exit(1)
    
    experiment_name = sys.argv[1]
    
    try:
        model, path, device, config = load_experiment_model(experiment_name)
        generate_samples(model, path, device, config, experiment_name)
        print(f"\nAll samples generated! Check the results in experiments/{experiment_name}/results/")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the experiment has been trained first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

