import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import torch
import sketchutils as util
import sketch_model
import sketch_unet
import json

def load_experiment_model(experiment_name: str):
    """Load model from experiment checkpoint"""
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load experiment config
    config_file = f"experiments/{experiment_name}/config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded config: {config['description']}")
    
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

def get_random_data_samples(category_path, num_samples=2, use_quickdraw=False):
    """Get random samples from a category directory"""
    all_files = os.listdir(category_path)
    
    if use_quickdraw:
        quickdraw_files = [f for f in all_files if f.startswith('quickdraw_')]
        if quickdraw_files:
            return random.sample(quickdraw_files, min(num_samples, len(quickdraw_files)))
        else:
            return random.sample(all_files, min(num_samples, len(all_files)))
    else:
        sketchy_files = [f for f in all_files if not f.startswith('quickdraw_')]
        if sketchy_files:
            return random.sample(sketchy_files, min(num_samples, len(sketchy_files)))
        else:
            return random.sample(all_files, min(num_samples, len(all_files)))

def load_and_resize_image(image_path, target_size=(64, 64)):
    """Load and resize image to target size"""
    try:
        img = Image.open(image_path).convert('L')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((target_size[0], target_size[1]), dtype=np.uint8)

def generate_samples_for_class(model, path, device, config, class_idx, num_samples=2):
    """Generate samples for a specific class"""
    model.eval()
    with torch.no_grad():
        # Setup ODE and simulator
        guidance_scale = 5.0
        ode = util.CFGVectorFieldODE(model, guidance_scale=guidance_scale)
        simulator = util.EulerSimulator(ode)
        
        # Create class tensor
        classes = torch.tensor([class_idx] * num_samples, device=device)
        
        # Sample initial conditions
        x0, _ = path.p_simple.sample(num_samples)
        x0 = x0.to(device)
        
        # Simulate
        ts = torch.linspace(0, 1, 100).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
        x1 = simulator.simulate(x0, ts, y=classes)
        
        # Convert to numpy and denormalize
        samples = x1.cpu().numpy()
        samples = (samples + 1) / 2  # Denormalize from [-1,1] to [0,1]
        
        return samples

def create_combined_comparison():
    """Create a combined comparison image with data and generated samples"""
    base_path = "/home/ialbornoz/tesis/flow_and_diffusion/sketch/data/sketch"
    
    # Define the 10 categories in order
    categories = [
        'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration',
        'ant', 'anvil', 'apple', 'arm', 'asparagus'
    ]
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load the model for generating samples
    print("Loading model...")
    model, path, device, config = load_experiment_model("sketch_64x64_10classes_20251003_021649")
    
    # Create figure with 10 rows and 4 columns (2 data + 2 generated per row)
    fig, axes = plt.subplots(10, 4, figsize=(8, 20))
    fig.suptitle('Data vs Generated Samples Comparison', fontsize=18, fontweight='bold', y=0.95)
    
    for i, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Category {category} not found")
            continue
        
        # Get data samples
        if i < 2:  # First 2 categories: QuickDraw + Sketchy
            # QuickDraw sample
            quickdraw_samples = get_random_data_samples(category_path, 1, use_quickdraw=True)
            if quickdraw_samples:
                img_path = os.path.join(category_path, quickdraw_samples[0])
                img = load_and_resize_image(img_path)
                axes[i, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[i, 0].axis('off')
            
            # Sketchy sample
            sketchy_samples = get_random_data_samples(category_path, 1, use_quickdraw=False)
            if sketchy_samples:
                img_path = os.path.join(category_path, sketchy_samples[0])
                img = load_and_resize_image(img_path)
                axes[i, 1].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[i, 1].axis('off')
        else:  # Other categories: 2 random samples
            samples = get_random_data_samples(category_path, 2)
            for j, sample in enumerate(samples):
                if j < 2:
                    img_path = os.path.join(category_path, sample)
                    img = load_and_resize_image(img_path)
                    axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=255)
                    axes[i, j].axis('off')
        
        # Generate samples for this class
        print(f"Generating samples for {category}...")
        generated_samples = generate_samples_for_class(model, path, device, config, i, 2)
        
        for j in range(2):
            axes[i, j + 2].imshow(generated_samples[j, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, j + 2].axis('off')
        
        # Add category title to the first column
        axes[i, 0].set_title(f'{category}', fontsize=10, fontweight='bold')
    
    # Add column headers
    fig.text(0.15, 0.92, 'Data Samples', fontsize=12, fontweight='bold', ha='center')
    fig.text(0.65, 0.92, 'Generated Samples', fontsize=12, fontweight='bold', ha='center')
    
    # Add sub-column headers
    fig.text(0.075, 0.89, 'QuickDraw', fontsize=10, ha='center')
    fig.text(0.225, 0.89, 'Sketchy', fontsize=10, ha='center')
    fig.text(0.575, 0.89, 'Gen 1', fontsize=10, ha='center')
    fig.text(0.725, 0.89, 'Gen 2', fontsize=10, ha='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    return fig

def main():
    """Main function to create the combined comparison image"""
    print("Creating combined comparison image...")
    
    try:
        fig = create_combined_comparison()
        
        # Save the figure
        output_path = "/home/ialbornoz/tesis/flow_and_diffusion/sketch/10_class/combined_data_vs_generated.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Combined comparison image saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating combined image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
