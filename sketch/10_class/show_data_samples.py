import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

def get_random_samples(category_path, num_samples=2, use_quickdraw=False):
    """Get random samples from a category directory"""
    all_files = os.listdir(category_path)
    
    if use_quickdraw:
        # Filter for QuickDraw files
        quickdraw_files = [f for f in all_files if f.startswith('quickdraw_')]
        if quickdraw_files:
            return random.sample(quickdraw_files, min(num_samples, len(quickdraw_files)))
        else:
            # Fallback to any files if no QuickDraw files found
            return random.sample(all_files, min(num_samples, len(all_files)))
    else:
        # Filter for Sketchy files (not QuickDraw)
        sketchy_files = [f for f in all_files if not f.startswith('quickdraw_')]
        if sketchy_files:
            return random.sample(sketchy_files, min(num_samples, len(sketchy_files)))
        else:
            # Fallback to any files if no Sketchy files found
            return random.sample(all_files, min(num_samples, len(all_files)))

def load_and_resize_image(image_path, target_size=(64, 64)):
    """Load and resize image to target size"""
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((target_size[0], target_size[1]), dtype=np.uint8)

def create_data_samples_grid():
    """Create a grid showing 2 random samples per category"""
    base_path = "/home/ialbornoz/tesis/flow_and_diffusion/sketch/data/sketch"
    
    # Define the 10 categories in order
    categories = [
        'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration',
        'ant', 'anvil', 'apple', 'arm', 'asparagus'
    ]
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create figure
    fig, axes = plt.subplots(10, 2, figsize=(4, 20))
    fig.suptitle('Data Samples: 2 Random Sketches per Category', fontsize=16, fontweight='bold', y=0.98)
    
    for i, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Category {category} not found")
            continue
            
        # For first 2 categories, use one QuickDraw and one Sketchy
        if i < 2:
            # Get one QuickDraw sample
            quickdraw_samples = get_random_samples(category_path, 1, use_quickdraw=True)
            if quickdraw_samples:
                img_path = os.path.join(category_path, quickdraw_samples[0])
                img = load_and_resize_image(img_path)
                axes[i, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
                axes[i, 0].set_title(f'{category}\n(QuickDraw)', fontsize=10, fontweight='bold')
                axes[i, 0].axis('off')
            
            # Get one Sketchy sample
            sketchy_samples = get_random_samples(category_path, 1, use_quickdraw=False)
            if sketchy_samples:
                img_path = os.path.join(category_path, sketchy_samples[0])
                img = load_and_resize_image(img_path)
                axes[i, 1].imshow(img, cmap='gray', vmin=0, vmax=255)
                axes[i, 1].set_title(f'{category}\n(Sketchy)', fontsize=10, fontweight='bold')
                axes[i, 1].axis('off')
        else:
            # For other categories, just get 2 random samples
            samples = get_random_samples(category_path, 2)
            for j, sample in enumerate(samples):
                if j < 2:  # Only show 2 samples
                    img_path = os.path.join(category_path, sample)
                    img = load_and_resize_image(img_path)
                    axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=255)
                    if j == 0:
                        axes[i, j].set_title(f'{category}', fontsize=10, fontweight='bold')
                    axes[i, j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for title
    return fig

def main():
    """Main function to create and save the data samples grid"""
    print("Creating data samples grid...")
    
    # Create the grid
    fig = create_data_samples_grid()
    
    # Save the figure
    output_path = "/home/ialbornoz/tesis/flow_and_diffusion/sketch/10_class/data_samples_grid.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Data samples grid saved to: {output_path}")
    print("Grid shows:")
    print("- First 2 categories: 1 QuickDraw + 1 Sketchy sample each")
    print("- Remaining 8 categories: 2 random samples each")

if __name__ == "__main__":
    main()
