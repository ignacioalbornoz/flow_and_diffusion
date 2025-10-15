import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

def concatenate_images_horizontally():
    """Concatenate data_samples_grid.png and samples_guidance_5.0.png side by side"""
    
    # Load the images
    data_samples_path = "/home/ialbornoz/tesis/flow_and_diffusion/sketch/10_class/data_samples_grid.png"
    generated_samples_path = "/home/ialbornoz/tesis/flow_and_diffusion/sketch/10_class/experiments/sketch_64x64_10classes_20251003_021649/results/samples_guidance_5.0.png"
    
    # Load images
    img1 = mpimg.imread(data_samples_path)
    img2 = mpimg.imread(generated_samples_path)
    
    print(f"Data samples image shape: {img1.shape}")
    print(f"Generated samples image shape: {img2.shape}")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 20))
    
    # Display first image (data samples)
    ax1.imshow(img1)
    ax1.set_title('Data Samples\n(2 Random Sketches per Category)', fontsize=14, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Display second image (generated samples)
    ax2.imshow(img2)
    ax2.set_title('Generated Samples\n(Guidance Scale: 5.0)', fontsize=14, fontweight='bold', pad=20)
    ax2.axis('off')
    
    # Add overall title
    fig.suptitle('Data vs Generated Samples Comparison', fontsize=18, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save the concatenated image
    output_path = "/home/ialbornoz/tesis/flow_and_diffusion/sketch/10_class/data_vs_generated_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Concatenated image saved to: {output_path}")
    return output_path

def main():
    """Main function to create the concatenated image"""
    print("Creating concatenated image...")
    
    try:
        output_path = concatenate_images_horizontally()
        print(f"Successfully created comparison image: {output_path}")
        
    except Exception as e:
        print(f"Error creating concatenated image: {e}")

if __name__ == "__main__":
    main()
