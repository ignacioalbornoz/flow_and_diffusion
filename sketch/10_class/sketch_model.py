import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import Optional, Tuple
import sketchutils as util

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class SketchDataset(Dataset):
    """
    Dataset for loading sketches from the Sketchy dataset
    """
    def __init__(self, root_dir: str, transform=None, max_classes: int = None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all available classes
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # Limit to max_classes if specified
        if max_classes is not None:
            self.classes = all_classes[:max_classes]
            print(f"Using first {max_classes} classes out of {len(all_classes)} available")
        else:
            self.classes = all_classes
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Build list of (image_path, class_idx) pairs
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx

class SketchSampler(nn.Module, util.Sampleable):
    """
    Sampleable wrapper for the Sketchy dataset
    """
    def __init__(self, data_dir: str = "/home/ialbornoz/tesis/flow_and_diffusion/sketch/data/sketch", max_classes: int = 10):
        super().__init__()
        self.data_dir = data_dir
        self.max_classes = max_classes
        
        # Define transforms for sketches
        image_size = self.get_image_size()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize to 64x64 for memory optimization
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        # Create dataset with limited classes
        self.dataset = SketchDataset(data_dir, transform=self.transform, max_classes=max_classes)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=1, 
            shuffle=True, 
            num_workers=4,
            drop_last=True
        )
        self.dataloader_iter = iter(self.dataloader)
        
        # Get number of classes
        self.num_classes = len(self.dataset.classes)
        print(f"Loaded {len(self.dataset)} sketches from {self.num_classes} classes")
        print(f"Classes: {self.dataset.classes[:10]}...")  # Show first 10 classes
    
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample sketches and their class labels
        Args:
            - num_samples: number of samples to generate
        Returns:
            - samples: (num_samples, 1, 64, 64) tensor of sketches
            - labels: (num_samples,) tensor of class labels
        """
        samples = []
        labels = []
        
        for _ in range(num_samples):
            try:
                sample, label = next(self.dataloader_iter)
            except StopIteration:
                # Restart iterator if we run out of samples
                self.dataloader_iter = iter(self.dataloader)
                sample, label = next(self.dataloader_iter)
            
            samples.append(sample)
            labels.append(label)
        
        samples = torch.cat(samples, dim=0)  # (num_samples, 1, 64, 64)
        labels = torch.cat(labels, dim=0)    # (num_samples,)
        
        return samples, labels
    
    def get_image_size(self):
        """Get the current image size"""
        return 64  # Optimized resolution for memory

if __name__ == '__main__':
    # Test the sampler
    sampler = SketchSampler().to(device)
    
    # Sample a few examples
    samples, labels = sampler.sample(10)
    print(f"Sample shape: {samples.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
    print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
