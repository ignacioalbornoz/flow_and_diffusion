# Sketch Diffusion

## Overview
Conditional diffusion model for generating sketches at 256x256 resolution using the Sketchy dataset.

## Model Architecture
- **Resolution**: 256x256 (full resolution)
- **Model**: U-Net with conditional flow matching
- **Channels**: [128, 256, 512, 1024] (deep architecture)
- **Residual Layers**: 6 layers
- **Conditioning**: Class-conditional (125 classes + 1 null class)
- **Guidance**: Classifier-Free Guidance (CFG)

## Dataset
- **Source**: Sketchy dataset
- **Classes**: 125 sketch categories
- **Original Size**: 256x256
- **Format**: Grayscale PNG
- **Normalization**: [-1, 1] range

## Training Configuration
- **Learning Rate**: 1e-5 (stable training)
- **Batch Size**: 32 (optimized for 256x256)
- **Epochs**: 2000 
- **Checkpoint Interval**: 50 epochs
- **Sample Interval**: 25 epochs
- **Weight Multiplier**: 3x for black pixels (sketch emphasis)

## File Structure
```
sketch/
├── experiments/
│   └── sketch_256x256_research/
│       ├── ckps/           # Model checkpoints
│       ├── samples/        # Training samples
│       ├── logs/          # Training logs
│       ├── results/       # Final generated samples
│       └── config.json    # Experiment configuration
├── sketch_model.py        # Dataset and sampler
├── sketch_unet.py         # U-Net model architecture
├── sketchutils.py         # Training utilities
├── train_experiment.py    # Training script
├── generate_256x256.py    # Generation script
├── start_experiment.sh    # Start training
└── monitor_experiments.sh # Monitor training
```

## Usage

### Start Training
```bash
./start_experiment.sh sketch_256x256_research
```

### Monitor Training
```bash
# Monitor all experiments
./monitor_experiments.sh

# Monitor specific experiment
./monitor_experiments.sh sketch_256x256_research
```

### Generate Samples
```bash
python generate_256x256.py
```

### Attach to Training Session
```bash
screen -r sketch_sketch_256x256_research
```

## Key Features

### 1. Architecture
- Deep U-Net with 4 channel levels
- 6 residual layers for better feature learning
- Optimized for 256x256 resolution

### 2. Conditional Generation
- Class-conditional training
- Classifier-Free Guidance for better quality
- Support for 125 sketch classes

### 3. Sketch-Optimized Loss
- Weighted loss function (3x weight for black pixels)
- Emphasizes sketch lines over background
- Prevents mode collapse to white images

### 4. Experiment Management
- Organized experiment structure
- Automatic checkpointing
- Comprehensive logging
- Easy monitoring and comparison

## Training Progress
- **Loss Monitoring**: Every 10 epochs
- **Sample Generation**: Every 25 epochs
- **Checkpointing**: Every 50 epochs
- **Best Model**: Automatically saved

## Expected Results
- **High-Quality Sketches**: 256x256 resolution
- **Class-Specific Generation**: Distinct sketches per class
- **Stable Training**: Low learning rate prevents divergence
- **Research-Ready**: Suitable for academic publication

## Technical Details
- **Framework**: PyTorch
- **Device**: CUDA (GPU-optimized)
- **Memory**: ~8GB VRAM required for 256x256
- **Training Time**: ~24-48 hours for full training

## Troubleshooting
- **Out of Memory**: Reduce batch size to 16
- **Slow Training**: Check GPU utilization
- **Poor Quality**: Monitor loss convergence
- **No Samples**: Check sample generation interval


🎨 CLASES EN LAS MUESTRAS GENERADAS:
Fila 1 (0-3):
0: airplane (avión)
1: alarm_clock (despertador)
2: ant (hormiga)
3: ape (simio)
Fila 2 (4-7):
4: apple (manzana)
5: armor (armadura)
6: axe (hacha)
7: banana (plátano)
Fila 3 (8-11):
8: bat (murciélago)
9: bear (oso)
10: bee (abeja)
11: beetle (escarabajo)
Fila 4 (12-15):
12: bell (campana)
13: bench (banco)
14: bicycle (bicicleta)
15: blimp (dirigible)