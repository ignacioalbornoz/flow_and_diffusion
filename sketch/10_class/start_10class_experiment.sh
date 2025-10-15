#!/bin/bash

# Start 10-class experiment training script
# Usage: ./start_10class_experiment.sh <experiment_name>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Available experiments:"
    echo "  sketch_32x32_10classes: 32x32 resolution, 10 classes, research-grade model"
    echo "  sketch_32x32_10classes_simple: 32x32 resolution, 10 classes, simpler model for faster training"
    echo "  sketch_64x64_10classes: 64x64 resolution, 10 classes, research-grade model"
    echo ""
    echo "Example: $0 sketch_32x32_10classes"
    exit 1
fi

EXPERIMENT_NAME=$1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_ID="${EXPERIMENT_NAME}_${TIMESTAMP}"
SCREEN_NAME="sketch_10class_${EXPERIMENT_ID}"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sketch_diffusion

# Kill existing screen session if it exists
screen -S $SCREEN_NAME -X quit 2>/dev/null

# Start training in screen session
screen -dmS $SCREEN_NAME bash -c "python train_experiment.py $EXPERIMENT_NAME; exec bash"

echo "10-Class Experiment '$EXPERIMENT_NAME' started in screen session '$SCREEN_NAME'"
echo "Experiment ID: $EXPERIMENT_ID"
echo ""
echo "To monitor training:"
echo "  screen -r $SCREEN_NAME"
echo ""
echo "To see logs in real-time:"
echo "  tail -f experiments/$EXPERIMENT_ID/logs/training_${EXPERIMENT_NAME}_*.log"
echo ""
echo "To list all screen sessions:"
echo "  screen -ls"
echo ""
echo "To kill the session:"
echo "  screen -S $SCREEN_NAME -X quit"
