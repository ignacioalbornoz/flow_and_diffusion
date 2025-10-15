#!/bin/bash

# Quick start script for 62-class training
# Usage: ./quick_start.sh <dataset_path>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_path>"
    echo ""
    echo "Arguments:"
    echo "  dataset_path: Path to your dataset directory with 62 class folders"
    echo ""
    echo "Example:"
    echo "  ./quick_start.sh /path/to/my/62classes/dataset"
    exit 1
fi

DATASET_PATH=$1

echo "🚀 Starting 62-class training with:"
echo "   Dataset: $DATASET_PATH"
echo "   Model: Optimized for 62 classes (6x larger than 10-class model)"
echo "   Resolution: 64x64"
echo "   Batch Size: 64"
echo ""

# Check if dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

# Count subdirectories (classes)
CLASS_COUNT=$(find "$DATASET_PATH" -maxdepth 1 -type d | wc -l)
CLASS_COUNT=$((CLASS_COUNT - 1))  # Subtract 1 for the parent directory

echo "📁 Found $CLASS_COUNT class directories in dataset"
echo ""

# Start the training
echo "🎯 Starting training..."
./start_62class_experiment.sh $DATASET_PATH
