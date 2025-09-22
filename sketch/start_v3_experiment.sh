#!/bin/bash

# Start V3 improved experiment with SD v3 enhancements
# This maintains Flow Matching as the core method

EXPERIMENT_NAME="sketch_32x32_v3_improved"

echo "Starting V3 improved experiment: $EXPERIMENT_NAME"
echo "This uses SD v3 improvements while maintaining Flow Matching"
echo ""

# Check if screen is available
if ! command -v screen &> /dev/null; then
    echo "Error: screen is not installed. Please install it first."
    exit 1
fi

# Check if experiment is already running
if screen -list | grep -q "sketch_${EXPERIMENT_NAME}"; then
    echo "Experiment is already running in screen session."
    echo "To attach: screen -r sketch_${EXPERIMENT_NAME}"
    echo "To kill: screen -S sketch_${EXPERIMENT_NAME} -X quit"
    exit 1
fi

# Create screen session name
SESSION_NAME="sketch_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"

# Start the experiment in a screen session
screen -dmS "$SESSION_NAME" bash -c "
    echo 'Starting V3 improved experiment...'
    echo 'Model: SketchUNetV3 with SD v3 improvements'
    echo 'Method: Flow Matching (maintained)'
    echo 'Architecture: Enhanced with attention, better embeddings, improved normalization'
    echo ''
    
    cd $(pwd)
    python train_v3.py $EXPERIMENT_NAME
    
    echo ''
    echo 'Experiment completed!'
    echo 'Press any key to close this session...'
    read -n 1
"

# Wait a moment for screen to start
sleep 2

# Check if screen session was created successfully
if screen -list | grep -q "$SESSION_NAME"; then
    echo ""
    echo "Experiment '$EXPERIMENT_NAME' started in screen session '$SESSION_NAME'"
    echo ""
    echo "Experiment ID: ${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
    echo ""
    echo "To monitor training:"
    echo "  screen -r $SESSION_NAME"
    echo ""
    echo "To see logs in real-time:"
    echo "  tail -f experiments/${EXPERIMENT_NAME}_*/logs/training_${EXPERIMENT_NAME}_*.log"
    echo ""
    echo "To list all screen sessions:"
    echo "  screen -ls"
    echo ""
    echo "To kill the session:"
    echo "  screen -S $SESSION_NAME -X quit"
    echo ""
    echo "V3 Improvements included:"
echo "  ✓ Cross-attention mechanisms"
echo "  ✓ Enhanced Fourier embeddings"
echo "  ✓ Improved normalization (GroupNorm)"
echo "  ✓ Better weight initialization"
echo "  ✓ Deeper architecture (6 residual layers)"
echo "  ✓ Larger model capacity (96-768 channels)"
echo "  ✓ Flow Matching maintained as core method"
echo ""
echo "✅ All Original Features Maintained:"
echo "  ✓ 10000 épocas máximo"
echo "  ✓ Early stopping 2500 épocas"
echo "  ✓ Sample cada 50 épocas"
echo "  ✓ Sample en new best model"
echo "  ✓ Checkpoint cada 50 épocas"
echo "  ✓ Learning rate 1e-4"
echo "  ✓ Batch size 32"
echo "  ✓ Guidance scale 1.5"
echo "  ✓ Eta 0.1"
else
    echo "Error: Failed to start screen session"
    exit 1
fi
