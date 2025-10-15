#!/bin/bash

# Start 62-class experiment training script with tmux
# Usage: ./start_62class_experiment.sh <dataset_path>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_path>"
    echo ""
    echo "Arguments:"
    echo "  dataset_path: Path to your dataset directory with 62 class folders"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/your/62/classes/dataset"
    exit 1
fi

EXPERIMENT_NAME="sketch_62classes"
DATASET_PATH=$1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_ID="${EXPERIMENT_NAME}_${TIMESTAMP}"
TMUX_SESSION="sketch_62class_${EXPERIMENT_ID}"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sketch_diffusion

# Kill existing tmux session if it exists
tmux kill-session -t $TMUX_SESSION 2>/dev/null

# Create tmux session and start training
tmux new-session -d -s $TMUX_SESSION

# Start training in the session
tmux send-keys -t $TMUX_SESSION "cd /home/ialbornoz/tesis/flow_and_diffusion/sketch/62_class" Enter
tmux send-keys -t $TMUX_SESSION "python train_experiment.py $EXPERIMENT_NAME --data_dir $DATASET_PATH" Enter

# Start system monitoring in background (no tmux pane)
cd /home/ialbornoz/tesis/flow_and_diffusion/sketch/62_class
python monitor_training.py $EXPERIMENT_ID > experiments/$EXPERIMENT_ID/monitoring/monitor.log 2>&1 &

echo "62-Class Experiment '$EXPERIMENT_NAME' started in tmux session '$TMUX_SESSION'"
echo "Experiment ID: $EXPERIMENT_ID"
echo ""
echo "To monitor training:"
echo "  tmux attach -t $TMUX_SESSION"
echo ""
echo "To see training logs:"
echo "  tail -f experiments/$EXPERIMENT_ID/logs/training_sketch_62classes.log"
echo ""
echo "To see system monitoring logs:"
echo "  tail -f experiments/$EXPERIMENT_ID/monitoring/monitor.log"
echo ""
echo "To see system metrics:"
echo "  cat experiments/$EXPERIMENT_ID/monitoring/system_metrics.json"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t $TMUX_SESSION"
echo ""
echo "To list all tmux sessions:"
echo "  tmux list-sessions"
