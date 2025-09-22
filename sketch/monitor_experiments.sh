#!/bin/bash

# Monitor all sketch diffusion experiments
# Usage: ./monitor_experiments.sh [experiment_name]

if [ $# -eq 1 ]; then
    # Monitor specific experiment
    EXPERIMENT_NAME=$1
    
    # Find the latest experiment with this name
    if [ -d "experiments" ]; then
        LATEST_EXP=$(ls -t experiments/ | grep "^${EXPERIMENT_NAME}_" | head -1)
        if [ -n "$LATEST_EXP" ]; then
            EXPERIMENT_ID=$LATEST_EXP
            SCREEN_NAME="sketch_${EXPERIMENT_ID}"
        else
            echo "No experiments found for '$EXPERIMENT_NAME'"
            exit 1
        fi
    else
        echo "No experiments directory found"
        exit 1
    fi
    
    echo "=== Monitoring Experiment: $EXPERIMENT_NAME ==="
    
    # Check if screen session is running
    if screen -list | grep -q "$SCREEN_NAME"; then
        echo "✅ Training is running in screen session '$SCREEN_NAME'"
        echo "To attach: screen -r $SCREEN_NAME"
    else
        echo "❌ No training session found for '$EXPERIMENT_NAME'"
    fi
    
    # Show latest log
    LOG_DIR="experiments/$EXPERIMENT_ID/logs"
    if [ -d "$LOG_DIR" ]; then
        LATEST_LOG=$(ls -t "$LOG_DIR"/training_${EXPERIMENT_NAME}_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo ""
            echo "📊 Latest log file: $LATEST_LOG"
            echo ""
            echo "=== Last 15 lines of training log ==="
            tail -15 "$LATEST_LOG"
        else
            echo "No log files found"
        fi
    else
        echo "Log directory not found: $LOG_DIR"
    fi
    
    # Show samples
    SAMPLES_DIR="experiments/$EXPERIMENT_ID/samples"
    if [ -d "$SAMPLES_DIR" ]; then
        echo ""
        echo "=== Generated Samples ==="
        ls -la "$SAMPLES_DIR" | tail -5
    fi
    
else
    # Monitor all experiments
    echo "=== Sketch Diffusion Experiments Monitor ==="
    echo ""
    
    # List all screen sessions
    echo "=== Active Screen Sessions ==="
    screen -list | grep "sketch_"
    echo ""
    
    # List all experiments
    if [ -d "experiments" ]; then
        echo "=== All Experiments ==="
        for exp_dir in experiments/*/; do
            if [ -d "$exp_dir" ]; then
                exp_name=$(basename "$exp_dir")
                config_file="$exp_dir/config.json"
                
                if [ -f "$config_file" ]; then
                    description=$(python -c "import json; print(json.load(open('$config_file'))['description'])" 2>/dev/null)
                    echo "  $exp_name: $description"
                else
                    echo "  $exp_name: No config found"
                fi
                
                # Check if training is running
                screen_name="sketch_${exp_name}"
                if screen -list | grep -q "$screen_name"; then
                    echo "    ✅ Running"
                else
                    echo "    ❌ Not running"
                fi
            fi
        done
    else
        echo "No experiments directory found"
    fi
    
    echo ""
    echo "=== Usage ==="
    echo "Monitor specific experiment: $0 <experiment_name>"
    echo "Start experiment: ./start_experiment.sh <experiment_name>"
    echo "List experiments: python experiment_manager.py"
fi
