#!/bin/bash

# Name of the tmux session
SESSION_NAME="franka_automation"

# Check if the session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Session $SESSION_NAME already exists. Attaching to it..."
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

# Start a new detached tmux session
tmux new-session -d -s $SESSION_NAME

# ---------------------------------------------------------
# Global Settings & Custom Keybindings
# ---------------------------------------------------------
# Enable mouse support (clicking panes, dragging to resize, scrolling)
tmux set-option -g mouse on

# Bind Ctrl+Q (C-q) to instantly kill the session without needing the prefix key
tmux bind-key -n C-q kill-session

# ---------------------------------------------------------
# Pane 1 (Top): ROS2 Mono Controller
# ---------------------------------------------------------
tmux send-keys -t $SESSION_NAME "cd /home/maksimgorki/DataCollectionScripts" C-m
tmux send-keys -t $SESSION_NAME "conda deactivate" C-m
tmux send-keys -t $SESSION_NAME "source install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME "ros2 run controllers mono_controller_sim" C-m

# ---------------------------------------------------------
# Pane 2: EEF Pose Publisher
# ---------------------------------------------------------
# Split vertically to place below the previous pane
tmux split-window -v -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME "cd /home/maksimgorki/lerobot/src/lerobot/robots/franka_panda" C-m
tmux send-keys -t $SESSION_NAME "conda deactivate" C-m
tmux send-keys -t $SESSION_NAME "python3 eef_pose_publisher.py" C-m

# ---------------------------------------------------------
# Pane 3: Bridge Server
# ---------------------------------------------------------
# Split vertically to place below the previous pane
tmux split-window -v -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME "cd /home/maksimgorki/lerobot/src/lerobot/robots/franka_panda" C-m
tmux send-keys -t $SESSION_NAME "conda deactivate" C-m
tmux send-keys -t $SESSION_NAME "python3 run_bridge_server.py" C-m

# ---------------------------------------------------------
# Pane 4 (Bottom): Test Panda Script
# ---------------------------------------------------------
# Split vertically to place below the previous pane
tmux split-window -v -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME "cd /home/maksimgorki/lerobot" C-m
tmux send-keys -t $SESSION_NAME "conda activate lerobot" C-m
tmux send-keys -t $SESSION_NAME "./bringup/test_panda.sh" C-m

# ---------------------------------------------------------
# Finalize Layout & Attach
# ---------------------------------------------------------
# Arrange all panes into equal top-to-bottom rows
tmux select-layout -t $SESSION_NAME even-vertical

# Attach your terminal to the new tmux session
tmux attach-session -t $SESSION_NAME