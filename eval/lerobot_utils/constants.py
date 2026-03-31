"""
Constants for LeRobot-based inference and evaluation.
"""

# --- ROBOT LIMITS & CONTROL ---
# Z-limit in meters to prevent collision with table
# Even the length of the gripper is about 0.1 m, so set it higher than 0.1 m
Z_LIMIT = 0.055

# --- GRIPPER NORMALIZATION ---
# Simulator (SIM) range: [0.0, 0.04]
SIM_MAX_VAL = 0.04
# Real Robot (REAL) range: [0.0, 0.068]
REAL_MAX_VAL = 0.068

# Gripper Thresholds (0-100 scale)
REAL_GRIPPER_THRESHOLD = 60
SIM_GRIPPER_THRESHOLD = 60
SIM_GRIPPER_CLOSED_VAL = 15

# --- MODEL CONSTANTS ---
PROPRIO_DIM = 7

# --- SPEED LIMITS ---
MIN_SPEED = 0.005
MAX_SPEED = 0.02
