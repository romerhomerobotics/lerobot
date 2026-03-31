import json
import os
from pathlib import Path

repo_id = "HomeRobotics/alotb_v0"
dataset_dir = Path("/home/romer-vla-sim/.cache/huggingface/lerobot/HomeRobotics/alotb_v0")

rename_map = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
    "state": "observation.state",
    "actions": "action"
}

# 1. Update stats.json
stats_path = dataset_dir / "meta/stats.json"
if stats_path.exists():
    with open(stats_path, "r") as f:
        stats = json.load(f)

    new_stats = {}
    for k, v in stats.items():
        new_stats[rename_map.get(k, k)] = v

    with open(stats_path, "w") as f:
        json.dump(new_stats, f, indent=4)
    print("Fixed stats.json!")
else:
    print("No stats.json found.")
