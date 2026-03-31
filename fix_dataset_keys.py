import json
import os
import glob
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

repo_id = "HomeRobotics/alotb_v0"
dataset_dir = Path("/home/romer-vla-sim/.cache/huggingface/lerobot/HomeRobotics/alotb_v0")

rename_map = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
    "state": "observation.state",
    "actions": "action"
}

# 1. Update info.json
info_path = dataset_dir / "meta/info.json"
with open(info_path, "r") as f:
    info = json.load(f)

new_features = {}
for k, v in info["features"].items():
    if k in rename_map:
        new_features[rename_map[k]] = v
        # Also fix the name inside the dictionary if it exists
        if v.get("names"):
            if v["names"] == [k]:
                v["names"] = [rename_map[k]]
    else:
        new_features[k] = v
info["features"] = new_features

with open(info_path, "w") as f:
    json.dump(info, f, indent=4)

# 2. Update Parquet files
parquet_files = glob.glob(str(dataset_dir / "data/**/*.parquet"), recursive=True)
for p in parquet_files:
    df = pd.read_parquet(p)
    df = df.rename(columns=rename_map)
    df.to_parquet(p, index=False)

print("Dataset keys successfully renamed!")
