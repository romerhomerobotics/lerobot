import os
import shutil
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

def convert(
    coupled_data_dir: str, 
    repo_name: str, 
    space: str,
    push_to_hub: bool
):
    
    # 1. Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Cleaning existing directory at {output_path}...")
        shutil.rmtree(output_path)

    # 2. Define the Schema (Strictly using tuples, not lists) using LeRobot canonical keys
    features = {
        "observation.images.image": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist_image": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,) if space == "joint" else (7,),
            "names": ["observation.state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,) if space == "joint" else (7,),
            "names": ["action"],
        },
    }

    # Locate files first so we can extract the FPS before dataset initialization
    data_dir_path = Path(coupled_data_dir)
    npz_files = list(data_dir_path.glob("*.npz"))
    
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {coupled_data_dir}")

    # Peek at the first file to get the target_hz
    preview_data = np.load(npz_files[0], allow_pickle=True)
    if 'target_hz' not in preview_data:
        raise KeyError("target_hz not found in .npz file")
    else:
        FPS = preview_data['target_hz'].item()
    preview_data.close()

    # 3. Initialize the Dataset Writer
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda", # You can leave this as panda or change to your robot's name
        fps=FPS,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 4. The Processing Loop
    print(f"Found {len(npz_files)} .npz files to process.")

    for npz_file in npz_files:
        print(f"Processing: {npz_file.name}")
        
        # Load the entire .npz into memory
        data = np.load(npz_file, allow_pickle=True)
        
        # Extract arrays
        images_front = data['image']      # Expected shape: (N, 224, 224, 3)
        images_wrist = data['image_hand'] # Expected shape: (N, 224, 224, 3)
        states = data['state']
        actions = data['actions']
        
        # Ensure task is a decoded python string
        if 'task' not in data:
            raise KeyError("Task not found in .npz file")
        else:
            task_string = data['task'].item()
        if isinstance(task_string, bytes):
            task_string = task_string.decode()
        else:
            task_string = str(task_string)

        num_frames = len(states)

        # Loop through each frame in the episode
        for i in range(num_frames):
            
            # Enforce strict float32 Numpy arrays for validation
            state_array = np.array(states[i], dtype=np.float32) 
            action_array = np.array(actions[i], dtype=np.float32)

            # Build the frame dict exactly matching the features keys
            # Notice we pass the raw numpy arrays directly, NO PyTorch conversion
            frame_dict = {
                "observation.images.image": images_front[i],
                "observation.images.wrist_image": images_wrist[i],
                "observation.state": state_array,
                "action": action_array,
                "task": task_string, # Task goes directly into every frame
            }
            
            dataset.add_frame(frame_dict)
            
        # Flush the trajectory to Parquet/MP4 on disk
        dataset.save_episode()
        data.close()

    # Consolidate ensures indices, info.json, and stats.json are all finalized locally 
    print("Consolidating dataset statistics and saving info files...")
    dataset.consolidate()

    print(f"\nConversion Complete! Output path: {output_path}")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_name}...")
        dataset.push_to_hub(
            tags=["custom", "vla", "rlds"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

def main(
    coupled_data_dir: str,
    repo_name: str,
    space: str = "eef", # "eef" or "joint"
    push_to_hub: bool = False
):
    """
    Convert .npz teleop data to LeRobot format.
    """
    convert(
        coupled_data_dir=coupled_data_dir,
        repo_name=repo_name,
        space=space,
        push_to_hub=push_to_hub
    )

if __name__ == "__main__":
    tyro.cli(main)
