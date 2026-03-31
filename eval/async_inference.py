import argparse
import os
import sys
import time
import traceback
from collections import deque
import multiprocessing as mp 

import cv2
import numpy as np
import rclpy

# Prevent OpenCV from stealing CPU from PyTorch
cv2.setNumThreads(1)

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ros_utils import ROSInterface 
from lerobot_utils.config import get_config 
from lerobot_utils.loader import load_model 
from lerobot_utils.utils import ( 
    normalize_gripper,
    normalize_speed,
    plot_timing_stats,
    plot_velocity_profiles,
    prepare_observation,
    save_annotated_videos,
    unnormalize_gripper,
    save_actions_json
)

def set_seed_everywhere(seed=42):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def inference_process_worker(cfg, task_description, input_queue, output_queue, stop_event):
    print("\n[InferenceProcess] Loading LeRobot model in subprocess...")
    model = load_model(cfg)
    print("\n[InferenceProcess] Model Loaded. Sending READY signal.")
    
    # Send handshake to main process
    output_queue.put("READY")

    while not stop_event.is_set():
        try:
            try:
                # Wait for data from main loop
                observation, spawn_index, ep_start_wall = input_queue.get(timeout=0.5)
            except Exception: # Matches mp.queues.Empty
                continue
            
            inf_start_wall = time.time()
            start_offset = inf_start_wall - ep_start_wall

            # --- HEAVY INFERENCE ---
            actions = model.predict_actions(observation, task_description)
            # -----------------------

            inf_end_wall = time.time()
            inf_duration = (inf_end_wall - inf_start_wall) * 1000.0
            
            output_queue.put({
                "actions": actions,
                "inf_ms": inf_duration,
                "timing_record": {
                    "type": "inference",
                    "start": start_offset,
                    "end": start_offset + (inf_end_wall - inf_start_wall),
                    "index": spawn_index
                }
            })
            
        except Exception as e:
            print(f"[InferenceProcess] Error: {e}")
            break
    print("[InferenceProcess] Shutting down.")


def run_episode(cfg, resize_size, ros: ROSInterface, args, task_description: str):
    manager = mp.Manager()
    input_queue = mp.Queue(maxsize=1)
    output_queue = mp.Queue()
    stop_event = mp.Event()
    inference_data_list = manager.list() 
    
    local_action_queue = deque()
    last_inf_ms = 0.0
    episode_start_wall = None

    # START SUBPROCESS
    inf_proc = mp.Process(
        target=inference_process_worker,
        args=(cfg, task_description, input_queue, output_queue, stop_event),
        daemon=True
    )
    inf_proc.start()

    # --- HANDSHAKE LOGIC ---
    print(f"\n[MainLoop] Waiting for Inference Process to load model (approx 15-20s)...")
    while True:
        try:
            # Check if user killed the script while waiting
            if not inf_proc.is_alive():
                print("[MainLoop] Inference process died during load. Exiting.")
                return False, {}, {}, []
            
            msg = output_queue.get(timeout=1.0)
            if msg == "READY":
                print("[MainLoop] Model is READY. Starting robot control now!")
                break
        except Exception:
            pass # Keep waiting

    t = 0
    max_steps = 10_000
    rollout_data = {
        "full_frames": [], "wrist_frames": [], "states": [],
        "actions": [], "raw_actions": [], "inf_times": [],
        "loop_dt": [], "timestamps": []
    }
    velocity_data = {
        "timestamps": [], "raw_linear_speeds": [], "clamped_linear_speeds": [],
        "delta_pos_xyz": [], "angular_speeds": [],
    }

    ros.go_to_default_joint()
    prev_gripper_state = None
    start_pos = None
    rate = ros.create_rate(args.control_freq)
    dt_fixed = 1.0 / args.control_freq

    print(f"Starting async loop at {args.control_freq}Hz...")

    try:
        while t < max_steps:
            loop_start_wall = time.time()
            
            # 1. Capture latest data
            fetch_start_wall = time.time()
            full_image, wrist_image, raw_proprio, gripper, full_stamp, _ = ros.get_latest()
            fetch_end_wall = time.time()

            # --- DEFENSIVE CHECK: Ensure all sensors have published ---
            if full_image is None or full_stamp is None or raw_proprio is None or gripper is None:
                if t % 20 == 0:
                    print(f"Waiting for all sensor data [Image: {full_image is not None}, TF: {raw_proprio is not None}, Gripper: {gripper is not None}] (Mode: {args.mode})...")
                rate.sleep()
                continue
            # ----------------------------------------------------------

            if episode_start_wall is None:
                episode_start_wall = loop_start_wall

            rel_time = loop_start_wall - episode_start_wall

            # Record Fetch Timing
            inference_data_list.append({
                "type": "fetch",
                "start": fetch_start_wall - episode_start_wall,
                "end": fetch_end_wall - episode_start_wall,
                "index": t
            })

            # Prepare observation
            pos = np.asarray(raw_proprio["eef_pos"], dtype=np.float32)
            quat = np.asarray(raw_proprio["eef_quat"], dtype=np.float32)
            gripper_val = unnormalize_gripper(float(gripper), args)
            
            observation, _ = prepare_observation(
                image=full_image, wrist_image=wrist_image,
                proprio={"eef_pos": pos, "eef_quat": quat, "gr_state": np.array([gripper_val])},
                resize_size=resize_size
            )

            # 2. Push to Inference Process
            try:
                while not input_queue.empty():
                    input_queue.get_nowait()
                input_queue.put_nowait((observation, t, episode_start_wall))
            except Exception:
                pass

            # 3. Pull new actions from Inference Process
            try:
                while not output_queue.empty():
                    result = output_queue.get_nowait()
                    if isinstance(result, dict) and "actions" in result:
                        local_action_queue = deque(result["actions"])
                        last_inf_ms = result["inf_ms"]
                        inference_data_list.append(result["timing_record"])
            except Exception:
                pass

            # 4. Action Logic
            current_action = local_action_queue.popleft() if local_action_queue else None
            
            if current_action is None:
                current_action = np.zeros(7)
                current_action[-1] = unnormalize_gripper(float(gripper), args)
                if t % 50 == 0:
                    print(f"Waiting for first model action... (Inference speed: {last_inf_ms:.1f}ms)")

            # Process action
            action_gripper = normalize_gripper(current_action[-1], args)
            
            # --- Check and Print Gripper State Change ---
            if prev_gripper_state is None:
                prev_gripper_state = action_gripper
                
            gripper_state_has_changed = abs(action_gripper - prev_gripper_state) > 20.0
            if gripper_state_has_changed:
                print(f"[MainLoop] Gripper state changed to {action_gripper:.2f}")
                prev_gripper_state = action_gripper
            # --------------------------------------------
            
            delta_pos = current_action[0:3]
            delta_rpy = current_action[3:6]
            delta_pos_norm = normalize_speed(delta_pos, args) if args.speed_limit else delta_pos
            
            # 5. Logging
            rollout_data["full_frames"].append(full_image.copy())
            rollout_data["wrist_frames"].append(wrist_image.copy() if wrist_image is not None else full_image.copy())
            rollout_data["states"].append({"pos": pos.copy(), "gripper": gripper_val})
            rollout_data["actions"].append({"dpos": delta_pos_norm, "drpy": delta_rpy, "gripper": action_gripper})
            rollout_data["raw_actions"].append({"dpos": delta_pos, "drpy": delta_rpy, "gripper": action_gripper})
            rollout_data["inf_times"].append(last_inf_ms)
            rollout_data["loop_dt"].append(dt_fixed)
            rollout_data["timestamps"].append(rel_time)

            velocity_data["timestamps"].append(rel_time)
            velocity_data["raw_linear_speeds"].append(np.linalg.norm(delta_pos))
            velocity_data["clamped_linear_speeds"].append(np.linalg.norm(delta_pos_norm))
            velocity_data["delta_pos_xyz"].append(delta_pos_norm.copy())
            velocity_data["angular_speeds"].append(np.linalg.norm(delta_rpy))

            # 6. Publish
            publish_start_wall = time.time()
            if args.control_mode == "pose":
                delta_action = np.concatenate([delta_pos_norm, delta_rpy])
                ros.publish_action_pose(delta_action, action_gripper, base_pos=pos, base_quat=quat, default_pos=start_pos)
            else:
                ros.publish_action(np.concatenate([delta_pos_norm * args.control_freq, np.zeros(4)]), action_gripper)
            publish_end_wall = time.time()

            # Record Publish Timing
            inference_data_list.append({
                "type": "publish",
                "start": publish_start_wall - episode_start_wall,
                "end": publish_end_wall - episode_start_wall,
                "index": t
            })

            rate.sleep()
            t += 1

    except KeyboardInterrupt:
        print("\\n[MainLoop] Interrupted by user. Cleaning up...")
    except Exception:
        traceback.print_exc()
    finally:
        stop_event.set()
        inf_proc.terminate()
        inf_proc.join(timeout=1.0)

    return True, rollout_data, velocity_data, list(inference_data_list)


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--mode", type=str, default="real")
    parser.add_argument("--task", type=str, default="pick up the object")
    parser.add_argument("--control_freq", type=float, default=10.0)
    parser.add_argument("--dir", type=str, default="async_results")
    parser.add_argument("--resize_size", type=int, default=224)
    args = parser.parse_args()

    cfg = get_config(args.config)
    os.makedirs(args.dir, exist_ok=True)
    
    set_seed_everywhere(42)

    # Use arguments combined with configs natively
    args.speed_limit = getattr(cfg, "speed_limit", False)
    args.control_mode = getattr(cfg, "control_mode", "velocity")

    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    print(f"Initializing ROS in {args.mode} mode...")
    ros = ROSInterface(mode=args.mode)
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(ros)
    
    import threading
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    resize_size = (args.resize_size, args.resize_size)

    success, rollout_data, velocity_data, inference_data = run_episode(
        cfg, resize_size, ros, args, task_description=args.task
    )

    if rollout_data and len(rollout_data.get("timestamps", [])) > 5:
        print(f"Saving results to {args.dir}...")
        save_annotated_videos(rollout_data, mode=args.mode, task_description=args.task, path=args.dir)
        plot_timing_stats(rollout_data, inference_data, mode=args.mode, task_description=args.task, path=args.dir)
        plot_velocity_profiles(velocity_data, mode=args.mode, task_description=args.task, path=args.dir)
        save_actions_json(rollout_data, path=args.dir)
    else:
        print("Episode too short. Skipping plots and video saving.")

    ros.destroy_node()
    rclpy.shutdown()
    spinner.join(timeout=1.0)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    evaluate()
