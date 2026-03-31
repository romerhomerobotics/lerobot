import argparse
import os
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np
import rclpy

# Add parent dir to path to allow importing modules
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
)

def set_seed_everywhere(seed=42):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_episode(cfg, model, resize_size, ros: ROSInterface, args, task_description: str):
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    max_steps = 10_000
    success = False
    flush = True
    frames_received = 0
    frames_used = 0

    episode_start_wall = None
    prev_ros_secs = None

    rollout_data = {
        "full_frames": [],
        "wrist_frames": [],
        "states": [],
        "actions": [],
        "loop_dt": [],
        "timestamps": []
    }

    inference_data = []

    velocity_data = {
        "timestamps": [],
        "raw_linear_speeds": [],
        "clamped_linear_speeds": [],
        "delta_pos_xyz": [],
        "angular_speeds": [],
    }

    ros.go_to_default_joint()
    ros.go_to_default_joint()
    prev_gripper_state = None
    start_pos = None

    rate = ros.create_rate(args.control_freq)

    try:
        while t < max_steps:
            fetch_start_wall = time.time()
            full_image, wrist_image, raw_proprio, gripper, full_stamp, _wrist_stamp = ros.get_latest()

            if full_image is None or wrist_image is None or full_stamp is None or raw_proprio is None or gripper is None:
                if t % 100 == 0:
                    print("Waiting for all sensor data (image, wrist, proprio, gripper)...")
                time.sleep(0.1)
                continue
            frames_received += 1

            curr_wall_time = time.time()

            try:
                curr_ros_secs = full_stamp.sec + (full_stamp.nanosec / 1e9)
            except AttributeError:
                curr_ros_secs = float(full_stamp)

            if episode_start_wall is None:
                episode_start_wall = fetch_start_wall
                prev_ros_secs = curr_ros_secs

            dt = curr_ros_secs - prev_ros_secs
            if dt <= 0:
                dt = 1.0 / args.control_freq

            prev_ros_secs = curr_ros_secs
            rel_time = fetch_start_wall - episode_start_wall

            pos = np.asarray(raw_proprio["eef_pos"], dtype=np.float32)
            quat = np.asarray(raw_proprio["eef_quat"], dtype=np.float32)
            gripper_val = unnormalize_gripper(float(gripper), args)

            proprio = {
                "eef_pos": pos,
                "eef_quat": quat,
                "gr_state": np.array([gripper_val])
            }

            if start_pos is None:
                start_pos = pos

            observation, _image = prepare_observation(
                image=full_image,
                wrist_image=wrist_image,
                proprio=proprio,
                resize_size=resize_size
            )
            
            fetch_end_wall = time.time()
            inference_data.append({
                "type": "fetch",
                "start": fetch_start_wall - episode_start_wall,
                "end": fetch_end_wall - episode_start_wall,
                "index": frames_used + 1 if len(action_queue) == 0 else frames_used
            })

            if len(action_queue) == 0:
                frames_used += 1

                inf_start_wall = time.time()
                inf_start_plot = inf_start_wall - episode_start_wall

                actions = model.predict_actions(observation, task_description)

                inf_end_wall = time.time()
                inf_duration = inf_end_wall - inf_start_wall

                inference_data.append({
                    "type": "inference",
                    "start": inf_start_plot,
                    "end": inf_start_plot + inf_duration,
                    "index": frames_used
                })

                if flush:
                   flush = False
                   continue
                action_queue.extend(actions)

            action = action_queue.popleft()
            action_gripper = normalize_gripper(action[-1], args)

            if prev_gripper_state is None:
                prev_gripper_state = action_gripper
                gripper_state_has_changed = False

            if abs(action_gripper - prev_gripper_state) > 30:
                gripper_state_has_changed = True
                prev_gripper_state = action_gripper
            else:
                gripper_state_has_changed = False

            delta_pos = action[0:3]
            delta_rpy = action[3:6]
            raw_speed = np.linalg.norm(delta_pos)
            delta_pos = normalize_speed(delta_pos, args)

            rollout_data["full_frames"].append(full_image.copy())
            rollout_data["wrist_frames"].append(wrist_image.copy())
            rollout_data["states"].append({"pos": pos.copy(), "gripper": gripper_val})
            rollout_data["actions"].append(
                {"dpos": delta_pos.copy(), "drpy": delta_rpy.copy(), "gripper": action_gripper}
            )
            rollout_data["loop_dt"].append(dt)
            rollout_data["timestamps"].append(rel_time)

            velocity_data["timestamps"].append(rel_time)
            velocity_data["raw_linear_speeds"].append(raw_speed)
            velocity_data["clamped_linear_speeds"].append(np.linalg.norm(delta_pos))
            velocity_data["delta_pos_xyz"].append(delta_pos.copy())
            velocity_data["angular_speeds"].append(np.linalg.norm(delta_rpy))

            pub_start_wall = time.time()
            if args.control_mode == "pose":
                 delta_action = np.concatenate([delta_pos, delta_rpy])
                 ros.publish_action_pose(
                     delta_action, action_gripper, base_pos=pos, base_quat=quat, default_pos=start_pos
                 )
            else:
                 ros.publish_action(np.concatenate([delta_pos * args.control_freq, np.zeros(4)]), action_gripper)

            if gripper_state_has_changed:
                print(f"Gripper state changed to {action_gripper}")
                time.sleep(0.5)
                gripper_state_has_changed = False
                
            pub_end_wall = time.time()
            inference_data.append({
                "type": "publish",
                "start": pub_start_wall - episode_start_wall,
                "end": pub_end_wall - episode_start_wall,
                "index": frames_used
            })

            rate.sleep()
            t += 1

    except KeyboardInterrupt:
        print("Episode interrupted by user.")
        return success, rollout_data, velocity_data, inference_data

    except Exception:
        print("Episode terminated with error:")
        traceback.print_exc()
        raise

    print(f"Episode finished! Frames Received: {frames_received} | Inferences: {frames_used} | Commands Sent: {t}")

    return success, rollout_data, velocity_data, inference_data


def evaluate():
    parser = argparse.ArgumentParser(description="Eval LeRobot")
    parser.add_argument("config", type=str, help="Name of the config file (in configs/)")
    parser.add_argument("--mode", type=str, choices=["sim", "real"], default="sim", help="Mode: 'sim' or 'real'")
    parser.add_argument(
        "--task", type=str, default="pick up the object", help="Task description"
    )
    parser.add_argument("--control_freq", type=float, default=10.0, help="Control frequency in Hz")
    parser.add_argument("--dir", type=str, default="1")
    parser.add_argument("--resize_size", type=int, default=224)

    args = parser.parse_args()
    cfg = get_config(args.config)
    set_seed_everywhere(42)
    os.makedirs(args.dir, exist_ok=True)
    
    args.speed_limit = getattr(cfg, "speed_limit", False)
    args.control_mode = getattr(cfg, "control_mode", "velocity")

    model = load_model(cfg)

    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    ros = ROSInterface(mode=args.mode)

    def safe_spin(node):
        try:
            rclpy.spin(node)
        except Exception:
            if rclpy.ok():
                traceback.print_exc()

    spinner_thread = threading.Thread(target=safe_spin, args=(ros,), daemon=True)
    spinner_thread.start()

    resize_size = (args.resize_size, args.resize_size)

    _success, rollout_data, velocity_data, inference_data = run_episode(
        cfg, model, resize_size, ros, args, task_description=args.task
    )

    save_annotated_videos(rollout_data, mode=args.mode, task_description=args.task, path=args.dir)
    plot_timing_stats(rollout_data, inference_data, mode=args.mode, task_description=args.task, path=args.dir)
    plot_velocity_profiles(velocity_data, mode=args.mode, task_description=args.task, path=args.dir)

    if rclpy.ok():
        rclpy.shutdown()
    spinner_thread.join(timeout=1.0)

if __name__ == "__main__":
    evaluate()
