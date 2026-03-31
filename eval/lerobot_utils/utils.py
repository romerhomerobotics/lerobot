import json
import os
import random
import traceback
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

from lerobot_utils.constants import (
    MAX_SPEED,
    MIN_SPEED,
    REAL_GRIPPER_THRESHOLD,
    REAL_MAX_VAL,
    SIM_GRIPPER_CLOSED_VAL,
    SIM_GRIPPER_THRESHOLD,
    SIM_MAX_VAL,
)

def prepare_observation(
    image: np.ndarray,
    wrist_image: Optional[np.ndarray],
    proprio: Dict[str, np.ndarray],
    resize_size: Union[int, Tuple[int, int]],
):
    """
    Prepare observation using standard LeRobot structure.
    Returns np.ndarray of shape (H, W, C) for images in uint8 format.
    The loader's preprocessor will handle torch stack/C,H,W later.
    """
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # 1. Main visual anchor
    image_resized = cv2.resize(image, resize_size, interpolation=cv2.INTER_LINEAR)
    
    # Map to canonical LeRobot feature keys
    observation = {
        "observation.images.image": image_resized,
        "observation.state": np.concatenate((proprio["eef_pos"], proprio["eef_quat"], proprio["gr_state"])),
    }

    # 2. Wrist view
    if wrist_image is not None:
        wrist_image_resized = cv2.resize(wrist_image, resize_size, interpolation=cv2.INTER_LINEAR)
        observation["observation.images.wrist_image"] = wrist_image_resized

    return observation, image

def normalize_gripper(gripper, args):
    gripper = np.clip(gripper, -1.0, 1.0)
    sim_value = (gripper + 1) * 50
    if args.mode == "real":
        sim_value = 0 if sim_value < REAL_GRIPPER_THRESHOLD else sim_value
    else:
        sim_value = SIM_GRIPPER_CLOSED_VAL if sim_value < SIM_GRIPPER_THRESHOLD else sim_value
    return float(sim_value)

def unnormalize_gripper(g_state, args):
    if args.mode == "real":
        max_sim_val = REAL_MAX_VAL
    else:
        max_sim_val = SIM_MAX_VAL
    g_state = np.clip(g_state, 0.0, max_sim_val)
    model_value = (g_state / max_sim_val) * 2.0 - 1.0
    return model_value

def normalize_speed(dpos, args):
    speed = np.linalg.norm(dpos)
    if speed < MIN_SPEED:
        if speed > 1e-8:
            dpos = (dpos / speed) * MIN_SPEED
    elif speed > MAX_SPEED:
        dpos = (dpos / speed) * MAX_SPEED
    return dpos

def plot_velocity_profiles(velocity_data, mode, task_description, path):
    ts = np.array(velocity_data["timestamps"])
    raw = np.array(velocity_data["raw_linear_speeds"])
    clamped = np.array(velocity_data["clamped_linear_speeds"])
    dxyz = np.array(velocity_data["delta_pos_xyz"])
    ang = np.array(velocity_data["angular_speeds"])

    if len(ts) == 0:
        print("No velocity data to plot.")
        return

    sort_idx = np.argsort(ts)
    ts, raw, clamped = ts[sort_idx], raw[sort_idx], clamped[sort_idx]
    dxyz, ang = dxyz[sort_idx], ang[sort_idx]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Velocity Profile — mode={mode}\\n{task_description}", fontsize=13)

    ax = axes[0]
    ax.plot(ts, raw, label="Raw ‖Δpos‖", alpha=0.7)
    ax.plot(ts, clamped, label="Clamped ‖Δpos‖", alpha=0.9)
    ax.axhline(MIN_SPEED, color="orange", linestyle="--", linewidth=1, label=f"MIN_SPEED={MIN_SPEED}")
    ax.axhline(MAX_SPEED, color="red", linestyle="--", linewidth=1, label=f"MAX_SPEED={MAX_SPEED}")
    ax.set_ylabel("Linear Speed (m/step)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    ax = axes[1]
    ax.plot(ts, dxyz[:, 0], label="Δx", alpha=0.8)
    ax.plot(ts, dxyz[:, 1], label="Δy", alpha=0.8)
    ax.plot(ts, dxyz[:, 2], label="Δz", alpha=0.8)
    ax.set_ylabel("Position Delta (m/step)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(ts, ang, label="‖Δrpy‖", color="purple", alpha=0.8)
    ax.set_ylabel("Angular Speed (rad/step)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"velocity_profile_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=150)
    plt.close(fig)

def plot_timing_stats(rollout_data, inference_data, mode, task_description, path):
    ts = np.array(rollout_data.get("timestamps", []))
    loop_dt = np.array(rollout_data.get("loop_dt", np.zeros_like(ts)))

    if len(ts) == 0:
        return

    sort_idx = np.argsort(ts)
    ts = ts[sort_idx]
    loop_dt = loop_dt[sort_idx]

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 1.2, 1.2, 1.2]})
    
    ax_dt, ax_comb, ax_fetch, ax_inf, ax_pub = axes

    ax_dt.plot(ts, loop_dt, label="Control Step (dt)", color="royalblue", alpha=0.9, linewidth=1.5)
    ax_dt.set_ylabel("Duration (s)")
    title_base = f"Timing & Inference Profile — mode={mode}\\n{task_description}"
    ax_dt.set_title(title_base)
    ax_dt.legend(loc="upper right")
    ax_dt.grid(True, alpha=0.3)
    ax_dt.set_xlim(left=0)

    stage_styles = {
        "fetch": {"color": "dodgerblue", "marker": "*", "label": "Fetch Data", "ax": ax_fetch},
        "inference": {"color": "firebrick", "marker": "s", "label": "Model Inference", "ax": ax_inf},
        "publish": {"color": "forestgreen", "marker": "^", "label": "Publish Action", "ax": ax_pub}
    }

    def draw_event(ax, x_start, x_end, y, color, marker):
        if (x_end - x_start) < 0.01:
            x_end = x_start + 0.01
        ax.plot([x_start, x_end], [y, y], color=color, linewidth=2, zorder=1)
        ax.plot(x_start, y, marker=marker, markerfacecolor='none', 
                markeredgecolor=color, markersize=7, linestyle='None', zorder=2)
        ax.plot(x_end, y, marker=marker, markerfacecolor=color, 
                markeredgecolor=color, markersize=7, linestyle='None', zorder=2)

    for inf in inference_data:
        itype = inf.get("type", "inference")
        style = stage_styles.get(itype, stage_styles["inference"])
        y_idx = inf["index"]
        x_start = inf["start"]
        x_end = inf["end"]
        draw_event(ax_comb, x_start, x_end, y_idx, style["color"], style["marker"])
        draw_event(style["ax"], x_start, x_end, y_idx, style["color"], style["marker"])

    ax_comb.set_ylabel("Inference Index\\n(Combined)")
    ax_fetch.set_ylabel("Fetch")
    ax_inf.set_ylabel("Inference")
    ax_pub.set_ylabel("Publish")
    ax_pub.set_xlabel("Wall-Clock Time (s)")
    
    for ax in axes[1:]:
        ax.grid(True, axis='both', alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    legend_elements = [
        Line2D([0], [0], color=style["color"], marker=style["marker"], 
               markerfacecolor=style["color"], markersize=8, label=style["label"])
        for style in stage_styles.values()
    ]
    ax_comb.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_filepath = os.path.join(path, f"inference_gantt_{mode}_{timestamp_str}_full.png")
    plt.savefig(full_filepath, dpi=150)
    plt.close(fig)

def save_annotated_videos(rollout_data, mode, task_description, path):
    if not rollout_data["full_frames"]:
        return

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"rollout_{mode}_{timestamp_str}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    h_f, w_f, _ = rollout_data["full_frames"][0].shape
    out_full = cv2.VideoWriter(os.path.join(path, f"{prefix}_full.mp4"), fourcc, 10.0, (w_f, h_f))

    h_w, w_w, _ = rollout_data["wrist_frames"][0].shape
    out_wrist = cv2.VideoWriter(os.path.join(path, f"{prefix}_wrist.mp4"), fourcc, 10.0, (w_w, h_w))

    scale_f = min(w_f / 1280.0, h_f / 720.0)
    font_scale_f = max(0.2, 0.66 * scale_f)
    thickness_f = max(1, int(2 * scale_f))
    x_off_f = max(2, int(10 * scale_f))
    y_off_f1 = max(10, int(25 * scale_f))
    y_off_f2 = max(20, int(55 * scale_f))
    y_off_f3 = max(30, int(85 * scale_f))

    scale_w = min(w_w / 640.0, h_w / 480.0)
    font_scale_w = max(0.15, 0.4 * scale_w)
    thickness_w = max(1, int(1 * scale_w))
    x_off_w = max(2, int(10 * scale_w))
    y_off_w1 = max(8, int(25 * scale_w))
    y_off_w2 = max(16, int(45 * scale_w))
    y_off_w3 = max(24, int(65 * scale_w))

    num_frames = min(
        len(rollout_data["full_frames"]),
        len(rollout_data["wrist_frames"]),
        len(rollout_data["states"]),
        len(rollout_data["actions"]),
        len(rollout_data["timestamps"]),
    )

    for i in range(num_frames):
        f_img = rollout_data["full_frames"][i].copy()
        w_img = rollout_data["wrist_frames"][i].copy()
        state = rollout_data["states"][i]
        action = rollout_data["actions"][i]
        rel_time = rollout_data["timestamps"][i]

        state_txt = f"S: P[{state['pos'][0]:.4f}, {state['pos'][1]:.4f}, {state['pos'][2]:.4f}] G[{state['gripper']:.4f}]"
        action_txt = f"A: dP[{action['dpos'][0]:.4f}, {action['dpos'][1]:.4f}, {action['dpos'][2]:.4f}] G[{action['gripper']:.4f}]"
        time_txt = f"Wall Time: {rel_time:.2f}s"

        if f_img.shape[-1] == 3:
            f_img = cv2.cvtColor(f_img, cv2.COLOR_RGB2BGR)
            w_img = cv2.cvtColor(w_img, cv2.COLOR_RGB2BGR)

        cv2.putText(f_img, state_txt, (x_off_f, y_off_f1), cv2.FONT_HERSHEY_SIMPLEX, font_scale_f, (0, 255, 0), thickness_f)
        cv2.putText(f_img, action_txt, (x_off_f, y_off_f2), cv2.FONT_HERSHEY_SIMPLEX, font_scale_f, (0, 0, 255), thickness_f)
        cv2.putText(f_img, time_txt, (x_off_f, y_off_f3), cv2.FONT_HERSHEY_SIMPLEX, font_scale_f, (255, 255, 0), thickness_f)

        cv2.putText(w_img, state_txt, (x_off_w, y_off_w1), cv2.FONT_HERSHEY_SIMPLEX, font_scale_w, (0, 255, 0), thickness_w)
        cv2.putText(w_img, action_txt, (x_off_w, y_off_w2), cv2.FONT_HERSHEY_SIMPLEX, font_scale_w, (0, 0, 255), thickness_w)
        cv2.putText(w_img, time_txt, (x_off_w, y_off_w3), cv2.FONT_HERSHEY_SIMPLEX, font_scale_w, (255, 255, 0), thickness_w)

        out_full.write(f_img)
        out_wrist.write(w_img)

    out_full.release()
    out_wrist.release()

def save_actions_json(rollout_data, path):
    if not rollout_data.get("actions"):
        return

    try:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(path, f"actions_{timestamp_str}.json")

        with open(filepath, "w") as f:
            num_actions = min(
                len(rollout_data["actions"]),
                len(rollout_data.get("raw_actions", rollout_data["actions"])),
                len(rollout_data.get("inf_times", []) or [0.0] * len(rollout_data["actions"])),
                len(rollout_data["timestamps"]),
            )
            for i in range(num_actions):
                action = rollout_data["actions"][i]
                raw = rollout_data.get("raw_actions", rollout_data["actions"])[i]
                inf_ms = rollout_data.get("inf_times", [0.0]*len(rollout_data["actions"]))[i]
                ts = rollout_data["timestamps"][i]
                dt = rollout_data.get("loop_dt", [0.0]*len(rollout_data["actions"]))[i]

                entry = {
                    "t": i,
                    "timestamp": float(ts),
                    "loop_dt_ms": float(dt * 1000.0),
                    "inf_ms": float(inf_ms),
                    "raw_delta": raw["dpos"].tolist() if hasattr(raw["dpos"], "tolist") else raw["dpos"],
                    "clamped_delta": action["dpos"].tolist() if hasattr(action["dpos"], "tolist") else action["dpos"],
                    "action": f"dx: {action['dpos'][0]:.4f}, dy: {action['dpos'][1]:.4f}, dz: {action['dpos'][2]:.4f}"
                }
                f.write(json.dumps(entry) + "\\n")
    except Exception as e:
        traceback.print_exc()
