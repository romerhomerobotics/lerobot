import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def visualize_log(latest_file):
    filename = os.path.basename(latest_file)
    print(f"\n{'='*60}")
    print(f" Raw Data Analysis: {filename}")
    print(f"{'='*60}")

    try:
        data = np.load(latest_file)
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        return
        
    if data.size == 0:
        print(f"File {filename} is empty.")
        return
        
    if data.ndim == 1:
        data = data.reshape(1, -1)

    timestamps = data[:, 0]
    relative_time = timestamps - timestamps[0]
    sample_indices = np.arange(len(data))
    
    if "cam_dt" in filename:
        # Cam DT shape: [N, 3] -> [timestamp, dt, type] (0=full, 1=wrist)
        dts_full = data[data[:, 2] == 0, 1]
        dts_wrist = data[data[:, 2] == 1, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Camera DT Analysis: {filename}", fontsize=16)
        
        # Full Cam DT
        axes[0, 0].plot(dts_full, color='blue', alpha=0.6, label='Full Cam DT')
        axes[0, 0].set_ylabel("Interval (s)")
        axes[0, 0].set_title("Full Camera Frame Intervals")
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(dts_full, bins=50, color='blue', alpha=0.5)
        axes[0, 1].set_title("Full Camera Jitter Distribution")
        
        # Wrist Cam DT
        axes[1, 0].plot(dts_wrist, color='orange', alpha=0.6, label='Wrist Cam DT')
        axes[1, 0].set_ylabel("Interval (s)")
        axes[1, 0].set_title("Wrist Camera Frame Intervals")
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(dts_wrist, bins=50, color='orange', alpha=0.5)
        axes[1, 1].set_title("Wrist Camera Jitter Distribution")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Stats
        for label, dts in [("Full Cam", dts_full), ("Wrist Cam", dts_wrist)]:
            if len(dts) > 0:
                avg_fps = 1.0 / np.mean(dts)
                print(f"{label:10} | Avg FPS: {avg_fps:6.2f} | Mean DT: {np.mean(dts)*1000:6.2f}ms | Max Stutter: {np.max(dts)*1000:6.2f}ms")
    
    elif "states" in filename:
        # State shape: [N, 9] -> [timestamp, x, y, z, qx, qy, qz, qw, gripper]
        raw_values = data[:, 1:9]
        labels = ["EEF X", "EEF Y", "EEF Z", "Qx", "Qy", "Qz", "Qw", "Gripper State"]
        num_plots = 8
        plot_type = "STATE"
    else:
        # Action shape: [N, 8] -> [timestamp, dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
        raw_values = data[:, 1:8]
        labels = ["Delta X", "Delta Y", "Delta Z", "D-Roll", "D-Pitch", "D-Yaw", "Gripper Command"]
        num_plots = 7
        plot_type = "ACTION"

    if "cam_dt" not in filename:
        # Standard Action/State visualization
        fig, axes = plt.subplots(num_plots, 2, figsize=(16, 2.5 * num_plots), gridspec_kw={'width_ratios': [3, 1]})
        fig.suptitle(f"{plot_type} Analysis: {filename}\nSamples: {len(data)}, Duration: {relative_time[-1]:.2f}s", fontsize=16)

        for i in range(num_plots):
            col_data = raw_values[:, i]
            axes[i, 0].step(sample_indices, col_data, where='post', color='blue', alpha=0.6)
            axes[i, 0].scatter(sample_indices, col_data, s=10, color='red', alpha=0.4)
            axes[i, 0].set_ylabel(labels[i], fontweight='bold', fontsize=9)
            axes[i, 0].grid(True, linestyle='--', alpha=0.4)
            
            axes[i, 1].hist(col_data, bins=30, color='green', alpha=0.6)
            axes[i, 1].grid(True, linestyle=':', alpha=0.5)

        axes[num_plots-1, 0].set_xlabel("Sample Index (Step #)")
        axes[num_plots-1, 1].set_xlabel("Value")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Statistics
        print(f"{'Feature':15} | {'Mean':8} | {'Std':8} | {'Min':8} | {'Max':8}")
        print("-" * 60)
        for i, label in enumerate(labels):
            col = raw_values[:, i]
            print(f"{label:15} | {np.mean(col):8.4f} | {np.std(col):8.4f} | {np.min(col):8.4f} | {np.max(col):8.4f}")

    # Save and Show
    plot_name = latest_file.replace(".npy", "_analysis.png")
    plt.savefig(plot_name)
    print(f"Analysis plot saved to: {plot_name}")

def main():
    log_dir = os.path.expanduser("~/lerobot/outputs/test_logs")
    if not os.path.exists(log_dir):
        print(f"Log directory {log_dir} does not exist.")
        return

    files = [f for f in os.listdir(log_dir) if f.endswith(".npy")]
    if not files:
        print("No log files found in " + log_dir)
        return

    files.sort(reverse=True)

    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            for f in files: visualize_log(os.path.join(log_dir, f))
        else:
            visualize_log(os.path.join(log_dir, sys.argv[1]))
    else:
        # Visualize latest of each type
        for pattern in ["actions", "states", "cam_dt"]:
            latest = next((f for f in files if pattern in f), None)
            if latest: visualize_log(os.path.join(log_dir, latest))

    if "DISPLAY" in os.environ and "--headless" not in sys.argv:
        plt.show()

if __name__ == "__main__":
    main()
