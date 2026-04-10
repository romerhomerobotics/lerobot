import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def play_video(data, image_keys):
    print("\n--- Playing Video (Press 'q' to quit, 'p' to pause) ---")
    num_frames = len(data)
    i = 0
    paused = False
    
    while i < num_frames:
        frames = []
        for ik in image_keys:
            img = data[i][ik]
            # Convert to (H, W, C)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            
            # Ensure it's uint8 BGR for OpenCV
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frames.append(img)
        
        # Combine images side-by-side
        combined = np.hstack(frames)
        
        # Overlay info
        cv2.putText(combined, f"Frame: {i}/{num_frames-1}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Inference Observations Playback", combined)
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resuming")
            while paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p'):
                    paused = False
                    print("Resuming")
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return

        if not paused:
            i += 1
            
    cv2.destroyAllWindows()

def visualize_inference_logs(log_dir="outputs/test_logs", play=True):
    # 1. Find the latest log file
    log_files = glob.glob(os.path.join(log_dir, "inference_obs_*.npy"))
    if not log_files:
        print(f"No inference logs found in {log_dir}")
        return

    latest_log = max(log_files, key=os.path.getctime)
    print(f"Analyzing: {latest_log}")
    
    # 2. Load data
    data = np.load(latest_log, allow_pickle=True)
    if len(data) == 0:
        print("Log file is empty.")
        return
    
    num_frames = len(data)
    keys = list(data[0].keys())
    print(f"Found {num_frames} frames with keys: {keys}")
    
    image_keys = [k for k in keys if "images" in k]
    state_keys = [k for k in keys if "state" in k]
    
    # 3. Statistics for states
    print("\n--- State Statistics ---")
    for sk in state_keys:
        state_data = np.array([d[sk] for d in data])
        if state_data.ndim > 1:
            avg = np.mean(state_data, axis=0)
            std = np.std(state_data, axis=0)
            print(f"{sk}: Mean={avg}, Std={std}")
        else:
            avg = np.mean(state_data)
            std = np.std(state_data)
            print(f"{sk}: Mean={avg:.4f}, Std={std:.4f}")

    # 4. Play Video
    if play and image_keys:
        play_video(data, image_keys)

    # 5. Static Summary Plot (Snapshots)
    if image_keys:
        num_cams = len(image_keys)
        fig, axes = plt.subplots(num_cams, 3, figsize=(15, 5 * num_cams))
        if num_cams == 1:
            axes = [axes]
            
        for i, ik in enumerate(image_keys):
            indices = [0, num_frames // 2, num_frames - 1]
            for j, idx in enumerate(indices):
                img = data[idx][ik]
                if img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                if img.max() > 1.0:
                    img = img / 255.0
                
                axes[i][j].imshow(img)
                axes[i][j].set_title(f"{ik}\nFrame {idx}")
                axes[i][j].axis('off')
        
        plt.tight_layout()
        img_plot_path = latest_log.replace(".npy", "_images.png")
        plt.savefig(img_plot_path)
        print(f"Image summary saved to: {img_plot_path}")
        plt.close()

    # 6. Plot State Trajectories
    if state_keys:
        plt.figure(figsize=(12, 6))
        for sk in state_keys:
            state_data = np.array([d[sk] for d in data])
            if state_data.ndim == 1:
                plt.plot(state_data, label=sk)
            else:
                for d in range(state_data.shape[1]):
                    plt.plot(state_data[:, d], label=f"{sk}[{d}]")
        
        plt.title("Inference State Trajectories")
        plt.legend()
        plt.grid(True)
        state_plot_path = latest_log.replace(".npy", "_states.png")
        plt.savefig(state_plot_path)
        print(f"State trajectories saved to: {state_plot_path}")
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs/test_logs", help="Directory containing logs")
    parser.add_argument("--no-play", action="store_true", help="Disable video playback")
    args = parser.parse_args()
    
    visualize_inference_logs(args.dir, play=not args.no_play)
