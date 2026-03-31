import json
import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class EvalConfig:
    pretrained_checkpoint: str
    num_open_loop_steps: int = 8
    speed_limit: bool = False
    control_mode: str = "velocity"

def get_config(config_name: str) -> EvalConfig:
    # 1. Resolve path
    base_dir = Path(__file__).resolve().parent.parent / "configs"
    fp = base_dir / config_name

    if fp.suffix != ".json":
        fp = fp.with_suffix(".json")

    # 2. Check existence and handle fallbacks
    if not fp.exists():
        fallback_path = Path(config_name)
        if fallback_path.exists():
            fp = fallback_path
        elif fallback_path.with_suffix(".json").exists():
            fp = fallback_path.with_suffix(".json")
        else:
            raise FileNotFoundError(f"Config file not found at {fp}")

    # 3. Load the user config file
    with open(fp, "r") as f:
        config_data = json.load(f)
        print(f"Loaded the config: {fp.stem}")

    # Expand ~ if present in checkpoint path
    if 'pretrained_checkpoint' in config_data:
        ckpt_path = config_data['pretrained_checkpoint']
        ckpt_path = os.path.expanduser(ckpt_path)
        config_data['pretrained_checkpoint'] = ckpt_path

    # Extract Eval parameters
    cfg = EvalConfig(
        pretrained_checkpoint=config_data.get("pretrained_checkpoint"),
        num_open_loop_steps=config_data.get("num_open_loop_steps", 8),
        speed_limit=config_data.get("speed_limit", False),
        control_mode=config_data.get("control_mode", "velocity"),
    )

    return cfg
