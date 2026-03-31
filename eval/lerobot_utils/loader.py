import os
import torch
import numpy as np
import logging
from pathlib import Path
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

logger = logging.getLogger(__name__)

class LeRobotModel:
    def __init__(self, cfg):
        self.cfg = cfg
        policy_path = cfg.pretrained_checkpoint

        logger.info(f"[LeRobot] Loading model from checkpoint: {policy_path}")
        
        # Load the policy config
        self.hf_config = PreTrainedConfig.from_pretrained(policy_path)
        self.hf_config.pretrained_path = policy_path
        
        # Check if PEFT (like Pi05 LoRA)
        is_peft = (Path(policy_path) / "adapter_config.json").exists()
        
        if is_peft:
            logger.info("[LeRobot] PEFT Adapter detected. Wrapping via PeftModel.")
            from peft import PeftConfig, PeftModel
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            
            peft_config = PeftConfig.from_pretrained(policy_path)
            base_name = peft_config.base_model_name_or_path
            
            self.policy = PI05Policy.from_pretrained(base_name, config=self.hf_config)
            self.policy = PeftModel.from_pretrained(self.policy, policy_path, config=peft_config)
            
        else:
            logger.info("[LeRobot] No PEFT Adapter. Loading natively (e.g. ACT full weights).")
            # Currently fallback dynamically checking config type to load either ACT or other full weight model
            if self.hf_config.type == "act":
                from lerobot.policies.act.modeling_act import ACTPolicy
                self.policy = ACTPolicy.from_pretrained(policy_path, config=self.hf_config)
            elif self.hf_config.type == "pi05":
                from lerobot.policies.pi05.modeling_pi05 import PI05Policy
                self.policy = PI05Policy.from_pretrained(policy_path, config=self.hf_config)
            else:
                from lerobot.policies.factory import get_policy_class
                policy_cls = get_policy_class(self.hf_config.type)
                self.policy = policy_cls.from_pretrained(policy_path, config=self.hf_config)
        
        self.policy.to("cuda")
        self.policy.eval()
        
        # Load Pre and Post Processors
        logger.info("[LeRobot] Loading Pre/Post Processors from fine-tuning stats.")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.hf_config, 
            pretrained_path=policy_path,
            dataset_stats={}
        )

    @torch.no_grad()
    def predict_actions(self, observation: dict, task_description: str = "") -> list:
        bs = 1
        tensor_obs = {}
        
        # 1. Convert Numpy Inputs to PyTorch Format Expected by LeRobot
        for key, value in observation.items():
            if "images" in key:
                # Numpy HWC uint8/float -> PyTorch CHW Float tensor in [0, 1] usually handled by preprocessor?
                # Actually, LeRobot preprocessor usually accepts native torch CHW Float directly, or numpy arrays.
                # Let's cleanly construct the dict expected by the preprocessor.
                # In standard usage: (C, H, W) scaled between 0 and 1, add batch dimension.
                if isinstance(value, np.ndarray):
                    # Transpose from (H, W, C) to (C, H, W)
                    v_t = np.transpose(value, (2, 0, 1))
                    # Scale to float32 [0., 1.] if it's uint8
                    if v_t.dtype == np.uint8:
                        v_t = v_t.astype(np.float32) / 255.0
                    tensor_obs[key] = torch.from_numpy(v_t).unsqueeze(0).float()
            elif "state" in key:
                if isinstance(value, np.ndarray):
                    tensor_obs[key] = torch.from_numpy(value).unsqueeze(0).float()

        if task_description:
            tensor_obs["task"] = [task_description] * bs

        # 2. Preprocess (Handles Normalization via loaded safetensors)
        processed_obs = self.preprocessor(tensor_obs)
        for k, v in processed_obs.items():
             if isinstance(v, torch.Tensor):
                 processed_obs[k] = v.to("cuda")

        # 3. Policy Inference
        raw_action = self.policy.select_action(processed_obs)

        # 4. Postprocess (Unnormalize and Unbatch)
        final_action_chunk = self.postprocessor(raw_action.cpu()) # Shape: (horizon, 7) or (1, horizon, 7)
        
        if final_action_chunk.ndim == 3: # (Batch, Horizon, ActionDim)
             final_action_chunk = final_action_chunk[0] # Drop batch output
             
        # 5. Output as list of 1D numpy arrays for the Robot execution loop
        return [a.numpy() for a in final_action_chunk]

def load_model(cfg):
    return LeRobotModel(cfg)
