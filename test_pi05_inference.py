import torch
from pathlib import Path
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

def run_dummy_inference():
    # Model path from your fine-tuning run
    policy_path = "/home/romer-vla-sim/Workspace/lerobot/outputs/train/2026-03-30/21-17-34_pi05/checkpoints/020000/pretrained_model"
    print(f"Loading configuration from {policy_path}...")
    
    # 1. Load the policy configuration
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    # We must explicitly set these for the factory pattern to pick up the local PEFT adapter
    cfg.pretrained_path = policy_path
    
    # Check if adapter_config exists to flag PEFT usage
    if (Path(policy_path) / "adapter_config.json").exists():
        cfg.use_peft = True
    
    # 2. Instantiate the policy directly (bypassing make_policy to avoid dataset meta requirements)
    print("Loading policy base weights and adapter weights into GPU...")
    if getattr(cfg, "use_peft", False):
        from peft import PeftConfig, PeftModel
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        
        peft_config = PeftConfig.from_pretrained(policy_path)
        base_name = peft_config.base_model_name_or_path
        
        print(f"Loading base model: {base_name}")
        policy = PI05Policy.from_pretrained(base_name, config=cfg)
        policy = PeftModel.from_pretrained(policy, policy_path, config=peft_config)
    else:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        policy = PI05Policy.from_pretrained(policy_path, config=cfg)
        
    policy.to("cuda")
    policy.eval()
    
    # Load the processors for normalization and tokenization if needed
    print("Loading pre/post processors...")
    dataset_stats_placeholder = {} # The processors loader will override this natively since it loads the safetensors from the folder
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, 
        pretrained_path=policy_path,
        dataset_stats=dataset_stats_placeholder
    )
    
    print("Policy successfully loaded! Ready for inference.")
    print("-" * 50)
    
    # 3. Create a dummy un-normalized observation
    # We create dummy batch data mimicking exactly what standard pipelines output (Batch Size = 1)
    bs = 1
    dummy_obs = {}
    
    for key, feature in cfg.input_features.items():
        # Typically image feature shapes are (C, H, W). We create zeros.
        # Images normally come in as floats [0.0, 1.0] when arriving at the pipeline.
        shape = (bs, *feature.shape)
        dummy_obs[key] = torch.zeros(shape, dtype=torch.float32)

    # If the policy expects a text prompt, include it.
    dummy_task = ["pick up the red cube"] * bs
    
    # Add task natively if the processor looks for it
    dummy_obs["task"] = dummy_task
    
    print("Input observation shapes (before preprocessing):")
    for k, v in dummy_obs.items():
        if isinstance(v, torch.Tensor):
            print(f"- {k}: {v.shape}")
        else:
            print(f"- {k}: [\"{v[0]}\"]")
            
    # 4. Run the inference pipeline
    with torch.no_grad():
        # Process the inputs (Normalizer, Text Tokenizers via Hugging Face inside preprocessor)
        processed_obs = preprocessor(dummy_obs)
        
        # We need to move the tensors to the GPU before policy inference as the preprocessor computes on CPU usually
        for k, v in processed_obs.items():
            if isinstance(v, torch.Tensor):
                processed_obs[k] = v.to("cuda")
        
        print("\nExecuting inference step...")
        # Get next action
        raw_action = policy.select_action(processed_obs)
        
        # Post-process the output (Un-normalize)
        final_action = postprocessor(raw_action.cpu())
        
    print("-" * 50)
    print("Inference successful!")
    print(f"Predicted Output Action Chunk Shape: {final_action.shape}")

if __name__ == "__main__":
    run_dummy_inference()
