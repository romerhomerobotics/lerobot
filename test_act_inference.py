import torch
from pathlib import Path
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

def run_dummy_inference():
    # Model path from your fine-tuning ACT run
    policy_path = "/home/romer-vla-sim/Workspace/lerobot/outputs/train/2026-03-31/08-30-59_act/checkpoints/100000/pretrained_model"
    
    print(f"Loading ACT configuration from {policy_path}...")
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    
    # 1. Instantiate the ACT policy
    # Unlike your PEFT pi05 model, ACT typically undergoes a full fine-tune (model.safetensors rather than adapter_model.safetensors)
    # This means the architecture has stored the entire policy locally, and from_pretrained handles everything internally 
    print("Loading native ACT policy weights into GPU...")
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.to("cuda")
    policy.eval()
    
    # Load the processors for normalization 
    print("Loading pre/post processors from fine-tuned statistics...")
    dataset_stats_placeholder = {}
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, 
        pretrained_path=policy_path,
        dataset_stats=dataset_stats_placeholder
    )
    
    print("ACT Policy successfully loaded! Ready for inference.")
    print("-" * 50)
    
    # 2. Create a dummy un-normalized observation
    # We create dummy batch data mimicking exactly what standard pipelines output (Batch Size = 1)
    bs = 1
    dummy_obs = {}
    
    for key, feature in cfg.input_features.items():
        # Typically image feature shapes are (C, H, W). We create zeros exactly matching the dimensions.
        shape = (bs, *feature.shape)
        dummy_obs[key] = torch.zeros(shape, dtype=torch.float32)

    # Note: ACT policy typically doesn't expect a text feature ('task') in its core dictionary like vision-language models do
    
    print("Input observation shapes (before preprocessing):")
    for k, v in dummy_obs.items():
        if isinstance(v, torch.Tensor):
            print(f"- {k}: {v.shape}")
        else:
            print(f"- {k}: [\"{v[0]}\"]")
            
    # 3. Run the inference pipeline
    with torch.no_grad():
        # Process the inputs (e.g., standardizing images and state distributions)
        processed_obs = preprocessor(dummy_obs)
        
        # We need to explicitly move the tensors to the GPU before passing them to the policy
        for k, v in processed_obs.items():
            if isinstance(v, torch.Tensor):
                processed_obs[k] = v.to("cuda")
        
        print("\nExecuting ACT inference chunk step...")
        # Get next chunk action
        raw_action = policy.select_action(processed_obs)
        
        # Post-process the output (Un-normalize to true physical space arrays)
        final_action = postprocessor(raw_action.cpu())
        
    print("-" * 50)
    print("Inference successful!")
    print(f"Predicted Output Action Chunk Shape: {final_action.shape}")

if __name__ == "__main__":
    run_dummy_inference()
