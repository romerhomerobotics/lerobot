# LeRobot Model Loading and Inference Pipeline Report

This report outlines the technical architecture of how the Hugging Face LeRobot repository handles loading pre-trained and fine-tuned policies, as well as its core inference loop.

## 1. Model Loading Architecture

Loading fine-tuned models is centrally managed by the **Policy Factory** pattern. It seamlessly coordinates between loading vanilla pre-trained weights, Parameter-Efficient Fine-Tuning (PEFT) adapters (like LoRA), and standard architectural instantiations.

### The Entry Point: `make_policy`
Location: [`src/lerobot/policies/factory.py`](file:///home/romer-vla-sim/Workspace/lerobot/src/lerobot/policies/factory.py#L406)

The primary entry point for loading models is the `make_policy` function. It looks at the policy configuration (`cfg`) and determines the loading strategy:

1. **Vanilla Pre-trained Loading**:
   If `cfg.pretrained_path` is set (but not using PEFT), the factory leverages the `from_pretrained` class method:
   ```python
   # Line 492
   policy = policy_cls.from_pretrained(**kwargs)
   ```

2. **PEFT Loading for Fine-Tuned Models**:
   If the model was fine-tuned using parameter-efficient methods (`cfg.use_peft=True` alongside a `pretrained_path`), the pipeline:
   - Loads the adapter's configuration to identify the base model name.
   - Loads the base policy via `from_pretrained`.
   - Wraps the base model with the adapter weights.
   ```python
   # Lines 501-514
   peft_config = PeftConfig.from_pretrained(peft_pretrained_path)
   kwargs["pretrained_name_or_path"] = peft_config.base_model_name_or_path
   policy = policy_cls.from_pretrained(**kwargs)
   policy = PeftModel.from_pretrained(policy, peft_pretrained_path, config=peft_config)
   ```

### Safetensors Loading: `from_pretrained`
Location: [`src/lerobot/policies/pretrained.py`](file:///home/romer-vla-sim/Workspace/lerobot/src/lerobot/policies/pretrained.py#L75)

All policies inherit from `PreTrainedPolicy`. The `from_pretrained` method in this base class handles checking if the requested weights are local or remote.
- If remote, it utilizes Hugging Face Hub APIs `hf_hub_download` to fetch the `model.safetensors` file.
- It then natively uses `load_model_as_safetensor` ([Line 146](file:///home/romer-vla-sim/Workspace/lerobot/src/lerobot/policies/pretrained.py#L146)) to strict-map the tensors to the PyTorch architecture.
- Finally, it explicitly sets the model to evaluation mode (`policy.eval()`, [Line 133](file:///home/romer-vla-sim/Workspace/lerobot/src/lerobot/policies/pretrained.py#L133)).

### Data Normalization Stats
Location: [`src/lerobot/policies/factory.py`](file:///home/romer-vla-sim/Workspace/lerobot/src/lerobot/policies/factory.py#L214)

Fine-tuned models are highly sensitive to normalization. LeRobot handles this gracefully using the `make_pre_post_processors` method. It builds the pre/post-processing pipelines necessary for image and state normalizations corresponding to the exact `stats` dictionary stored in the fine-tuned dataset directory.

---

## 2. The Inference Pipeline

The inference loop expects the pre/post-processing constraints to be met. Scripts like `lerobot_eval.py` execute this pipeline during rollout procedures.

### Initializing during Evaluation
Location: [`src/lerobot/scripts/lerobot_eval.py`](file:///home/romer-vla-sim/Workspace/lerobot/src/lerobot/scripts/lerobot_eval.py#L528)

At the start of the evaluation, the policy is constructed once via the `make_policy` call. The environment is reset, and the initial `observation` dict is captured. This `observation` dict must map to the expected schema (e.g. `observation.state`, `observation.images.image`).

### The Inference Step: `select_action`
Location: [`src/lerobot/scripts/lerobot_eval.py`](file:///home/romer-vla-sim/Workspace/lerobot/src/lerobot/scripts/lerobot_eval.py#L177)

During the environment loop, the inference is triggered by:
```python
action = policy.select_action(observation)
```

The underlying abstract method is defined in:
Location: [`src/lerobot/policies/pretrained.py`](file:///home/romer-vla-sim/Workspace/lerobot/src/lerobot/policies/pretrained.py#L199)

The `select_action` footprint states that:
> "When the model uses a history of observations, or outputs a sequence of actions, this method deals with caching."

This means the unified interface completely abstracts away complexities like Action Chunking (for Diffusion or ACT) or observation history buffers. The script simply provides the current step's observations as `Tensor` batches across the expected keys, and `select_action` internally manages returning just the single subsequent action to execute in the simulation or real world.
