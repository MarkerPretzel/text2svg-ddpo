import ml_collections
import imp
import os

# Import base configuration
base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config():
    config = base.get_config()

    # Base model to use
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"  # Same as aesthetic config

    # Training duration
    config.num_epochs = 200  # Match aesthetic config's 200 epochs
    config.use_lora = True   # Use LoRA for memory efficiency (recommended for any GPU count)
    config.save_freq = 5     # Save a checkpoint every 5 epochs

    # Sampling configuration
    # Original used 8 GPUs with batch_size=8, so total 64 images per batch
    # Adjust batch_size based on your GPU memory (Lower for <24GB VRAM):
    # - 8+ for A100/H100 (40GB+)
    # - 4-6 for RTX 3090/4090 (24GB)
    # - 2-3 for RTX 3080/4080 (10-16GB)
    # - 1 for smaller GPUs
    config.sample.batch_size = 8  # Per GPU, adjust down if you have less GPU memory
    
    # Original used 4 batches with 8 GPUs for 256 samples per epoch total
    # If reducing batch_size, consider increasing this to maintain similar total samples
    # For N GPUs: num_batches_per_epoch = 32/N (to get ~256 samples)
    config.sample.num_batches_per_epoch = 4  # Adjust up if using fewer GPUs

    # Training configuration
    # Original used batch_size=4 per GPU for training
    # For GPUs with less memory:
    # - 2-4 for A100/H100/RTX 3090
    # - 1-2 for RTX 3080 and smaller
    config.train.batch_size = 4  # Per GPU, adjust based on your GPU memory
    
    # Original used 4 gradient accumulation steps (important for harder rewards)
    # If using fewer GPUs, increase this to maintain effective batch size
    # For N GPUs with B train.batch_size: gradient_accumulation_steps = 256/(N*B)
    config.train.gradient_accumulation_steps = 4  # Keep at 4 as in aesthetic config

    # Prompting
    config.prompt_fn = "styled_animals"  # function that adds to prompt
    config.prompt_fn_kwargs = {}

    # Use clip_score reward instead of aesthetic_score
    config.reward_fn = "clip_compressibility_combined"

    # Statistics tracking for more stable training
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,  # Same as aesthetic config
        "min_count": 16,    # Same as aesthetic config
    }

    return config
