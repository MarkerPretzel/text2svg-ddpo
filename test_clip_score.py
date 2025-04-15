import torch
from PIL import Image
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import ddpo_pytorch
sys.path.append(os.path.abspath('.'))

# Import the clip_score function
from ddpo_pytorch.rewards import clip_score

def main():
    # Create a simple test image or load one
    try:
        # Try to load an existing image if available
        test_image = Image.open("test_image.jpg")
        print("Loaded existing test image")
    except FileNotFoundError:
        # Create a simple colored square as a test image
        print("Creating test image")
        test_image = Image.new('RGB', (512, 512), color='blue')
        test_image.save("test_image.jpg")

    # Convert to numpy array
    test_image_np = np.array(test_image)

    # Add batch dimension (1, H, W, C)
    test_image_batch = np.expand_dims(test_image_np, axis=0)

    # Create a few test prompts
    test_prompts = ["a blue square", "a red circle", "a beautiful landscape photo", "a cow", "a frog with a fedora"]
    
    print(f"Number of test prompts: {len(test_prompts)}")
    print(f"Test prompts: {test_prompts}")

    # Get the reward function
    reward_fn = clip_score()

    # Process each prompt individually to debug
    print("\nTesting each prompt individually:")
    for i, prompt in enumerate(test_prompts):
        single_rewards, _ = reward_fn(test_image_batch, [prompt], {})
        
        # Print debugging info about single_rewards
        print(f"Type: {type(single_rewards)}, Shape: {single_rewards.shape if hasattr(single_rewards, 'shape') else 'N/A'}")
        
        # More robust printing that handles various array shapes
        if isinstance(single_rewards, np.ndarray):
            if single_rewards.size == 1:  # Single value
                score = float(single_rewards.flatten()[0])
                print(f"[{i+1}] Prompt: '{prompt}', CLIP Score: {score:.4f}")
            else:  # Multiple values
                print(f"[{i+1}] Prompt: '{prompt}', CLIP Scores: {single_rewards}")
        else:
            print(f"[{i+1}] Prompt: '{prompt}', CLIP Score: {single_rewards}")
    
    # Now try with all prompts together
    print("\nTesting all prompts together:")
    rewards, metadata = reward_fn(test_image_batch, test_prompts, {})
    
    # Debug information
    print(f"Type of rewards: {type(rewards)}")
    print(f"Shape or length of rewards: {rewards.shape if hasattr(rewards, 'shape') else len(rewards)}")
    print(f"Rewards data: {rewards}")
    
    # Print results for all prompts more robustly
    print("\nResults for all prompts:")
    if len(rewards) == len(test_prompts):
        # If we have one reward per prompt
        for i, (prompt, reward) in enumerate(zip(test_prompts, rewards)):
            print(f"[{i+1}] Prompt: '{prompt}', CLIP Score: {float(reward):.4f}")
    else:
        # Otherwise just print what we have
        print(f"Rewards: {rewards}")
        print(f"Note: Number of rewards ({len(rewards)}) doesn't match number of prompts ({len(test_prompts)})")

if __name__ == "__main__":
    main()
