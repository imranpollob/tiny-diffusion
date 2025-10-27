"""
Sampling Script for Diffusion Model
Generate images using a trained diffusion model
"""

import torch
import argparse
import os
from tqdm import tqdm

from diffusion import DiffusionProcess
from models.unet import create_model
from utils import save_images, get_device


@torch.no_grad()
def sample_images_with_progress(
    model,
    diffusion,
    device,
    num_images=64,
    show_progress=True
):
    """
    Generate images with progress visualization
    
    Args:
        model: Trained model
        diffusion: Diffusion process
        device: Device
        num_images: Number of images to generate
        show_progress: Whether to show progress bar
        
    Returns:
        Generated images
    """
    model.eval()
    
    # Start from pure noise
    x_t = torch.randn(num_images, 1, 28, 28).to(device)
    
    # Save intermediate steps for visualization
    intermediate_steps = []
    steps_to_save = [999, 800, 600, 400, 200, 100, 50, 0]
    
    # Iteratively denoise
    iterator = reversed(range(diffusion.num_timesteps))
    if show_progress:
        iterator = tqdm(list(iterator), desc="Generating images")
    
    for i in iterator:
        t = torch.full((num_images,), i, dtype=torch.long, device=device)
        x_t = diffusion.reverse_diffusion_step(model, x_t, t, i)
        
        # Save intermediate steps
        if i in steps_to_save:
            intermediate_steps.append((i, x_t.clone()))
    
    return x_t, intermediate_steps


def generate(
    checkpoint_path,
    num_images=64,
    output_path="outputs/generated_samples.png",
    show_intermediate=False,
    num_timesteps=1000
):
    """
    Generate images from a trained model
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_images: Number of images to generate
        output_path: Where to save generated images
        show_intermediate: Whether to save intermediate denoising steps
        num_timesteps: Number of diffusion timesteps (should match training)
    """
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = create_model(image_size=28, in_channels=1, base_channels=64)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if 'epoch' in checkpoint:
        print(f"Model trained for {checkpoint['epoch']} epochs")
    if 'loss' in checkpoint:
        print(f"Final training loss: {checkpoint['loss']:.6f}")
    
    # Create diffusion process
    diffusion = DiffusionProcess(
        num_timesteps=num_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="linear",
        device=device
    )
    
    # Generate images
    print(f"\nGenerating {num_images} images...")
    samples, intermediate_steps = sample_images_with_progress(
        model, diffusion, device, num_images, show_progress=True
    )
    
    # Save generated images
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_images(samples, output_path, nrow=8)
    print(f"Generated images saved to {output_path}")
    
    # Save intermediate steps if requested
    if show_intermediate and intermediate_steps:
        print("\nSaving intermediate denoising steps...")
        intermediate_dir = os.path.join(os.path.dirname(output_path), "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        for step, images in intermediate_steps:
            step_path = os.path.join(intermediate_dir, f"step_{step:04d}.png")
            save_images(images, step_path, nrow=8)
            print(f"  Saved timestep {step} to {step_path}")
    
    print("\nGeneration completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using trained diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--num_images", type=int, default=64,
                       help="Number of images to generate")
    parser.add_argument("--output", type=str, default="outputs/generated_samples.png",
                       help="Output path for generated images")
    parser.add_argument("--show_intermediate", action="store_true",
                       help="Save intermediate denoising steps")
    parser.add_argument("--timesteps", type=int, default=1000,
                       help="Number of diffusion timesteps")
    
    args = parser.parse_args()
    
    generate(
        checkpoint_path=args.checkpoint,
        num_images=args.num_images,
        output_path=args.output,
        show_intermediate=args.show_intermediate,
        num_timesteps=args.timesteps
    )
