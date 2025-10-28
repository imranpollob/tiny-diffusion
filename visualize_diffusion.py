"""
Visualize Forward Diffusion Process
Shows how images gradually become noise over time
"""

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from diffusion import DiffusionProcess


def visualize_forward_diffusion(
    num_images=8,
    timesteps_to_show=[0, 50, 100, 200, 400, 600, 800, 999],
    dataset="mnist",
    save_path="outputs/forward_diffusion.png",
):
    """
    Visualize the forward diffusion process

    Args:
        num_images: Number of different images to show
        timesteps_to_show: Which timesteps to visualize
        dataset: Dataset to use ("mnist" or "fashion_mnist")
        save_path: Where to save the visualization
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize diffusion process
    diffusion = DiffusionProcess(num_timesteps=1000, device=device)

    # Load dataset
    print(f"Loading {dataset} dataset...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ]
    )

    if dataset == "mnist":
        dataset_obj = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset == "fashion_mnist":
        dataset_obj = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Get random images
    indices = np.random.choice(len(dataset_obj), num_images, replace=False)
    images = torch.stack([dataset_obj[i][0] for i in indices]).to(device)

    # Create visualization
    fig, axes = plt.subplots(
        num_images, len(timesteps_to_show), figsize=(20, 2.5 * num_images)
    )

    print("Generating noisy images...")
    for i, img in enumerate(images):
        for j, t in enumerate(timesteps_to_show):
            # Add noise at timestep t
            t_tensor = torch.tensor([t], device=device)
            noisy_img, _ = diffusion.forward_diffusion(img.unsqueeze(0), t_tensor)

            # Convert to numpy for plotting
            noisy_img = noisy_img.squeeze().cpu().numpy()

            # Plot
            ax = axes[i, j] if num_images > 1 else axes[j]
            ax.imshow(noisy_img, cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")

            # Add title on first row
            if i == 0:
                ax.set_title(f"t = {t}", fontsize=12)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to {save_path}")
    plt.show()

    # Print some statistics
    print("\n=== Forward Diffusion Statistics ===")
    print(f"Number of timesteps: {diffusion.num_timesteps}")
    print(f"Beta start: {diffusion.betas[0]:.6f}")
    print(f"Beta end: {diffusion.betas[-1]:.6f}")
    print(f"Alpha cumprod at t=0: {diffusion.alphas_cumprod[0]:.6f}")
    print(f"Alpha cumprod at t=999: {diffusion.alphas_cumprod[-1]:.6f}")

    # Show noise levels at different timesteps
    print("\n=== Signal-to-Noise Ratio at Different Timesteps ===")
    for t in timesteps_to_show:
        signal = diffusion.sqrt_alphas_cumprod[t].item()
        noise = diffusion.sqrt_one_minus_alphas_cumprod[t].item()
        print(f"t={t:3d}: signal weight={signal:.4f}, noise weight={noise:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize forward diffusion process")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--num_images", type=int, default=8, help="Number of images to show"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/forward_diffusion.png",
        help="Where to save visualization",
    )

    args = parser.parse_args()

    visualize_forward_diffusion(
        num_images=args.num_images, dataset=args.dataset, save_path=args.save_path
    )
