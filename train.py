"""
Training Script for Diffusion Model
Train a U-Net to predict noise in images
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import argparse

from diffusion import DiffusionProcess, get_timesteps
from models.unet import create_model
from utils import save_images, plot_losses, count_parameters, get_device
from anime_dataset import get_anime_dataloader


def train_epoch(model, dataloader, optimizer, diffusion, device):
    """
    Train for one epoch

    Args:
        model: U-Net model
        dataloader: Training data loader
        optimizer: Optimizer
        diffusion: Diffusion process
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0

    for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        batch_size = images.shape[0]

        # Sample random timesteps
        t = get_timesteps(batch_size, diffusion.num_timesteps, device)

        # Add noise to images (forward diffusion)
        noisy_images, noise = diffusion.forward_diffusion(images, t)

        # Predict noise using the model
        predicted_noise = model(noisy_images, t)

        # Calculate loss (MSE between predicted and actual noise)
        loss = nn.functional.mse_loss(predicted_noise, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def sample_images(model, diffusion, device, num_images=64, image_size=28, channels=1):
    """
    Generate sample images during training

    Args:
        model: Trained model
        diffusion: Diffusion process
        device: Device
        num_images: Number of images to generate
        image_size: Size of images to generate
        channels: Number of channels (1 or 3)

    Returns:
        Generated images
    """
    model.eval()
    samples = diffusion.sample(
        model, image_size=image_size, batch_size=num_images, channels=channels
    )
    return samples


def train(
    dataset_name="mnist",
    batch_size=128,
    num_epochs=20,
    learning_rate=2e-4,
    num_timesteps=1000,
    save_interval=5,
    sample_interval=1,
    output_dir="outputs",
    min_size=64,
):
    """
    Main training function

    Args:
        dataset_name: "mnist", "fashion_mnist", or "anime"
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        num_timesteps: Number of diffusion timesteps
        save_interval: Save checkpoint every N epochs
        sample_interval: Generate samples every N epochs
        output_dir: Directory to save outputs
        min_size: Minimum image size for anime dataset (filters smaller images)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading {dataset_name} dataset...")

    # Determine image size and channels based on dataset
    if dataset_name == "anime":
        # Anime faces: RGB images, typically 64x64 or larger
        image_size = 64  # Can be changed to 32, 128, etc.
        in_channels = 3  # RGB

        # Check if data path exists
        anime_data_path = "./data/anime_faces"
        if not os.path.exists(anime_data_path):
            print(f"\n⚠️  ERROR: Anime dataset not found at {anime_data_path}")
            print("\nTo use the anime dataset:")
            print(
                "1. Download from: https://www.kaggle.com/datasets/splcher/animefacedataset"
            )
            print("2. Extract images to: ./data/anime_faces/")
            print("3. Make sure images are directly in the folder (not in subfolders)")
            raise FileNotFoundError(f"Anime dataset not found at {anime_data_path}")

        dataloader, train_dataset = get_anime_dataloader(
            data_path=anime_data_path,
            batch_size=batch_size,
            image_size=image_size,
            min_size=min_size,  # Filter out images smaller than this threshold
            num_workers=2,
            shuffle=True,
        )
    elif dataset_name == "mnist":
        # MNIST: grayscale, 28x28
        image_size = 28
        in_channels = 1

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ]
        )

        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
    elif dataset_name == "fashion_mnist":
        # Fashion-MNIST: grayscale, 28x28
        image_size = 28
        in_channels = 1

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ]
        )

        train_dataset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )

        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Use 'mnist', 'fashion_mnist', or 'anime'"
        )

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Channels: {in_channels} ({'RGB' if in_channels == 3 else 'Grayscale'})")

    # Create model
    print("\nCreating model...")
    model = create_model(
        image_size=image_size, in_channels=in_channels, base_channels=64
    )
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params:,}")

    # Create diffusion process
    diffusion = DiffusionProcess(
        num_timesteps=num_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="linear",
        device=device,
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\nStarting training...")
    losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")

        # Train for one epoch
        avg_loss = train_epoch(model, dataloader, optimizer, diffusion, device)
        losses.append(avg_loss)

        print(f"Average Loss: {avg_loss:.6f}")

        # Generate samples
        if epoch % sample_interval == 0:
            print("Generating samples...")
            samples = sample_images(
                model,
                diffusion,
                device,
                num_images=64,
                image_size=image_size,
                channels=in_channels,
            )
            save_images(samples, f"{output_dir}/samples/epoch_{epoch:03d}.png", nrow=8)

        # Save checkpoint
        if epoch % save_interval == 0:
            checkpoint_path = f"{output_dir}/checkpoints/model_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    final_model_path = f"{output_dir}/model_final.pt"
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": losses[-1],
        },
        final_model_path,
    )
    print(f"\nFinal model saved to {final_model_path}")

    # Plot losses
    plot_losses(losses, f"{output_dir}/training_loss.png")
    print(f"Loss plot saved to {output_dir}/training_loss.png")

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "anime"],
        help="Dataset to train on (mnist, fashion_mnist, or anime)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training (use 32 for anime with limited VRAM)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (50+ recommended for anime)",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--save_interval", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--sample_interval", type=int, default=1, help="Generate samples every N epochs"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=64,
        help="Minimum image size for anime dataset - filters out smaller images (default: 64)",
    )

    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_timesteps=args.timesteps,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        output_dir=args.output_dir,
        min_size=args.min_size,
    )
