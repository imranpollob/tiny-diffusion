"""
Custom Dataset Loader for Anime Face Images
Loads images from a folder for training the diffusion model
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob


class AnimeFaceDataset(Dataset):
    """
    Dataset loader for anime face images
    Loads all images from a folder and applies transformations
    """

    def __init__(self, root_dir, transform=None, image_size=64):
        """
        Args:
            root_dir: Path to folder containing anime face images
            transform: Optional transforms to apply
            image_size: Size to resize images to (will be square)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size

        # Find all image files
        self.image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
            self.image_files.extend(glob.glob(os.path.join(root_dir, ext)))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {root_dir}. Please check the path.")

        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Load and return an image

        Returns:
            image: Transformed image tensor
            label: Dummy label (0) since we don't use labels for diffusion
        """
        img_path = self.image_files[idx]

        try:
            # Load image and convert to RGB
            image = Image.open(img_path).convert("RGB")

            # Apply transformations if provided
            if self.transform:
                image = self.transform(image)

            # Return image with dummy label (diffusion doesn't need labels)
            return image, 0

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank, 0


def get_anime_dataloader(
    data_path, batch_size=32, image_size=64, num_workers=2, shuffle=True
):
    """
    Create a DataLoader for anime face images

    Args:
        data_path: Path to folder containing images
        batch_size: Number of images per batch
        image_size: Size to resize images to
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data

    Returns:
        DataLoader object
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),  # Resize to target size
            transforms.CenterCrop(image_size),  # Center crop to ensure square
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.ToTensor(),  # Convert to tensor [0, 1]
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # Normalize to [-1, 1]
            ),
        ]
    )

    # Create dataset
    dataset = AnimeFaceDataset(
        root_dir=data_path, transform=transform, image_size=image_size
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch
    )

    return dataloader, dataset
