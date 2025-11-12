"""
U-Net Model for Diffusion
A simplified U-Net architecture with time embeddings for noise prediction
"""

import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding
    Encodes timestep information into a high-dimensional vector
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: Timesteps [batch_size]
        Returns:
            Time embeddings [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class DownBlock(nn.Module):
    """Downsampling block in U-Net"""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.act = nn.SiLU()

    def forward(self, x, t):
        """
        Args:
            x: Input features [batch, in_channels, height, width]
            t: Time embeddings [batch, time_emb_dim]
        Returns:
            Output features [batch, out_channels, height, width]
        """
        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Add time embedding
        time_emb = self.act(self.time_mlp(t))
        h = h + time_emb[:, :, None, None]

        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h


class UpBlock(nn.Module):
    """Upsampling block in U-Net"""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.act = nn.SiLU()

    def forward(self, x, skip, t):
        """
        Args:
            x: Input features [batch, in_channels, height, width]
            skip: Skip connection from encoder [batch, skip_channels, height, width]
            t: Time embeddings [batch, time_emb_dim]
        Returns:
            Output features [batch, out_channels, height, width]
        """
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Add time embedding
        time_emb = self.act(self.time_mlp(t))
        h = h + time_emb[:, :, None, None]

        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for noise prediction in diffusion models
    Designed for 28x28 images (MNIST, Fashion-MNIST)
    """

    def __init__(
        self, in_channels=1, out_channels=1, base_channels=64, time_emb_dim=256
    ):
        super().__init__()

        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim)

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder (downsampling path)
        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)  # 28x28 -> 14x14

        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)  # 14x14 -> 7x7

        # Bottleneck
        self.bottleneck = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        # Decoder (upsampling path)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 2, stride=2
        )  # 7x7 -> 14x14
        self.up_block1 = UpBlock(base_channels * 4, base_channels * 2, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 2, stride=2
        )  # 14x14 -> 28x28
        self.up_block2 = UpBlock(base_channels * 2, base_channels, time_emb_dim)

        # Output convolution
        self.conv_out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):
        """
        Predict noise in the input

        Args:
            x: Noisy images [batch, in_channels, height, width]
            t: Timesteps [batch]
        Returns:
            Predicted noise [batch, out_channels, height, width]
        """
        # Get time embeddings
        t_emb = self.time_embed(t)

        # Initial convolution
        x = self.conv_in(x)

        # Encoder
        skip1 = self.down1(x, t_emb)
        x = self.pool1(skip1)

        skip2 = self.down2(x, t_emb)
        x = self.pool2(skip2)

        # Bottleneck
        x = self.bottleneck(x, t_emb)

        # Decoder
        x = self.up1(x)
        x = self.up_block1(x, skip2, t_emb)

        x = self.up2(x)
        x = self.up_block2(x, skip1, t_emb)

        # Output
        x = self.conv_out(x)

        return x


def create_model(image_size=28, in_channels=1, base_channels=64):
    """
    Factory function to create a U-Net model

    Args:
        image_size: Size of input images (28, 32, 64, 128, etc.)
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        base_channels: Base number of channels (controls model size)

    Returns:
        U-Net model
    """
    # U-Net works with any power-of-2 image size >= 28
    # The architecture handles different sizes automatically
    model = SimpleUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        time_emb_dim=256,
    )

    return model


if __name__ == "__main__":
    # Test the model
    model = create_model()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== Model Architecture ===")
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    t = torch.randint(0, 1000, (batch_size,))

    output = model(x, t)
    print(f"\nInput shape: {x.shape}")
    print(f"Timesteps shape: {t.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✓ Model test passed!")
