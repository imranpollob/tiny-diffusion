"""
Diffusion Process Implementation
Implements forward and reverse diffusion for image generation
"""

import torch
import torch.nn as nn
import numpy as np


class DiffusionProcess:
    """
    Implements DDPM (Denoising Diffusion Probabilistic Models)
    
    Forward process: gradually add Gaussian noise to images
    Reverse process: learn to denoise and generate images
    """
    
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="linear",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            num_timesteps: Number of diffusion steps (T)
            beta_start: Starting variance for noise schedule
            beta_end: Ending variance for noise schedule
            schedule_type: "linear" or "cosine" schedule
            device: Device to run on
        """
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Create noise schedule (beta_t)
        if schedule_type == "linear":
            self.betas = self._linear_schedule(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute useful values for diffusion
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Move to device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
    
    def _linear_schedule(self, beta_start, beta_end, num_timesteps):
        """Linear variance schedule"""
        return torch.linspace(beta_start, beta_end, num_timesteps)
    
    def _cosine_schedule(self, num_timesteps, s=0.008):
        """
        Cosine variance schedule as proposed in https://arxiv.org/abs/2102.09672
        More stable for longer diffusion processes
        """
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward_diffusion(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to clean images according to timestep t
        
        Uses the reparameterization trick:
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
        
        Args:
            x_0: Clean images [batch_size, channels, height, width]
            t: Timesteps [batch_size]
            noise: Optional pre-generated noise
            
        Returns:
            x_t: Noisy images at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod for timestep t
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        # Apply noise according to the formula
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise
    
    def _extract(self, a, t, x_shape):
        """
        Extract values from a at timestep t and reshape for broadcasting
        
        Args:
            a: 1D tensor of values
            t: Timesteps [batch_size]
            x_shape: Shape to broadcast to
            
        Returns:
            Extracted values reshaped for broadcasting
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    @torch.no_grad()
    def reverse_diffusion_step(self, model, x_t, t, t_index):
        """
        Single step of reverse diffusion: p(x_{t-1} | x_t)
        Uses the trained model to predict noise and denoise
        
        Args:
            model: Trained denoising model
            x_t: Noisy image at timestep t
            t: Current timestep
            t_index: Index of current timestep (for indexing arrays)
            
        Returns:
            x_{t-1}: Slightly less noisy image
        """
        # Get batch size
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self._extract(self.alphas, t, x_t.shape))
        
        # Predict noise using the model
        predicted_noise = model(x_t, t)
        
        # Use predicted noise to compute mean of p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            # No noise at the last step
            return model_mean
        else:
            # Add noise
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=1, channels=1):
        """
        Generate images by running the full reverse diffusion process
        Starting from pure noise, iteratively denoise to generate images
        
        Args:
            model: Trained denoising model
            image_size: Size of images to generate (height, width)
            batch_size: Number of images to generate
            channels: Number of image channels (1 for grayscale, 3 for RGB)
            
        Returns:
            Generated images
        """
        model.eval()
        
        # Start from pure noise
        x_t = torch.randn(batch_size, channels, image_size, image_size).to(self.device)
        
        # Iteratively denoise
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=self.device)
            x_t = self.reverse_diffusion_step(model, x_t, t, i)
        
        return x_t


def get_timesteps(batch_size, num_timesteps, device):
    """
    Sample random timesteps for training
    
    Args:
        batch_size: Number of timesteps to sample
        num_timesteps: Maximum timestep value (T)
        device: Device to create tensor on
        
    Returns:
        Random timesteps [batch_size]
    """
    return torch.randint(0, num_timesteps, (batch_size,), device=device)
