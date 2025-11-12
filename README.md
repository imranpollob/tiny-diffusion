# Tiny Diffusion

A compact, educational implementation of Denoising Diffusion Probabilistic Models (DDPM). This repo implements the forward (noising) and reverse (denoising) diffusion process and provides scripts to visualize, train, and sample images.

**Supports:** MNIST, Fashion-MNIST, and **Anime Faces** (RGB images)

**Detailed explanation:** [SUMMARY.md](SUMMARY.md) | **Anime setup guide:** [ANIME_SETUP.md](ANIME_SETUP.md)

## Table of contents
- Quick start
- Project structure
- Usage (demo & manual)
- Key concepts (forward / reverse / U-Net)
- Experiments and tips
- Troubleshooting
- Contributing & license

---

## Quick start

### Option A: MNIST/Fashion-MNIST (Grayscale Digits)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Try the interactive demo (recommended):

```bash
python demo.py
```

3. Or run the three main steps manually:

```bash
python visualize_diffusion.py          # see forward diffusion
python train.py --epochs 20           # train model (quick)
python sample.py --checkpoint outputs/model_final.pt --show_intermediate
```

### Option B: Anime Faces (RGB Images) 🎨

**You have 63,565 anime face images ready!**

1. Train the model (recommended settings for your hardware):

```bash
# Quick test (20 epochs, ~30-40 minutes)
python train.py --dataset anime --epochs 20 --batch_size 32

# Recommended quality (50 epochs, ~1.5-2 hours)
python train.py --dataset anime --epochs 50 --batch_size 32

# Best quality (100 epochs, ~3-4 hours)
python train.py --dataset anime --epochs 100 --batch_size 32
```

2. Generate anime faces:

```bash
python sample.py --checkpoint outputs/model_final.pt --num_images 64
```

**See [ANIME_SETUP.md](ANIME_SETUP.md) for detailed instructions and tips.**

---

## Project structure

```
tiny-diffusion/
├── diffusion.py              # Core diffusion logic (forward & reverse)
├── models/
│   └── unet.py               # U-Net architecture with time embeddings
├── train.py                  # Training script (supports MNIST, Fashion-MNIST, Anime)
├── sample.py                 # Image generation script
├── visualize_diffusion.py    # Visualize forward diffusion process
├── demo.py                   # Interactive demo script (recommended)
├── anime_dataset.py          # Custom dataset loader for anime faces
├── utils.py                  # Helper functions
├── data/                     # Datasets
│   ├── MNIST/                # Auto-downloaded MNIST
│   └── anime_faces/          # Your anime face images (63,565 images) ✅
├── outputs/                  # Generated images and checkpoints
│   ├── samples/              # Training samples
│   ├── checkpoints/          # Model checkpoints
│   └── intermediate/         # Denoising process visualization
├── SUMMARY.md                # Comprehensive beginner's guide
└── ANIME_SETUP.md            # Anime faces setup guide
└── requirements.txt          # Python dependencies
```

---

## Usage

### Interactive demo (beginner-friendly)

```bash
python demo.py
```

The demo walks you through the pipeline with explanations and optional skips.

### Manual execution (more control)

Step 1 — Visualize forward diffusion:

```bash
python visualize_diffusion.py --dataset mnist --num_images 8
```

Step 2 — Train the model:

```bash
python train.py --dataset mnist --epochs 20 --batch_size 128
```

Useful options:
- `--dataset`: `mnist`, `fashion_mnist`, or `anime`
- `--epochs` (default 20)
- `--batch_size` (default 128; use 32-64 for anime with 8GB VRAM)
- `--lr` (default 2e-4)
- `--timesteps` (default 1000)
- `--output_dir` (default `outputs`)

Step 3 — Sample / generate images:

```bash
# For MNIST/Fashion-MNIST (28x28 grayscale)
python sample.py --checkpoint outputs/model_final.pt --num_images 64 --show_intermediate

# For anime faces (64x64 RGB) - auto-detected from checkpoint
python sample.py --checkpoint outputs/anime_model_final.pt --num_images 64 --show_intermediate
```

Options:
- `--checkpoint`: path to checkpoint (required)
- `--num_images`: number of images (default 64)
- `--show_intermediate`: save intermediate denoising steps
- `--image_size`: image dimensions (auto-detected from checkpoint)
- `--channels`: 1 for grayscale, 3 for RGB (auto-detected from checkpoint)

---

## Key concepts

### Forward diffusion
The forward process gradually adds Gaussian noise to an image:

$$x_t = \\sqrt{\\alpha_t} x_0 + \\sqrt{1 - \\alpha_t} \\epsilon$$

where `x_0` is the clean image and `\\epsilon` is random noise.

### Reverse diffusion
The model learns to predict the noise at each timestep so we can iteratively remove it and recover an image. Sampling typically runs for 1000 steps (configurable).

### U-Net architecture

- Encoder/decoder with skip connections
- Time embeddings (sinusoidal) to condition the network on timestep `t`
- Predicts noise rather than the clean image directly
- Supports both grayscale (1 channel) and RGB (3 channels) images
- Flexible architecture works with various image sizes (28×28 for MNIST, 64×64 for anime faces)

---

## Experiments & tips

Try these small experiments to see effects on quality and speed:

1. Training length:

```bash
# Short
python train.py --epochs 10 --output_dir outputs/exp_10

# Longer
python train.py --epochs 50 --output_dir outputs/exp_50
```

2. Different datasets:

```bash
# Fashion-MNIST (28x28 grayscale)
python train.py --dataset fashion_mnist --epochs 20 --output_dir outputs/fashion

# Anime faces (64x64 RGB) - requires dataset in data/anime_faces/
python train.py --dataset anime --epochs 50 --batch_size 32 --output_dir outputs/anime
```

3. Model sizes (edit `train.py` / model params): try base_channels=32 / 64 / 128

Tips:
- 20 epochs shows basic structure; 50+ epochs improves quality
- Larger batch sizes (128-256) stabilize training
- Use GPU if available for much faster runs

---

## Troubleshooting

Out of memory:
- Reduce `--batch_size`
- Reduce model size (base channels)

Too slow:
- Decrease timesteps (e.g., 500)
- Use smaller model

Blurry images:
- Train longer
- Lower learning rate and fine-tune

---

## Outputs

- `outputs/samples/epoch_XXX.png` — samples during training
- `outputs/checkpoints/model_epoch_XXX.pt` — checkpoints
- `outputs/training_loss.png` — loss curve
- `outputs/intermediate/` — intermediate denoising images (if requested)

---

## Contributing

This repo is for learning. Welcome contributions that are educational and low-risk: clearer docs, extra visualization, small performance improvements, or experimental scripts.

---

## License

Use and modify freely for educational purposes.

