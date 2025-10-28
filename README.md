# Tiny Diffusion

A compact, educational implementation of Denoising Diffusion Probabilistic Models (DDPM). This repo implements the forward (noising) and reverse (denoising) diffusion process and provides scripts to visualize, train, and sample images (MNIST / Fashion-MNIST).

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

See the Options & Examples sections below for full command flags and variations.

---

## Project structure

```
tiny-diffusion/
├── diffusion.py              # Core diffusion logic (forward & reverse)
├── models/
│   └── unet.py               # U-Net architecture with time embeddings
├── train.py                  # Training script
├── sample.py                 # Image generation script
├── visualize_diffusion.py    # Visualize forward diffusion process
├── demo.py                   # Interactive demo script (recommended)
├── utils.py                  # Helper functions
├── data/                     # Downloaded datasets (auto-created)
├── outputs/                  # Generated images and checkpoints
│   ├── samples/              # Training samples
│   ├── checkpoints/          # Model checkpoints
│   └── intermediate/         # Denoising process visualization
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
- `--dataset`: `mnist` or `fashion_mnist`
- `--epochs` (default 20)
- `--batch_size` (default 128)
- `--lr` (default 2e-4)
- `--timesteps` (default 1000)
- `--output_dir` (default `outputs`)

Step 3 — Sample / generate images:

```bash
python sample.py --checkpoint outputs/model_final.pt --num_images 64 --show_intermediate
```

Options:
- `--checkpoint`: path to checkpoint (required)
- `--num_images`: number of images (default 64)
- `--show_intermediate`: save intermediate denoising steps

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
python train.py --dataset fashion_mnist --epochs 20 --output_dir outputs/fashion
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

