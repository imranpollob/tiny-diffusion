# Tiny Diffusion

A minimal implementation of Denoising Diffusion Probabilistic Models (DDPM) for learning purposes. This project implements the complete diffusion process from scratch, including forward diffusion (adding noise) and reverse diffusion (denoising to generate images).

## 📚 What You'll Learn

- **Forward Diffusion**: How images gradually become noise over time
- **Reverse Diffusion**: How a neural network learns to denoise and generate images
- **Noise Schedules**: Linear and cosine variance schedules
- **U-Net Architecture**: Time-conditioned U-Net for noise prediction
- **Training Process**: How to train a diffusion model with MSE loss
- **Sampling**: Iterative denoising to generate new images from pure noise

## 🛠️ Setup

```bash
# Clone or navigate to the project
cd tiny-diffusion

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
tiny-diffusion/
├── diffusion.py              # Core diffusion logic (forward & reverse)
├── models/
│   └── unet.py              # U-Net architecture with time embeddings
├── train.py                 # Training script
├── sample.py                # Image generation script
├── visualize_diffusion.py   # Visualize forward diffusion process
├── utils.py                 # Helper functions
├── data/                    # Downloaded datasets (auto-created)
├── outputs/                 # Generated images and checkpoints
│   ├── samples/            # Training samples
│   ├── checkpoints/        # Model checkpoints
│   └── intermediate/       # Denoising process visualization
└── requirements.txt         # Python dependencies
```

## 🚀 Usage

### Step 1: Visualize Forward Diffusion

First, understand how the forward diffusion process works by visualizing how clean images gradually become noise:

```bash
python visualize_diffusion.py --dataset mnist --num_images 6
```

This creates a visualization showing images at different noise levels (t=0, 50, 100, 200, 400, 600, 800, 999).

**Options:**
- `--dataset`: Choose `mnist` or `fashion_mnist`
- `--num_images`: Number of sample images to show (default: 8)
- `--save_path`: Where to save the visualization

### Step 2: Train the Model

Train a U-Net model to predict and remove noise:

```bash
# Quick training (20 epochs, ~5-10 minutes on CPU)
python train.py --dataset mnist --epochs 20 --batch_size 128

# Longer training for better results (50 epochs)
python train.py --dataset mnist --epochs 50 --batch_size 128
```

**Options:**
- `--dataset`: Dataset to use (`mnist` or `fashion_mnist`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 2e-4)
- `--timesteps`: Number of diffusion steps (default: 1000)
- `--sample_interval`: Generate samples every N epochs (default: 1)
- `--save_interval`: Save checkpoint every N epochs (default: 5)
- `--output_dir`: Output directory (default: outputs)

**What happens during training:**
- The model learns to predict noise at various timesteps
- Sample images are generated every epoch to track progress
- Model checkpoints are saved periodically
- Training loss is plotted and saved

### Step 3: Generate Images

Generate new images using your trained model:

```bash
# Generate 64 images
python sample.py --checkpoint outputs/model_final.pt --num_images 64

# Generate with intermediate steps visualization
python sample.py --checkpoint outputs/model_final.pt --num_images 64 --show_intermediate
```

**Options:**
- `--checkpoint`: Path to trained model checkpoint (required)
- `--num_images`: Number of images to generate (default: 64)
- `--output`: Output path for generated images
- `--show_intermediate`: Save intermediate denoising steps
- `--timesteps`: Number of diffusion timesteps (default: 1000)

## 🧪 Example Workflow

```bash
# 1. Visualize the forward diffusion process
python visualize_diffusion.py

# 2. Train the model
python train.py --epochs 20

# 3. Generate images from the trained model
python sample.py --checkpoint outputs/model_final.pt --show_intermediate
```

## 📊 Understanding the Output

### During Training
- `outputs/samples/epoch_XXX.png`: Generated samples after each epoch
- `outputs/checkpoints/model_epoch_XXX.pt`: Model checkpoints
- `outputs/training_loss.png`: Loss curve over training
- `outputs/model_final.pt`: Final trained model

### After Generation
- `outputs/generated_samples.png`: Final generated images
- `outputs/intermediate/step_XXXX.png`: Denoising process at different timesteps (if `--show_intermediate` is used)

## 🔍 Key Concepts

### Forward Diffusion
The forward process gradually adds Gaussian noise to images:
```
x_t = √(α_t) * x_0 + √(1 - α_t) * ε
```
where `x_0` is the clean image, `ε` is random noise, and `α_t` controls the noise level.

### Reverse Diffusion
The reverse process learns to remove noise step by step:
- Model predicts the noise at each timestep
- Noise is subtracted to recover a slightly cleaner image
- Process repeats for 1000 steps (from pure noise to clean image)

### U-Net Architecture
- **Encoder**: Downsamples the image to extract features
- **Time Embedding**: Sinusoidal embeddings tell the model which timestep it's at
- **Decoder**: Upsamples to reconstruct the image
- **Skip Connections**: Preserve spatial information

## 💡 Tips for Better Results

1. **Training Duration**: 20 epochs is enough to see results, but 50+ epochs gives much better quality
2. **Batch Size**: Larger batch sizes (128-256) train faster and more stably
3. **Dataset Choice**: MNIST is simpler and trains faster; Fashion-MNIST is more challenging
4. **Monitoring**: Check `outputs/samples/` during training to see progress
5. **GPU**: If available, training will be much faster (the code automatically uses CUDA/MPS if available)

## 🎓 Learning Resources

This implementation is based on:
- **Paper**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- **Key Idea**: Train a model to reverse a gradual noising process

## 🔧 Model Architecture

- **U-Net**: ~2M parameters
- **Input/Output**: 28×28 grayscale images
- **Time Embedding**: 256 dimensions
- **Base Channels**: 64 (configurable)

## ⚡ Performance

- **Training**: ~5-10 minutes for 20 epochs on CPU, <2 minutes on GPU
- **Generation**: ~30 seconds for 64 images on CPU, ~5 seconds on GPU
- **Memory**: ~2GB RAM for training with batch size 128

## 🤝 Contributing

This is a learning project! Feel free to experiment with:
- Different network architectures
- Various noise schedules (cosine, learned)
- Conditional generation (class-conditional)
- Different datasets (CIFAR-10, custom images)
- Faster sampling methods (DDIM)

## 📝 License

This project is for educational purposes. Feel free to use and modify!
