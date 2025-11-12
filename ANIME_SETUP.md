# Training Diffusion Model on Anime Faces

This guide will walk you through setting up and training your diffusion model to generate anime faces using the Anime Face Dataset from Kaggle.

## 📋 Prerequisites

- 16GB RAM, 8GB VRAM (your setup is perfect!)
- Python with PyTorch, torchvision installed
- Kaggle account (free)

---

## 🎯 Step 1: Download the Anime Face Dataset

### Option A: Download from Kaggle (Recommended)

1. **Go to the dataset page:**
   - Visit: https://www.kaggle.com/datasets/splcher/animefacedataset

2. **Download the dataset:**
   - Click the "Download" button (you may need to create a free Kaggle account)
   - You'll get a file named `animefacedataset.zip` (~300MB)

3. **Extract the dataset:**
   ```bash
   # Navigate to your project directory
   cd /home/pollmix/Coding/tiny-diffusion
   
   # Create the data directory if it doesn't exist
   mkdir -p data/anime_faces
   
   # Extract the downloaded zip file
   unzip ~/Downloads/animefacedataset.zip -d data/anime_faces/
   
   # The images should now be in data/anime_faces/
   # You should see many .jpg or .png files
   ```

4. **Verify the setup:**
   ```bash
   # Check if images are in the right place
   ls data/anime_faces/ | head -10
   
   # Count how many images you have
   ls data/anime_faces/*.jpg | wc -l
   ```
   
   You should see thousands of anime face images (typically ~20,000-60,000 images).

### Option B: Using Kaggle API (Advanced)

If you prefer using the command line:

```bash
# Install Kaggle API
pip install kaggle

# Set up your Kaggle API credentials (follow Kaggle's instructions)
# Then download the dataset
kaggle datasets download -d splcher/animefacedataset -p data/

# Extract
unzip data/animefacedataset.zip -d data/anime_faces/
rm data/animefacedataset.zip  # Clean up the zip file
```

---

## 🚀 Step 2: Train the Model

### Quick Start (Recommended Settings for Your Hardware)

```bash
# Train for 20 epochs (quick test, ~30-40 minutes)
python train.py --dataset anime --epochs 20 --batch_size 32

# Train for better quality (50 epochs, ~1.5-2 hours)
python train.py --dataset anime --epochs 50 --batch_size 32

# Train for best quality (100 epochs, ~3-4 hours)
python train.py --dataset anime --epochs 100 --batch_size 32
```

### Understanding the Parameters

- `--dataset anime`: Use the anime face dataset
- `--epochs 50`: Number of times to go through the entire dataset
  - 20 epochs: Quick test, recognizable faces
  - 50 epochs: Good quality, recommended
  - 100+ epochs: Best quality, very realistic
- `--batch_size 32`: Process 32 images at once
  - Your 8GB VRAM can handle 32-64
  - Lower if you get "out of memory" errors
  - Higher for faster training if you have more VRAM

### Additional Options

```bash
# Customize training further
python train.py \
  --dataset anime \
  --epochs 50 \
  --batch_size 32 \
  --lr 2e-4 \
  --timesteps 1000 \
  --sample_interval 5 \
  --save_interval 10 \
  --output_dir outputs_anime
```

Parameters explained:
- `--lr 2e-4`: Learning rate (0.0002) - how fast the model learns
- `--timesteps 1000`: Number of denoising steps (more = better quality but slower)
- `--sample_interval 5`: Generate sample images every 5 epochs
- `--save_interval 10`: Save model checkpoint every 10 epochs
- `--output_dir`: Where to save outputs

---

## 📊 Step 3: Monitor Training Progress

### What to Watch

1. **Training loss**: Should decrease over time
   - Check `outputs/training_loss.png` after training
   - Good: Steadily decreasing from ~0.10 to ~0.02
   - Bad: Stuck at high values or increasing

2. **Sample images**: Generated during training
   - Check `outputs/samples/epoch_XXX.png`
   - Early epochs: Blurry, vague shapes
   - Middle epochs: Recognizable faces, some details
   - Late epochs: Clear, detailed anime faces

3. **Terminal output**: Shows progress
   ```
   Epoch 10/50
   Training: 100%|████████| 625/625 [02:15<00:00]
   Average Loss: 0.0345
   Generating samples...
   ```

### Expected Timeline (Your Hardware)

- **Epoch 1-10**: Loss drops quickly (~0.10 → ~0.04), blurry faces
- **Epoch 10-30**: Gradual improvement (~0.04 → ~0.03), clearer features
- **Epoch 30-50**: Fine-tuning (~0.03 → ~0.025), high quality faces
- **Epoch 50+**: Diminishing returns, very high quality

---

## 🎨 Step 4: Generate Anime Faces

### After Training

Once training is complete, generate new anime faces:

```bash
# Generate 64 anime faces
python sample.py --checkpoint outputs/model_final.pt --num_images 64

# Generate with intermediate steps (see the denoising process)
python sample.py --checkpoint outputs/model_final.pt --num_images 64 --show_intermediate

# Generate more images
python sample.py --checkpoint outputs/model_final.pt --num_images 100 --output my_anime_faces.png
```

### Understanding the Output

- **Generated images**: Saved to `outputs/generated_samples.png` by default
- **Intermediate steps** (if `--show_intermediate`): 
  - Saved to `outputs/intermediate/step_XXXX.png`
  - Shows the denoising process from noise to clear image
  - Timesteps: 999 (pure noise) → 0 (clear face)

### Generation Time

- 64 images: ~30-40 seconds on your GPU
- 100 images: ~50-60 seconds on your GPU

---

## 💡 Tips for Best Results

### Training Tips

1. **Start small, scale up:**
   ```bash
   # Quick 10 epoch test to verify everything works
   python train.py --dataset anime --epochs 10 --batch_size 32
   
   # If it looks good, train for real
   python train.py --dataset anime --epochs 50 --batch_size 32
   ```

2. **Use data augmentation** (already enabled in `anime_dataset.py`):
   - Random horizontal flips
   - Center cropping
   - Helps model generalize better

3. **Monitor VRAM usage:**
   ```bash
   # In another terminal, watch GPU usage
   watch -n 1 nvidia-smi
   ```
   - If VRAM is maxed out, lower batch size
   - If VRAM is underutilized, increase batch size

4. **Save checkpoints regularly:**
   - Default: saves every 5 epochs
   - Checkpoints are in `outputs/checkpoints/`
   - Can resume from any checkpoint if training interrupts

### Quality Tips

1. **Train longer for better results:**
   - 20 epochs: OK quality, good for testing
   - 50 epochs: Good quality, recommended
   - 100 epochs: Excellent quality
   - 200+ epochs: Diminishing returns, very high quality

2. **Check sample images during training:**
   - If faces look good at epoch 30, you can stop early
   - If faces are still blurry at epoch 30, train longer

3. **Experiment with batch size:**
   - Larger batches (64) = more stable training
   - Smaller batches (16-32) = more "creative" results

---

## 🐛 Troubleshooting

### Issue: "No images found in data/anime_faces"

**Solution:** Check that images are in the right folder
```bash
ls data/anime_faces/ | head
# Should show image files (.jpg, .png)

# If empty, re-extract the dataset
unzip ~/Downloads/animefacedataset.zip -d data/anime_faces/
```

---

### Issue: "CUDA out of memory"

**Solution:** Lower the batch size
```bash
# Try batch_size 16
python train.py --dataset anime --epochs 50 --batch_size 16

# Or even smaller
python train.py --dataset anime --epochs 50 --batch_size 8
```

---

### Issue: Generated faces look bad after training

**Possible causes:**

1. **Not trained enough:**
   ```bash
   # Train for more epochs
   python train.py --dataset anime --epochs 100 --batch_size 32
   ```

2. **Learning rate too high:**
   ```bash
   # Lower the learning rate
   python train.py --dataset anime --epochs 50 --batch_size 32 --lr 1e-4
   ```

3. **Dataset issues:**
   - Check if images are actually anime faces
   - Make sure images aren't corrupted
   ```bash
   # Test by opening a few random images
   ls data/anime_faces/ | shuf -n 5
   ```

---

### Issue: Training is very slow

**Solutions:**

1. **Make sure GPU is being used:**
   - Should see "Using device: cuda" at start
   - If it says "cpu", install CUDA-enabled PyTorch

2. **Increase batch size (if memory allows):**
   ```bash
   python train.py --dataset anime --epochs 50 --batch_size 64
   ```

3. **Reduce image size** (edit `train.py`, line ~118):
   ```python
   image_size = 32  # Instead of 64
   ```
   - Smaller images = faster training
   - But lower quality output

---

## 📈 Expected Results

### After 20 Epochs (~30-40 min)
- **Loss**: ~0.030-0.035
- **Quality**: Recognizable anime faces, somewhat blurry
- **Usable**: Yes, for testing/understanding

### After 50 Epochs (~1.5-2 hours)
- **Loss**: ~0.025-0.030
- **Quality**: Clear anime faces with good details
- **Usable**: Yes, recommended quality level

### After 100 Epochs (~3-4 hours)
- **Loss**: ~0.020-0.025
- **Quality**: High-quality, diverse anime faces
- **Usable**: Yes, excellent results

### After 200 Epochs (~6-8 hours)
- **Loss**: ~0.018-0.023
- **Quality**: Very high quality, subtle details
- **Usable**: Yes, best quality (but diminishing returns)

---

## 🎓 What's Different from MNIST?

| Aspect             | MNIST (Digits)       | Anime Faces           |
| ------------------ | -------------------- | --------------------- |
| Image size         | 28×28                | 64×64                 |
| Channels           | 1 (grayscale)        | 3 (RGB color)         |
| Complexity         | Simple shapes        | Complex features      |
| Training time      | 5-10 min (20 epochs) | 30-40 min (20 epochs) |
| Recommended epochs | 20-30                | 50-100                |
| Model size         | ~2M parameters       | ~2M parameters        |
| Memory usage       | Low                  | Higher                |

---

## 🚀 Next Steps

Once you have good results:

1. **Generate lots of faces:**
   ```bash
   python sample.py --checkpoint outputs/model_final.pt --num_images 200
   ```

2. **Experiment with different epochs:**
   - Compare results from epoch 20, 50, 100
   - See how quality improves

3. **Try different settings:**
   - Different batch sizes
   - Different learning rates
   - More/fewer timesteps

4. **Share your results:**
   - Post on social media
   - Show friends
   - Join AI art communities

5. **Extend the project:**
   - Try other datasets (faces, landscapes, art)
   - Implement conditional generation (specify attributes)
   - Add text conditioning
   - Try larger image sizes (128×128)

---

## 📁 File Structure After Setup

```
tiny-diffusion/
├── data/
│   └── anime_faces/          # Your anime face images here
│       ├── image_001.jpg
│       ├── image_002.jpg
│       └── ... (thousands more)
├── outputs/
│   ├── samples/              # Training progress samples
│   │   ├── epoch_001.png
│   │   ├── epoch_005.png
│   │   └── ...
│   ├── checkpoints/          # Model checkpoints
│   │   ├── model_epoch_005.pt
│   │   ├── model_epoch_010.pt
│   │   └── ...
│   ├── intermediate/         # Denoising process (if --show_intermediate)
│   ├── training_loss.png     # Loss curve
│   ├── generated_samples.png # Final generated images
│   └── model_final.pt        # Final trained model
├── anime_dataset.py          # Anime dataset loader
├── train.py                  # Training script
├── sample.py                 # Sampling script
└── ...
```

---

## ❓ FAQ

**Q: How long does it take to train?**  
A: On your hardware (16GB RAM, 8GB VRAM):
- 20 epochs: ~30-40 minutes
- 50 epochs: ~1.5-2 hours
- 100 epochs: ~3-4 hours

**Q: How many images should I have in the dataset?**  
A: The Anime Face Dataset has ~20,000-60,000 images, which is plenty. Generally:
- Minimum: ~1,000 images
- Good: 10,000+ images
- Excellent: 50,000+ images

**Q: Can I use my own images?**  
A: Yes! Just put your images in `data/anime_faces/` and train. Make sure:
- Images are similar style/subject
- Reasonable quality
- At least a few hundred images

**Q: Can I pause and resume training?**  
A: Not directly, but you can load a checkpoint and continue:
1. Stop training (Ctrl+C)
2. Note which checkpoint was last saved
3. Modify train.py to load that checkpoint
4. Continue training

**Q: The faces look weird/distorted. Is this normal?**  
A: In early epochs (1-20), yes. They should improve significantly by epoch 30-50. If they're still bad after 50 epochs, try:
- Training longer
- Lowering learning rate
- Checking your dataset

**Q: Can I train on larger images?**  
A: Yes! Edit `train.py` line ~118:
```python
image_size = 128  # Instead of 64
```
But note:
- Much slower training
- Uses more VRAM (may need smaller batch size)
- Better quality output

---

## 🎉 You're Ready!

You now have everything you need to train your diffusion model on anime faces. Start with a quick test run, then scale up for best results.

**Recommended first command:**
```bash
python train.py --dataset anime --epochs 20 --batch_size 32
```

Good luck and have fun generating anime faces! 🎨✨
