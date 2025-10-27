# Quick Start Guide

## 🎯 Goal
Learn diffusion models by building one from scratch! This project will help you understand:
- How images turn into noise (forward diffusion)
- How neural networks learn to denoise (reverse diffusion)
- How to generate new images from pure noise

## 📝 Quick Commands

```bash
# 1. See forward diffusion in action
python visualize_diffusion.py

# 2. Train your model (20 epochs, ~5-10 min on CPU)
python train.py --epochs 20

# 3. Generate new images!
python sample.py --checkpoint outputs/model_final.pt --show_intermediate
```

## 🎓 Learning Path

### Phase 1: Understanding Forward Diffusion (5 min)
Run the visualization to see how images become noise:
```bash
python visualize_diffusion.py --num_images 8
```

**What to observe:**
- At t=0: Clean digit images
- At t=50-100: Slight noise added
- At t=400-600: Heavy noise, barely visible
- At t=999: Pure Gaussian noise

**Key insight**: The forward process is deterministic and doesn't require training!

### Phase 2: Training the Reverse Process (10-15 min)
Train the U-Net to predict noise:
```bash
python train.py --epochs 20 --batch_size 128
```

**What happens:**
- Model sees noisy images at random timesteps
- Learns to predict what noise was added
- Gets better at denoising over epochs
- Check `outputs/samples/` to see improvement!

**Key insight**: The model learns ONE function that works for ALL noise levels!

### Phase 3: Generating Images (2 min)
Use your trained model to create new images:
```bash
python sample.py --checkpoint outputs/model_final.pt --num_images 64 --show_intermediate
```

**What happens:**
- Starts from pure random noise
- Iteratively denoises for 1000 steps
- Each step removes a bit of noise
- Final result: new digit images!

**Key insight**: Generation is deterministic given the noise, but random noise gives random results!

## 🔬 Experiments to Try

### Experiment 1: Compare Training Lengths
```bash
# Short training
python train.py --epochs 10 --output_dir outputs/exp1_10epochs
python sample.py --checkpoint outputs/exp1_10epochs/model_final.pt --output outputs/exp1_samples.png

# Longer training
python train.py --epochs 50 --output_dir outputs/exp1_50epochs
python sample.py --checkpoint outputs/exp1_50epochs/model_final.pt --output outputs/exp1_samples_50.png
```
**Question**: How does training duration affect image quality?

### Experiment 2: Different Datasets
```bash
# MNIST (handwritten digits)
python train.py --dataset mnist --epochs 20 --output_dir outputs/exp2_mnist
python sample.py --checkpoint outputs/exp2_mnist/model_final.pt --output outputs/exp2_mnist_samples.png

# Fashion-MNIST (clothing items)
python train.py --dataset fashion_mnist --epochs 20 --output_dir outputs/exp2_fashion
python sample.py --checkpoint outputs/exp2_fashion/model_final.pt --output outputs/exp2_fashion_samples.png
```
**Question**: Which dataset is harder to learn? Why?

### Experiment 3: Watch the Denoising Process
```bash
python sample.py --checkpoint outputs/model_final.pt --show_intermediate
```
Then check `outputs/intermediate/` to see how noise gradually becomes digits!

**Question**: At what timestep do digits start becoming recognizable?

### Experiment 4: Different Model Sizes
Edit `train.py` to use different `base_channels`:
- 32 channels: Smaller, faster, less quality
- 64 channels: Default, good balance
- 128 channels: Larger, slower, better quality

**Question**: Is bigger always better? Consider training time vs. quality.

## 📊 What to Monitor

### During Training
1. **Loss curve**: Should decrease over time
   - Check `outputs/training_loss.png`
   - Early epochs: loss drops quickly
   - Later epochs: slower improvement

2. **Sample quality**: Gets better each epoch
   - Check `outputs/samples/epoch_XXX.png`
   - Early: mostly noise
   - Middle: blurry digits
   - Late: clear, recognizable digits

3. **Console output**: Shows progress
   - Epoch number and loss
   - GPU/CPU usage
   - Time per epoch

## 🐛 Troubleshooting

### "Out of memory" error
```bash
# Reduce batch size
python train.py --batch_size 64

# Or use fewer epochs
python train.py --epochs 10
```

### Training is too slow
```bash
# Use smaller model
# Edit train.py: create_model(..., base_channels=32)

# Or reduce timesteps
python train.py --timesteps 500
```

### Generated images are blurry
```bash
# Train longer
python train.py --epochs 50

# Or adjust learning rate
python train.py --lr 1e-4 --epochs 30
```

## 🎯 Success Criteria

After completing the learning path, you should understand:

✅ **Forward diffusion**: How to add noise mathematically  
✅ **Noise schedules**: Why we use β_t and α_t  
✅ **Model training**: Why we predict noise (not images directly)  
✅ **Reverse diffusion**: How iterative denoising works  
✅ **Time embeddings**: Why the model needs to know timestep t  
✅ **U-Net architecture**: Why we use skip connections  

## 📚 Next Steps

Once you're comfortable with the basics:

1. **Read the paper**: "Denoising Diffusion Probabilistic Models" by Ho et al.
2. **Try CIFAR-10**: Extend to 32×32 RGB images
3. **Add conditioning**: Make class-conditional generation
4. **Faster sampling**: Implement DDIM for fewer steps
5. **Better schedules**: Try cosine noise schedule
6. **Larger models**: Scale up for better quality

## 💡 Key Takeaways

1. **Diffusion = Gradual Process**: Both forward (noising) and reverse (denoising) happen step by step
2. **One Model, All Steps**: The U-Net handles all timesteps with time embeddings
3. **Noise Prediction**: Predicting noise is easier than predicting clean images
4. **Iterative Refinement**: Generation takes many steps, each improving slightly
5. **Simple but Powerful**: This simple approach generates high-quality images!

---

**Ready to start?** Run the three commands at the top and dive in! 🚀
