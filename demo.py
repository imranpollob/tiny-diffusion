#!/usr/bin/env python3
"""
Demo Script - Run all components of the diffusion model
This script demonstrates the complete pipeline step by step
"""

import os
import sys
import time

def print_header(text):
    """Print a nice header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_step(step_num, title, command, explanation):
    """Run a single demonstration step"""
    print_header(f"Step {step_num}: {title}")
    print(f"📝 {explanation}\n")
    print(f"🔧 Running: {command}\n")
    
    response = input("Press Enter to continue (or 's' to skip): ")
    if response.lower() == 's':
        print("⏭️  Skipped!\n")
        return False
    
    print()
    os.system(command)
    print(f"\n✅ Step {step_num} completed!\n")
    time.sleep(1)
    return True

def main():
    print_header("🚀 Tiny Diffusion - Complete Demo")
    
    print("""
This demo will walk you through the complete diffusion model pipeline:

1. Visualize forward diffusion (how images become noise)
2. Train a model to reverse the process
3. Generate new images from noise

Each step is optional - you can skip steps if you want.
The training step will take about 5-10 minutes on CPU.

Note: Generated files will be saved in the 'outputs/' directory.
    """)
    
    response = input("Ready to start? (y/n): ")
    if response.lower() != 'y':
        print("Demo cancelled. Run this script again when you're ready!")
        return
    
    # Step 1: Visualize forward diffusion
    run_step(
        1,
        "Visualize Forward Diffusion",
        "python visualize_diffusion.py --num_images 6",
        "This shows how clean images gradually become pure noise over 1000 steps."
    )
    
    print("""
💡 What you should see:
   - A grid showing 6 different digit images
   - Each row shows the same image at different noise levels
   - By t=999, all images look like random noise
   - Check: outputs/forward_diffusion.png
    """)
    
    input("Press Enter when you're ready for the next step...")
    
    # Step 2: Train the model
    trained = run_step(
        2,
        "Train the Diffusion Model",
        "python train.py --epochs 10 --batch_size 128 --sample_interval 2",
        "Train a U-Net to predict and remove noise. This will take 5-10 minutes.\n"
        "   We're using just 10 epochs for a quick demo (50+ epochs recommended for best results)."
    )
    
    if trained:
        print("""
💡 What you should see:
   - Progress bars showing training progress
   - Loss values that gradually decrease
   - Sample images generated every 2 epochs
   - Files saved in outputs/samples/ and outputs/checkpoints/
   
   As training progresses, the sample images should get clearer and more digit-like!
        """)
    
    input("Press Enter when you're ready for the next step...")
    
    # Step 3: Generate images
    if trained or os.path.exists("outputs/model_final.pt"):
        run_step(
            3,
            "Generate New Images",
            "python sample.py --checkpoint outputs/model_final.pt --num_images 64 --show_intermediate",
            "Generate 64 new digit images by starting from pure noise and iteratively denoising.\n"
            "   This will also save the intermediate denoising steps."
        )
        
        print("""
💡 What you should see:
   - A progress bar as the model denoises over 1000 steps
   - Final generated images saved to outputs/generated_samples.png
   - Intermediate steps saved to outputs/intermediate/
   
   Check the intermediate folder to see how noise gradually becomes digits!
        """)
    else:
        print("⚠️  No trained model found. Skipping generation step.")
        print("   Train a model first with: python train.py")
    
    # Summary
    print_header("🎉 Demo Complete!")
    
    print("""
Great job! You've completed the full pipeline:

✅ Visualized forward diffusion (noise addition)
✅ Trained a model to reverse the process
✅ Generated new images from pure noise

📁 Check these files:
   - outputs/forward_diffusion.png - Forward diffusion visualization
   - outputs/samples/ - Training progress samples
   - outputs/generated_samples.png - Final generated images
   - outputs/intermediate/ - Denoising process step-by-step
   - outputs/training_loss.png - Loss curve

📚 Next steps:
   1. Read QUICKSTART.md for experiments to try
   2. Read SUMMARY.md to understand the implementation
   3. Try training for more epochs: python train.py --epochs 50
   4. Experiment with Fashion-MNIST: python train.py --dataset fashion_mnist
   5. Modify the code and see what happens!

🔬 Understanding diffusion models:
   - Forward: deterministic noise addition
   - Reverse: learned iterative denoising
   - Key: predict noise, not images directly
   - Result: high-quality generative model!

Happy learning! 🚀
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted. You can run it again anytime!")
        sys.exit(0)
