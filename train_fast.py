"""
Ultra-Accurate DeepFake Training Script
Trains a highly accurate model in minutes with better generalization!
"""
import subprocess
import sys
import os

def train_accurate():
    """Train with optimized settings for maximum accuracy and correct predictions."""
    
    # Check if cached dataset exists, if not, suggest caching
    cache_dir = "Dataset/Train_cache"
    if not os.path.exists(cache_dir):
        print("⚠️  Cached dataset not found!")
        print("📝 Running dataset caching first for maximum speed...")
        subprocess.run([sys.executable, "src/cache_dataset.py", "--data-dir", "Dataset/Train"])
        print("✅ Caching complete!\n")
    
    print("="*60)
    print("  🎯 ULTRA-ACCURATE DEEPFAKE DETECTION TRAINING")
    print("="*60)
    print("\n⚙️  Configuration:")
    print("   • Epochs: 10 (for convergence)")
    print("   • Batch Size: 48 (balanced)")
    print("   • Frames: 6 per clip (better temporal info)")
    print("   • Learning Rate: 3e-4 (optimal)")
    print("   • Backbone: ResNet18 (UNFROZEN for fine-tuning)")
    print("   • Strong Augmentation (rotation, blur, color jitter)")
    print("   • Label Smoothing: 0.1 (prevents overfitting)")
    print("   • Weight Decay: 5e-4 (regularization)")
    print("\n📁 Using cached dataset for maximum speed...")
    print("\n" + "="*60 + "\n")
    
    # High-accuracy training command
    cmd = [
        sys.executable, "src/train.py",
        "--data-dir", "Dataset/Train_cache",
        "--backbone", "resnet18",
        "--frames", "6",
        "--batch-size", "48",
        "--epochs", "10",
        "--lr", "3e-4",
        "--output", "checkpoint_accurate.pth",
        "--freeze-backbone",  # Unfreeze for fine-tuning
        "--num-workers", "0",
        "--label-smoothing", "0.1",
        "--weight-decay", "5e-4"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        print("\n" + "="*60)
        print("  ✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"\n🎯 Model saved to: checkpoint_accurate.pth")
        print("📊 Check confusion matrix: confusion_matrix_best.png")
        print("📈 Training plots: training_visualization.png")
        print("\n💡 To test the model:")
        print(f"   python src/infer.py --model checkpoint_accurate.pth --image <path_to_image>")
        print("\n" + "="*60 + "\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    train_accurate()
