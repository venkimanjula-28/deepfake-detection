"""
Quick High-Accuracy Training - Optimized for correct predictions
Uses frozen backbone + simple transforms + focused training
"""
import subprocess
import sys

def quick_train():
    print("="*60)
    print("  🎯 QUICK HIGH-ACCURACY TRAINING")
    print("="*60)
    print("\n⚙️  Strategy:")
    print("   • Frozen backbone (pretrained features)")
    print("   • Simple augmentations (avoid overfitting)")
    print("   • 15 epochs for full convergence")
    print("   • Lower learning rate for stability")
    print("   • Early stopping on validation loss")
    print("\n📁 Using cached dataset...")
    print("="*60 + "\n")
    
    cmd = [
        sys.executable, "src/train.py",
        "--data-dir", "Dataset/Train_cache",
        "--backbone", "resnet18",
        "--frames", "6",
        "--batch-size", "32",
        "--epochs", "15",
        "--lr", "1e-4",
        "--output", "checkpoint_best.pth",
        "--freeze-backbone",
        "--num-workers", "0",
        "--label-smoothing", "0.05",
        "--weight-decay", "1e-4",
        "--patience", "20"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training complete! Model saved to: checkpoint_best.pth")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    quick_train()
