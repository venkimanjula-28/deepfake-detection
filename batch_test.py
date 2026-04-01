"""
Batch test the DeepFake model on multiple real and fake images
"""
import torch
import os
import sys
from pathlib import Path

sys.path.append('src')
from model import get_model
from test_model import predict as single_predict

def batch_test(model_path, test_dir, num_samples=10):
    """Test model on multiple samples from a directory."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    best_acc = checkpoint.get('best_acc', 0)
    
    print("="*60)
    print(f"  BATCH TESTING - Model Accuracy Verification")
    print("="*60)
    print(f"\n📦 Model: {model_path}")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}%")
    print(f"💻 Device: {device}")
    print(f"\n📁 Test Directory: {test_dir}")
    print(f"📊 Samples to test: {num_samples}")
    print("="*60 + "\n")
    
    # Get image files
    img_files = list(Path(test_dir).glob('*.jpg'))[:num_samples]
    
    if not img_files:
        print(f"❌ No images found in {test_dir}")
        return
    
    correct = 0
    total = len(img_files)
    
    for idx, img_path in enumerate(img_files, 1):
        try:
            label, conf = single_predict(model_path, str(img_path))
            
            # Determine expected label from path
            expected = "REAL" if "Real" in str(img_path) or "real_" in str(img_path) else "FAKE"
            is_correct = (label == expected)
            
            if is_correct:
                correct += 1
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG"
            
            print(f"[{idx}/{total}] {status} | Expected: {expected}, Predicted: {label} ({conf:.1f}%)")
            
        except Exception as e:
            print(f"[{idx}/{total}] ⚠️  Error: {e}")
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print("\n" + "="*60)
    print(f"  RESULTS")
    print("="*60)
    print(f"✓ Correct: {correct}/{total}")
    print(f"✗ Wrong: {total - correct}/{total}")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print("="*60 + "\n")
    
    return accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Test DeepFake Model')
    parser.add_argument('--model', type=str, default='checkpoint_best.pth', help='Model checkpoint')
    parser.add_argument('--real-dir', type=str, default='Dataset/Train/Real', help='Directory with real images')
    parser.add_argument('--fake-dir', type=str, default='Dataset/Train/Fake', help='Directory with fake images')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples per class')
    
    args = parser.parse_args()
    
    print("\n🔍 Testing on REAL images:")
    real_acc = batch_test(args.model, args.real_dir, args.samples)
    
    print("\n🔍 Testing on FAKE images:")
    fake_acc = batch_test(args.model, args.fake_dir, args.samples)
    
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(f"Real Images Accuracy: {real_acc:.2f}%")
    print(f"Fake Images Accuracy: {fake_acc:.2f}%")
    overall = (real_acc + fake_acc) / 2
    print(f"🎯 Overall Accuracy: {overall:.2f}%")
    print("="*60 + "\n")
