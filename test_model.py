"""
Test the trained DeepFake detection model on sample images
"""
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import sys

try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(image_size=112, margin=20, select_largest=True, post_process=False)
    print("✓ MTCNN loaded")
except Exception:
    mtcnn = None
    print("⚠ MTCNN not available, using full image")

# Import model
sys.path.append('src')
from model import get_model

def preprocess_image(image_path, size=112):
    """Preprocess a single image for prediction."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect face (optional)
    if mtcnn is not None:
        face = mtcnn(img_rgb)
        if face is not None:
            # Face detected - resize to target size
            face_np = face.numpy().transpose(1, 2, 0)
            face_resized = cv2.resize(face_np, (size, size))
            img_tensor = torch.from_numpy(face_resized.transpose(2, 0, 1)).float() / 255.0
        else:
            # No face - use full image
            img_resized = cv2.resize(img_rgb, (size, size))
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    else:
        # No MTCNN - resize full image
        img_resized = cv2.resize(img_rgb, (size, size))
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    # Add batch dimension and repeat for 6 frames
    img_batch = img_normalized.unsqueeze(0).repeat(6, 1, 1, 1).unsqueeze(0)
    
    return img_batch

def predict(model_path, image_path):
    """Make prediction on a single image."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = get_model(backbone='resnet18', pretrained=False, freeze_backbone=False, use_lstm=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded: {model_path}")
    print(f"  Best validation accuracy: {checkpoint.get('best_acc', 0):.2f}%")
    
    # Preprocess
    input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = probs.max(1)
        
    label = "FAKE" if predicted.item() == 1 else "REAL"
    confidence_pct = confidence.item() * 100
    
    print(f"\n📷 Image: {image_path}")
    print(f"🎯 Prediction: {label}")
    print(f"💯 Confidence: {confidence_pct:.2f}%")
    
    return label, confidence_pct

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test DeepFake Detection Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    
    args = parser.parse_args()
    
    try:
        predict(args.model, args.image)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
