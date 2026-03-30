import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms

from model import get_model
from dataset import extract_frames_from_video, detect_faces, pad_or_truncate


def preprocess_frames(frames, num_frames=12):
    frames = frames[:num_frames]
    frames = detect_faces(frames, target_size=224)
    frames = pad_or_truncate(frames, num_frames)
    frames = torch.stack(frames, dim=0)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    frames = normalize(frames)
    return frames.unsqueeze(0)


def infer_video(model, path, num_frames=12, device='cpu'):
    model.eval()
    frames = extract_frames_from_video(path, num_frames)
    tensor = preprocess_frames(frames, num_frames).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0]
    classes = ['real', 'fake']
    pred = classes[prob.argmax().item()]
    return pred, prob.detach().cpu().numpy().tolist()


def main():
    parser = argparse.ArgumentParser(description='Inference for DeepFake Temporal Attention')
    parser.add_argument('--input', type=str, required=True, help='input video file')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'efficientnet_b0'], default='resnet18')
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--use-lstm', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(backbone=args.backbone, pretrained=False, freeze_backbone=False, use_lstm=args.use_lstm)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Try to load state dict, handling potential architecture mismatches
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Warning: Model architecture mismatch. Loading compatible weights only...")
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from checkpoint")
    
    model = model.to(device)

    pred, prob = infer_video(model, args.input, num_frames=args.frames, device=device)
    print(f'Prediction: {pred} - confidence real={prob[0]:.4f}, fake={prob[1]:.4f}')


if __name__ == '__main__':
    main()
