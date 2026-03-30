import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import get_model
from dataset import DeepFakeDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DeepFake Temporal Attention model')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'efficientnet_b0'], default='resnet18')
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-lstm', action='store_true', default=True)
    return parser.parse_args()


def evaluate():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = DeepFakeDataset(args.data_dir, num_frames=args.frames, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model(backbone=args.backbone, pretrained=False, freeze_backbone=False, use_lstm=args.use_lstm)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Try to load state dict, handling potential architecture mismatches
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Warning: Model architecture mismatch. Loading compatible weights only...")
        # Load only matching keys
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        # Update current model dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from checkpoint")
    
    model.to(device).eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    acc = 100.0 * correct / total
    print(f'Evaluation accuracy: {acc:.2f}% on {total} samples')


if __name__ == '__main__':
    evaluate()
