import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import get_model
from dataset import DeepFakeDataset


def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Temporal Attention Training - Fast & Accurate')
    parser.add_argument('--data-dir', type=str, required=True, help='path to dataset root (real, fake subfolders)')
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'efficientnet_b0'], default='resnet18')
    parser.add_argument('--frames', type=int, default=8, help='Reduced frames for faster training (default: 8)')
    parser.add_argument('--batch-size', type=int, default=32, help='Larger batch size for faster training (default: 32)')
    parser.add_argument('--epochs', type=int, default=20, help='More epochs with early stopping (default: 20)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Higher initial LR with warmup (default: 5e-4)')
    parser.add_argument('--output', type=str, default='checkpoint_fast.pth')
    parser.add_argument('--use-lstm', action='store_true', help='Enable LSTM for better temporal modeling')
    parser.add_argument('--freeze-backbone', action='store_true', default=True, help='Freeze backbone initially (default: True)')
    parser.add_argument('--unfreeze-epoch', type=int, default=5, help='Epoch to unfreeze backbone (default: 5)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (default: 4)')
    parser.add_argument('--mixed-precision', action='store_true', default=True, help='Use mixed precision (default: True)')
    parser.add_argument('--val-split', type=float, default=0.15, help='Validation split (default: 0.15)')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience (default: 7)')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay for regularization (default: 1e-3)')
    parser.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing (default: 0.05)')
    return parser.parse_args()


def train():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation transform - no augmentation
    val_transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset without transform first
    print(f"Loading dataset from {args.data_dir}...")
    dataset = DeepFakeDataset(args.data_dir, num_frames=args.frames, transform=None)
    print(f"Total samples: {len(dataset)}")
    
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_indices, val_indices = random_split(range(len(dataset)), [train_len, val_len], 
                                               generator=torch.Generator().manual_seed(42))
    
    # Create datasets with appropriate transforms
    train_ds = torch.utils.data.Subset(DeepFakeDataset(args.data_dir, num_frames=args.frames, transform=train_transform), train_indices.indices)
    val_ds = torch.utils.data.Subset(DeepFakeDataset(args.data_dir, num_frames=args.frames, transform=val_transform), val_indices.indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, 
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")

    model = get_model(backbone=args.backbone, pretrained=True, freeze_backbone=args.freeze_backbone, use_lstm=args.use_lstm)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    best_acc = 0.0
    patience_counter = 0
    start_unfreeze = args.unfreeze_epoch if args.freeze_backbone else 0
    
    # Lists to store metrics for visualization
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, args.epochs + 1):
        # Progressive unfreezing: unfreeze backbone after specified epoch
        if epoch == start_unfreeze and args.freeze_backbone:
            print(f"\n>>> Unfreezing backbone at epoch {epoch}")
            for p in model.backbone.parameters():
                p.requires_grad = True
            # Add backbone params to optimizer with lower LR
            optimizer.add_param_group({'params': model.backbone.parameters(), 'lr': args.lr * 0.1})
            print(f"Trainable params now: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        model.train()
        epoch_loss = 0.0
        total = 0
        correct = 0

        loop = tqdm(train_loader, desc=f'Epoch [{epoch}/{args.epochs}]', unit='batch')
        for frames, labels in loop:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                outputs = model(frames)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            loop.set_postfix(loss=f'{epoch_loss/total:.4f}', acc=f'{100.0*correct/total:.2f}%')

        train_acc = 100.0 * correct / total

        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(frames)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                val_total += labels.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        print(f'[Epoch {epoch}] train_loss={epoch_loss/total:.4f} train_acc={train_acc:.2f}% val_loss={val_loss/val_total:.4f} val_acc={val_acc:.2f}%')

        # Store metrics for visualization
        train_losses.append(epoch_loss / total)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping check
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                        'args': vars(args)}, args.output)
            print(f'✓ New best model saved with val_acc={best_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break

    print(f'Training complete. best_val_acc={best_acc:.2f}%')
    
    # Create visualization charts
    epochs = list(range(1, args.epochs + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training visualization saved as 'training_visualization.png'")


if __name__ == '__main__':
    train()
