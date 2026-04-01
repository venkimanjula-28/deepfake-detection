"""
Clean, Simple, HIGH-ACCURACY Training Script
Removes all problematic augmentations for maximum accuracy
"""
import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from model import get_model
from dataset import CachedDeepFakeDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Clean High-Accuracy Training')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--frames', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output', type=str, default='checkpoint_clean.pth')
    parser.add_argument('--freeze-backbone', action='store_true', default=True)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=999)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    return parser.parse_args()


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = probs.max(1)
            correct += preds.eq(labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    return avg_loss, acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics(labels, preds, probs, split='Validation'):
    classes = ['Real', 'Fake']
    print(f"\n{'='*50}")
    print(f"  {split} Metrics")
    print(f"{'='*50}")
    print(classification_report(labels, preds, target_names=classes, digits=4))
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(f"              Predicted Real  Predicted Fake")
    print(f"Actual Real       {cm[0][0]:>6}          {cm[0][1]:>6}")
    print(f"Actual Fake       {cm[1][0]:>6}          {cm[1][1]:>6}")
    try:
        auc = roc_auc_score(labels, probs)
        print(f"\nROC-AUC Score: {auc:.4f}")
    except Exception:
        pass
    print(f"{'='*50}\n")
    return cm


def train():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # SIMPLE, CLEAN transforms - NO strong augmentations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    full_ds = CachedDeepFakeDataset(args.data_dir, num_frames=args.frames, transform=None)
    print(f"Total samples: {len(full_ds)}")
    
    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_indices, val_indices = random_split(
        range(len(full_ds)), [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_ds = torch.utils.data.Subset(
        CachedDeepFakeDataset(args.data_dir, num_frames=args.frames, transform=train_transform),
        train_indices.indices
    )
    val_ds = torch.utils.data.Subset(
        CachedDeepFakeDataset(args.data_dir, num_frames=args.frames, transform=val_transform),
        val_indices.indices
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Batch: {args.batch_size}")
    
    # Model
    model = get_model(backbone='resnet18', pretrained=True, freeze_backbone=args.freeze_backbone, use_lstm=False)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training
    best_acc = 0.0
    patience_ctr = 0
    best_cm = None
    
    print(f"\nStarting training for {args.epochs} epochs...\n")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f'Epoch [{epoch}/{args.epochs}]')
        
        for frames, labels in loop:
            frames = frames.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            loop.set_postfix(loss=f'{epoch_loss/total:.4f}', acc=f'{100.0*correct/total:.2f}%')
        
        train_acc = 100.0 * correct / total
        train_loss = epoch_loss / total
        
        val_loss, val_acc, v_labels, v_preds, v_probs = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        
        print(f"[Epoch {epoch:>3}] train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")
        
        if val_acc > best_acc or epoch % 2 == 0:
            cm = print_metrics(v_labels, v_preds, v_probs, split=f'Epoch {epoch}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_ctr = 0
            best_cm = confusion_matrix(v_labels, v_preds)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, args.output)
            print(f'  ✓ Best model saved → val_acc={best_acc:.2f}%')
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f'\nEarly stopping at epoch {epoch}')
                break
    
    # Final results
    print(f"\n{'='*50}")
    print(f"  FINAL RESULTS (Best Val Acc = {best_acc:.2f}%)")
    print(f"{'='*50}")
    
    ckpt = torch.load(args.output, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    _, final_acc, f_labels, f_preds, f_probs = evaluate(model, val_loader, device, criterion)
    print_metrics(f_labels, f_preds, f_probs, split='FINAL')
    
    if best_cm is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=['Real', 'Fake'])
        disp.plot(ax=ax, colorbar=True, cmap='Blues')
        ax.set_title(f'Confusion Matrix (Best)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('confusion_matrix_clean.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Confusion matrix saved → confusion_matrix_clean.png")
    
    print(f"\nTraining complete! Best Val Acc = {best_acc:.2f}%")
    print(f"Model saved → {args.output}")


if __name__ == '__main__':
    train()
