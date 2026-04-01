"""
BALANCED High-Accuracy Training
Fixes the real/fake imbalance issue
"""
import sys
import os
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
    parser = argparse.ArgumentParser(description='Balanced High-Accuracy Training')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--frames', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--output', type=str, default='checkpoint_balanced.pth')
    parser.add_argument('--freeze-backbone', action='store_true', default=True)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=999)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--class-weight', type=str, default='balanced', help='balanced or none')
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
    print(f"\n{'='*60}")
    print(f"  {split} Metrics")
    print(f"{'='*60}")
    print(classification_report(labels, preds, target_names=classes, digits=4))
    cm = confusion_matrix(labels, preds)
    
    # Calculate per-class accuracy
    real_acc = cm[0][0] / (cm[0][0] + cm[0][1]) * 100 if (cm[0][0] + cm[0][1]) > 0 else 0
    fake_acc = cm[1][1] / (cm[1][0] + cm[1][1]) * 100 if (cm[1][0] + cm[1][1]) > 0 else 0
    
    print("Confusion Matrix:")
    print(f"              Predicted Real  Predicted Fake")
    print(f"Actual Real       {cm[0][0]:>6}          {cm[0][1]:>6}  → Real Accuracy: {real_acc:.2f}%")
    print(f"Actual Fake       {cm[1][0]:>6}          {cm[1][1]:>6}  → Fake Accuracy: {fake_acc:.2f}%")
    
    try:
        auc = roc_auc_score(labels, probs)
        print(f"\nROC-AUC Score: {auc:.4f}")
    except Exception:
        pass
    print(f"{'='*60}\n")
    return cm, real_acc, fake_acc


def train():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # SIMPLE transforms - NO augmentation bias
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
    
    # Class weights to balance real/fake
    if args.class_weight == 'balanced':
        # Calculate class weights from dataset
        class_counts = torch.bincount(torch.tensor([label for _, label in train_ds]))
        class_weights = len(train_ds) / (2.0 * class_counts.float())
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Optimizer with lower LR for fine-tuning
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training
    best_acc = 0.0
    best_real_acc = 0.0
    best_fake_acc = 0.0
    patience_ctr = 0
    best_cm = None
    
    print(f"\nStarting training for {args.epochs} epochs...\n")
    print("🎯 GOAL: Balance Real & Fake accuracy (both should be >85%)\n")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f'Epoch [{epoch}/{args.epochs}]')
        
        for idx, (frames, labels) in enumerate(loop):
            frames = frames.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            loop.set_postfix(loss=f'{epoch_loss/total:.4f}', acc=f'{100.0*correct/total:.2f}%')
        
        train_acc = 100.0 * correct / total
        train_loss = epoch_loss / total
        
        val_loss, val_acc, v_labels, v_preds, v_probs = evaluate(model, val_loader, device, criterion)
        
        print(f"[Epoch {epoch:>3}] train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")
        
        if epoch % 2 == 0 or epoch == 1:
            cm, real_acc, fake_acc = print_metrics(v_labels, v_preds, v_probs, split=f'Epoch {epoch}')
            
            # Save if BOTH real and fake accuracy are good
            combined_score = (real_acc + fake_acc) / 2
            if combined_score > best_acc and abs(real_acc - fake_acc) < 15:  # Balanced
                best_acc = combined_score
                best_real_acc = real_acc
                best_fake_acc = fake_acc
                patience_ctr = 0
                best_cm = cm
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'best_real_acc': best_real_acc,
                    'best_fake_acc': best_fake_acc,
                }, args.output, _use_new_zipfile_serialization=False)
                print(f'  ✓ Best BALANCED model saved → Real: {best_real_acc:.2f}%, Fake: {best_fake_acc:.2f}%')
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f'\nEarly stopping at epoch {epoch}')
                    break
    
    # Final results
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Combined Accuracy: {best_acc:.2f}%")
    print(f"Best Real Accuracy: {best_real_acc:.2f}%")
    print(f"Best Fake Accuracy: {best_fake_acc:.2f}%")
    print(f"Balance: {abs(best_real_acc - best_fake_acc):.2f}% difference")
    print(f"{'='*60}")
    
    if best_cm is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=['Real', 'Fake'])
        disp.plot(ax=ax, colorbar=True, cmap='Blues')
        ax.set_title(f'Balanced Confusion Matrix', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('confusion_matrix_balanced.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Confusion matrix saved → confusion_matrix_balanced.png")
    
    print(f"\nTraining complete!")
    print(f"Model saved → {args.output}")


if __name__ == '__main__':
    train()
