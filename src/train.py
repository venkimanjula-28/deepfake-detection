import argparse
import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

from model import get_model
from dataset import DeepFakeDataset, CachedDeepFakeDataset


def parse_args():
    parser = argparse.ArgumentParser(description='DeepFake Detection Training - Fast & Accurate')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'efficientnet_b0'], default='resnet18')
    parser.add_argument('--frames', type=int, default=6, help='Frames per clip')
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output', type=str, default='checkpoint_final.pth')
    parser.add_argument('--use-lstm', action='store_true', default=False)
    parser.add_argument('--freeze-backbone', action='store_true', default=False)
    parser.add_argument('--unfreeze-epoch', type=int, default=999)
    parser.add_argument('--num-workers', type=int, default=0, help='0 = main process (faster on Windows CPU)')
    parser.add_argument('--img-size',    type=int, default=112, help='Image size used during caching')
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=999)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    return parser.parse_args()


def evaluate_model(model, loader, device, criterion):
    """Evaluate model and return loss, accuracy, all labels, all predictions, all probs."""
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
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of fake class

    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    return avg_loss, acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics(labels, preds, probs, split='Validation'):
    """Print confusion matrix and classification report."""
    classes = ['Real', 'Fake']
    print(f"\n{'='*50}")
    print(f"  {split} Metrics")
    print(f"{'='*50}")

    # Classification report
    print(classification_report(labels, preds, target_names=classes, digits=4))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(f"              Predicted Real  Predicted Fake")
    print(f"Actual Real       {cm[0][0]:>6}          {cm[0][1]:>6}")
    print(f"Actual Fake       {cm[1][0]:>6}          {cm[1][1]:>6}")

    # ROC-AUC
    try:
        auc = roc_auc_score(labels, probs)
        print(f"\nROC-AUC Score: {auc:.4f}")
    except Exception:
        pass
    print(f"{'='*50}\n")
    return cm


def save_confusion_matrix(cm, epoch, save_path='confusion_matrix.png'):
    """Save confusion matrix as image."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(f'Confusion Matrix (Epoch {epoch})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_training_plots(train_losses, val_losses, train_accs, val_accs, save_path='training_visualization.png'):
    """Save loss and accuracy curves."""
    epochs = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'b-o', markersize=4, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-o', markersize=4, label='Val Loss', linewidth=2)
    ax1.set_title('Loss Curves', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-o', markersize=4, label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-o', markersize=4, label='Val Acc', linewidth=2)
    ax2.set_title('Accuracy Curves', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim([40, 105])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved → {save_path}")


def train():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Transforms – stronger augmentation for better generalization
    # ------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ------------------------------------------------------------------
    # Dataset – auto-detect cached vs raw
    # ------------------------------------------------------------------
    print(f"Loading dataset from {args.data_dir}...")

    # Check if cached .npy files exist
    _sample_cls = os.path.join(args.data_dir, 'real')
    _has_cache  = os.path.exists(_sample_cls) and any(
        f.endswith('.npy') for f in os.listdir(_sample_cls)
    ) if os.path.isdir(_sample_cls) else False

    DatasetClass = CachedDeepFakeDataset if _has_cache else DeepFakeDataset
    print(f"Dataset mode: {'CACHED (.npy)' if _has_cache else 'RAW (images/videos)'}")

    full_ds = DatasetClass(args.data_dir, num_frames=args.frames, transform=None)
    print(f"Total samples: {len(full_ds)}")

    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_indices, val_indices = random_split(
        range(len(full_ds)), [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds = torch.utils.data.Subset(
        DatasetClass(args.data_dir, num_frames=args.frames, transform=train_transform),
        train_indices.indices
    )
    val_ds = torch.utils.data.Subset(
        DatasetClass(args.data_dir, num_frames=args.frames, transform=val_transform),
        val_indices.indices
    )

    # num_workers=0 avoids Windows multiprocessing overhead on CPU
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Batch: {args.batch_size} | Workers: {args.num_workers}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = get_model(backbone=args.backbone, pretrained=True,
                      freeze_backbone=args.freeze_backbone, use_lstm=args.use_lstm)
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # ------------------------------------------------------------------
    # Loss / Optimizer / Scheduler
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    # CosineAnnealing gives smoother convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_acc        = 0.0
    patience_ctr    = 0
    train_losses, val_losses   = [], []
    train_accs,   val_accs     = [], []
    best_cm = None

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):

        # Progressive unfreezing
        if args.freeze_backbone and epoch == args.unfreeze_epoch:
            print(f"\n>>> Unfreezing backbone at epoch {epoch}")
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                'params': [p for p in model.backbone.parameters() if p.requires_grad],
                'lr': args.lr * 0.1
            })

        # --- Train ---
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f'Epoch [{epoch}/{args.epochs}]', unit='batch')

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
            total      += labels.size(0)
            _, preds    = outputs.max(1)
            correct    += preds.eq(labels).sum().item()
            loop.set_postfix(loss=f'{epoch_loss/total:.4f}',
                             acc=f'{100.0*correct/total:.2f}%')

        train_acc  = 100.0 * correct / total
        train_loss = epoch_loss / total

        # --- Validate ---
        val_loss, val_acc, v_labels, v_preds, v_probs = evaluate_model(
            model, val_loader, device, criterion
        )

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[Epoch {epoch:>3}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        # Print confusion matrix + classification report every 5 epochs or on best
        if val_acc > best_acc or epoch % 5 == 0 or epoch == args.epochs:
            cm = print_metrics(v_labels, v_preds, v_probs, split=f'Epoch {epoch} Validation')

        if val_acc > best_acc:
            best_acc = val_acc
            patience_ctr = 0
            best_cm = confusion_matrix(v_labels, v_preds)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': vars(args)
            }, args.output)
            save_confusion_matrix(best_cm, epoch, save_path='confusion_matrix_best.png')
            print(f'  ✓ Best model saved  →  val_acc={best_acc:.2f}%  |  confusion_matrix_best.png saved')
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f'\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)')
                break

    # ------------------------------------------------------------------
    # Final evaluation on entire validation set with best model
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  FINAL RESULTS  (Best Val Acc = {best_acc:.2f}%)")
    print(f"{'='*50}")

    # Load best weights for final report
    ckpt = torch.load(args.output, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    _, final_acc, f_labels, f_preds, f_probs = evaluate_model(model, val_loader, device, criterion)
    print_metrics(f_labels, f_preds, f_probs, split='FINAL Validation (Best Model)')

    # Save plots
    save_training_plots(train_losses, val_losses, train_accs, val_accs)
    if best_cm is not None:
        save_confusion_matrix(best_cm, ckpt['epoch'], 'confusion_matrix_best.png')
        print("Confusion matrix saved → confusion_matrix_best.png")

    print(f"\nTraining complete.  Best Val Acc = {best_acc:.2f}%")
    print(f"Model saved → {args.output}")


if __name__ == '__main__':
    train()
