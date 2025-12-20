import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import ModelNetDataLoader

# OFFICIAL-style Point Transformer classifiers
from models.model_v3 import (
    pointtransformer_cls38,
    pointtransformer_cls50
)

# ============================================================
# Utils
# ============================================================

def batch_to_pointcloud(points):
    """
    Convert (B, N, C) -> (p, x, o) for pointops
    """
    B, N, C = points.shape
    p = points[:, :, :3].contiguous().view(-1, 3)
    if C > 3:
        x = points[:, :, 3:].contiguous().view(-1, C - 3)
    else:
        x = None
    o = torch.arange(1, B + 1, device=points.device, dtype=torch.int32) * N
    return p, x, o


# ============================================================
# Plot Training History
# ============================================================

def plot_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['test_loss'], label='Test')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_oa'], label='Train')
    plt.plot(epochs, history['test_oa'], label='Test')
    plt.title('Overall Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_macc'], label='Train')
    plt.plot(epochs, history['test_macc'], label='Test')
    plt.title('Mean Class Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_path, 'training_plot.png'))
    plt.close()


# ============================================================
# Evaluation
# ============================================================

def calculate_metrics(loader, model, criterion, device, num_classes=40, desc="Eval"):
    model.eval()
    loss_sum = 0.0
    total_correct = 0
    total_seen = 0
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():
        for points, target in tqdm(loader, desc=desc, leave=False):
            points = points.to(device).float()
            target = target.to(device).squeeze().long()

            p, x, o = batch_to_pointcloud(points)
            pred = model([p, x, o])

            loss = criterion(pred, target)
            loss_sum += loss.item()

            _, predicted = pred.max(1)
            total_correct += (predicted == target).sum().item()
            total_seen += target.size(0)

            for i in range(target.size(0)):
                label = target[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    oa = 100 * total_correct / total_seen
    class_acc = np.divide(
        class_correct, class_total,
        out=np.zeros_like(class_correct),
        where=class_total > 0
    )
    macc = 100 * class_acc.mean()
    return loss_sum / len(loader), oa, macc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pointtransformer38')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--normal', action='store_true')
    parser.add_argument('--accum_iter', type=int, default=1)

    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--decay_epoch', nargs='+', type=int, default=[70, 120])
    parser.add_argument('--val_epoch', type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.checkpoint_dir, 'log.txt'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    logging.getLogger('').addHandler(logging.StreamHandler())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = 'data/modelnet40_normal_resampled'
    train_dataset = ModelNetDataLoader(data_path, split='train', normal_channel=args.normal)
    test_dataset = ModelNetDataLoader(data_path, split='test', normal_channel=args.normal)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    in_channels = 6 if args.normal else 3

    # MODEL
    if args.model == 'pointtransformer50':
        model = pointtransformer_cls50(c=in_channels, num_classes=40)
    else:
        model = pointtransformer_cls38(c=in_channels, num_classes=40)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.decay_epoch, gamma=args.gamma
    )

    best_oa = 0.0
    history = {
        'train_loss': [], 'test_loss': [],
        'train_oa': [], 'test_oa': [],
        'train_macc': [], 'test_macc': []
    }

    last_ckpt = os.path.join(args.checkpoint_dir, 'last.pth')
    if os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_oa = ckpt['best_oa']
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 0

    # TRAINING LOOP
    for epoch in range(start_epoch, args.epoch):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        class_correct = np.zeros(40)
        class_total = np.zeros(40)

        optimizer.zero_grad()

        for i, (points, target) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")
        ):
            points = points.to(device).float()
            target = target.to(device).squeeze().long()

            p, x, o = batch_to_pointcloud(points)
            pred = model([p, x, o])

            loss = criterion(pred, target) / args.accum_iter
            loss.backward()

            if (i + 1) % args.accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.accum_iter
            _, predicted = pred.max(1)

            total_correct += (predicted == target).sum().item()
            total_seen += target.size(0)

            for j in range(target.size(0)):
                label = target[j].item()
                class_correct[label] += (predicted[j] == label).item()
                class_total[label] += 1

        scheduler.step()

        train_oa = 100 * total_correct / total_seen
        train_macc = 100 * np.mean(
            np.divide(class_correct, class_total,
                      out=np.zeros_like(class_correct),
                      where=class_total > 0)
        )
        train_loss = total_loss / len(train_loader)

        if epoch % args.val_epoch == 0:
            test_loss, test_oa, test_macc = calculate_metrics(
                test_loader, model, criterion, device
            )

            logging.info(
                f"Epoch {epoch+1}: "
                f"Train Loss {train_loss:.4f}, "
                f"Test Loss {test_loss:.4f}, "
                f"Train OA {train_oa:.2f}%, "
                f"Test OA {test_oa:.2f}%, "
                f"Train mAcc {train_macc:.2f}%, "
                f"Test mAcc {test_macc:.2f}%"
            )

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_oa'].append(train_oa)
            history['test_oa'].append(test_oa)
            history['train_macc'].append(train_macc)
            history['test_macc'].append(test_macc)

            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_oa': best_oa,
                'history': history
            }
            torch.save(state, last_ckpt)

            if test_oa > best_oa:
                best_oa = test_oa
                state['best_oa'] = best_oa
                torch.save(state, os.path.join(args.checkpoint_dir, 'best_OA.pth'))
                logging.info(f"🔥 New best OA: {best_oa:.2f}%")

            plot_history(history, args.checkpoint_dir)


if __name__ == '__main__':
    main()
