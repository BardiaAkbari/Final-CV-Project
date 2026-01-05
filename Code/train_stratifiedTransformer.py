import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .dataset import ModelNetDataLoader
from .models.stratified_transformer import Stratified

from .models.pytorch_pointops import ball_query


# ============================================================
# Utils
# ============================================================

def build_batch_inputs(points):
    """
    Convert (B, N, 3) -> Stratified Transformer inputs

    Returns:
        feats:  (BN, 3)
        xyz:    (BN, 3)
        offset: (B,)
        batch:  (BN,)
    """
    B, N, _ = points.shape
    xyz = points.view(-1, 3)
    feats = xyz.clone()

    batch = (
        torch.arange(B, device=points.device)
        .view(B, 1)
        .repeat(1, N)
        .view(-1)
    )

    offset = torch.arange(
        1, B + 1,
        device=points.device,
        dtype=torch.int32
    ) * N

    return feats, xyz, offset, batch


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

@torch.no_grad()
def calculate_metrics(loader, model, criterion, device, num_classes=40):
    model.eval()
    loss_sum = 0.0
    total_correct = 0
    total_seen = 0
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for points, target in tqdm(loader, desc="Eval", leave=False):
        points = points.to(device).float()
        target = target.to(device).long()

        feats, xyz, offset, batch = build_batch_inputs(points)

        neighbor_idx = ball_query(
            radius=0.1,       # must match KPConv config
            nsample=32,
            xyz=xyz,
            new_xyz=xyz,
            batch_x=batch,
            batch_y=batch
        )

        pred = model(feats, xyz, offset, batch, neighbor_idx)
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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--decay_epoch', nargs='+', type=int, default=[80, 110])
    parser.add_argument('--resume_ckpt', type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    last_ckpt = os.path.join(args.checkpoint_dir, 'last.pth')

    logging.basicConfig(
        filename=os.path.join(args.checkpoint_dir, 'log.txt'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    logging.getLogger('').addHandler(logging.StreamHandler())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============================================================
    # Dataset
    # ============================================================

    data_path = 'data/modelnet40_normal_resampled'
    train_dataset = ModelNetDataLoader(data_path, split='train', normal_channel=False)
    test_dataset = ModelNetDataLoader(data_path, split='test', normal_channel=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # ============================================================
    # Model
    # ============================================================

    model = StratifiedCls(
        depths=[2, 2, 6, 2],
        channels=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        window_sizes=[0.4, 0.8, 1.6, 3.2],
        grid_sizes=[0.04, 0.08, 0.16, 0.32],
        quant_sizes=[0.02, 0.04, 0.08, 0.16],
        num_classes=40
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.endswith('bias')
            or 'norm' in name.lower()
            or 'bn' in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay, 'weight_decay': args.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ],
        lr=args.lr
    )


    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.decay_epoch,
        gamma=args.gamma
    )

    # ============================================================
    # Resume
    # ============================================================

    start_epoch = 0
    best_oa = 0.0
    history = {
        'train_loss': [], 'test_loss': [],
        'train_oa': [], 'test_oa': [],
        'train_macc': [], 'test_macc': []
    }

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        history = ckpt.get('history', history)
        best_oa = ckpt.get('best_oa', 0.0)
        start_epoch = ckpt['epoch'] + 1

    # ============================================================
    # Training Loop
    # ============================================================

    for epoch in range(start_epoch, args.epoch):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        class_correct = np.zeros(40)
        class_total = np.zeros(40)

        for points, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}"):
            points = points.to(device).float()
            target = target.to(device).long()

            feats, xyz, offset, batch = build_batch_inputs(points)

            neighbor_idx = ball_query(
                radius=0.1,
                nsample=32,
                xyz=xyz,
                new_xyz=xyz,
                batch_x=batch,
                batch_y=batch
            )

            pred = model(feats, xyz, offset, batch, neighbor_idx)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = pred.max(1)

            total_correct += (predicted == target).sum().item()
            total_seen += target.size(0)

            for i in range(target.size(0)):
                label = target[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

        scheduler.step()

        train_oa = 100 * total_correct / total_seen
        train_macc = 100 * np.mean(
            np.divide(
                class_correct,
                class_total,
                out=np.zeros_like(class_correct),
                where=class_total > 0
            )
        )
        train_loss = total_loss / len(train_loader)

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

        plot_history(history, args.checkpoint_dir)


if __name__ == '__main__':
    main()
