import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ModelNetDataLoader
from models.model import point_transformer_38, point_transformer_50
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def plot_history(train_losses, test_losses, train_accs, test_accs, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_path, 'training_plot.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Point Transformer Training')
    parser.add_argument('--model', type=str, default='pointtransformer38', choices=['pointtransformer38', 'pointtransformer50'], help='model name')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    
    # Extra args to match bash script format (compat)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--decay_epoch', nargs='+', type=int, default=[70, 120])
    parser.add_argument('--val_epoch', type=int, default=1)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Dataset
    data_path = 'data/modelnet40_normal_resampled/' # UPDATE THIS PATH for Kaggle
    train_dataset = ModelNetDataLoader(root=data_path, split='train')
    test_dataset = ModelNetDataLoader(root=data_path, split='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    if args.model == 'pointtransformer38':
        model = point_transformer_38(num_classes=40)
    else:
        model = point_transformer_50(num_classes=40)
    
    model = model.to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_epoch, gamma=args.gamma)
    
    start_epoch = 0
    best_acc = 0.0
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    # Resume Logic
    last_ckpt_path = os.path.join(args.checkpoint_dir, 'last.pth')
    if args.resume and os.path.exists(last_ckpt_path):
        checkpoint = torch.load(last_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {start_epoch} with best acc {best_acc}")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epoch):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for points, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epoch} [Train]'):
            points, target = points.to(device).float(), target.to(device).squeeze().long()
            
            optimizer.zero_grad()
            # Input (B, N, 6) -> split to xyz (B, N, 3) and x (B, N, 6)
            xyz = points[:, :, :3]
            pred = model(xyz, points)
            
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # Validation
        if epoch % args.val_epoch == 0:
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for points, target in tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.epoch} [Test]'):
                    points, target = points.to(device).float(), target.to(device).squeeze().long()
                    xyz = points[:, :, :3]
                    pred = model(xyz, points)
                    loss = criterion(pred, target)
                    test_loss += loss.item()
                    _, predicted = torch.max(pred.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            avg_test_loss = test_loss / len(test_loader)
            avg_test_acc = 100 * correct / total
            test_losses.append(avg_test_loss)
            test_accs.append(avg_test_acc)
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.2f}%')

            # Save checkpoints
            save_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(save_state, os.path.join(args.checkpoint_dir, 'last.pth'))
            
            if avg_test_acc > best_acc:
                best_acc = avg_test_acc
                save_state['best_acc'] = best_acc
                torch.save(save_state, os.path.join(args.checkpoint_dir, 'best.pth'))
                print(f"New best model saved with accuracy: {best_acc:.2f}%")

        plot_history(train_losses, test_losses, train_accs, test_accs, args.checkpoint_dir)

if __name__ == '__main__':
    main()