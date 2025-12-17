import argparse
import os
import logging
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ModelNetDataLoader
from models.model import point_transformer_38, point_transformer_50
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def plot_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['test_acc'], 'r-', label='Test Acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_path, 'training_plot.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pointtransformer38', help='model name')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    
    # Dummy args to match your train.sh request style
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--decay_epoch', nargs='+', type=int, default=[70, 120])
    parser.add_argument('--val_epoch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()

    # Logging Setup
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    logging.basicConfig(
        filename=os.path.join(args.checkpoint_dir, 'log.txt'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Loading
    data_path = 'data/modelnet40_normal_resampled/'
    train_dataset = ModelNetDataLoader(root=data_path, split='train', normal_channel=args.normal)
    test_dataset = ModelNetDataLoader(root=data_path, split='test', normal_channel=args.normal)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model Loading
    in_channels = 6 if args.normal else 3
    if args.model == 'pointtransformer50':
        model = point_transformer_50(num_classes=40, in_channels=in_channels)
    else:
        model = point_transformer_38(num_classes=40, in_channels=in_channels)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_epoch, gamma=args.gamma)

    # Resume Logic
    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    last_ckpt = os.path.join(args.checkpoint_dir, 'last.pth')
    if os.path.exists(last_ckpt):
        logging.info(f"Loading checkpoint from {last_ckpt}")
        checkpoint = torch.load(last_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        if 'history' in checkpoint:
            history = checkpoint['history']

    # Training Loop
    for epoch in range(start_epoch, args.epoch):
        # --- TRAIN ---
        model.train()
        train_loss_accum = 0.0
        correct_train = 0
        total_train = 0
        
        for points, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epoch} [Train]'):
            points, target = points.to(device).float(), target.to(device).squeeze().long()
            optimizer.zero_grad()
            pred = model(points)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
        scheduler.step()
        
        train_loss = train_loss_accum / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # --- TEST ---
        if epoch % args.val_epoch == 0 or epoch == args.epoch - 1:
            model.eval()
            test_loss_accum = 0.0
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for points, target in tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.epoch} [Test]'):
                    points, target = points.to(device).float(), target.to(device).squeeze().long()
                    pred = model(points)
                    loss = criterion(pred, target)
                    test_loss_accum += loss.item()
                    _, predicted = torch.max(pred.data, 1)
                    total_test += target.size(0)
                    correct_test += (predicted == target).sum().item()

            test_loss = test_loss_accum / len(test_loader)
            test_acc = 100 * correct_test / total_test
            
            logging.info(f'Epoch {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | Test Loss {test_loss:.4f} Acc {test_acc:.2f}%')
            
            # Update History
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            # Save Last
            save_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }
            torch.save(save_state, last_ckpt)
            
            # Save Best
            if test_acc > best_acc:
                best_acc = test_acc
                save_state['best_acc'] = best_acc
                torch.save(save_state, os.path.join(args.checkpoint_dir, 'best.pth'))
                logging.info(f'New Best Accuracy: {best_acc:.2f}%')

            # Plot
            plot_history(history, args.checkpoint_dir)

if __name__ == '__main__':
    main()