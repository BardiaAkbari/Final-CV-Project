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

    plt.figure(figsize=(18, 5))
    
    # 1. Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    # 2. Overall Accuracy (OA)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_oa'], 'b-', label='Train OA')
    plt.plot(epochs, history['test_oa'], 'r-', label='Test OA')
    plt.title('Overall Accuracy (OA)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # 3. Mean Class Accuracy (mAcc)
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_macc'], 'b--', label='Train mAcc')
    plt.plot(epochs, history['test_macc'], 'r--', label='Test mAcc')
    plt.title('Mean Class Accuracy (mAcc)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_path, 'training_plot.png'))
    plt.close()

def calculate_metrics(loader, model, criterion, device, num_classes=40, desc="Eval"):
    loss_accum = 0.0
    
    # OA Metrics
    total_correct = 0
    total_samples = 0
    
    # mAcc Metrics
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    model.eval()
    with torch.no_grad():
        for points, target in tqdm(loader, desc=desc, leave=False):
            points, target = points.to(device).float(), target.to(device).squeeze().long()
            
            pred = model(points)
            loss = criterion(pred, target)
            loss_accum += loss.item()
            
            # Get predictions
            _, predicted = torch.max(pred.data, 1)
            
            # Update OA
            total_samples += target.size(0)
            total_correct += (predicted == target).sum().item()
            
            # Update mAcc
            c = (predicted == target).squeeze()
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    avg_loss = loss_accum / len(loader)
    oa = 100 * total_correct / total_samples
    
    # Calculate mAcc (handle division by zero for classes that might not appear in a batch)
    class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc[i] = 100 * class_correct[i] / class_total[i]
    macc = np.mean(class_acc)
    
    return avg_loss, oa, macc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pointtransformer38', help='model name')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    
    # Arguments for compatibility with bash script
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--decay_epoch', nargs='+', type=int, default=[70, 120])
    parser.add_argument('--val_epoch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    # Setup Logger
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
    best_oa = 0.0
    best_macc = 0.0
    history = {'train_loss': [], 'test_loss': [], 'train_oa': [], 'test_oa': [], 'train_macc': [], 'test_macc': []}
    
    last_ckpt = os.path.join(args.checkpoint_dir, 'last.pth')
    if os.path.exists(last_ckpt):
        logging.info(f"Loading checkpoint from {last_ckpt}")
        checkpoint = torch.load(last_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_oa = checkpoint['best_oa']
        if 'history' in checkpoint:
            history = checkpoint['history']

    # Training Loop
    for epoch in range(start_epoch, args.epoch):
        model.train()
        train_loss = 0.0
        
        # --- TRAIN ONE EPOCH ---
        # Note: We calculate simplistic OA during training loop for speed, 
        # but for history we can run a quick eval or just use the moving average.
        # To save time on Kaggle, we use the moving average from the training loop for Train Metrics.
        correct_train = 0
        total_train = 0
        class_correct = np.zeros(40)
        class_total = np.zeros(40)
        
        for points, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epoch} [Train]'):
            points, target = points.to(device).float(), target.to(device).squeeze().long()
            optimizer.zero_grad()
            pred = model(points)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # mAcc tracking for train
            c = (predicted == target).squeeze()
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        train_oa = 100 * correct_train / total_train
        
        # Calc Train mAcc
        train_class_acc = np.zeros(40)
        for i in range(40):
            if class_total[i] > 0:
                train_class_acc[i] = 100 * class_correct[i] / class_total[i]
        train_macc = np.mean(train_class_acc)

        # --- VALIDATION ---
        if epoch % args.val_epoch == 0 or epoch == args.epoch - 1:
            test_loss, test_oa, test_macc = calculate_metrics(test_loader, model, criterion, device, num_classes=40, desc=f'Epoch {epoch+1} [Test]')
            
            logging.info(f'Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Train OA {train_oa:.2f}% mAcc {train_macc:.2f}% | Test OA {test_oa:.2f}% Test mAcc {test_macc:.2f}%')
            
            # Update History
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(test_loss)
            history['train_oa'].append(train_oa)
            history['test_oa'].append(test_oa)
            history['train_macc'].append(train_macc)
            history['test_macc'].append(test_macc)
            
            # Save Checkpoints
            save_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_oa': best_oa,
                'history': history
            }
            torch.save(save_state, last_ckpt)
            
            # Save Best OA model
            if test_oa > best_oa:
                best_oa = test_oa
                save_state['best_oa'] = best_oa
                torch.save(save_state, os.path.join(args.checkpoint_dir, 'best_OA.pth'))
                logging.info(f'New Best OA: {best_oa:.2f}%')
            
            # Plot
            plot_history(history, args.checkpoint_dir)

if __name__ == '__main__':
    main()