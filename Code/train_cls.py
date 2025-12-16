import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from dataset import ModelNetDataLoader
from point_transformer import PointTransformerCls

def parse_args():
    parser = argparse.ArgumentParser('Point Transformer Classification')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training')
    parser.add_argument('--epoch', default=200, type=int, help='Epoch to run')
    parser.add_argument('--learning_rate', default=0.05, type=float, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--data_path', type=str, default='data/modelnet40_normal_resampled', help='Data path')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def main(args):
    # Logging Setup
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = 'log/'
    if args.log_dir is None:
        exp_dir = exp_dir + timestr
    else:
        exp_dir = exp_dir + args.log_dir
        
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (exp_dir, 'log_train'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.info('PARAMETER ...')
    logger.info(args)

    # Data Loading
    logger.info('Load Dataset ...')
    train_dataset = ModelNetDataLoader(root=args.data_path, args=args, split='train')
    test_dataset = ModelNetDataLoader(root=args.data_path, args=args, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    num_class = args.num_category
    input_dim = 6 if args.use_normals else 3
    
    # Model Setup
    classifier = PointTransformerCls(num_class=num_class, num_point=args.num_point, input_dim=input_dim, k=16).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    # Optimizer Setup
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_epoch = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    # Training Loop
    logger.info('Start Training ...')
    for epoch in range(global_epoch, args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        classifier.train()
        mean_correct = []
        
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            
            # Data Augmentation: Rotation and Jitter
            points = points.data.numpy()
            points = np.transpose(points, (0, 2, 1)) # B, C, N
            
            # Simple random rotation around Y axis
            theta = np.random.uniform(0, 2 * np.pi, size=points.shape[0])
            rotation_matrix = np.zeros((points.shape[0], 3, 3))
            rotation_matrix[:, 0, 0] = np.cos(theta)
            rotation_matrix[:, 0, 2] = np.sin(theta)
            rotation_matrix[:, 1, 1] = 1
            rotation_matrix[:, 2, 0] = -np.sin(theta)
            rotation_matrix[:, 2, 2] = np.cos(theta)
            
            points[:, :3, :] = np.matmul(rotation_matrix, points[:, :3, :]) # Apply to xyz
            if args.use_normals:
                 points[:, 3:, :] = np.matmul(rotation_matrix, points[:, 3:, :]) # Apply to normals

            points = torch.Tensor(points).cuda()
            target = target[:, 0].long().cuda()

            # Forward pass
            pred, _ = classifier(points)
            loss = criterion(pred, target)
            
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)
        scheduler.step()

        # Evaluation Loop
        with torch.no_grad():
            classifier.eval()
            mean_correct = []
            class_acc = np.zeros((num_class, 3))
            
            for batch_id, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.permute(0, 2, 1).float().cuda()
                target = target[:, 0].long().cuda()
                
                pred, _ = classifier(points)
                pred_choice = pred.data.max(1)[1]
                
                for cat in np.unique(target.cpu()):
                    classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                    class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                    class_acc[cat, 1] += 1

                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))

            class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
            class_acc = np.mean(class_acc[:, 2])
            instance_acc = np.mean(mean_correct)

            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if class_acc >= best_class_acc:
                best_class_acc = class_acc

            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Best Class Accuracy: %f' % (best_instance_acc, best_class_acc))

    logger.info('End of training...')