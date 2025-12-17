import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNetDataLoader(Dataset):
    def __init__(self, root, split='train', num_points=1024, uniform=False, process_data=False):
        self.root = root
        self.npoints = num_points
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        
        # Download logic can be added here, but usually on Kaggle you add the dataset via UI
        # Assuming data is at root/modelnet40_normal_resampled/
        
        self.cat = [line.strip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        
        shape_ids = {}
        shape_ids['train'] = [line.strip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.strip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        
        assert split in ['train', 'test']
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i in range(len(shape_ids[split]))]
        self.cache = {} 
        self.cache_size = 20000

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
                
        return point_set, cls

    def __getitem__(self, index):
        point_set, cls = self._get_item(index)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        # Augmentation
        # jitter, rotate, scale
        return point_set, cls

def farthest_point_sample(point, npoint):
    """
    Input:
        point: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point