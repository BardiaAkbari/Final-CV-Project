import os
import numpy as np
import torch
from torch.utils.data import Dataset

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    
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

class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size 
        self.cache = {} 

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[fn[0]]
        cls = np.array([cls]).astype(np.int32)
        
        # Read the point cloud
        # The dataset is comma separated: x,y,z,nx,ny,nz
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

        if self.normal_channel:
            point_set = point_set[:, 0:6]
        else:
            point_set = point_set[:, 0:3]

        # Resample to 1024 points
        # Using random choice if points > 1024, or with replacement if < 1024
        try:
            choice = np.random.choice(len(point_set), self.npoints, replace=True)
            point_set = point_set[choice, :]
        except ValueError:
            # Fallback if empty
            point_set = np.zeros((self.npoints, 3), dtype=np.float32)

        # Normalize
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        return point_set, cls

            