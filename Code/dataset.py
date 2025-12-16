import os
import numpy as np
import torch
from torch.utils.data import Dataset

def pc_normalize(pc):
    """
    Center and scale the point cloud to unit sphere.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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

class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        self.split = split

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        cls = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

if __name__ == '__main__':
    import torch
    import argparse
    
    # Mock args for testing
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_uniform_sample', type=bool, default=True, help='Use uniform sampling')
    parser.add_argument('--use_normals', type=bool, default=False, help='Use normals')
    parser.add_argument('--num_category', type=int, default=40, help='Category Number')
    args = parser.parse_args()

    print("Dataset ready to be initialized with path to ModelNet40")