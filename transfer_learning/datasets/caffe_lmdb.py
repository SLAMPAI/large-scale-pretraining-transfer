import torch
import lmdb
import caffe2
from caffe2.proto import caffe2_pb2
from PIL import Image
import io

import numpy as np

from joblib import Parallel, delayed

class CachedCaffeLMDB:
    def __init__(self, root, rank=0, world_size=1, transform=None, target_transform=None, max_readers=1):
        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py
        self.root = root
        self.env = lmdb.open(root,readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        self.nb = self.env.stat()["entries"]
        self.transform = transform
        self.target_transform = target_transform
        self.shuffled_data = np.arange(self.nb)
        np.random.shuffle(self.shuffled_data)
        indices = np.arange(self.nb)
        self.indices = indices[(indices % world_size) == rank]
        self.data = []
        import horovod.torch as hvd
        self.load()
        hvd.join()
    
    def __getitem__(self, i):
        img, target = self.data[i]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def load(self):
        print(f"Caching data into memory. Nb Of Examples: {len(self.indices)}")
        for ind in self.indices:
            ind = self.shuffled_data[ind]
            img, target = self.get(ind)
            self.data.append((img, target))
        print("Finished caching data into memory")

    def get(self, i):
        with self.env.begin(write=False) as txn:
            key = f"{i}"
            value = txn.get(key.encode("ascii"))
        image_data, label_data = caffe2_pb2.TensorProtos.FromString(value).protos
        imgbuf = image_data.string_data[0]
        target = (label_data.int32_data[0])
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img, target

    def __len__(self):
        return len(self.indices)


class CaffeLMDB:
    def __init__(self, root, rank=0, world_size=1, transform=None, target_transform=None, max_readers=1, cache=False):
        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py
        self.root = root
        self.env = lmdb.open(root,readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        self.nb = self.env.stat()["entries"]
        self.transform = transform
        self.target_transform = target_transform
        # self.shuffled_data = np.arange(self.nb)
        # self.sampler = torch.utils.data.distributed.DistributedSampler(self, num_replicas=world_size, rank=rank)
        self.cache = cache
        self.cache_dict = {}
    
    def __getitem__(self, i):
        if self.cache:
            if i in self.cache_dict:
                img, target = self.cache_dict[i]
                # print("use cache", len(self.cache_dict))
            else:
                img, target = self.get(i)
                self.cache_dict[i] = img, target
        else:
            img, target = self.get(i)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def get(self, i):
        with self.env.begin(write=False) as txn:
            key = f"{i}"
            value = txn.get(key.encode("ascii"))
        image_data, label_data = caffe2_pb2.TensorProtos.FromString(value).protos
        imgbuf = image_data.string_data[0]
        target = (label_data.int32_data[0])
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img, target

    def __len__(self):
        return (self.nb)

def caffe_lmdb_multilabel_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.envs = [lmdb.open(path,readonly=True, max_readers=1, lock=False, readahead=False, meminit=False) for path in dataset.paths]


class CaffeLMDBMultiLabel:
    def __init__(self, root, rank=0, world_size=1, transform=None, target_transform=None, max_readers=1, nb_classes=1000):
        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py
        from glob import glob
        import os
        import random
        from joblib import load
        self.nb_classes = nb_classes # ImageNet-21K my splits
        paths = root
        random.shuffle(paths)
        paths = [path for i, path in enumerate(paths) if (i % world_size) == rank]
        self.paths = paths
        self.envs = [lmdb.open(path,readonly=True, max_readers=1, lock=False, readahead=False, meminit=False) for path in paths]
        self.sizes = [env.stat()["entries"] for env in self.envs]
        self.inds = []
        for i, size in enumerate(self.sizes):
            for j in range(size):
                self.inds.append((i, j))
        # self.envs = None
        self.nb = len(self.inds)
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, i):
        img, target = self.get(i)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def get(self, i):
        env_id, example_id = self.inds[i]
        env = self.envs[env_id]
        with env.begin(write=False) as txn:
            key = f"{example_id}"
            value = txn.get(key.encode("ascii"))
        image_data, label_data = caffe2_pb2.TensorProtos.FromString(value).protos
        imgbuf = image_data.string_data[0]
        target = torch.Tensor(label_data.int32_data).long()
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        target_vector = torch.zeros(self.nb_classes).float()
        target_vector[target] = 1
        return img, target_vector

    def __len__(self):
        return (self.nb)
