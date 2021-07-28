try:
    import horovod.torch as hvd
    HOROVOD = True
except Exception:
    HOROVOD = False
import os
import torch
import torchvision.transforms as transforms
import torchvision
from transfer_learning.datasets.builder import build_dataset

from .dataloader_threads import DataLoader as DataLoaderThreads

def build_dataloader(config):
    train_batch_size = config.optim.train_local_batch_size * config.optim.gradient_accumulate
    val_batch_size = config.optim.val_local_batch_size
    dataset_type = config.data.type
    backend = config.data.backend
    workers = config.data.workers
    train_dir = config.data.train_dir
    val_dir = config.data.val_dir
    augmentation = config.data.augmentation
    image_size = config.data.image_size
    if not image_size:
        image_size = 224
    if not augmentation:
        augmentation = "random_resized_crop"
    if dataset_type == "tfrecords" and backend == "dali":
        from glob import glob
        from transfer_learning.dataloaders.dali import build_dali_data_loader_from_tfrecords
        assert (augmentation != "rand_augment"), ("rand_augment not supported for tfrecords/dali")

        class Loader:

            def __init__(self, loader):
                self.loader = loader

            def __iter__(self):
                for x, y in self.loader:
                    yield x, y
            def __len__(self):
                return len(self.loader)

        index_folder = os.path.join(train_dir, "idx_files")
        tfrecord_files = glob(os.path.join(train_dir, "*"))
        tfrecord_files = list(filter(os.path.isfile, tfrecord_files))
        idx_files = [
            os.path.join(index_folder, os.path.basename(path) + ".idx")
            for path in tfrecord_files
        ]
        mean = [255*0.485, 255*0.456, 255*0.406]
        std = [255*0.229, 255*0.224, 255*0.225]
        train_loader = build_dali_data_loader_from_tfrecords(
            tfrecord_files,
            idx_files,
            device_id=hvd.local_rank(),
            rank=hvd.rank(),
            world_size=hvd.size(),
            prefetch_queue_depth=16,
            batch_size=train_batch_size,
            workers=workers,
            image_size=image_size,
            shuffle=True,
            data_augmentation=True,
            mean=mean,
            std=std,
            shuffle_buffer_size=512,
        )
        train_loader = Loader(train_loader)
        if val_dir:
            tfrecord_files = glob(os.path.join(val_dir, "*"))
            tfrecord_files = list(filter(os.path.isfile, tfrecord_files))
            idx_files = [
                os.path.join(index_folder, os.path.basename(path) + ".idx")
                for path in tfrecord_files
            ]
            val_loader = build_dali_data_loader_from_tfrecords(
                tfrecord_files,
                idx_files,
                device_id=hvd.local_rank(),
                rank=hvd.rank(),
                world_size=hvd.size(),
                prefetch_queue_depth=16,
                batch_size=val_local_batch_size,
                workers=workers,
                image_size=image_size,
                shuffle=False,
                data_augmentation=False,
                mean=mean,
                std=std,
                shuffle_buffer_size=512,
            )
            val_loader = Loader(val_loader)
        else:
            val_loader = None
    elif dataset_type == "lmdb" and backend == "dali":
        from transfer_learning.dataloaders.dali import build_dali_data_loader_from_caffe_lmdb
        from glob import glob
        assert (augmentation != "rand_augment"), ("rand_augment not supported for lmdb/dali")
        mean = [255*0.485, 255*0.456, 255*0.406]
        std = [255*0.229, 255*0.224, 255*0.225]
        label_type = 0 # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html?highlight=label_type
        train_loader = build_dali_data_loader_from_caffe_lmdb(
            path=glob(train_dir),
            device_id=hvd.local_rank(),
            rank=hvd.rank(),
            world_size=hvd.size(),
            prefetch_queue_depth=16,
            batch_size=train_batch_size,
            workers=workers,
            image_size=image_size,
            shuffle=True,
            data_augmentation=True,
            mean=mean,
            std=std,
            dali_device="cpu",
            label_type=label_type,
        )
        val_loader = build_dali_data_loader_from_caffe_lmdb(
            path=glob(val_dir),
            device_id=hvd.local_rank(),
            rank=hvd.rank(),
            world_size=hvd.size(),
            prefetch_queue_depth=16,
            batch_size=val_batch_size,
            workers=workers,
            image_size=image_size,
            shuffle=False,
            data_augmentation=False,
            mean=mean,
            std=std,
            dali_device="cpu",
            label_type=label_type,
        )
    elif backend in ("native", "threads"):
        backend_cls = DataLoaderThreads if backend == "threads" else torch.utils.data.DataLoader
        train_dataset, val_dataset = build_dataset(config)
        if HOROVOD:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        else:
            train_sampler = None
            val_sampler = None
        train_loader = backend_cls(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=workers,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            multiprocessing_context="forkserver",
        )
        val_loader = backend_cls(
            val_dataset,
            batch_size=val_batch_size,
            num_workers=workers,
            sampler=val_sampler,
            shuffle=False,
            multiprocessing_context="forkserver",
        )
    else:
        raise ValueError("Cannot recognize dataloader setup.")
    return train_loader, val_loader
