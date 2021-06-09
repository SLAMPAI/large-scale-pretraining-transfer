# Adapted from :
# - https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py
# - https://docs.nvidia.com/deeplearning/dali/archives/dali_012_beta/dali-master-branch-user-guide/docs/examples/dataloading_tfrecord.html
import torch
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import lmdb
import caffe2
from caffe2.proto import caffe2_pb2
from PIL import Image
import io

import numpy as np

from joblib import Parallel, delayed

class BasePipeline(Pipeline):
    def __init__(
        self,
        input,
        batch_size=128,
        workers=4,
        device_id=0,
        image_size=224,
        no_data_augmentation_base_image_size=256,
        dali_device="cpu",
        rank=0,
        world_size=1,
        data_augmentation=False,
        mean=[0, 0, 0],
        std=[1, 1, 1],
    ):
        """
        Basic DALI pipeline implementing usual ImageNet preprocessing
        and data augmentation

        input: reader 
            depends on whether it's a file loader, TFRecords, or LMDB)

        batch_size: int
            batch size to use (per GPU)
        
        workers: int
            number of workers for data loading

        device_id: int
            gpu device id. For horovod, this should be `hvd.local_rank()`

        image_size: int
            image size, defaults to 224 (ImageNet setting)
            the pipeline when data augmentation is enabled is 
            to take a random `(image_size, image_size)` crop.
            When data augmentation is not enabled, we resize
            the shorter size of the image to `no_data_augmentation_base_image_size`
            then we take a center crop to get `(image_size, image_size)`

        no_data_augmentation_base_image_size: int
            see image_size parameter doc
    
        dali_device: cpu/gpu (default:cpu)
            whether to use cpu or gpu for data processing and augmentation

        rank: int
            rank (used for sharding). For horovod, this should be `hvd.rank()`

        world_size: int
            world_size (used for sharding). For horovod, this should be `hvd.size()`

        data_augmentation: bool
            whether to use data augmentation.  see the description of
            `image_size` parameter.

        mean: list of 3 elements
            [mean_r, mean_g, mean_b] for standardization

        std: list of 3 elements
            [std_r, std_g, std_b] for standardization
        """
        super().__init__(batch_size, workers, device_id)
        self.input = input
        self.data_augmentation = data_augmentation
        self.dali_device = dali_device
        if dali_device == "cpu":
            self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        else:
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.ImageDecoder(
                device="mixed",
                output_type=types.RGB,
                device_memory_padding=211025920,
                host_memory_padding=140544512,
            )

        if data_augmentation:
            self.res = ops.RandomResizedCrop(
                device=dali_device,
                size=(image_size, image_size),
                interp_type=types.INTERP_LINEAR,
                random_aspect_ratio=[0.75, 4.0 / 3.0],
                random_area=[0.08, 1.0],
                num_attempts=100,
            )

            self.cmnp = ops.CropMirrorNormalize(
                device=dali_device,
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(image_size, image_size),
                image_type=types.RGB,
                mean=mean,
                std=std,
            )
            self.coin = ops.CoinFlip(probability=0.5)
        else:
            self.res = ops.Resize(
                device=dali_device, resize_shorter=no_data_augmentation_base_image_size
            )
            self.cmnp = ops.CropMirrorNormalize(
                device=dali_device,
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(image_size, image_size),
                image_type=types.RGB,
                mean=mean,
                std=std,
            )

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        labels = self.labels
        images = self.decode(self.jpegs)
        images = self.res(images)
        mirror = self.coin() if self.data_augmentation else None
        if self.dali_device == "gpu":
            images = images.gpu()
            labels =  self.labels.gpu()
        output = self.cmnp(images, mirror=mirror)
        return (output, labels)


class DALIDataLoader(object):

    def gen_wrapper(pipeline):
        for data in pipeline:
            input = data[0]["data"]
            target = torch.reshape(data[0]["label"], [-1]).cuda().long()
            yield input, target
        pipeline.reset()

    def __init__(self, pipeline, batch_size):
        self.pipeline = pipeline
        self.batch_size = batch_size

    def __iter__(self):
        return DALIDataLoader.gen_wrapper(self.pipeline)

    def __len__(self):
        return self.pipeline._size // self.batch_size


def build_dali_data_loader_from_image_folder(
    data_dir,
    rank=0,
    world_size=1,
    prefetch_queue_depth=1,
    shuffle=False,
    **params,
):
    """
    This returns a DALI data loader from an image folder.

    data_dir: str
        image folder. Each subfolder `data_dir` contain
        the images of a class.

    rank: int
        rank (used for sharding). For horovod, this should be `hvd.rank()`

    world_size: int
        world_size (used for sharding). For horovod, this should be `hvd.size()`
    
    prefetch_queue_depth: int
        number of batches to prefetch.
        See doc of https://docs.nvidia.com/deeplearning/dali/user-guide/docs/advanced_topics.html#prefetching-queue-depth
        for more explanation

    shuffle: bool
        whether to shuffle data

    For the rest of the params (**params), they are redirected to `BasePipeline`,
    see `BasePipeline` for a doc about them.


    Returns
    -------

    DALIDataLoader object

    """
    input = ops.FileReader(
        file_root=data_dir,
        shard_id=rank,
        num_shards=world_size,
        prefetch_queue_depth=prefetch_queue_depth,
        random_shuffle=shuffle,
    )
    pipeline = BasePipeline(input=input, **params)
    pipeline.build()
    batch_size = params["batch_size"]
    dataloader = DALIClassificationIterator(pipeline, reader_name="Reader")
    dataloader = DALIDataLoader(dataloader, batch_size=batch_size)
    return dataloader


def build_dali_data_loader_from_tfrecords(
    tfrecord_files,
    idx_files,
    rank=0,
    world_size=1,
    prefetch_queue_depth=1,
    shuffle=False,
    shuffle_buffer_size=8192,
    **params,
):
    """
    This returns a DALI data loader from TFRecords.
    
    tfrecord_files: list of str
        list of tfrecord files

    idx_files: list of str
        list of tfrecord index files for each
        tfrecord. That is, `idx_files[i]` should be
        the path of the index file of `tfrecord_files[i]`.
        See <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/dataloading_tfrecord.html>
        for doc about how to create index files.

    rank: int
        rank (used for sharding). For horovod, this should be `hvd.rank()`

    world_size: int
        world_size (used for sharding). For horovod, this should be `hvd.size()`
    
    prefetch_queue_depth: int
        number of batches to prefetch.
        See doc of https://docs.nvidia.com/deeplearning/dali/user-guide/docs/advanced_topics.html#prefetching-queue-depth
        for more explanation

    shuffle: bool
        whether to shuffle data
    
    shuffle_buffer_size: int
        buffer size used for shuffling the data
        
    For the rest of the params (**params), they are redirected to `BasePipeline`,
    see `BasePipeline` for a doc about them.


    Returns
    -------

    DALIDataLoader object

    """
    features = {
        "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
        "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
        # "image/class/text": tfrec.FixedLenFeature([], tfrec.string, ""),
    }
    input = ops.TFRecordReader(
        # more info about the params here:
        # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html#nvidia.dali.ops.TFRecordReader
        path=tfrecord_files,
        index_path=idx_files,
        features=features,
        prefetch_queue_depth=prefetch_queue_depth,
        shard_id=rank,
        num_shards=world_size,
        random_shuffle=shuffle,
        initial_fill=shuffle_buffer_size,
    )

    def wrapped_input(name=None):
        dictionary = input()
        image = dictionary["image/encoded"]
        label = dictionary["image/class/label"]
        return image, label

    pipeline = BasePipeline(input=wrapped_input, **params)
    pipeline.build()
    nb_examples_per_epoch = 0
    for idx_file in idx_files:
        with open(idx_file) as fd:
            nb_examples_per_epoch += len(fd.readlines())
    nb_examples_per_epoch = nb_examples_per_epoch // world_size
    batch_size = params["batch_size"]
    dataloader = DALIClassificationIterator(pipeline, size=nb_examples_per_epoch)
    # dataloader = DALIClassificationIterator(pipeline, reader_name="Reader")
    dataloader = DALIDataLoader(dataloader, batch_size=batch_size)
    print(nb_examples_per_epoch, len(dataloader))
    return dataloader


def build_dali_data_loader_from_caffe_lmdb(
    path,
    rank=0,
    world_size=1,
    prefetch_queue_depth=1,
    shuffle=False,
    shuffle_buffer_size=8192,
    label_type=0,
    **params,
):
    """
    This returns a DALI data loader from an LMDB dataset.
    
    path: str
        folder of LMDB dataset

    rank: int
        rank (used for sharding). For horovod, this should be `hvd.rank()`

    world_size: int
        world_size (used for sharding). For horovod, this should be `hvd.size()`
    
    prefetch_queue_depth: int
        number of batches to prefetch.
        See doc of https://docs.nvidia.com/deeplearning/dali/user-guide/docs/advanced_topics.html#prefetching-queue-depth
        for more explanation

    shuffle: bool
        whether to shuffle data
    
    shuffle_buffer_size: int
        buffer size used for shuffling the data
        
    For the rest of the params (**params), they are redirected to `BasePipeline`,
    see `BasePipeline` for a doc about them.


    Returns
    -------

    DALIDataLoader object
    """

    input = ops.Caffe2Reader(
        path=path,
        prefetch_queue_depth=prefetch_queue_depth,
        shard_id=rank,
        num_shards=world_size,
        random_shuffle=shuffle,
        initial_fill=shuffle_buffer_size,
        # label_type=label_type,
    )
    pipeline = BasePipeline(input=input, **params)
    pipeline.build()
    batch_size = params["batch_size"]
    dataloader = DALIClassificationIterator(pipeline, reader_name="Reader")
    dataloader = DALIDataLoader(dataloader, batch_size=batch_size)
    print(len(dataloader))
    return dataloader


# def worker_init_fn(worker_id):
    # worker_info = torch.utils.data.get_worker_info()
    # dataset = worker_info.dataset
    # dataset.data = []
    # env = lmdb.open(dataset.root,readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
    # dataset.env = env
    # dataset.load()
    # env.close()

def collate_fn(batch, transform=None, parallel=None, n_jobs=16):
    with Parallel(n_jobs=n_jobs, backend="threading") as parallel:
        xs = parallel(delayed(transform)(x) for x, y in batch)
        xs = torch.stack(xs)
        ys = torch.Tensor([y for x, y in batch]).long()
        return xs, ys
