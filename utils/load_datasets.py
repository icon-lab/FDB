import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle, os
import read_data

def load_data(
    *, data_dir, batch_size, image_size, data_type
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None

    dataset = ImageDataset(
        image_size,
        data_dir,
        all_files,
        data_type,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )

    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["mat"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, data_dir, image_paths, data_type, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        if data_type == "singlecoil":
            self.local_images = read_data.get_fs_singlecoil(data_dir)
        elif data_type == "multicoil":
            self.local_images = read_data.get_fs_multicoil(data_dir)
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        idx = np.mod(idx, (len(self.local_images)-1))
        arr = self.local_images[idx]
        real = np.real(arr)
        imag = np.imag(arr)
        out = np.stack([real,imag]).astype(np.float32)
        return out, {}
