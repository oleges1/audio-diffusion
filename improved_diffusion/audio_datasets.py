from PIL import Image
import blobfile as bf

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .vctk import DRVCTK
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T

def audio_data_defaults():
    """
    Defaults for audio training.
    """
    return dict(
        root='data/',
        segment_size=8192,
        # num_mels=80,
        # "num_freq": 1025,
        n_fft=1024,
        hop_size=256,
        win_size=1024,
        raw_wave=False,
        # sampling_rate=16000
    )


def load_data(
    batch_size, subset = 'train', *args, deterministic=False, use_ddp = False, **kwargs
):
    """
    For a dataset, create a generator over (mels, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    """

    transform = T.Resize((1024, 32))
    
    dataset = DRVCTK(*args, subset=subset, transform = transform, **kwargs)
    sampler=None
    if deterministic:
        if use_ddp:
            sampler =  DistributedSampler(dataset, shuffle=False)
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=1, drop_last=True, sampler=sampler
        )
    else:
        if use_ddp:
            sampler =  DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=1, drop_last=True, sampler=sampler
        )
    while True:
        yield from loader
