from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .vctk import DRVCTK

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
        raw_wave=True,
        # sampling_rate=16000
    )


def load_data(
    batch_size, subset = 'train', *args, deterministic=False, **kwargs
):
    """
    For a dataset, create a generator over (mels, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    """


    dataset = DRVCTK(*args, **kwargs) #DRVCTK(*args, subset='train', **kwargs)


    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader
