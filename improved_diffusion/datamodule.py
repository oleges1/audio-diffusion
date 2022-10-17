from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from .vctk import VCTK
#from torchaudio.datasets import VCTK_092

class AudioDatamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.root = config.root
        self.batch_size = config.batch_size
        self.segment_size = config.segment_size
        self.n_fft = config.n_fft
        self.hop_size = config.hop_size
        self.win_size = config.win_size
        self.raw_wave = config.raw_wave
                

    def setup(self, stage: str):

        data = VCTK(self.root,  self.segment_size,  self.n_fft, 
         self.hop_size,  self.win_size,  self.raw_wave,
         zero_out_percent=None)
        # data = DRVCTK(self.root, self.segment_size, self.n_fft, 
        #     self.hop_size, self.win_size, self.raw_wave,
        #     subset='train', zero_out_percent=None)
        # self.val = RealESRGANDataset(self.opt_params, self.val_dir)
        # self.mnist_val = MNIST(self.data_dir, train=False, transform=self.transforms)
        # mnist_full = MNIST(self.data_dir, train=True)
        # print('length of dataset', len(div2k))
        n = len(data)
        self.train, self.val = random_split(data, [int(n*0.9), n-int(n*0.9)])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False,  drop_last=True)
