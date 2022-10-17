"""
Train a diffusion model on images.
"""

import argparse
import sys
import copy
sys.path.append('.')
import functools
import torch

from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.audio_datasets import audio_data_defaults
from improved_diffusion.train_util import TrainLoop
import pytorch_lightning as pl
from improved_diffusion.datamodule import AudioDatamodule
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from improved_diffusion.nn import update_ema
from improved_diffusion.resample import LossAwareSampler, UniformSampler
from torch.optim import AdamW
import torchaudio
import os


def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(audio_data_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def audio_data_defaults():
    return dict(
        root='data/',
        segment_size=8192,
        # num_mels=80,
        # "num_freq": 1025,
        n_fft=1024,
        hop_size=256,
        win_size=1024,
        raw_wave=True,
        batch_size=8
        # sampling_rate=16000
    )
    # parser = argparse.ArgumentParser()
    # add_dict_to_argparser(parser, defaults)
    # return parser


INITIAL_LOG_LOSS_SCALE = 20.0    
    
class DiffusionLit(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.model, self.diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        print(os.getcwd())
        self.schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, self.diffusion)
        # self.ema_rate = (
        #     [args.ema_rate]
        #     if isinstance(args.ema_rate, float)
        #     else [float(x) for x in args.ema_rate.split(",")]
        # )
        self.batch_size = args.batch_size
        self.microbatch = args.microbatch if args.microbatch > 0 else args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        # self.schedule_sampler = self.schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0

        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.ema_params = []
        # if self.resume_checkpoint:
        #     self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
        #     if  self.global_rank == 0:
        #         print(f"loading model from checkpoint: {self.resume_checkpoint}...")
        #     self.model.load_state_dict(
        #         torch.load(
        #             self.resume_checkpoint, map_location='cpu'
        #         )
        #     )
        #     self.ema_params = []
        #     for rate in self.ema_rate: 
        #         ema_params = copy.deepcopy(self.model.parameters())
        #         main_checkpoint = self.resume_checkpoint
        #         ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        #         if ema_checkpoint:
        #             if self.global_rank == 0:
        #                 print(f"loading EMA from checkpoint: {ema_checkpoint}...")
        #             state_dict = torch.load(
        #                 ema_checkpoint, map_location='cpu'
        #             )
        #             ema_params = self._state_dict_to_master_params(state_dict)
        #             self.ema_params.append(ema_params)
        # else:
        # for _ in range(len(self.ema_rate)):
        #     for i in self.model.parameters():
        #         self.ema_params.append(copy.deepcopy(i))                 


    def training_step(self, batch, batch_idx):
        x, y = batch
        # if self.global_rank == 0:
            # torchaudio.save(f'samples/test_sample_at_{self.current_epoch}_epoch.wav', x[0].cpu(), 16000)
        for i in range(0, x.shape[0], self.microbatch):
            micro = x[i : i + self.microbatch]
            micro_cond = {
                k: v[i : i + self.microbatch]
                for k, v in y.items()
            }
            last_batch = (i + self.microbatch) >= x.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch:
                losses = compute_losses()
            else:
                losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self.log("training_loss", loss.item(), prog_bar=True)    
            self.step += 1

        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
#         if self.global_rank == 0:
#             torchaudio.save(f'samples/test_sample_at_{self.current_epoch}_epoch.wav', x[0].cpu(), 48000)
        
        for i in range(0, x.shape[0], self.microbatch):
            micro = x[i : i + self.microbatch]
            micro_cond = {
                k: v[i : i + self.microbatch]
                for k, v in y.items()
            }
            last_batch = (i + self.microbatch) >= x.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch:
                losses = compute_losses()
            else:
                losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            self.log("validation_loss", loss.item(), prog_bar=True)  
            if self.global_rank == 0:
                output = self.model(micro, self.diffusion._scale_timesteps(t), **micro_cond)
                torchaudio.save(f'samples_new_2/sample_at_{self.current_epoch}_epoch.wav', output[3].cpu(), 48000)
                torch.save(self.model.state_dict(), 'logs/vctk_gpu/latest.pt')

        return loss
    
    
    def configure_optimizers(self):
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            opt_checkpoint = bf.join(
            bf.dirname(self.resume_checkpoint), f"opt{resume_step:06}.pt"
            )
            if bf.exists(opt_checkpoint):
                if  self.global_rank == 0:
                    print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
                state_dict = torch.load(
                    opt_checkpoint, map_location='cpu'
                )
                opt.load_state_dict(state_dict)

        return opt
    
    def on_before_optimizer_step(self,optimizer, optimizer_idx):
        if not self.lr_anneal_steps:
            pass
        else:
            frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
            lr = self.lr * (1 - frac_done)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        # for rate, params in zip(self.ema_rate, self.ema_params):
        #     update_ema(params, self.model.parameters(), rate=rate)



if __name__ == "__main__":
    defaults = audio_data_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    data_args = parser.parse_args()
    datamodule = AudioDatamodule(data_args)
                
    args = create_argparser().parse_args()
    model = DiffusionLit(args)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath='logs/vctk_gpu',
        filename='vctk-{epoch:02d}-{validation_loss:.2f}',
        save_top_k = 5)
    
    trainer = Trainer(logger=TensorBoardLogger(save_dir='outputs/test'), accelerator='gpu', devices=6, callbacks=[checkpoint_callback], sync_batchnorm=True)
    trainer.fit(model, datamodule=datamodule, ckpt_path="logs/vctk_test/vctk-epoch=641-validation_loss=0.00.ckpt")
