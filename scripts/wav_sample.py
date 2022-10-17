"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

# FIXME: this is not working

import argparse
import os
import sys
sys.path.append('.')

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.audio_datasets import audio_data_defaults
from einops import rearrange
import torchaudio

hann_window = {}

save_path = 'wavs/'

def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.cuda()
    model.eval()

    logger.log("sampling...")
    all_wavs = []
    all_labels = []
    while len(all_wavs) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.cuda()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        wavs = sample_fn(
            model,
            # (args.batch_size, args.in_specs * 2, 32),
            # (args.batch_size, args.in_specs, 8192 * 4),
            (args.batch_size, args.in_specs, 8192),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )

        # gathered_samples = [th.zeros_like(wavs) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, wavs)  # gather not supported with NCCL
        os.makedirs(save_path, exist_ok=True)
        for i, wave in enumerate(wavs):
            print(wave.size())
            torchaudio.save(os.path.join(save_path, f'{i}.wav'), wave.cpu(), 48000)
        all_wavs.extend([sample.cpu() for sample in wavs])
        logger.log(f"created {len(all_wavs) * args.batch_size} samples")

    arr = th.cat(all_wavs, dim=0)
    arr = arr[: args.num_samples]
    # if dist.get_rank() == 0:
        # shape_str = "x".join([str(x) for x in arr.shape])
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        # logger.log(f"saving to {out_path}")
        # if args.class_cond:
        #     np.savez(out_path, arr, label_arr)
        # else:
        #     np.savez(out_path, arr)
    # os.makedirs(save_path, exist_ok=True)
    # for i, wave in enumerate(arr):
    #     torchaudio.save(os.path.join(save_path, f'{i}.wav'), wave.unsqueeze(0), 48000)


    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=16,
        use_ddim=False,
        model_path="logs/vctk_test/latest.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(audio_data_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
