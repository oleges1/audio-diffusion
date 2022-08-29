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

from improved_diffusion import  logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.audio_datasets import audio_data_defaults
from einops import rearrange
from improved_diffusion.audio_datasets import load_data
import torchaudio

hann_window = {}

def to_waveform(sample, n_fft, hop_size, win_size, center=True):
    global hann_window
    if str(sample.device) not in hann_window:
        hann_window[str(sample.device)] = th.hann_window(win_size).to(sample.device)
    sample = rearrange(sample, 'B (S D) T -> B S T D', D=2)
    wave = th.istft(sample, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(sample.device)],
                      center=center, normalized=False, onesided=True)
    return wave.unsqueeze(1)

save_path = 'wavs/'

def main():
    args = create_argparser().parse_args()

    #dist_util.setup_dist()
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

    if args.use_ddrm:
        sample_fn = diffusion.ddrm_sample            
    elif args.use_ddim:
        sample_fn = diffusion.ddim_sample_loop
    else:
        sample_fn = diffusion.p_sample_loop


    if args.use_ddrm:
        data = load_data(batch_size=args.batch_size, subset='test', **args_to_dict(args, audio_data_defaults().keys()), deterministic=True)
        for batch, cond in data:
            print('Using DDRM sampling')
            if len(all_wavs) * args.batch_size > args.num_samples:
                break
            model_kwargs = {}
            batch.cuda()
            wavs = sample_fn(model,
                (args.batch_size, args.in_specs, 8192),
                progress=True,
                device='cpu',
                clear_signal=batch,
                sigma_0=args.sigma_0,
                etaA=args.etaA,
                etaB=args.etaB,
                etaC=args.etaC,
                model_kwargs=model_kwargs,
                trained_on = 'speca'
            )

            gathered_samples = [th.zeros_like(wavs) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, wavs)  # gather not supported with NCCL
            all_wavs.extend([sample.cpu() for sample in gathered_samples])
            logger.log(f"created {len(all_wavs) * args.batch_size} samples")        


    else:

        while len(all_wavs) * args.batch_size < args.num_samples:
            model_kwargs = {}
            # if args.class_cond:
            #     classes = th.randint(
            #         low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            #     )
            #     model_kwargs["y"] = classes
            # sample_fn = (
            #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            # )
            sample = sample_fn(
                model,
                # (args.batch_size, args.in_specs * 2, 32),
                (args.batch_size, args.in_specs, 32),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True
            )
            print(sample.shape, "sample, should be only one batch dimension")
            wavs = to_waveform(sample, args.n_fft, args.hop_size, args.win_size).contiguous()
            print(wavs.shape, "wavs, should be only one batch dimension")

            gathered_samples = [th.zeros_like(wavs) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, wavs)  # gather not supported with NCCL
            all_wavs.extend([sample.cpu() for sample in gathered_samples])
            logger.log(f"created {len(all_wavs) * args.batch_size} samples")

    arr = th.cat(all_wavs, dim=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        # shape_str = "x".join([str(x) for x in arr.shape])
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        # logger.log(f"saving to {out_path}")
        # if args.class_cond:
        #     np.savez(out_path, arr, label_arr)
        # else:
        #     np.savez(out_path, arr)
        os.makedirs(save_path, exist_ok=True)
        for i, wave in enumerate(arr):
            torchaudio.save(os.path.join(save_path, f'{i}.wav'), wave, 16000)


    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=2,
        use_ddim=False,
        use_ddrm=True,
        model_path="logs/model1100000.pt",
        sigma_0=0.00000001,
        etaA=0.85,
        etaB=0.85,
        etaC=0.85
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(audio_data_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
