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
from improved_diffusion.audio_datasets import load_data
from einops import rearrange
import torchaudio

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

hann_window = {}

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

        print('Using DDRM sampling')
        segment_size=8192
        data = load_data(batch_size=args.batch_size, subset='test', **args_to_dict(args, audio_data_defaults().keys()), deterministic=True)
        
        batch, _ = next(data)
        batch, _ = next(data)

        if True:
            # if len(all_wavs) * args.batch_size > args.num_samples:
            #     break
            model_kwargs = {}
            batch.cuda()

            start = 0
            end = segment_size
            audio_denoised = []
            audio_noisy = []
            torchaudio.save('initial_wav.wav', batch[0].cpu(), 16000)
            print('batch', batch.size())

            while start < batch.size(2):
                if end > batch.size(2):
                    print('last batch', batch[:, :, start:].size())
                    subset = th.nn.functional.pad(batch[:, :, start:], (0, segment_size - batch[:, :, start:].size(-1)), 'constant')
                else:
                    subset = batch[:, :, start:end]
                    out_denoised, out_noisy = sample_fn(model,
                    (args.batch_size, args.in_specs, segment_size),
                    progress=True,
                    device='cpu',
                    clear_signal=subset,
                    sigma_0=args.sigma_0,
                    etaA=args.etaA,
                    etaB=args.etaB,
                    etaC=args.etaC,
                    model_kwargs=model_kwargs,
                    trained_on = 'wav'
                )
                print('noisy2', out_noisy.size())
                print('den', out_denoised.size())
                audio_denoised.append(out_denoised.squeeze(-1).transpose(1, 2))
                audio_noisy.append(out_noisy)
                start += segment_size
                end += segment_size

            final_noisy = th.cat(audio_noisy, dim=1)
            print('noisy', final_noisy.size())
            torchaudio.save('degraded_wav.wav', final_noisy[0].transpose(0, 1).cpu(), 16000)
            final_sample = th.cat(audio_denoised, dim=1)
            final_sample = final_sample[:, :batch.size(-1), :]
            print(final_sample.size(), 'final size')
            torchaudio.save('restored_wav.wav', final_sample[0].transpose(0, 1).cpu(), 16000)

            fig, axs = plt.subplots(3, 1,  figsize=(40, 40))
            values, ybins, xbins, im1 = axs[0].specgram(batch[0][0].cpu().numpy(), Fs=16000,  NFFT = 1024)
            axs[0].set_title('Initial spectrogram')
            values, ybins, xbins, im2 = axs[1].specgram(final_noisy.squeeze(-1)[0].cpu().numpy(), Fs=16000, NFFT = 1024)
            axs[1].set_title('Degraded spectrogram')
            values, ybins, xbins, im3 = axs[2].specgram(final_sample[0].transpose(0, 1)[0].cpu().numpy(), Fs=16000,  NFFT = 1024)
            axs[2].set_title('Restored spectrogram')
            fig.colorbar(im1, ax=axs[0])
            fig.colorbar(im2, ax=axs[1])
            fig.colorbar(im3, ax=axs[2])
            plt.savefig(f'speca_result_model_wav.png')
            plt.close()
            # # break
            # gathered_samples = [th.zeros_like(wavs) for _ in range(dist.get_world_size())]
            # dist.all_gather(gathered_samples, wavs)  # gather not supported with NCCL
            # all_wavs.extend([sample.cpu() for sample in gathered_samples])
            # logger.log(f"created {len(all_wavs) * args.batch_size} samples")        


    else:
        while len(all_wavs) * args.batch_size < args.num_samples:
            model_kwargs = {}
            # if args.class_cond:
            #     classes = th.randint(
            #         low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            #     )
            #     model_kwargs["y"] = classes

            wavs = sample_fn(
                model,
                # (args.batch_size, args.in_specs * 2, 32),
                # (args.batch_size, args.in_specs, 8192 * 4),
                (args.batch_size, args.in_specs, 8192),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True
            )

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
        batch_size=1,
        use_ddim=False,
        use_ddrm=False,
        model_path="logs/model1530000.pt",
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
