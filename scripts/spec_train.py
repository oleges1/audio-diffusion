"""
Train a diffusion model on images.
"""
import os
gpu_list = "0,1,2,3,4,5"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

import argparse
import sys
import torch
import torch.distributed as dist
sys.path.append('.')
from pathlib import Path

from improved_diffusion import dist_util, logger

from improved_diffusion.audio_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.audio_datasets import audio_data_defaults
from improved_diffusion.train_util import TrainLoop



def get_available_devices():
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids 


def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


def main():
    args = create_argparser().parse_args()
    
    Path(args.experiment_name).mkdir(parents=True, exist_ok=True)
    Path(f'{args.experiment_name}/checkpoints').mkdir(parents=True, exist_ok=True)
   # dist_util.setup_dist()

    if dist.get_rank() == 0:
        with open(f'{args.experiment_name}/args.txt', 'wt') as file:
            for k, v in sorted(vars(args).items()):
                file.write('%s: %s\n' % (str(k), str(v)))


    logger.configure()
    init_distributed()
    # num_of_gpus = torch.cuda.device_count()
    # print(num_of_gpus)
    # dev, gpus = get_available_devices()
    # print('available devices')
    # for g in gpus:
    #     print(g)

    logger.log("creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    local_rank = int(os.environ['LOCAL_RANK'])
    model = torch.nn.parallel.DistributedDataParallel(
                  model, 
                  device_ids=[local_rank]) 
                #   roadcast_buffers=False,
                #   bucket_cap_mb=128,
                #   find_unused_parameters=True)


    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        batch_size=args.batch_size,
        **args_to_dict(args, audio_data_defaults().keys())
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        # num_gpus=args.num_gpus,
        experiment_name = args.experiment_name
    ).run_loop()


def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(audio_data_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--local_rank', type=int)
    # parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--experiment_name', type=str)
    return parser


if __name__ == "__main__":
    main()
