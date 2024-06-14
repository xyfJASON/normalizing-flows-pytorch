import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import tqdm
import argparse
from omegaconf import OmegaConf

import accelerate
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.data import load_data
from utils.logger import get_logger
from utils.misc import image_norm_to_float, instantiate_from_config, amortize, discard_label


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['eval_bpd', 'sample'], help='Choose a mode')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=2024, help='Set random seed')
    parser.add_argument('--weights', type=str, required=True, help='Path to pretrained model weights')
    parser.add_argument('--n_samples', type=int, help='Number of samples')
    parser.add_argument('--save_dir', type=str, help='Path to directory saving samples')
    parser.add_argument('--bspp', type=int, default=256, help='Batch size per process')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    # INITIALIZE LOGGER
    logger = get_logger(use_tqdm_handler=True, is_main_process=accelerator.is_main_process)

    # SET SEED
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD MODEL
    model = instantiate_from_config(conf.model).eval()

    # LOAD WEIGHTS
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Successfully load model from {args.weights}')
    logger.info(f'Number of parameters of model: {sum(p.numel() for p in model.parameters()):,}')
    logger.info('=' * 50)

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model = accelerator.prepare(model)  # type: ignore
    unwrapped_model = accelerator.unwrap_model(model)

    accelerator.wait_for_everyone()

    data_size = (conf.data.img_channels, conf.data.img_size, conf.data.img_size)

    # EVALUATE BPD
    if args.mode == 'eval_bpd':
        logger.info('Start evaluating...')
        # build test dataloader
        test_set = load_data(conf.data, split='test')
        test_loader = DataLoader(test_set, batch_size=args.bspp, shuffle=False, **conf.dataloader)
        test_loader = accelerator.prepare(test_loader)  # type: ignore
        # evaluate nll and bpd
        nlls, bpds = [], []
        with torch.no_grad():
            for x in tqdm.tqdm(test_loader, desc='Evaluating', disable=not accelerator.is_main_process):
                x = discard_label(x)
                z, log_abs_jac = model(x)
                nll_prior = F.softplus(z) + F.softplus(-z)
                nll_prior = nll_prior.flatten(start_dim=1).sum(dim=1)
                nll = nll_prior - log_abs_jac
                bpd = nll / (data_size[0] * data_size[1] * data_size[2])
                bpd = (bpd + math.log(128)) / math.log(2)
                nll = accelerator.gather_for_metrics(nll)
                bpd = accelerator.gather_for_metrics(bpd)
                nlls.append(nll)
                bpds.append(bpd)
        nll = torch.cat(nlls, dim=0).mean().item()
        bpd = torch.cat(bpds, dim=0).mean().item()
        logger.info(f'Test NLL: {nll:.4f}')
        logger.info(f'Test BPD: {bpd:.4f}')

    # SAMPLE
    elif args.mode == 'sample':
        logger.info('Start sampling...')
        # make directory
        assert args.n_samples is not None, 'Number of samples should be specified'
        assert args.save_dir is not None, 'Directory to save samples should be specified'
        os.makedirs(args.save_dir, exist_ok=True)
        logger.info(f'Samples will be saved to {args.save_dir}')
        # sample
        idx = 0
        with torch.no_grad():
            bslist = amortize(args.n_samples, args.bspp * accelerator.num_processes)
            for bs in tqdm.tqdm(bslist, desc='Sampling', disable=not accelerator.is_main_process):
                bspp = min(args.bspp, math.ceil(bs / accelerator.num_processes))
                z = torch.rand((bspp, *data_size), device=device)
                z = -torch.log(1 / z - 1)
                sample = unwrapped_model.backward(z).clamp(-1, 1)
                sample = accelerator.gather(sample)[:bs]
                if accelerator.is_main_process:
                    for x in sample:
                        x = image_norm_to_float(x).cpu()
                        save_image(x, os.path.join(args.save_dir, f'{idx}.png'))
                        idx += 1
        logger.info(f'Samples are saved to {args.save_dir}')
        logger.info('End of sampling')

    else:
        raise ValueError(f'Invalid mode: {args.mode}')


if __name__ == '__main__':
    main()
