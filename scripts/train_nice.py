import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import argparse
from omegaconf import OmegaConf

import accelerate
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.data import load_data
from utils.logger import get_logger
from utils.tracker import StatusTracker
from utils.misc import create_exp_dir, find_resume_checkpoint, instantiate_from_config
from utils.misc import get_time_str, check_freq, amortize, get_data_generator, discard_label


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to configuration file',
    )
    parser.add_argument(
        '-e', '--exp_dir', type=str,
        help='Path to the experiment directory. Default to be ./runs/exp-{current time}/',
    )
    parser.add_argument(
        '-r', '--resume', type=str,
        help='Resume from a checkpoint. Could be a path or `best` or `latest`',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir, conf_yaml=OmegaConf.to_yaml(conf), time_str=args.time_str,
            exist_ok=args.resume is not None, no_interaction=args.no_interaction,
        )

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True, is_main_process=accelerator.is_main_process,
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger, exp_dir=exp_dir, print_freq=conf.train.print_freq,
        is_main_process=accelerator.is_main_process,
    )

    # SET SEED
    accelerate.utils.set_seed(conf.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    if conf.train.batch_size % accelerator.num_processes != 0:
        raise ValueError(
            f'Batch size should be divisible by number of processes, '
            f'get {conf.train.batch_size} % {accelerator.num_processes} != 0'
        )
    bspp = conf.train.batch_size // accelerator.num_processes
    train_set = load_data(conf.data)
    train_loader = DataLoader(
        dataset=train_set, batch_size=bspp,
        shuffle=True, drop_last=True, **conf.dataloader,
    )
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {bspp}')
    logger.info(f'Total batch size: {conf.train.batch_size}')

    # BUILD MODEL AND OPTIMIZERS
    model = instantiate_from_config(conf.model)
    optimizer = instantiate_from_config(conf.train.optim, params=model.parameters())
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Number of parameters of model: {sum(p.numel() for p in model.parameters()):,}')
    logger.info('=' * 50)

    # RESUME TRAINING
    step, best_nll = 0, float('inf')
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        ckpt_model = torch.load(os.path.join(resume_path, 'model.pt'), map_location='cpu')
        model.load_state_dict(ckpt_model['model'])
        logger.info(f'Successfully load model from {resume_path}')
        ckpt_optimizer = torch.load(os.path.join(resume_path, 'optimizer.pt'), map_location='cpu')
        optimizer.load_state_dict(ckpt_optimizer['optimizer'])
        logger.info(f'Successfully load optimizer from {resume_path}')
        ckpt_meta = torch.load(os.path.join(resume_path, 'meta.pt'), map_location='cpu')
        step = ckpt_meta['step'] + 1
        best_nll = ckpt_meta['best_nll']
        logger.info(f'Restart training at step {step}, current best nll is {best_nll}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)  # type: ignore
    unwrapped_model = accelerator.unwrap_model(model)

    accelerator.wait_for_everyone()

    # TRAINING FUNCTIONS
    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        accelerator.save(dict(model=unwrapped_model.state_dict()), os.path.join(save_path, 'model.pt'))
        accelerator.save(dict(optimizer=optimizer.state_dict()), os.path.join(save_path, 'optimizer.pt'))
        accelerator.save(dict(step=step, best_nll=best_nll), os.path.join(save_path, 'meta.pt'))

    def run_step(batch):
        x = discard_label(batch)
        z, log_abs_jac = model(x)
        # compute loss
        nll_prior = F.softplus(z) + F.softplus(-z)
        nll_prior = nll_prior.flatten(start_dim=1).sum(dim=1)
        loss = torch.mean(nll_prior - log_abs_jac)
        # optimize
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        # reduce loss for logging
        loss = accelerator.reduce(loss.detach(), reduction='mean')
        return dict(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    @torch.no_grad()
    @accelerator.on_main_process
    def sample(savepath: str):
        data_size = (conf.data.img_channels, conf.data.img_size, conf.data.img_size)
        samples = []
        for bs in amortize(conf.train.n_samples, bspp):
            z = torch.rand((bs, *data_size), device=device)
            z = -torch.log(1 / z - 1)
            samples.append(unwrapped_model.backward(z).clamp(-1, 1))
        samples = torch.cat(samples, dim=0)
        nrow = math.ceil(math.sqrt(conf.train.n_samples))
        save_image(samples, savepath, nrow=nrow, normalize=True, value_range=(-1, 1))

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        tqdm_kwargs=dict(desc='Epoch', leave=False, disable=not accelerator.is_main_process),
    )
    while step < conf.train.n_steps:
        # get a batch
        _batch = next(train_data_generator)
        # run a step
        model.train()
        train_status = run_step(_batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()
        # validate
        model.eval()
        # save checkpoint
        if check_freq(conf.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>7d}'))
            accelerator.wait_for_everyone()
        # save best checkpoint
        if train_status['loss'] < best_nll:
            best_nll = train_status['loss']
            save_ckpt(os.path.join(exp_dir, 'ckpt', 'best'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(conf.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>7d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>7d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    logger.info('End of training')


if __name__ == '__main__':
    main()
