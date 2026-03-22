# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import math
from contextlib import ExitStack

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError:
    Progress = None

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    use_grad_checkpoint = args['meta'].get('use_grad_checkpoint', False)
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    dataset_backend = args['data'].get('dataset_backend', 'imagefolder')
    hf_dataset_path = args['data'].get('hf_dataset_path')
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    accum_steps = max(1, int(args['optimization'].get('accum_steps', 1)))

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    checkpoint_every = args['logging'].get('checkpoint_freq', checkpoint_freq)

    os.makedirs(folder, exist_ok=True)
    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    tb_folder = os.path.join(folder, 'tensorboard')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))
    tb_writer = SummaryWriter(log_dir=tb_folder) if rank == 0 else None

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        use_grad_checkpoint=use_grad_checkpoint)
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            dataset_backend=dataset_backend,
            hf_dataset_path=hf_dataset_path,
            copy_data=copy_data,
            drop_last=True)
    ipe = len(unsupervised_loader)
    logger.info(f'Iterations per epoch: {ipe}')
    optimizer_updates_per_epoch = math.ceil(ipe / accum_steps)
    logger.info(f'Optimizer updates per epoch: {optimizer_updates_per_epoch}')

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=optimizer_updates_per_epoch,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1
    if use_ddp:
        # Be conservative when using gradient accumulation/checkpointing. DDP's
        # static-graph fast path can conflict with repeated backward passes.
        ddp_static_graph = (accum_steps == 1) and (not use_grad_checkpoint)
        encoder = DistributedDataParallel(encoder, static_graph=ddp_static_graph)
        predictor = DistributedDataParallel(predictor, static_graph=ddp_static_graph)
        target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    total_optimizer_updates = int(optimizer_updates_per_epoch * num_epochs * ipe_scale)
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/max(total_optimizer_updates, 1)
                          for i in range(total_optimizer_updates + 1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*optimizer_updates_per_epoch):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if epoch % checkpoint_every == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    def format_metrics(loss_meter, maskA_meter, maskB_meter, lr_value, wd_value, time_meter):
        mem_gb = torch.cuda.max_memory_allocated() / 1024.**3 if torch.cuda.is_available() else 0.0
        return (
            f'loss {loss_meter.avg:.3f} | masks {maskA_meter.avg:.1f}/{maskB_meter.avg:.1f} | '
            f'wd {wd_value:.2e} | lr {lr_value:.2e} | mem {mem_gb:.2f} GB | '
            f'{time_meter.avg:.1f} ms'
        )

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d/%d (iterations: %d)' % (epoch + 1, num_epochs, ipe))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()
        optimizer.zero_grad()
        accum_count = 0
        accum_target = accum_steps
        last_lr = optimizer.param_groups[0]['lr']
        last_wd = next(
            (group['weight_decay'] for group in optimizer.param_groups
             if group.get('weight_decay', 0) > 0),
            0.0)
        progress = None
        progress_task = None
        if rank == 0 and Progress is not None:
            progress = Progress(
                TextColumn('[bold cyan]{task.description}'),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn('{task.fields[metrics]}'),
                refresh_per_second=2,
            )
            progress.start()
            progress_task = progress.add_task(
                f'Epoch {epoch + 1}/{num_epochs}',
                total=ipe,
                metrics=format_metrics(loss_meter, maskA_meter, maskB_meter, last_lr, last_wd, time_meter),
            )

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
            global_step = epoch * ipe + itr
            if accum_count == 0:
                accum_target = min(accum_steps, ipe - itr)

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                should_update = (accum_count + 1) == accum_target

                # Step 1. Forward
                with ExitStack() as sync_stack:
                    if use_ddp and not should_update:
                        sync_stack.enter_context(encoder.no_sync())
                        sync_stack.enter_context(predictor.no_sync())
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                        h = forward_target()
                        z = forward_context()
                        loss = loss_fn(z, h)
                        scaled_loss = loss / accum_target

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(scaled_loss).backward()
                    if should_update:
                        _new_lr = scheduler.step()
                        _new_wd = wd_scheduler.step()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        _new_lr = last_lr
                        _new_wd = last_wd
                else:
                    scaled_loss.backward()
                    if should_update:
                        _new_lr = scheduler.step()
                        _new_wd = wd_scheduler.step()
                        optimizer.step()
                    else:
                        _new_lr = last_lr
                        _new_wd = last_wd
                grad_stats = None
                if should_update:
                    grad_stats = grad_logger(encoder.named_parameters())
                    optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                if should_update:
                    with torch.no_grad():
                        m = next(momentum_scheduler)
                        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                            param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            accum_count = (accum_count + 1) % accum_target
            if accum_count == 0:
                last_lr = _new_lr
                last_wd = _new_wd
            loss_meter.update(loss)
            time_meter.update(etime)
            if progress is not None:
                progress.update(
                    progress_task,
                    advance=1,
                    metrics=format_metrics(
                        loss_meter,
                        maskA_meter,
                        maskB_meter,
                        last_lr,
                        last_wd,
                        time_meter,
                    ),
                )

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if tb_writer is not None:
                    tb_writer.add_scalar('train/loss', loss, global_step)
                    tb_writer.add_scalar('train/lr', _new_lr, global_step)
                    tb_writer.add_scalar('train/weight_decay', _new_wd, global_step)
                    tb_writer.add_scalar('train/mask_enc', maskA_meter.val, global_step)
                    tb_writer.add_scalar('train/mask_pred', maskB_meter.val, global_step)
                    tb_writer.add_scalar('train/iter_time_ms', etime, global_step)
                    tb_writer.add_scalar(
                        'train/max_memory_gb',
                        torch.cuda.max_memory_allocated() / 1024.**3,
                        global_step)
                    if grad_stats is not None:
                        tb_writer.add_scalar('train/grad_first_layer', grad_stats.first_layer, global_step)
                        tb_writer.add_scalar('train/grad_last_layer', grad_stats.last_layer, global_step)
                        tb_writer.add_scalar('train/grad_min', grad_stats.min, global_step)
                        tb_writer.add_scalar('train/grad_max', grad_stats.max, global_step)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    if progress is None:
                        logger.info('[%d, %5d] loss: %.3f '
                                    'masks: %.1f %.1f '
                                    '[wd: %.2e] [lr: %.2e] '
                                    '[mem: %.2f GB] '
                                    '(%.1f ms)'
                                    % (epoch + 1, itr,
                                       loss_meter.avg,
                                       maskA_meter.avg,
                                       maskB_meter.avg,
                                       _new_wd,
                                       _new_lr,
                                       torch.cuda.max_memory_allocated() / 1024.**3,
                                       time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        if progress is not None:
            progress.stop()
        if tb_writer is not None:
            tb_writer.add_scalar('epoch/loss_avg', loss_meter.avg, epoch + 1)
            tb_writer.add_scalar('epoch/mask_enc_avg', maskA_meter.avg, epoch + 1)
            tb_writer.add_scalar('epoch/mask_pred_avg', maskB_meter.avg, epoch + 1)
            tb_writer.add_scalar('epoch/time_ms_avg', time_meter.avg, epoch + 1)
        save_checkpoint(epoch+1)

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
