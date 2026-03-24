import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
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

from src.classification.checkpoint import (
    load_pretrained_backbone,
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.classification.data import (
    make_classification_dataset,
    make_classification_loader,
)
from src.classification.models import build_classification_model
from src.classification.optim import (
    LARS,
    WarmupCosineLRScheduler,
    WarmupStepLRScheduler,
    build_finetune_param_groups,
    build_linear_probe_param_groups,
)
from src.classification.transforms import (
    MixupCutmix,
    build_eval_transform,
    build_train_transform,
)
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger, gpu_timer


try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def compute_topk_accuracy(logits, target, topk=(1, 5)):
    if target.ndim != 1:
        target = target.argmax(dim=1)
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    results = []
    for k in topk:
        k = min(k, logits.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        acc = 100.0 * correct_k / max(target.size(0), 1)
        results.append(float(AllReduce.apply(acc)))
    return results


def soft_target_cross_entropy(logits, target):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return (-target * log_probs).sum(dim=-1).mean()


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def build_progress(rank, total, description):
    if rank != 0 or Progress is None:
        return None, None
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
    task = progress.add_task(description, total=total, metrics='')
    return progress, task


def format_train_metrics(loss_meter, acc1_meter, acc5_meter, lr_value, time_meter):
    mem_gb = torch.cuda.max_memory_allocated() / 1024.0 ** 3 if torch.cuda.is_available() else 0.0
    return (
        f'loss {loss_meter.avg:.3f} | acc1 {acc1_meter.avg:.2f} | acc5 {acc5_meter.avg:.2f} | '
        f'lr {lr_value:.2e} | mem {mem_gb:.2f} GB | {time_meter.avg:.1f} ms'
    )


def evaluate(model, data_loader, device, use_bfloat16):
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    model.eval()

    with torch.no_grad():
        for images, target in data_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                logits = model(images)
                loss = criterion(logits, target)
            loss = float(AllReduce.apply(loss))
            acc1, acc5 = compute_topk_accuracy(logits, target)
            batch_size = images.size(0)
            loss_meter.update(loss, batch_size)
            acc1_meter.update(acc1, batch_size)
            acc5_meter.update(acc5, batch_size)

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


def main(args, resume_preempt=False):
    meta = args['meta']
    data = args['data']
    optimization = args['optimization']
    logging_cfg = args['logging']

    freeze_backbone = bool(meta['freeze_backbone'])
    use_bfloat16 = bool(meta.get('use_bfloat16', False))
    use_grad_checkpoint = bool(meta.get('use_grad_checkpoint', False))
    model_name = meta['model_name']
    pretrained_checkpoint = meta['pretrained_checkpoint']
    pretrained_key = meta.get('pretrained_key', 'target_encoder')
    patch_size = meta['patch_size']
    crop_size = data['crop_size']
    feature_mode = meta.get('feature_mode', 'avgpool')
    head_type = meta.get('head_type', 'linear')
    num_classes = int(data['num_classes'])
    load_checkpoint = bool(meta.get('load_checkpoint', False) or resume_preempt)
    resume_checkpoint = meta.get('read_checkpoint')

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    folder = logging_cfg['folder']
    tag = logging_cfg['write_tag']
    checkpoint_freq = int(logging_cfg.get('checkpoint_freq', 1))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'params-classification.yaml'), 'w') as f:
        yaml.dump(args, f)

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info('Initialized (rank/world-size) %s/%s', rank, world_size)
    if rank > 0:
        logger.setLevel(logging.ERROR)

    model = build_classification_model(
        model_name=model_name,
        crop_size=crop_size,
        patch_size=patch_size,
        num_classes=num_classes,
        feature_mode=feature_mode,
        head_type=head_type,
        freeze_backbone=freeze_backbone,
        use_grad_checkpoint=use_grad_checkpoint and not freeze_backbone,
    )
    load_pretrained_backbone(
        backbone=model.feature_extractor.backbone,
        checkpoint_path=pretrained_checkpoint,
        checkpoint_key=pretrained_key,
    )
    model.to(device)

    train_transform = build_train_transform(
        crop_size=crop_size,
        resize_scale=tuple(data.get('train_resize_scale', [0.08, 1.0])),
        hflip=float(data.get('horizontal_flip_prob', 0.5)),
        randaugment=bool(data.get('randaugment', False)),
        crop_mode=data.get('train_crop_mode', 'rrc'),
        resize_size=data.get('train_resize_size'),
    )
    eval_transform = build_eval_transform(
        crop_size=crop_size,
        resize_size=data.get('eval_resize_size'),
        crop_mode=data.get('eval_crop_mode', 'center_crop'),
    )

    train_dataset = make_classification_dataset(
        transform=train_transform,
        root_path=data['root_path'],
        image_folder=data['image_folder'],
        training=True,
        copy_data=bool(meta.get('copy_data', False)),
        subset_file=data.get('subset_file'),
        dataset_backend=data.get('dataset_backend', 'imagefolder'),
        hf_dataset_path=data.get('hf_dataset_path'),
    )
    val_dataset = make_classification_dataset(
        transform=eval_transform,
        root_path=data['root_path'],
        image_folder=data['image_folder'],
        training=False,
        copy_data=bool(meta.get('copy_data', False)),
        subset_file=None,
        dataset_backend=data.get('dataset_backend', 'imagefolder'),
        hf_dataset_path=data.get('hf_dataset_path'),
    )

    train_loader, train_sampler = make_classification_loader(
        dataset=train_dataset,
        batch_size=int(data['batch_size']),
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=int(data['num_workers']),
        pin_mem=bool(data['pin_mem']),
        drop_last=bool(data.get('drop_last', False)),
    )
    val_loader, val_sampler = make_classification_loader(
        dataset=val_dataset,
        batch_size=int(data.get('eval_batch_size', data['batch_size'])),
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=int(data['num_workers']),
        pin_mem=bool(data['pin_mem']),
        drop_last=False,
    )

    if freeze_backbone:
        param_groups = build_linear_probe_param_groups(model, weight_decay=float(optimization['weight_decay']))
        optimizer = LARS(
            param_groups,
            lr=float(optimization['lr']),
            weight_decay=float(optimization['weight_decay']),
            momentum=float(optimization.get('momentum', 0.9)),
        )
    else:
        param_groups = build_finetune_param_groups(
            model=model,
            weight_decay=float(optimization['weight_decay']),
            layer_decay=float(optimization.get('layer_decay', 0.75)),
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=float(optimization['lr']),
            betas=tuple(optimization.get('betas', [0.9, 0.999])),
        )

    num_epochs = int(optimization['epochs'])
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_epochs = float(optimization.get('warmup_epochs', 0))
    if freeze_backbone:
        scheduler = WarmupStepLRScheduler(
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs,
            milestone_epochs=optimization.get('step_decay_epochs', [15, 30, 45]),
            gamma=float(optimization.get('step_decay_gamma', 0.1)),
            base_lr=float(optimization['lr']),
        )
    else:
        scheduler = WarmupCosineLRScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_epochs * steps_per_epoch),
            base_lr=float(optimization['lr']),
            min_lr=float(optimization.get('min_lr', 0.0)),
        )

    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    mixup_cutmix = None
    if optimization.get('mixup_alpha', 0.0) > 0.0 or optimization.get('cutmix_alpha', 0.0) > 0.0:
        mixup_cutmix = MixupCutmix(
            num_classes=num_classes,
            mixup_alpha=float(optimization.get('mixup_alpha', 0.0)),
            cutmix_alpha=float(optimization.get('cutmix_alpha', 0.0)),
            prob=float(optimization.get('mixup_prob', 1.0)),
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(optimization.get('label_smoothing', 0.0)))

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        model = DistributedDataParallel(model, static_graph=freeze_backbone)

    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    best_path = os.path.join(folder, f'{tag}-best.pth.tar')
    tb_folder = os.path.join(folder, 'tensorboard')
    tb_writer = SummaryWriter(log_dir=tb_folder) if rank == 0 else None
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'train_loss'),
        ('%.5f', 'train_acc1'),
        ('%.5f', 'train_acc5'),
        ('%.5f', 'val_loss'),
        ('%.5f', 'val_acc1'),
        ('%.5f', 'val_acc5'),
        ('%.8f', 'lr'),
        ('%d', 'time_ms'),
    )

    start_epoch = 0
    best_acc1 = 0.0
    if load_checkpoint and resume_checkpoint is not None:
        start_epoch, best_acc1 = load_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            checkpoint_path=resume_checkpoint,
            device=device,
        )

    for epoch in range(start_epoch, num_epochs):
        unwrap_model(model).train(True)
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        time_meter = AverageMeter()
        last_lr = optimizer.param_groups[0]['lr']

        progress, progress_task = build_progress(rank, len(train_loader), f'Epoch {epoch + 1}/{num_epochs}')

        for itr, (images, target) in enumerate(train_loader):
            global_step = epoch * len(train_loader) + itr
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if mixup_cutmix is not None:
                images, target_for_loss = mixup_cutmix(images, target)
            else:
                target_for_loss = target

            def train_step():
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    logits = model(images)
                    if target_for_loss.ndim == 2:
                        loss = soft_target_cross_entropy(logits, target_for_loss)
                    else:
                        loss = criterion(logits, target_for_loss)

                if use_bfloat16:
                    scaler.scale(loss).backward()
                    if optimization.get('clip_grad') is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimization['clip_grad']))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if optimization.get('clip_grad') is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimization['clip_grad']))
                    optimizer.step()

                new_lr = scheduler.step()
                train_target = target
                acc1, acc5 = compute_topk_accuracy(logits, train_target)
                return float(AllReduce.apply(loss)), acc1, acc5, new_lr

            (loss, acc1, acc5, last_lr), etime = gpu_timer(train_step)
            batch_size = images.size(0)
            loss_meter.update(loss, batch_size)
            acc1_meter.update(acc1, batch_size)
            acc5_meter.update(acc5, batch_size)
            time_meter.update(etime, 1)

            if progress is not None:
                progress.update(
                    progress_task,
                    advance=1,
                    metrics=format_train_metrics(loss_meter, acc1_meter, acc5_meter, last_lr, time_meter),
                )

            if tb_writer is not None:
                tb_writer.add_scalar('train/loss', loss, global_step)
                tb_writer.add_scalar('train/acc1', acc1, global_step)
                tb_writer.add_scalar('train/acc5', acc5, global_step)
                tb_writer.add_scalar('train/lr', last_lr, global_step)

        if progress is not None:
            progress.stop()

        val_loss, val_acc1, val_acc5 = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            use_bfloat16=use_bfloat16,
        )

        csv_logger.log(
            epoch + 1,
            len(train_loader),
            loss_meter.avg,
            acc1_meter.avg,
            acc5_meter.avg,
            val_loss,
            val_acc1,
            val_acc5,
            last_lr,
            int(time_meter.avg if time_meter.count > 0 else 0),
        )

        if rank == 0:
            logger.info(
                '[%d/%d] train_loss=%.4f train_acc1=%.2f train_acc5=%.2f val_loss=%.4f val_acc1=%.2f val_acc5=%.2f lr=%.3e',
                epoch + 1,
                num_epochs,
                loss_meter.avg,
                acc1_meter.avg,
                acc5_meter.avg,
                val_loss,
                val_acc1,
                val_acc5,
                last_lr,
            )

        if tb_writer is not None:
            tb_writer.add_scalar('epoch/train_loss', loss_meter.avg, epoch + 1)
            tb_writer.add_scalar('epoch/train_acc1', acc1_meter.avg, epoch + 1)
            tb_writer.add_scalar('epoch/train_acc5', acc5_meter.avg, epoch + 1)
            tb_writer.add_scalar('epoch/val_loss', val_loss, epoch + 1)
            tb_writer.add_scalar('epoch/val_acc1', val_acc1, epoch + 1)
            tb_writer.add_scalar('epoch/val_acc5', val_acc5, epoch + 1)

        is_best = val_acc1 > best_acc1
        best_acc1 = max(best_acc1, val_acc1)
        if rank == 0:
            state = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': None if scaler is None else scaler.state_dict(),
                'best_acc1': best_acc1,
            }
            save_training_checkpoint(
                state=state,
                latest_path=latest_path,
                best_path=best_path,
                is_best=is_best,
            )
            if (epoch + 1) % checkpoint_freq == 0:
                epoch_path = os.path.join(folder, f'{tag}-ep{epoch + 1}.pth.tar')
                save_training_checkpoint(
                    state=state,
                    latest_path=epoch_path,
                )

    if tb_writer is not None:
        tb_writer.close()
