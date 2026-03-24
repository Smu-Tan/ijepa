import logging
import os
import sys

import torch


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    return {
        key[len('module.'):] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }


def load_pretrained_backbone(backbone, checkpoint_path, checkpoint_key='target_encoder'):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(checkpoint_key)
        if state_dict is None:
            fallback_keys = ['target_encoder', 'encoder', 'state_dict', 'model']
            for key in fallback_keys:
                state_dict = checkpoint.get(key)
                if state_dict is not None:
                    logger.info('checkpoint key %s missing, falling back to %s', checkpoint_key, key)
                    break
    if state_dict is None:
        raise ValueError(f'Could not find backbone weights in checkpoint: {checkpoint_path}')

    state_dict = _strip_module_prefix(state_dict)
    msg = backbone.load_state_dict(state_dict, strict=True)
    logger.info('loaded pretrained backbone from %s with msg: %s', checkpoint_path, msg)
    return checkpoint


def load_training_checkpoint(model, optimizer, scaler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scaler is not None and checkpoint.get('scaler') is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    start_epoch = checkpoint.get('epoch', 0)
    best_acc1 = checkpoint.get('best_acc1', 0.0)
    logger.info('resumed classification checkpoint from %s (epoch=%s)', checkpoint_path, start_epoch)
    return start_epoch, best_acc1


def save_training_checkpoint(state, latest_path, best_path=None, is_best=False):
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    torch.save(state, latest_path)
    if is_best and best_path is not None:
        torch.save(state, best_path)
