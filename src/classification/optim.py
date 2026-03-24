import math

import torch


class LARS(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0,
        weight_decay=0,
        momentum=0.9,
        trust_coefficient=0.001,
        eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coefficient = group['trust_coefficient']
            eps = group['eps']
            lr = group['lr']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError('LARS does not support sparse gradients')

                if param.ndim > 1:
                    grad = grad.add(param, alpha=weight_decay)
                    param_norm = torch.norm(param)
                    update_norm = torch.norm(grad)
                    trust_ratio = 1.0
                    if param_norm > 0 and update_norm > 0:
                        trust_ratio = trust_coefficient * param_norm / (update_norm + eps)
                    grad = grad.mul(trust_ratio)

                state = self.state[param]
                if 'mu' not in state:
                    state['mu'] = torch.zeros_like(param)
                mu = state['mu']
                mu.mul_(momentum).add_(grad)
                param.add_(mu, alpha=-lr)

        return loss


class WarmupCosineLRScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps, base_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            scale = group.get('lr_scale', 1.0)
            group['lr'] = lr * scale
        return lr


class WarmupStepLRScheduler:
    def __init__(self, optimizer, steps_per_epoch, warmup_epochs, milestone_epochs, gamma, base_lr):
        self.optimizer = optimizer
        self.steps_per_epoch = max(steps_per_epoch, 1)
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)
        self.milestone_steps = [int(epoch * steps_per_epoch) for epoch in milestone_epochs]
        self.gamma = gamma
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            decay_count = sum(self.step_num >= milestone for milestone in self.milestone_steps)
            lr = self.base_lr * (self.gamma ** decay_count)
        for group in self.optimizer.param_groups:
            scale = group.get('lr_scale', 1.0)
            group['lr'] = lr * scale
        return lr


def get_vit_layer_id(name, num_layers):
    if name.startswith('classifier'):
        return num_layers + 1
    if name.startswith('feature_extractor.backbone.patch_embed'):
        return 0
    if name.startswith('feature_extractor.backbone.pos_embed'):
        return 0
    if name.startswith('feature_extractor.backbone.blocks'):
        block_id = int(name.split('.')[3])
        return block_id + 1
    return num_layers


def build_finetune_param_groups(model, weight_decay, layer_decay):
    param_groups = {}
    num_layers = len(model.feature_extractor.backbone.blocks)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_vit_layer_id(name, num_layers)
        group_name = f'layer_{layer_id}_{"no_decay" if param.ndim == 1 or name.endswith(".bias") else "decay"}'
        if group_name not in param_groups:
            lr_scale = layer_decay ** (num_layers + 1 - layer_id)
            param_groups[group_name] = {
                'params': [],
                'weight_decay': 0.0 if (param.ndim == 1 or name.endswith('.bias')) else weight_decay,
                'lr_scale': lr_scale,
            }
        param_groups[group_name]['params'].append(param)
    return list(param_groups.values())


def build_linear_probe_param_groups(model, weight_decay):
    decay_params = []
    no_decay_params = []
    for name, param in model.classifier.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith('.bias'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
