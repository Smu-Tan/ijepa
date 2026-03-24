import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_train_transform(
    crop_size,
    resize_scale=(0.08, 1.0),
    hflip=0.5,
    randaugment=False,
    crop_mode='rrc',
    resize_size=None,
):
    if crop_mode == 'rrc':
        transform_list = [
            transforms.RandomResizedCrop(
                crop_size,
                scale=resize_scale,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
    elif crop_mode == 'resize':
        transform_list = [
            transforms.Resize(
                (resize_size or crop_size, resize_size or crop_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
    else:
        raise ValueError(f'Unsupported train crop mode: {crop_mode}')

    transform_list.append(transforms.RandomHorizontalFlip(p=hflip))
    if randaugment:
        transform_list.append(transforms.RandAugment())
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return transforms.Compose(transform_list)


def build_eval_transform(crop_size, resize_size=None, crop_mode='center_crop'):
    if crop_mode == 'center_crop':
        resize_size = resize_size or int(round(crop_size / 0.875))
        transform_list = [
            transforms.Resize(
                resize_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(crop_size),
        ]
    elif crop_mode == 'resize':
        transform_list = [
            transforms.Resize(
                (resize_size or crop_size, resize_size or crop_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
    else:
        raise ValueError(f'Unsupported eval crop mode: {crop_mode}')

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return transforms.Compose(transform_list)


class MixupCutmix:
    def __init__(self, num_classes, mixup_alpha=0.0, cutmix_alpha=0.0, prob=1.0):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def _one_hot(self, target):
        return torch.nn.functional.one_hot(target, num_classes=self.num_classes).float()

    def _rand_bbox(self, size, lam):
        _, _, height, width = size
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        cx = np.random.randint(width)
        cy = np.random.randint(height)

        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        return x1, y1, x2, y2

    def __call__(self, images, targets):
        if random.random() > self.prob:
            return images, self._one_hot(targets)

        if self.cutmix_alpha > 0.0 and self.mixup_alpha > 0.0:
            use_cutmix = random.random() < 0.5
        else:
            use_cutmix = self.cutmix_alpha > 0.0

        batch_size = images.size(0)
        perm = torch.randperm(batch_size, device=images.device)
        targets_a = self._one_hot(targets)
        targets_b = self._one_hot(targets[perm])

        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            x1, y1, x2, y2 = self._rand_bbox(images.size(), lam)
            images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
            lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        elif self.mixup_alpha > 0.0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            images = images * lam + images[perm] * (1.0 - lam)
        else:
            return images, targets_a

        mixed_targets = targets_a * lam + targets_b * (1.0 - lam)
        return images, mixed_targets
