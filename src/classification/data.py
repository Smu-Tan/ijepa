from logging import getLogger
import os

import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from src.datasets.imagenet1k import HFImageNet, ImageNet, ImageNetSubset


logger = getLogger()


class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, root, image_folder='tiny-imagenet-200', transform=None, train=True):
        self.transform = transform
        self.root = os.path.join(root, image_folder)
        self.train = train
        self.wnids = self._read_wnids()
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        self.classes = list(self.wnids)
        self.loader = default_loader
        self.samples = self._build_samples()
        self.targets = [target for _, target in self.samples]

    def _read_wnids(self):
        wnids_path = os.path.join(self.root, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _build_train_samples(self):
        train_root = os.path.join(self.root, 'train')
        samples = []
        for wnid in self.wnids:
            image_dir = os.path.join(train_root, wnid, 'images')
            for fname in sorted(os.listdir(image_dir)):
                if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    samples.append((os.path.join(image_dir, fname), self.class_to_idx[wnid]))
        return samples

    def _build_val_samples(self):
        val_root = os.path.join(self.root, 'val')
        image_dir = os.path.join(val_root, 'images')
        annotations_path = os.path.join(val_root, 'val_annotations.txt')
        samples = []
        with open(annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                image_name, wnid = parts[:2]
                if wnid not in self.class_to_idx:
                    continue
                samples.append((os.path.join(image_dir, image_name), self.class_to_idx[wnid]))
        return samples

    def _build_samples(self):
        if self.train:
            samples = self._build_train_samples()
        else:
            samples = self._build_val_samples()
        logger.info('Tiny-ImageNet dataset created [%s] with %d samples and %d classes', 'train' if self.train else 'val', len(samples), len(self.classes))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def make_classification_dataset(
    transform,
    root_path,
    image_folder,
    training,
    copy_data=False,
    subset_file=None,
    dataset_backend='imagefolder',
    hf_dataset_path=None,
):
    if dataset_backend == 'hf':
        dataset = HFImageNet(
            dataset_path=hf_dataset_path or root_path,
            transform=transform,
            train=training,
        )
    elif image_folder == 'tiny-imagenet-200':
        dataset = TinyImageNet(
            root=root_path,
            image_folder=image_folder,
            transform=transform,
            train=training,
        )
    else:
        dataset = ImageNet(
            root=root_path,
            image_folder=image_folder,
            transform=transform,
            train=training,
            copy_data=copy_data,
            index_targets=False,
        )
    if subset_file is not None and training:
        if dataset_backend == 'hf':
            raise NotImplementedError('subset_file is only supported with the imagefolder backend')
        dataset = ImageNetSubset(dataset, subset_file)
    logger.info('classification dataset created [%s]', 'train' if training else 'val')
    return dataset


def make_classification_loader(
    dataset,
    batch_size,
    world_size,
    rank,
    training,
    num_workers=8,
    pin_mem=True,
    drop_last=False,
):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last and training,
        persistent_workers=False,
    )
    return loader, sampler
