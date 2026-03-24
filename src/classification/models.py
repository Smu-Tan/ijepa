import torch
import torch.nn as nn

import src.models.vision_transformer as vit


def build_backbone(model_name, crop_size, patch_size, use_grad_checkpoint=False):
    backbone = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        use_grad_checkpoint=use_grad_checkpoint,
    )
    return backbone


class BatchNormLinearHead(nn.Module):
    def __init__(self, in_dim, num_classes, use_batch_norm=False):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim, affine=False) if use_batch_norm else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(self.norm(x))


class ViTFeatureExtractor(nn.Module):
    def __init__(self, backbone, feature_mode='avgpool', num_last_blocks=4):
        super().__init__()
        self.backbone = backbone
        self.feature_mode = feature_mode
        self.num_last_blocks = num_last_blocks

    @property
    def embed_dim(self):
        return self.backbone.embed_dim

    @property
    def output_dim(self):
        if self.feature_mode == 'concat_avgpool_last4':
            return self.backbone.embed_dim * self.num_last_blocks
        return self.backbone.embed_dim

    def _forward_features(self, x):
        x = self.backbone.patch_embed(x)
        pos_embed = self.backbone.interpolate_pos_encoding(x, self.backbone.pos_embed)
        x = x + pos_embed

        outputs = []
        for block in self.backbone.blocks:
            x = block(x)
            outputs.append(self.backbone.norm(x))
        return outputs

    def forward(self, x):
        outputs = self._forward_features(x)
        if self.feature_mode == 'avgpool':
            return outputs[-1].mean(dim=1)
        if self.feature_mode == 'concat_avgpool_last4':
            selected = outputs[-self.num_last_blocks:]
            pooled = [out.mean(dim=1) for out in selected]
            return torch.cat(pooled, dim=-1)
        raise ValueError(f'Unsupported feature mode: {self.feature_mode}')


class ClassificationModel(nn.Module):
    def __init__(self, feature_extractor, classifier, freeze_backbone=True):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.feature_extractor.eval()
        return self

    def forward(self, images):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.feature_extractor(images)
        else:
            features = self.feature_extractor(images)
        return self.classifier(features)


def build_classification_model(
    model_name,
    crop_size,
    patch_size,
    num_classes,
    feature_mode,
    head_type,
    freeze_backbone,
    use_grad_checkpoint=False,
):
    backbone = build_backbone(
        model_name=model_name,
        crop_size=crop_size,
        patch_size=patch_size,
        use_grad_checkpoint=use_grad_checkpoint,
    )
    feature_extractor = ViTFeatureExtractor(
        backbone=backbone,
        feature_mode=feature_mode,
    )
    classifier = BatchNormLinearHead(
        in_dim=feature_extractor.output_dim,
        num_classes=num_classes,
        use_batch_norm=(head_type == 'bn_linear'),
    )
    return ClassificationModel(
        feature_extractor=feature_extractor,
        classifier=classifier,
        freeze_backbone=freeze_backbone,
    )
