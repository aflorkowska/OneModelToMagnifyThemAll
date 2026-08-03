import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from dataclasses import dataclass, field


@dataclass
class TransformConfig:
    """
    Configuration for HistopathologyTransform.
    """

    hflip_enable: bool = False
    hflip_prob: float = 0.5

    vflip_enable: bool = False
    vflip_prob: float = 0.5

    rotation_enable: bool = False
    rotation_degrees: list = field(default_factory=lambda: [0, 90, 180, 270])

    z_norm: bool = False

    apply_mask_mapping: bool = False
    mask_mapping: dict = (
        None  # Dictionary for arbitrary mask mapping {old_label: new_label}
    )

    background_label: int = 0


class HistopathologyTransform:
    """
    Flexible transform class for classification and segmentation.
    """

    def __init__(self, config: TransformConfig = None, gt_labels=None):
        if config is None:
            config = TransformConfig()
        self.config = config
        self.to_tensor = transforms.ToTensor()

        self._apply_mask_mapping = config.apply_mask_mapping
        self._mask_mapping = config.mask_mapping or {}
        self._gt_labels = gt_labels

    def _apply_z_normalization(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply per-channel Z-score normalization.
        """
        epsilon = 1e-6
        image_float = image.float()
        mean = image_float.mean(dim=(1, 2), keepdim=True)
        std = image_float.std(dim=(1, 2), keepdim=True) + epsilon
        return (image_float - mean) / std

    def apply_mask_mapping_only(self, mask) -> torch.Tensor:
        """
        Apply only mask mapping without any augmentations.
        """
        mask_tensor = (
            F.pil_to_tensor(mask).long()
            if not isinstance(mask, torch.Tensor)
            else mask.long()
        )
        mask_tensor = self._ensure_channel_first(mask_tensor, is_mask=True)
        if self._gt_labels:
            mask_tensor = self._filter_mask_by_gt_labels(
                mask_tensor, self._gt_labels, self.config.background_label
            )
        mask_tensor = self._apply_mask_mapping_tensor(mask_tensor)
        return mask_tensor

    def _filter_mask_by_gt_labels(
        self,
        mask: torch.Tensor,
        gt_labels: list,
        fill_value: int = 0,
    ) -> torch.Tensor:
        """
        Remove labels not in gt_labels BEFORE mapping.
        mask: (1,H,W) long
        """
        if not gt_labels:
            return mask

        valid = torch.zeros_like(mask, dtype=torch.bool)
        for v in gt_labels:
            valid |= mask == v

        mask = mask.clone()
        mask[~valid] = fill_value
        return mask

    def _ensure_channel_first(
        self, tensor_tile: torch.Tensor, is_mask: bool = False
    ) -> torch.Tensor:
        """
        Ensure image is (C,H,W) with channel first (1 or 3),
        and mask is (1,H,W)
        """
        if tensor_tile.dim() == 2:
            tensor_tile = tensor_tile.unsqueeze(0)
        elif tensor_tile.dim() == 3:
            if is_mask:
                if tensor_tile.size(0) != 1:
                    if tensor_tile.size(-1) == 1:
                        tensor_tile = tensor_tile.permute(2, 0, 1)
                    else:
                        print("Warning: mask tensor has unexpected number of channels")
            else:
                if tensor_tile.size(0) in [1, 3]:
                    pass
                elif tensor_tile.size(-1) in [1, 3]:
                    tensor_tile = tensor_tile.permute(2, 0, 1)
                else:
                    print("Warning: image tensor has unexpected number of channels")
        else:
            print("Warning: tensor must have 2 or 3 dimensions (H,W,[C])")
        return tensor_tile

    def _apply_mask_mapping_tensor(self, label_tile: torch.Tensor) -> torch.Tensor:
        """
        Apply arbitrary mask mapping to a 2D or 3D torch tensor of labels (already long and 1,H,W).
        Returns mapped tensor with same shape and dtype=torch.long.
        """
        if not self._apply_mask_mapping:
            return label_tile

        original_shape = label_tile.shape
        flat = label_tile.view(-1)
        max_val = int(flat.max().item())
        lookup = torch.arange(max_val + 1, device=label_tile.device, dtype=flat.dtype)
        for k, v in self._mask_mapping.items():
            if k <= max_val:
                lookup[k] = v

        mapped_flat = lookup[flat]
        mapped = mapped_flat.view(original_shape)
        return mapped

    def __call__(self, image, mask=None):
        """
        Apply configured transformations to image (and mask, if provided).

        Returns:
        --------
        tuple(torch.Tensor, torch.Tensor|None)
        """
        do_hflip = (
            np.random.random() < self.config.hflip_prob
            if self.config.hflip_enable
            else False
        )
        do_vflip = (
            np.random.random() < self.config.vflip_prob
            if self.config.vflip_enable
            else False
        )
        angle = (
            np.random.choice(self.config.rotation_degrees)
            if self.config.rotation_enable
            else 0
        )

        if do_hflip:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)

        if do_vflip:
            image = F.vflip(image)
            if mask is not None:
                mask = F.vflip(mask)

        if angle != 0:
            image = F.rotate(image, angle)
            if mask is not None:
                mask = F.rotate(mask, angle)

        img_tensor = self.to_tensor(image)

        if self.config.z_norm:
            img_tensor = self._apply_z_normalization(img_tensor)

        img_tensor = img_tensor.float()
        img_tensor = self._ensure_channel_first(img_tensor, is_mask=False)

        mask_tensor = None
        if mask is not None:
            mask_tensor = F.pil_to_tensor(mask).long()
            mask_tensor = self._ensure_channel_first(mask_tensor, is_mask=True)
            if self._gt_labels:
                mask_tensor = self._filter_mask_by_gt_labels(
                    mask_tensor, self._gt_labels, self.config.background_label
                )
            mask_tensor = self._apply_mask_mapping_tensor(mask_tensor)

        return img_tensor, mask_tensor
