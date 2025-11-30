from pathlib import Path
from typing import Optional
import torch
from tools.util.format import raise_on_none
from tools.io.image import save_image_stack, save_image
from tools.segmentation.masking import save_channel_masks
from tools.util.torch import flatten_batch_dims
from tools.util.numpy import numpyify_image


class SyntheticDataFolder():

    images: torch.Tensor
    """Images of the synthetic data. Shape (T, C, H, W)"""

    masks: torch.Tensor
    """Channel Masks of the synthetic data. Shape (T, C, H, W)"""

    depths: Optional[torch.Tensor]
    """Depths of the synthetic data. Shape (T, H, W) Range [0, 1] where 0 is near and 1 is far."""

    save_directory: str
    """Directory to save the synthetic data."""

    name: str
    """Name of the synthetic dataset."""

    masks_directory: str
    """Sub-Directory to save the masks."""

    images_directory: str
    """Sub-Directory to save the images."""

    depths_directory: str
    """Sub-Directory to save the depths."""

    @property
    def data_path(self) -> Path:
        path = f"{self.save_directory}/{self.name}"
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def masks_path(self) -> Path:
        path = f"{self.data_path}/{self.masks_directory}"
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def images_path(self) -> Path:
        path = f"{self.data_path}/{self.images_directory}"
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def depths_path(self) -> Path:
        path = f"{self.data_path}/{self.depths_directory}"
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __init__(self,
                 name: str,
                 save_directory: str,
                 images: torch.Tensor,
                 masks: torch.Tensor,
                 depths: Optional[torch.Tensor] = None,
                 images_directory: str = "images",
                 masks_directory: str = "masks",
                 depths_directory: str = "depth") -> None:
        self.name = raise_on_none(name)
        self.save_directory = raise_on_none(save_directory)
        self.images = flatten_batch_dims(raise_on_none(images), -4)[0]
        self.masks = flatten_batch_dims(raise_on_none(masks), -4)[0]
        self.images_directory = raise_on_none(images_directory)
        self.masks_directory = raise_on_none(masks_directory)
        self.depths_directory = raise_on_none(depths_directory)
        if depths is not None:
            depths = flatten_batch_dims(depths, -3)[0]
        self.depths = depths

    def estimate_depths(self):
        from nag.scripts.compute_depth import ComputeDepthConfig, main as depth_main
        cfg = ComputeDepthConfig(
            images_folder=self.images_path,
            depth_output_folder=self.depths_path,
            depth_filename_format="{index}.tiff",
        )
        depth_main(cfg)

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        # Save the images and masks
        images = numpyify_image(self.images)
        images *= 255
        images = images.astype("uint8")
        masks = self.masks.cpu().permute(0, 2, 3, 1).bool().numpy()
        save_image_stack(images, self.images_path /
                         "{index}.png", override=True)
        save_channel_masks(masks, self.masks_path)
        if self.depths is not None:
            depths = self.depths.cpu().numpy()
            for i, depth in enumerate(depths):
                save_image(depth, self.depths_path /
                           f"{i}.tiff", override=True)
        else:
            self.estimate_depths()
