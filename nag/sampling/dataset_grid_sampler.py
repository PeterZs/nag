from typing import Any, Callable, Dict, List, Optional, Tuple
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
from torch.utils.data import Dataset
from tools.logger.logging import logger
from nag.config.nag_config import NAGConfig
from nag.dataset.nag_dataset import NAGDataset
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from tools.util.format import raise_on_none
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims
from torch.nn import functional as F

from nag.sampling.regular_uv_grid_sampler import RegularUVGridSampler
from nag.sampling.timed_weighted_uv_grid_sampler import TimedWeightedUVGridSampler
from nag.sampling.uv_grid_sampler import UVGridSampler
from tools.util.format import parse_type
from nag.model.epoch_state_mixin import EpochStateMixin
from tools.viz.matplotlib import saveable, plot_as_image, plot_coords_as_image, assemble_coords_to_image, get_mpl_figure, plot_mask
from tools.util.typing import DEFAULT
from tools.util.numpy import numpyify


class DatasetGridSampler(Dataset, EpochStateMixin):

    _dataset: NAGDataset

    _sampler: UVGridSampler

    _load_masks: bool
    """If masks should be loaded into the data context as well."""

    def __init__(self,
                 dataset: NAGDataset,
                 sampler: UVGridSampler,
                 load_masks: bool = False
                 ) -> None:
        super().__init__()
        self._dataset = raise_on_none(dataset)
        self._sampler = raise_on_none(sampler)
        self._load_masks = load_masks

    @property
    def flatten(self) -> bool:
        return self._sampler.flatten

    @property
    def t(self) -> torch.Tensor:
        return self._sampler.t

    @flatten.setter
    def flatten(self, value: bool) -> None:
        self._sampler.flatten = value

    @t.setter
    def t(self, value: torch.Tensor) -> None:
        self._sampler.t = value

    @property
    def dataset(self) -> NAGDataset:
        return self._dataset

    @property
    def sampler(self) -> UVGridSampler:
        return self._sampler

    @classmethod
    def for_camera(cls,
                   camera: TimedCameraSceneNode3D,
                   dataset: NAGDataset) -> "DatasetGridSampler":
        from nag.loss.mask_loss_mixin import MaskLossMixin

        load_masks = issubclass(dataset.config.loss_type, MaskLossMixin)

        resolution = camera._image_resolution.flip(-1).detach().cpu().numpy()
        uv_max = camera._image_resolution.flip(-1).detach().cpu().numpy()
        t = dataset.frame_timestamps
        res_fac = dataset.config.learn_resolution_factor
        resolution = (res_fac * resolution).astype(int)

        sampler_type = parse_type(dataset.config.sampler_type, UVGridSampler)
        args = dict(dataset.config.sampler_kwargs)
        args["config"] = dataset.config

        if dataset.config.num_batches is not None:
            if "num_batches" in args:
                old_num_batches = args["num_batches"]
                logger.warning(
                    f"Overwriting num_batches in sampler_kwargs from {old_num_batches} to {dataset.config.num_batches} as it is set in num_batches in the config.")
            args["num_batches"] = dataset.config.num_batches

        sampler = sampler_type(
            resolution=resolution,
            uv_max=uv_max,
            t=t,
            **args)

        gs = cls(dataset=dataset, sampler=sampler, load_masks=load_masks)
        if isinstance(sampler, EpochStateMixin):
            gs.notify_on_epoch_change(sampler)
        return gs

    def __len__(self) -> int:
        return len(self._sampler)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        return self.sample_grid(
            idx, flatten=self.flatten, t=self.t)

    def sample_grid(self,
                    batch_idx: int,
                    flatten: bool = True,
                    t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Sample grid for the given batch index.

        Parameters
        ----------

        batch_idx : int
            Index of the batch to sample rays for.
            Each batch contains a spatially subsampled set of rays e.g. 1/num_batches * image_dimension (H, W) of the image.

        flatten : bool, optional
            Whether to flatten the batch dimensions, by default True
            If false, the batch dimensions are kept. (H, W, 2) instead of (H * W, 2)

        t : Optional[torch.Tensor], optional
            Timestamps to get the object position for, by default None
            If none, the timestamps for the camera are used. Should be of shape (T,) in range [0, 1]


        Returns
        -------

        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing:
            1. Sample grid
            The UV / Grid coordinates of the rays. Shape: (B, 2)

            2. Resampled images
            The resampled images / pixel, corresponding to the position of the ray origins. Shape: (B, T, C)

            3. Times
            The sample times at the time dimension. Shape: (T,)

            B is H * W, T is the number of time samples, C is the number of channels in the image.

            4. Weight Times
            The weight of each time dimension sample. Shape: (T,)
            Will be int within range [0, N] and indicate how much weight should be given to each time sample.
        """
        t_weight = None
        if isinstance(self._sampler, TimedWeightedUVGridSampler):
            uv, t, t_weight = self._sampler.sample_grid(
                batch_idx, t=t, flatten=False)
        else:
            uv, t = self._sampler.sample_grid(batch_idx, t=t, flatten=False)
            t_weight = torch.ones_like(t, dtype=torch.int32)

        data = dict()

        rel_uv = uv / self._sampler.uv_max
        # Norm to -1, 1 for gridsample
        grid_uv = (rel_uv - 0.5) * 2

        image_idx: torch.tensor = torch.arange(
            len(self._dataset), dtype=torch.int32)
        # if len(t) != len(dataset) we need a subset of images
        if len(t) != len(self._dataset):
            mat = torch.isin(self._dataset._frame_timestamps, t).squeeze()
            if mat.int().sum() != len(t):
                raise ValueError(
                    "Could not find all timestamps in the dataset.")
            image_idx = image_idx[mat]

        # Use Gridsample to sample the images
        images = self._dataset[image_idx].to(t.device)

        t_grid_uv = grid_uv.unsqueeze(0).repeat(
            len(t), 1, 1, 1)  # Shape T, H, W, 2 Whereby H = 1

        resampled_images = F.grid_sample(
            images, t_grid_uv, mode="bilinear", padding_mode="border", align_corners=self.dataset.config.plane_align_corners)

        # Move channel dimension to the end, time to second last (H, W, T, C)
        resampled_images = resampled_images.permute(2, 3, 0, 1)
        if flatten:
            # Flatten the first two dimensions
            resampled_images, _ = flatten_batch_dims(resampled_images, -3)
            uv, _ = flatten_batch_dims(uv, -2)

        if self._load_masks:
            # Load masks as well
            masks = self._dataset.load_mask(image_idx).to(t.device)
            # Shape: (T, C, H, W) wherby C corresponds to the acutally used objects.

            resampled_masks = F.grid_sample(masks.float(
            ), t_grid_uv, mode="nearest", padding_mode="border", align_corners=self.dataset.config.plane_align_corners).bool()

            resampled_masks = resampled_masks.permute(
                2, 3, 0, 1)  # Shape: (T, C, H, W) to (H, W, T, C)

            if flatten:
                resampled_masks, _ = flatten_batch_dims(resampled_masks, -3)

            data["masks"] = resampled_masks  # Shape: (B, T, C)
            data["masks_object_idx"] = torch.arange(
                resampled_masks.shape[-1], device=t.device)

        return uv, resampled_images, t, t_weight, data

    @saveable()
    def plot_output(
        self,
        uv: torch.Tensor,
        images: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        t_weight: Optional[torch.Tensor] = None,
        data: Dict[str, Any] = None,
        outputs: Optional[torch.Tensor] = None,
        downscale_factor: int = 4,
        only_weighted: bool = True,
        tight: bool = False,
        inpaint_title: bool = DEFAULT,
        **kwargs
    ):
        """Plots sample_grid outputs into image format.

        Parameters
        ----------
        uv : torch.Tensor
            Camera coordinates. Shape: (B, 2)
        images : torch.Tensor
            _description_
        t : Optional[torch.Tensor], optional
            _description_, by default None
        t_weight : Optional[torch.Tensor], optional
            _description_, by default None
        data : Dict[str, Any], optional
            _description_, by default None
        """
        from tools.util.torch import flatten_batch_dims, unflatten_batch_dims
        from tools.segmentation.masking import channel_masks_to_value_mask

        uv, shp = flatten_batch_dims(uv, -2)  # Shape: (B, 2)
        images, _ = flatten_batch_dims(images, -3)  # Shape: (B, T, C)
        if t is not None:
            t, _ = flatten_batch_dims(t, -1)
        if t_weight is not None:
            t_weight, _ = flatten_batch_dims(t_weight, -1)

        B, T, C = images.shape

        if t is None:
            t = torch.arange(T, dtype=torch.float32, device=uv.device)

        masks = None
        object_alphas = None

        if data is not None:
            masks = data.get("masks", None)
            if masks is not None:
                masks, _ = flatten_batch_dims(masks, -3)

            object_alphas = data.get("object_alpha", None)
            if object_alphas is not None:
                object_alphas, _ = flatten_batch_dims(
                    object_alphas.detach(), -3)

        if outputs is not None:
            outputs, _ = flatten_batch_dims(outputs.detach(), -3)

        resolution = torch.tensor(
            self._dataset._learning_image_shape).flip(-1) // downscale_factor  # H, W to W, H
        uv_max = self._sampler.uv_max

        imgs = assemble_coords_to_image(coords=uv, data=images.moveaxis(0, 1),
                                        resolution=resolution,
                                        domain=torch.tensor([[0, 0], uv_max]),
                                        )

        if masks is not None:
            mask_imgs = assemble_coords_to_image(coords=uv, data=masks.moveaxis(0, 1),
                                                 resolution=resolution,
                                                 domain=torch.tensor(
                                                     [[0, 0], uv_max]),
                                                 )

        if object_alphas is not None:
            object_alphas = assemble_coords_to_image(coords=uv, data=object_alphas.moveaxis(0, 1),
                                                     resolution=resolution,
                                                     domain=torch.tensor(
                                                         [[0, 0], uv_max]),
                                                     )

        if outputs is not None:
            outputs = assemble_coords_to_image(coords=uv, data=outputs.moveaxis(0, 1),
                                               resolution=resolution,
                                               domain=torch.tensor(
                                                   [[0, 0], uv_max]),
                                               )

        if only_weighted and t_weight is not None:
            t_weight = numpyify(t_weight)
            filt = t_weight > 0
            imgs = imgs[filt]
            if masks is not None:
                mask_imgs = mask_imgs[filt]
            t = t[filt]
            if outputs is not None:
                outputs = outputs[filt]
            if object_alphas is not None:
                object_alphas = object_alphas[filt]

        cols = 1
        if masks is not None:
            cols += 1
        if object_alphas is not None:
            cols += 1
        if outputs is not None:
            cols += 1

        mask_classes = None
        if masks is not None:
            mask_classes = torch.unique(masks)

        inpaint_title_kwargs = dict()
        if inpaint_title == DEFAULT:
            inpaint_title = tight

        if inpaint_title:
            inpaint_title_kwargs["size"] = .6
            inpaint_title_kwargs["margin"] = 5
            inpaint_title_kwargs["padding"] = 5

        fig, axes = get_mpl_figure(
            len(t), cols, tight=tight, ratio_or_img=imgs[0], ax_mode="2d")
        for i, _t in enumerate(t):
            col_idx = 0
            ax = axes[i, col_idx]
            plot_as_image(imgs[i], axes=ax, variable_name=f"GT t={_t:.3f} W={t_weight[i].item()}",
                          inpaint_title=inpaint_title, inpaint_title_kwargs=inpaint_title_kwargs)
            col_idx += 1

            if outputs is not None:
                ax = axes[i, col_idx]
                plot_as_image(outputs[i], axes=ax, variable_name=f"Output t={_t:.3f} W={t_weight[i].item()}",
                              inpaint_title=inpaint_title, inpaint_title_kwargs=inpaint_title_kwargs)
                col_idx += 1

            if masks is not None:
                ax = axes[i, col_idx]
                cmap = plt.get_cmap("tab20") if len(
                    mask_classes) > 10 else plt.get_cmap("tab10")
                plot_mask(imgs[i], mask_imgs[i], ax=ax,
                          cmap=cmap, title=f"GT+Mask t={_t:.3f}",
                          darkening_background=0.4,
                          filled_contours=True, lined_contours=False, inpaint_title=inpaint_title, inpaint_title_kwargs=inpaint_title_kwargs)
                col_idx += 1

            if object_alphas is not None:
                ax = axes[i, col_idx]
                cmap = plt.get_cmap("tab20") if len(
                    mask_classes) > 10 else plt.get_cmap("tab10")
                plot_mask(imgs[i], object_alphas[i] > 0.5, ax=ax,
                          cmap=cmap, title=f"GT+OBJ_Alpha t={_t:.3f}",
                          darkening_background=0.4,
                          filled_contours=True, lined_contours=False, inpaint_title=inpaint_title, inpaint_title_kwargs=inpaint_title_kwargs)
                col_idx += 1

        return fig
