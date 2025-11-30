from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from matplotlib import pyplot as plt
from tools.serialization.json_convertible import JsonConvertible

from nag.homography.match_finder import MatchFinder, MatchFinderConfig
from nag.homography.loftr_match_finder import LoftrMatchFinder, LoftrFinderConfig
from tools.util.format import parse_type
import cv2
import numpy as np
import torch
from tools.transforms.to_tensor_image import ToTensorImage
from tools.util.typing import VEC_TYPE
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims, tensorify, cummatmul
from tools.viz.matplotlib import get_mpl_figure, plot_as_image, saveable, plot_mask
from tools.util.progress_factory import ProgressFactory
from tools.util.format import parse_enum
from nag.homography.homography_finder import HomographyFinderConfig, HomographyFinder, is_coord_in_mask
from matplotlib.figure import Figure
import os


@dataclass
class SequenceHomographyFinderConfig(HomographyFinderConfig):

    plot_visualizations: bool = field(compare=False, default=False)
    """If True, visualizations will be plotted during the homography finding process."""

    plot_visualization_path: Optional[str] = field(compare=False, default=None)
    """Path to save the visualizations. If None, visualizations will not be saved."""


class SequenceHomographyFinder(HomographyFinder):
    """Finds homographies between a sequence of images,
    by identifying matching keypoints between them."""

    config: SequenceHomographyFinderConfig

    def __init__(self,
                 config: SequenceHomographyFinderConfig,
                 progress_bar: bool = False,
                 progress_factory: Optional[ProgressFactory] = None
                 ) -> None:
        super().__init__(config, progress_bar=progress_bar, progress_factory=progress_factory)

    @classmethod
    def config_type(cls) -> Type[SequenceHomographyFinderConfig]:
        return SequenceHomographyFinderConfig

    def find_cumulative_homographies(self,
                                     images: VEC_TYPE,
                                     masks: VEC_TYPE,
                                     frame_indices: VEC_TYPE,
                                     object_indices: VEC_TYPE,
                                     ) -> np.ndarray:

        img_source = images[:-1]
        img_target = images[1:]
        mask_source = masks[:-1]
        mask_target = masks[1:]
        fi = frame_indices[:-1]
        target_frame_indices = frame_indices[1:]
        homo = self.find_homography(image1=img_source,
                                    image2=img_target,
                                    mask1=mask_source,
                                    mask2=mask_target,
                                    frame_indices=fi,
                                    target_frame_indices=target_frame_indices,
                                    object_indices=object_indices
                                    )

        homo = tensorify(homo)
        target = torch.zeros_like(homo)
        for i in range(homo.shape[1]):
            target[:, i] = cummatmul(homo[:, i])

        if self.config.plot_visualizations and self.config.plot_visualization_path is not None:
            with plt.ioff():
                self.plot_sequence_homography_warps(images, masks=masks, homographies=homo,
                                                    frame_indices=frame_indices,
                                                    object_indices=object_indices, save_directory=self.config.plot_visualization_path)
        return target

    def plot_sequence_homography_warps(
            self,
            images: VEC_TYPE,
            masks: VEC_TYPE,
            homographies: np.ndarray,
            frame_indices: VEC_TYPE,
            object_indices: VEC_TYPE,
            save_directory: Optional[str] = None
    ) -> List[Figure]:
        from tools.util.format import consecutive_indices_string
        B, C, H, W = images.shape
        max_images_per_plot = 20

        starts = np.arange(0, B, max_images_per_plot)
        figs = []
        for i, start in enumerate(starts):
            end = min(start + max_images_per_plot, B)
            img_source = images[0][None, ...].repeat(end - start - 1, axis=0)
            img_target = images[start + 1: end]
            mask_source = masks[0][None, ...].repeat(end - start - 1, axis=0)
            mask_target = masks[start + 1: end]
            fi = frame_indices[0][None, ...].repeat(end - start - 1)
            target_frame_indices = frame_indices[start + 1: end]
            h = homographies[start: end]

            start_id_str = consecutive_indices_string(fi)
            target_id_str = consecutive_indices_string(target_frame_indices)

            save_file = f"cum_homography_warp_{start_id_str}_to_{target_id_str}.png"
            save_path = None
            if save_directory is not None:
                save_path = os.path.join(save_directory, save_file)
            fig = self.plot_homography_warp(img_source,
                                            img_target, mask1=mask_source, mask2=mask_target, homography=h,
                                            img1_index=fi, img2_index=target_frame_indices,
                                            object_index=object_indices,
                                            tight_layout=True,
                                            save=(save_path is not None),
                                            path=save_path,
                                            override=True)
            figs.append(fig)
        return figs

    def track_masks(
        self,
        masks: VEC_TYPE,
        cum_homography: VEC_TYPE,
        oidx: int = 0,
        tidx: int = 0,
        sparsity: int = 1
    ) -> torch.Tensor:
        """
        Plots a (sparse) point grid in the region of the first mask,
        and project these points using cum_homograpies in every other frame.

        Parameters
        ----------
        masks : VEC_TYPE
            Masks of the objects. Shape (T, H, W, O) if numpy or (T, O, H, W) if tensor.

        cum_homography : VEC_TYPE
            Cumulative homography for projecting the first coordinates into every other.
            Homography for the tidx frame should be the identity.
            Shape (T, O, 3, 3)

        oidx : int, optional
            Object idx (O) for the masks objects to consider, by default 0

        tidx : int, optional
            Time idx (T) for the masks which should be used as the starting point, by default 0

        sparsity : int, optional
            Sparsity of the created point grid, by default 1

        Returns
        -------
        torch.Tensor
            Tensor of shape (T, FP, 2) where FP is the number of feature points.
            Defined by the size of the first mask and the sparsity.
        """

        from tools.transforms.to_tensor_image import ToTensorImage
        from tools.util.torch import tensorify

        masks = ToTensorImage(output_dtype=torch.bool)(masks)
        cum_homography = tensorify(cum_homography, dtype=torch.float32)

        cum_homography = cum_homography[:, oidx]

        B, H, W, O = masks.shape
        HB, _, _ = cum_homography.shape

        if HB != B:
            raise ValueError(f"Homography must be of shape: (T, O, 3, 3)")

        mask_coords = torch.argwhere(masks[tidx, oidx])
        mask_min = mask_coords.amin(dim=0)
        mask_max = mask_coords.amax(dim=0)

        grid_size = mask_max - mask_min  # Shape (y, x)

        MH, MW = grid_size

        grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, MW, sparsity),
                torch.arange(0, MH, sparsity), indexing="xy"), dim=-1)
        # grid = grid + mask_min[None, None, :].flip(-1).float()

        SH, SW, _ = grid.shape

        grid = grid + mask_min[None, None, :].flip(-1).float()
        # Flatten Grid to shape (SH*SW, 2)
        grid = grid.reshape(SH*SW, 2)

        # Check if coords are in the grid
        is_valid = is_coord_in_mask(grid.flip(-1).int(), masks[tidx, oidx])
        coords = grid[is_valid]

        FP, _ = coords.shape

        affine_coords = torch.cat(
            (coords, torch.ones(FP, 1)), dim=-1
        )
        # Flatten the grid

        g, shp = flatten_batch_dims(affine_coords, -2)
        g = g.unsqueeze(0).expand(HB, -1, -1)  # Shape (HB, FP, 3)

        exp_homography = cum_homography.unsqueeze(
            1).expand(-1, FP, -1, -1)  # Shape (HB, FP, 3, 3)

        deformed_points = torch.bmm(
            exp_homography.reshape(HB*FP, 3, 3),
            g.reshape(HB*FP, -1).unsqueeze(-1)).squeeze(-1).reshape(HB, FP, -1)  # Shape (HB, FP, 3)

        deformed_points = deformed_points[..., :2] / deformed_points[..., 2:3]
        deformed_points = deformed_points.reshape(HB, FP, 2)

        all_points = torch.cat([coords.unsqueeze(0), deformed_points], dim=0)

        return all_points

    @saveable()
    def plot_tracked_points(
            self,
            images: VEC_TYPE,
            masks: VEC_TYPE,
            points: VEC_TYPE,
            tight: bool = True, **kwargs):
        plot_tracked_points(images, masks, points, tight=tight, **kwargs)


@saveable()
def plot_tracked_points(
    images: VEC_TYPE,
    masks: VEC_TYPE,
    points: VEC_TYPE,
    colors: Optional[Union[VEC_TYPE]] = None,
    tight: bool = True,
    frame_numbers: bool = True
):
    """
    Plots the tracked points on the images.

    Parameters
    ----------
    images : VEC_TYPE
        Images to plot the points on. Shape (T, H, W, C) if numpy or (T, C, H, W) if tensor.

    masks : VEC_TYPE
        Masks of the objects. Shape (T, H, W, O) if numpy or (T, O, H, W) if tensor.

    points : VEC_TYPE
        Points to plot on the images. Shape (T, NP, 2) (x, y).


    colors : Optional[Union[VEC_TYPE]], optional
        Colors for the points. Shape (T, NP, 4) if numpy or (T, 4, NP) if tensor, by default None.
        If None, colors will be generated based on the point index.
        Colors should be in RGBA format in the range [0, 1].

    tight : bool, optional
        Whether to plot the images tightly, by default True

    Returns
    -------
    plt.Figure
        Matplotlib figure with the images and points plotted.
    """

    from tools.transforms.to_tensor_image import ToTensorImage
    from tools.util.torch import tensorify
    from tools.viz.matplotlib import plot_mask, get_mpl_figure
    import math

    images = ToTensorImage(output_dtype=torch.float32)(images)
    masks = ToTensorImage(output_dtype=torch.bool)(masks)
    points = tensorify(points)

    B, C, H, W = images.shape
    _, NP, _ = points.shape

    if colors is None:
        cmap = plt.get_cmap("rainbow")
        ci = torch.arange(NP).unsqueeze(0).expand(B, -1)
        ci = ci % cmap.N
        colors = cmap(ci)
    else:
        colors = tensorify(colors)
        if len(colors.shape) != 3:
            raise ValueError(
                f"Colors must be of shape (B, NP, 4). Got {colors.shape}")
        if colors.shape[0] != B:
            raise ValueError(
                f"Colors must have the same number of points as points. Got {colors.shape[0]} and {NP}")
        if colors.shape[-1] == 3:
            colors = torch.cat([colors, torch.ones(NP, 1)], dim=-1)

    rows = round(math.sqrt(B))
    cols = math.ceil((B / rows))

    fig, axes = get_mpl_figure(
        cols=cols, rows=rows, ratio_or_img=images[0], ax_mode="2d", tight=tight)

    for row in range(rows):
        for col in range(cols):
            i = (row * cols) + col
            if i >= B:
                axes[row, col].axis("off")
                continue
            args = dict()
            if frame_numbers:
                args["title"] = str(i)
                args["inpaint_title"] = True
                args["inpaint_title_kwargs"] = dict(
                    margin=5,
                    padding=5,
                    thickness=1,
                    size=1.,
                    background_stroke=0
                )
            plot_mask(images[i], masks[i], ax=axes[row, col], **args)
            # Scatter the points on the axis
            p = points[i]
            # Filter out points which are not in the image bounds
            f = ((p[..., 0] >= 0) & (p[..., 0] < W) &
                 (p[..., 1] >= 0) & (p[..., 1] < H))
            p = p[f]

            axes[row, col].scatter(p[..., 0], p[..., 1], c=colors[i, f], s=8)
            axes[row, col].set_title(f"Frame {i}")
    return fig
