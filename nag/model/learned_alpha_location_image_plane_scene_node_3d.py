import math
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from nag.model.camera_scene_node_3d import CameraSceneNode3D
from nag.model.learned_offset_scene_node_3d import LearnedOffsetSceneNode3D
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_plane_scene_node_3d import TimedPlaneSceneNode3D
from nag.model.timed_discrete_scene_node_3d import global_to_local, local_to_global

import torch
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims, compose_transformation_matrix
from nag.config.encoding_config import EncodingConfig
from nag.config.network_config import NetworkConfig
from nag.transforms.transforms_timed_3d import interpolate_vector, linear_interpolate_vector
from tools.viz.matplotlib import plot_mask, plot_as_image
from tools.util.numpy import numpyify
from nag.utils import utils
try:
    import tinycudann as tcnn
except (ModuleNotFoundError, OSError) as err:
    from tools.logger.logging import logger
    from tools.util.mock_import import MockImport
    if not TYPE_CHECKING:
        logger.warning(f"Could not import tinycudann: {err}")
        tcnn = MockImport(mocked_property="tcnn")

import pytorch_lightning as pl
from tools.transforms.geometric.transforms3d import rotmat_to_unitquat, unitquat_to_rotmat
from nag.model.discrete_plane_scene_node_3d import local_to_plane_coordinates, plane_coordinates_to_local
from tools.util.torch import shadow_ones, shadow_zeros, shadow_identity_2d
from nag.model.learned_image_plane_scene_node_3d import LearnedImagePlaneSceneNode3D
from tools.util.format import raise_on_none


class LearnedAlphaLocationImagePlaneSceneNode3D(
    LearnedImagePlaneSceneNode3D
):

    def __init__(
        self,
            num_rigid_control_points: int,
            num_flow_control_points: int,
            encoding_image_config: Union[EncodingConfig, str] = "small",
            encoding_alpha_config: Union[EncodingConfig, str] = "small",
            encoding_flow_config: Union[EncodingConfig, str] = "small",
            network_image_config: Union[NetworkConfig, str] = "small",
            network_alpha_config: Union[NetworkConfig, str] = "small",
            network_flow_config: Union[NetworkConfig, str] = "small",
            translation_offset_weight: float = 0.03,
            rotation_offset_weight: float = 0.03,
            rgb_rescaling: bool = True,
            alpha_rescaling: bool = True,
            flow_weight: float = 0.01,
            plane_scale: Optional[VEC_TYPE] = None,
            plane_scale_offset: Optional[VEC_TYPE] = None,
            translation: Optional[VEC_TYPE] = None,
            orientation: Optional[VEC_TYPE] = None,
            initial_rgb: Optional[torch.Tensor] = None,
            initial_alpha: Optional[torch.Tensor] = None,
            rgb_weight: Optional[torch.Tensor] = None,
            alpha_weight: Optional[torch.Tensor] = None,
            position: Optional[torch.Tensor] = None,
            times: Optional[VEC_TYPE] = None,
            name: Optional[str] = None,
            children: Optional[Iterable['AbstractSceneNode']] = None,
            decoding: bool = False,
            dtype: torch.dtype = torch.float32,
            index: Optional[int] = None,
            proxy_init: bool = False,
            align_corners: bool = True,
            **kwargs
    ):

        if not proxy_init:
            raise_on_none(initial_alpha, "initial_alpha")
        else:
            if initial_alpha is None:
                initial_alpha = torch.zeros(1, 1, 1)
        if initial_alpha.shape[0] != 1:
            raise ValueError(
                "Initial alpha must have only one channel and Shape (1, H, W)")
        if len(initial_alpha.shape) != 3:
            raise ValueError("Initial alpha must have shape (1, H, W).")
        super().__init__(
            num_rigid_control_points=num_rigid_control_points,
            num_flow_control_points=num_flow_control_points,
            encoding_image_config=encoding_image_config,
            encoding_alpha_config=encoding_alpha_config,
            encoding_flow_config=encoding_flow_config,
            network_image_config=network_image_config,
            network_alpha_config=network_alpha_config,
            network_flow_config=network_flow_config,
            flow_weight=flow_weight,
            translation_offset_weight=translation_offset_weight,
            rotation_offset_weight=rotation_offset_weight,
            rgb_rescaling=rgb_rescaling,
            alpha_rescaling=alpha_rescaling,
            plane_scale=plane_scale,
            plane_scale_offset=plane_scale_offset,
            translation=translation,
            orientation=orientation,
            initial_rgb=initial_rgb,
            initial_alpha=initial_alpha,
            rgb_weight=rgb_weight,
            alpha_weight=alpha_weight,
            position=position,
            times=times,
            name=name,
            children=children,
            decoding=decoding,
            dtype=dtype,
            index=index,
            proxy_init=proxy_init,
            **kwargs)
        # Check if alpha has Shape (C, H, W)
        self.initial_alpha.data = flatten_batch_dims(self.initial_alpha, -3)[0]
        self.align_corners = align_corners

    def get_initial_alpha(self, uv: torch.Tensor) -> torch.Tensor:
        """Gets the initial alpha for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            Uv coordinates of the point.
            Shape: (B, 2) x, y should be in range [-0.5, 0.5]

        Returns
        -------
        torch.Tensor
            The alpha values in shape (B, 1)
        """
        grid = (uv * 2)[None, None, ...]  # (1, 1, B, 2) to match gridsample
        out = torch.nn.functional.grid_sample(self.initial_alpha.unsqueeze(0),
                                              grid, mode="bilinear", padding_mode="border", align_corners=self.align_corners)[0].reshape(1, -1)
        # Swap channel dimension to the end
        out = out.permute(1, 0)
        return out
