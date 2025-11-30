from typing import Iterable, Optional

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from nag.model.camera_scene_node_3d import CameraSceneNode3D
from nag.model.learned_offset_scene_node_3d import LearnedOffsetSceneNode3D
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D

import torch
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims, compose_transformation_matrix

from nag.transforms.transforms_timed_3d import interpolate_vector


class LearnedCameraSceneNode3D(
    LearnedOffsetSceneNode3D,
    TimedCameraSceneNode3D
):
    """A scene node representing a camera in 3D space with timed discrete position and orientation and learnable offsets."""

    def __init__(
        self,
            num_rigid_control_points: int,
            image_resolution: VEC_TYPE,
            lens_distortion: VEC_TYPE,
            intrinsics: VEC_TYPE,
            translation_offset_weight: float = 0.03,
            rotation_offset_weight: float = 0.03,
            translation: Optional[VEC_TYPE] = None,
            orientation: Optional[VEC_TYPE] = None,
            position: Optional[torch.Tensor] = None,
            times: Optional[VEC_TYPE] = None,
            name: Optional[str] = None,
            children: Optional[Iterable['AbstractSceneNode']] = None,
            decoding: bool = False,
            dtype: torch.dtype = torch.float32,
            **kwargs
    ):
        super().__init__(
            num_control_points=num_rigid_control_points,
            image_resolution=image_resolution,
            lens_distortion=lens_distortion,
            intrinsics=intrinsics,
            translation_offset_weight=translation_offset_weight,
            rotation_offset_weight=rotation_offset_weight,
            translation=translation,
            orientation=orientation,
            position=position,
            times=times,
            name=name,
            children=children,
            decoding=decoding,
            dtype=dtype,
            **kwargs)
