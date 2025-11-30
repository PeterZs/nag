from typing import Iterable, Optional
from nag.model.camera_scene_node_3d import CameraSceneNode3D
from tools.model.discrete_module_scene_node_3d import DiscreteModuleSceneNode3D
import torch
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify


class DiscreteCameraSceneNode3D(DiscreteModuleSceneNode3D, CameraSceneNode3D):
    """A scene node representing a camera in 3D space with discrete position and orientation."""

    def __init__(
        self,
            image_resolution: VEC_TYPE,
            lens_distortion: VEC_TYPE,
            intrinsics: VEC_TYPE,
            translation: Optional[VEC_TYPE] = None,
            orientation: Optional[VEC_TYPE] = None,
            position: Optional[torch.Tensor] = None,
            name: Optional[str] = None,
            children: Optional[Iterable['AbstractSceneNode']] = None,
            decoding: bool = False,
            dtype: torch.dtype = torch.float32,
            **kwargs
    ):
        super().__init__(
            image_resolution=image_resolution,
            lens_distortion=lens_distortion,
            intrinsics=intrinsics,
            translation=translation,
            orientation=orientation,
            position=position,
            name=name,
            children=children,
            decoding=decoding,
            dtype=dtype,
            **kwargs)
