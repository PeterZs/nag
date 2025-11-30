from nag.strategy.plane_position_strategy import PlanePositionStrategy, compute_plane_scale, compute_proto_plane_position_centeroid, gaussian_smoothing, get_plane_support_points, interpolate_plane_position, mask_to_camera_coordinates
import torch
from nag.strategy.strategy import Strategy
from typing import Any, Dict, Optional, Tuple
from nag.config.nag_config import NAGConfig
from nag.model.discrete_plane_scene_node_3d import local_to_plane_coordinates
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
import torch
from tools.transforms.geometric.quaternion import quat_composition, quat_subtraction
from nag.transforms.transforms3d import _linear_interpolate_position_rotation
from nag.transforms.transforms_timed_3d import linear_interpolate_vector
from nag.strategy.base_plane_initialization_strategy import BasicMaskProperties
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims
from kornia.utils.draw import draw_convex_polygon
from tools.transforms.geometric.transforms3d import compute_ray_plane_intersections_from_position_matrix
from nag.model.timed_discrete_scene_node_3d import global_to_local, local_to_global
from nag.model.discrete_plane_scene_node_3d import default_plane_scale_offset
import torch
import torch.nn.functional as F
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims
from tools.transforms.geometric.mappings import rotmat_to_unitquat, rotvec_to_unitquat, unitquat_to_rotvec, unitquat_to_rotmat
from tools.transforms.geometric.transforms3d import calculate_rotation_matrix, compute_ray_plane_intersections_from_position_matrix
from tools.transforms.to_tensor import tensorify
from nag.strategy.fixed_plane_position_strategy import FixedPlanePositionStrategy
from nag.strategy.in_image_plane_position_strategy import InImagePlanePositionStrategy


class BorderConditionPlanePositionStrategy(PlanePositionStrategy):
    """Simple decision rule to decide between FixedPlanePositionStrategy and InImagePlanePositionStrategy based on border contact."""

    def __init__(self,
                 border_padding_threshold: int = 15,
                 **kwargs):
        super().__init__(**kwargs)
        self.border_padding_threshold = border_padding_threshold
        """Pixel threshold which still counts as border contact"""

    def execute(self,
                images: torch.Tensor,
                mask: torch.Tensor,
                depths: torch.Tensor,
                times: torch.Tensor,
                config: NAGConfig,
                camera: TimedCameraSceneNode3D,
                mask_properties: BasicMaskProperties,
                plane_properties: Dict[str, Any],
                dtype: torch.dtype = torch.float32,
                device: torch.device = None,
                **kwargs
                ):
        # Crop the image at border contacxt
        bth = self.border_padding_threshold
        cmasks = mask[:, :, bth:-bth, bth:-bth]
        hit_left = cmasks[..., 0].any(dim=(-1, -2))
        hit_right = cmasks[..., -1].any(dim=(-1, -2))
        hit_top = cmasks[..., 0].any(dim=(-1, -2))
        hit_bottom = cmasks[..., -1, :].any(dim=(-1, -2))

        # If an objects touches the border in 2 directions, the object is probably larger than the image and we should use a fixed plane
        opposing_touch = torch.stack(
            [(hit_left & hit_right).any(), (hit_top & hit_bottom).any()], dim=-1)
        strategy_type = None

        if self.box_available:
            from nag.strategy.box_plane_position_strategy import BoxPlanePositionStrategy
            strategy_type = BoxPlanePositionStrategy
        else:
            if opposing_touch.any():
                strategy_type = FixedPlanePositionStrategy
            else:
                strategy_type = InImagePlanePositionStrategy

        strategy = strategy_type(
            depth_ray_thickness=self.depth_ray_thickness,
            min_depth_ray_thickness=self.min_depth_ray_thickness,
            relative_plane_margin=self.relative_plane_margin,
            position_spline_fitting=self.position_spline_fitting,
            position_spline_control_points=self.position_spline_control_points,
            translation_smoothing=self.translation_smoothing,
            orientation_smoothing=self.orientation_smoothing,
            orientation_locking=self.orientation_locking,
            smoothing_kernel_size=self.smoothing_kernel_size,
            smoothing_sigma=self.smoothing_sigma,
            plot_tracked_points=self.plot_tracked_points,
            plot_tracked_points_path=self.plot_tracked_points_path,
        )
        return strategy.execute(images=images, mask=mask, depths=depths, times=times, config=config, camera=camera, mask_properties=mask_properties, plane_properties=plane_properties, dtype=dtype, device=device, **kwargs)
