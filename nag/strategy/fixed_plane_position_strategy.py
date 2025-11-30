from nag.strategy.plane_position_strategy import PlanePositionStrategy, compute_plane_scale, compute_proto_plane_position_centeroid, gaussian_smoothing, get_plane_support_points, interpolate_plane_position, mask_to_camera_coordinates
import torch
from nag.strategy.strategy import Strategy
from typing import Any, Dict, Optional, Tuple
from nag.config.nag_config import NAGConfig
from nag.model.discrete_plane_scene_node_3d import local_to_plane_coordinates
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
import torch
from tools.transforms.geometric.quaternion import quat_composition, quat_subtraction, quat_average
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


class FixedPlanePositionStrategy(PlanePositionStrategy):

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

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
        T, C_img, H, W = images.shape
        global_position, global_center, global_orientation, interp_times = compute_plane_position(
            images=images,
            mask=mask,
            depths=depths,
            times=times,
            camera=camera,
            mask_properties=mask_properties,
            dtype=dtype,
            device=device,
            position_spline_fitting=self.position_spline_fitting,
            position_spline_control_points=self.position_spline_control_points,
            translation_smoothing=self.translation_smoothing,
            orientation_smoothing=self.orientation_smoothing,
            orientation_locking=self.orientation_locking,
            smoothing_kernel_size=self.smoothing_kernel_size,
            smoothing_sigma=self.smoothing_sigma,
            depth_ray_thickness=self.depth_ray_thickness,
            min_depth_ray_thickness=self.min_depth_ray_thickness,
        )
        # plane_properties["times"] = interp_times
        mask_resolution = torch.tensor(
            [H, W], dtype=dtype, device=mask.device)

        plane_scale = get_fixed_plane_scale(
            mask_properties=mask_properties,
            global_plane_position=global_position,
            camera=camera,
            times=times,
            mask_resolution=mask_resolution,
            relative_plane_margin=config.relative_plane_margin
        )
        plane_properties["position_init_strategy"] = self
        return global_position, global_center, global_orientation, interp_times, plane_scale


def compute_fixed_native_plane_scale(
    border_coords: torch.Tensor,
    resolution: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    times: torch.Tensor,
    global_plane_position: torch.Tensor,
) -> torch.Tensor:
    """Compute the plane scale for a fixed plane.

    Parameters
    ----------
    border_coords: torch.Tensor
        The border coordinates. Shape: (4, T, 2) (y, x)
    In counter clockwise order.
        bl, br, tr, tl

    resolution : torch.Tensor
        Resolution of the Mask. (H, W)
    camera : torch.Tensor
        The camera
    times : torch.Tensor
        The times. Shape (T,)
    global_plane_position : torch.Tensor
        The global plane position. Shape (T, 4, 4)
    relative_plane_margin : torch.Tensor
        Relative plane margin, as a percentage of the mask size.
        Used to enlarge the plane on every side.

    Returns
    -------
    torch.Tensor
        The plane scale. Shape (2, ) (x, y)
    """
    border_coords_cam = mask_to_camera_coordinates(
        border_coords, resolution, camera._image_resolution)

    # Switch to xy
    border_coords_cam = torch.flip(border_coords_cam, dims=(-1,))

    coord_ro, coord_rd = camera.get_global_rays(
        uv=border_coords_cam.swapaxes(0, 1), t=times, uv_includes_time=True)  # Shape B, T, 3

    intersection_points = compute_ray_plane_intersections_from_position_matrix(
        global_plane_position.unsqueeze(0).repeat(4, 1, 1, 1), coord_ro, coord_rd)  # B, T, 3
    #
    local_intersection_points = global_to_local(
        global_plane_position, intersection_points, v_include_time=True)[..., :2]

    # Flatten border points and times should give us the min and max at any time, factoring in the camera movement
    local_intersection_points = local_intersection_points.reshape(
        4 * len(times), 2)

    min_x = torch.amin(local_intersection_points[..., 0], dim=0)
    max_x = torch.amax(local_intersection_points[..., 0], dim=0)
    min_y = torch.amin(local_intersection_points[..., 1], dim=0)
    max_y = torch.amax(local_intersection_points[..., 1], dim=0)
    x_scale = max_x - min_x
    y_scale = max_y - min_y
    scale = torch.stack([x_scale, y_scale], dim=-1)
    return scale


def compute_fixed_plane_scale(
    border_coords: torch.Tensor,
    resolution: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    times: torch.Tensor,
    global_plane_position: torch.Tensor,
    relative_plane_margin: torch.Tensor,
) -> torch.Tensor:
    """Compute the plane scale for a fixed plane.

    Parameters
    ----------
    border_coords: torch.Tensor
        The border coordinates. Shape: (4, T, 2) (y, x)
    In counter clockwise order.
        bl, br, tr, tl

    resolution : torch.Tensor
        Resolution of the Mask. (H, W)
    camera : torch.Tensor
        The camera
    times : torch.Tensor
        The times. Shape (T,)
    global_plane_position : torch.Tensor
        The global plane position. Shape (T, 4, 4)
    relative_plane_margin : torch.Tensor
        Relative plane margin, as a percentage of the mask size.
        Used to enlarge the plane on every side.

    Returns
    -------
    torch.Tensor
        The plane scale. Shape (2, ) (x, y)
    """
    native_scale = compute_fixed_native_plane_scale(
        border_coords=border_coords,
        resolution=resolution,
        camera=camera,
        times=times,
        global_plane_position=global_plane_position
    )
    padded_scale = native_scale * (1 + relative_plane_margin)
    return padded_scale


def get_fixed_plane_scale(
    mask_properties: BasicMaskProperties,
    global_plane_position: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    times: torch.Tensor,
    mask_resolution: torch.Tensor,
    relative_plane_margin: torch.Tensor,
) -> torch.Tensor:
    """Method to get the plane scale for a fixed plane

    Plane scale is determined by the mask size and the relative plane margin and the position of the planes.

    Parameters
    ----------
    mask_properties : BasicMaskProperties
        Basic mask properties.
    camera : TimedCameraSceneNode3D
        The camera.
    times : torch.Tensor
        The times of the masks.
    mask_resolution : torch.Tensor
        The resolution of the mask.

    Returns
    -------
    torch.Tensor
        The plane scale. Shape: (2, ) (x, y)
    """
    missing_in_frame = mask_properties.missing_in_frame
    plane_scale = compute_fixed_plane_scale(
        border_coords=mask_properties.border_points[:, ~missing_in_frame],
        resolution=mask_resolution,
        camera=camera,
        times=times[~missing_in_frame],
        global_plane_position=global_plane_position[~missing_in_frame],
        relative_plane_margin=tensorify(relative_plane_margin)
    )
    return plane_scale


def compute_plane_position(
    images: torch.Tensor,
    mask: torch.Tensor,
    depths: torch.Tensor,
    times: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    mask_properties: BasicMaskProperties,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,

    position_spline_fitting: bool = False,
    position_spline_control_points: Optional[int] = None,
    translation_smoothing: bool = True,
    orientation_smoothing: bool = True,
    orientation_locking: bool = False,
    smoothing_kernel_size: int = 7,
    smoothing_sigma: float = 5.0,
    depth_ray_thickness: float = 0.05,
    min_depth_ray_thickness: float = 10.,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the plane position.

    Treats the plane as a fixed plane keeping the orientation and position constant over time.

    Parameters
    ----------
    images : torch.Tensor
        The images. Shape: (T, C, H, W)

    mask : torch.Tensor
        The mask. Shape: (T, 1, H, W)

    depths : torch.Tensor
        The depths. Shape: (T, 1, H, W)

    times : torch.Tensor
        The times. Shape: (T,)

    camera : TimedCameraSceneNode3D
        The camera.

    mask_properties : BasicMaskProperties
        The mask properties.

    dtype : torch.dtype
        The dtype.

    device : torch.device
        The device.

    position_spline_fitting : bool
        If a spline fitting should be applied to the position and orientation to smooth out the values.

    position_spline_control_points : Optional[int]
        The number of control points for the spline fitting. If None, it will be half of the times.

    translation_smoothing : bool
        If the translation should be smoothed.

    orientation_smoothing : bool
        If the orientation should be smoothed.

    orientation_locking : bool
        If the orientation should be locked to the median orientation.

    smoothing_kernel_size : int
        The kernel size for the smoothing.

    smoothing_sigma : float
        The sigma for the smoothing.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        1. The plane positions as global position matrix. Shape: (T, 4, 4)
        2. The plane global center position. Shape: (T^, 3) (x, y, z)
            As the plane positions could be interpolated, this is in the non-interpolated form of having maybe a different length.
        3. The plane global unit quarternions for the orientation. Shape: (T^, 4) (x, y, z, w)
        4. The times of the plane positions. Shape: (T^,) If not interpolation is performed, this is the same as the input times.

    """
    global_plane_position = compute_proto_plane_position_centeroid(
        images=images,
        mask=mask,
        depths=depths,
        times=times,
        camera=camera,
        mask_properties=mask_properties,
        dtype=dtype,
        device=device,
        depth_ray_thickness=depth_ray_thickness,
        min_depth_ray_thickness=min_depth_ray_thickness,
    )
    missing_in_frame = mask_properties.missing_in_frame
    plane_center = global_plane_position[:, :3, 3]

    # As the plane should be fixed, we take the mean
    quat = quat_average(rotmat_to_unitquat(
        global_plane_position[~missing_in_frame, :3, :3]))
    center = plane_center[~missing_in_frame].mean(dim=0)

    global_plane_position[:, :3, 3] = center
    global_plane_position[:, :3, :3] = unitquat_to_rotmat(quat)

    interp_t = times
    T = global_plane_position.shape[0]
    if position_spline_fitting:
        if position_spline_control_points is None:
            position_spline_control_points = T // 2
        T = position_spline_control_points
        interp_t = torch.linspace(
            0, 1, position_spline_control_points, dtype=dtype, device=device)
    quat = quat.repeat(T, 1)
    center = center.repeat(T, 1)

    return global_plane_position, center, quat, interp_t
