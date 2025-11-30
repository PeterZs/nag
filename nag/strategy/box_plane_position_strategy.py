from enum import Enum

from matplotlib import pyplot as plt
from nag.homography.sequence_homography_finder import (SequenceHomographyFinder,
                                                       SequenceHomographyFinderConfig,
                                                       plot_tracked_points)
from nag.homography.homography_finder import is_coord_in_mask
from nag.model.timed_box_scene_node_3d import TimedBoxSceneNode3D
from nag.strategy.plane_position_strategy import (PlanePositionStrategy, compute_plane_scale,
                                                  compute_proto_plane_position_centeroid,
                                                  gaussian_smoothing, get_plane_support_points,
                                                  interpolate_plane_position, mask_to_camera_coordinates,
                                                  plot_proto_points)
import torch
from nag.strategy.strategy import Strategy
from typing import Any, Dict, List, Optional, Tuple
from nag.config.nag_config import NAGConfig
from nag.model.discrete_plane_scene_node_3d import local_to_plane_coordinates
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
import torch
from tools.transforms.geometric.quaternion import quat_composition, quat_subtraction, quat_average, quat_product_scalar, quat_product
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
from nag.model.timed_plane_scene_node_3d import TimedPlaneSceneNode3D
from tools.util.format import parse_enum
from tools.viz.matplotlib import plot_as_image, saveable
from tools.logger.logging import logger
from nag.strategy.in_image_plane_position_strategy import get_plane_scale, refine_plane_position
from tools.labels.timed_box_3d import TimedBox3D
from tools.util.torch import index_of_first
from nag.transforms.transforms_timed_3d import align_rectangles


class BoxPlanePositionStrategy(PlanePositionStrategy):

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def execute(self,
                images: torch.Tensor,
                mask: torch.Tensor,
                depths: torch.Tensor,
                times: torch.Tensor,
                box: TimedBox3D,
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
            box=box,
            camera=camera,
            mask_properties=mask_properties,
            dtype=dtype,
            device=device,
            plane_properties=plane_properties,
            position_spline_fitting=self.position_spline_fitting,
            position_spline_control_points=self.position_spline_control_points,
            translation_smoothing=self.translation_smoothing,
            orientation_smoothing=self.orientation_smoothing,
            orientation_locking=self.orientation_locking,
            smoothing_kernel_size=self.smoothing_kernel_size,
            smoothing_sigma=self.smoothing_sigma,
            depth_ray_thickness=self.depth_ray_thickness,
            min_depth_ray_thickness=self.min_depth_ray_thickness,
            plot_tracked_points=self.plot_tracked_points,
            plot_tracked_points_path=self.plot_tracked_points_path,
        )
        # plane_properties["times"] = interp_times
        mask_resolution = torch.tensor(
            [H, W], dtype=dtype, device=mask.device)

        plane_scale = get_plane_scale(
            mask_properties=mask_properties,
            global_plane_position=global_position,
            camera=camera,
            times=times,
            mask_resolution=mask_resolution,
            relative_plane_margin=config.relative_plane_margin
        )
        plane_properties["position_init_strategy"] = self
        return global_position, global_center, global_orientation, interp_times, plane_scale


def compute_plane_position(
    images: torch.Tensor,
    mask: torch.Tensor,
    depths: torch.Tensor,
    box: TimedBox3D,
    times: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    mask_properties: BasicMaskProperties,
    plane_properties: Dict[str, Any],
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
    plot_tracked_points: bool = False,
    plot_tracked_points_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the plane position.

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
    from tools.transforms.geometric.transforms3d import component_position_matrix, vector_angle_3d, vector_angle, norm_rotation_angles
    proto_position_error = torch.zeros(
        len(times), dtype=torch.bool, device=device)
    global_plane_position = torch.eye(
        4, dtype=dtype, device=device).repeat(len(times), 1, 1)

    missing_in_frame = mask_properties.missing_in_frame

    box_node = TimedBoxSceneNode3D.from_timed_box_3d(box)

    if plane_properties is not None:
        plane_properties["box"] = box_node

    unknown_position = index_of_first(box_node._times, times)
    proto_position_error = (unknown_position == -1)

    # Get the center
    global_plane_position = box_node.get_global_position(t=times)

    # Check if the 90 degree rotation is needed (angle w.r.t the camera is better)
    rot_y = component_position_matrix(
        angle_y=90, mode="deg", device=device, dtype=dtype)
    global_plane_position_rot = global_plane_position @ rot_y

    # Compute the angle between the camera and the plane in the box.

    z_vector = torch.tensor(
        [0., 0., 1., 1.], dtype=dtype, device=device)[None, ...]
    camera_position = camera.get_global_position(
        t=times[~missing_in_frame]).detach()

    T = camera_position.shape[0]
    cam_z_pos = torch.bmm(camera_position, z_vector[..., None].expand(
        T, -1, -1))[:, :3, 0]  # Shape: (T, 3)
    cam_z_vector = cam_z_pos - camera_position[:, :3, 3]

    global_plane_position_existed = global_plane_position[~missing_in_frame]
    plane_z_pos = torch.bmm(global_plane_position_existed, z_vector[..., None].expand(
        T, -1, -1))[:, :3, 0]  # Shape: (T, 3)
    plane_z_vector = plane_z_pos - global_plane_position_existed[:, :3, 3]
    angle = norm_rotation_angles(vector_angle_3d(cam_z_vector, plane_z_vector))

    global_plane_position_rot_existed = global_plane_position_rot[~missing_in_frame]
    plane_rot_z_pos = torch.bmm(global_plane_position_rot_existed, z_vector[..., None].expand(
        T, -1, -1))[:, :3, 0]  # Shape: (T, 3)
    plane_rot_z_vector = plane_rot_z_pos - \
        global_plane_position_rot_existed[:, :3, 3]
    angle_rot = norm_rotation_angles(
        vector_angle_3d(cam_z_vector, plane_rot_z_vector))

    # Check if the box diagonals are better for the rotated plane
    edges_one = [4, 6, 2, 0]
    edges_two = [7, 5, 1, 3]

    exist_t = times[~missing_in_frame][0]
    box_corners = box_node.get_local_corners()

    corners_one = box_corners[edges_one][:, :3]
    corners_two = box_corners[edges_two][:, :3]
    box_diag_width = torch.norm(corners_one[0] - corners_one[1], dim=-1)

    plane_points = torch.tensor([[-1, -1],
                                 [1, -1],
                                 [1,  1],
                                 [-1,  1],
                                 ])
    raw_plane_size = torch.stack(
        [box_diag_width.squeeze(), box_node.size[1]], dim=-1)
    plane_node = TimedPlaneSceneNode3D(
        position=global_plane_position_existed, plane_scale=raw_plane_size, times=times[~missing_in_frame])
    plane_corners = plane_node.get_plane_corners()[:, :3]

    # local_plane_corners = plane_points * raw_plane_size[None, :]
    # local_plane_corners = torch.cat([local_plane_corners, torch.zeros(4, 1, dtype=dtype, device=device)], dim=-1)
    diag_rot_one = align_rectangles(plane_corners, corners_one)
    diag_rot_two = align_rectangles(plane_corners, corners_two)

    global_plane_position_diag_one = global_plane_position @ diag_rot_one
    global_plane_position_diag_two = global_plane_position @ diag_rot_two

    global_plane_position_diag_one_existed = global_plane_position_diag_one[~missing_in_frame]
    plane_z_pos_diag_one = torch.bmm(global_plane_position_diag_one_existed, z_vector[..., None].expand(
        T, -1, -1))[:, :3, 0]  # Shape: (T, 3)
    plane_z_vector_diag_one = plane_z_pos_diag_one - \
        global_plane_position_diag_one_existed[:, :3, 3]
    angle_diag_one = norm_rotation_angles(
        vector_angle_3d(cam_z_vector, plane_z_vector_diag_one))

    global_plane_position_diag_two_existed = global_plane_position_diag_two[~missing_in_frame]
    plane_z_pos_diag_two = torch.bmm(global_plane_position_diag_two_existed, z_vector[..., None].expand(
        T, -1, -1))[:, :3, 0]  # Shape: (T, 3)
    plane_z_vector_diag_two = plane_z_pos_diag_two - \
        global_plane_position_diag_two_existed[:, :3, 3]
    angle_diag_two = norm_rotation_angles(
        vector_angle_3d(cam_z_vector, plane_z_vector_diag_two))

    # Check if the angle is better for the rotated plane
    abs_angle = torch.abs(angle)
    abs_angle_rot = torch.abs(angle_rot)
    abs_angle_diag_one = torch.abs(angle_diag_one)
    abs_angle_diag_two = torch.abs(angle_diag_two)

    delta_angle = torch.rad2deg(torch.where(abs_angle < (
        torch.pi / 2), abs_angle, torch.pi - abs_angle))
    delta_angle_rot = torch.rad2deg(torch.where(abs_angle_rot < (
        torch.pi / 2), abs_angle_rot, torch.pi - abs_angle_rot))
    delta_angle_diag_one = torch.rad2deg(torch.where(abs_angle_diag_one < (
        torch.pi / 2), abs_angle_diag_one, torch.pi - abs_angle_diag_one))
    delta_angle_diag_two = torch.rad2deg(torch.where(abs_angle_diag_two < (
        torch.pi / 2), abs_angle_diag_two, torch.pi - abs_angle_diag_two))

    all_global_position_estimates = torch.stack([
        global_plane_position,
        global_plane_position_rot,
        global_plane_position_diag_one,
        global_plane_position_diag_two], dim=0)

    # Plane projection which is the best
    labels = ["XY", "YZ", "Diagonal 1", "Diagonal 2"]
    # fig = plane_plot_test(all_global_position_estimates[:, 0], box_node, labels=["Original", "Rotated", "Diagonal 1", "Diagonal 2"],
    #                       corners_native=plane_node.local_to_global(plane_corners, t=exist_t)[:, 0],
    #                       corners_diag_one=plane_node.local_to_global(corners_one, t=exist_t)[:, 0],
    #                       corners_diag_two=plane_node.local_to_global(corners_two, t=exist_t)[:, 0])
    # fig.show()

    all_angles = torch.stack([
        delta_angle,
        delta_angle_rot,
        delta_angle_diag_one,
        delta_angle_diag_two], dim=0)

    min_select = torch.argmin(all_angles.mean(dim=(1, 2)))
    global_plane_position = all_global_position_estimates[min_select]
    used_delta = all_angles[min_select]
    logger.info(
        f"Selected projection: {labels[min_select]} for plane '{plane_properties.get('name')}'")

    # Display a warning if the delta angle is to large as this would indicate a close-to-orthogonal plane w.r.t the camera
    if torch.any(used_delta[:, :-1] > 75): # TODO: Need to verify this.
        max_val = torch.amax(used_delta)
        max_t_idx = torch.argwhere(used_delta == max_val).squeeze()[0]
        logger.warning(f"Normal of plane '{plane_properties.get('name')}' is nearly to orthogonal to the camera, this makes projection very hard. \
                        Max-Delta angles: {[str(x) for x in used_delta[max_t_idx].tolist()]} at time {times[~missing_in_frame][max_t_idx].tolist()}")

    from nag.model.timed_discrete_scene_node_3d import save_tensor

    # save_tensor(global_plane_position, "global_plane_position", times=times, index=plane_properties.get("index"), path="temp/position_spline_tests")
    # save_tensor(missing_in_frame, "missing_in_frame", times=times, index=plane_properties.get("index"), path="temp/position_spline_tests")
    # save_tensor(proto_position_error, "proto_position_error", times=times, index=plane_properties.get("index"), path="temp/position_spline_tests")
    # save_tensor(times, "times", times=times, index=plane_properties.get("index"), path="temp/position_spline_tests")

    return refine_plane_position(
        global_plane_position=global_plane_position,
        proto_position_error=proto_position_error,
        times=times,
        camera=camera,
        mask_properties=mask_properties,
        dtype=dtype,
        device=device,
        position_spline_fitting=position_spline_fitting,
        position_spline_control_points=position_spline_control_points,
        translation_smoothing=translation_smoothing,
        orientation_smoothing=orientation_smoothing,
        orientation_locking=orientation_locking,
        smoothing_kernel_size=smoothing_kernel_size,
        smoothing_sigma=smoothing_sigma,
        only_correct_proto_error=True,
    )


def plane_plot_test(
    global_plane_positions: torch.Tensor,
    node: TimedBoxSceneNode3D,
    labels: Optional[List[str]] = None,
    corners_native: Optional[torch.Tensor] = None,
    corners_diag_one: Optional[torch.Tensor] = None,
    corners_diag_two: Optional[torch.Tensor] = None,
    **kwargs
) -> plt.Figure:
    from nag.model.discrete_plane_scene_node_3d import DiscretePlaneSceneNode3D
    from tools.model.discrete_module_scene_node_3d import DiscreteModuleSceneNode3D

    world = DiscreteModuleSceneNode3D(name="world")

    for i in range(global_plane_positions.shape[0]):
        name = f"{i}"
        if labels is not None and i < len(labels):
            name += ': ' + labels[i]
        plane = DiscretePlaneSceneNode3D(
            name=name,
            position=global_plane_positions[i])
        world.add_scene_children(plane)

    world.add_scene_children(node)

    fig = world.plot_scene(t=0., **kwargs)
    ax = fig.gca()
    if corners_native is not None and corners_diag_one is not None:
        target = corners_diag_one - corners_native
        # Ax is 3D axis
        ax.quiver(corners_native[:, 0], corners_native[:, 1], corners_native[:, 2],
                  target[:, 0], target[:, 1], target[:, 2], color='r', length=1, normalize=False)

    if corners_native is not None and corners_diag_two is not None:
        target = corners_diag_two - corners_native
        # Ax is 3D axis
        ax.quiver(corners_native[:, 0], corners_native[:, 1], corners_native[:, 2],
                  target[:, 0], target[:, 1], target[:, 2], color='g', length=1, normalize=False)

    return fig
