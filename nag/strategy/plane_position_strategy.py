from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from kornia.utils.draw import draw_convex_polygon
from matplotlib import pyplot as plt
from tools.transforms.geometric.transforms3d import (
    calculate_rotation_matrix,
    compute_ray_plane_intersections_from_position_matrix, flatten_batch_dims,
    unflatten_batch_dims)
from tools.transforms.to_tensor import tensorify
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims
from tools.viz.matplotlib import plot_as_image, saveable

from nag.config.nag_config import NAGConfig
from nag.dataset.nag_dataset import NAGDataset
from nag.homography.sequence_homography_finder import plot_tracked_points
from nag.model.discrete_plane_scene_node_3d import (default_plane_scale_offset,
                                                    local_to_plane_coordinates)
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import global_to_local
from nag.strategy.base_plane_initialization_strategy import BasicMaskProperties
from nag.strategy.strategy import Strategy
from tools.transforms.geometric.mappings import (rotmat_to_unitquat, rotvec_to_unitquat,
                                                 unitquat_to_rotmat, unitquat_to_rotvec)
from nag.transforms.transforms3d import _linear_interpolate_position_rotation
from nag.transforms.transforms_timed_3d import linear_interpolate_vector


class PlanePositionStrategy(Strategy):
    """Plane positioning strategy to initialize the plane position."""

    def __init__(self,
                 depth_ray_thickness: float = 0.05,
                 min_depth_ray_thickness: float = 10.,
                 relative_plane_margin: float = 0.1,
                 position_spline_fitting: bool = False,
                 position_spline_control_points: Optional[int] = None,
                 translation_smoothing: bool = True,
                 orientation_smoothing: bool = True,
                 orientation_locking: bool = False,
                 smoothing_kernel_size: int = 7,
                 smoothing_sigma: float = 5.0,
                 plot_tracked_points: bool = False,
                 plot_tracked_points_path: Optional[str] = None,
                 box_available: bool = False,
                 ):
        super().__init__()
        self.depth_ray_thickness = depth_ray_thickness
        """The thickness of the depth ray. Ray along a mask trajectory on the depth mask to sample plane support points."""
        self.min_depth_ray_thickness = min_depth_ray_thickness
        """The minimum thickness of the depth ray."""
        self.relative_plane_margin = relative_plane_margin
        """Relative plane margin to add to the estimated scale in encount for errors."""
        self.position_spline_fitting = position_spline_fitting
        self.position_spline_control_points = position_spline_control_points
        self.translation_smoothing = translation_smoothing
        self.orientation_smoothing = orientation_smoothing
        self.orientation_locking = orientation_locking
        self.smoothing_kernel_size = smoothing_kernel_size
        self.smoothing_sigma = smoothing_sigma
        self.plot_tracked_points = plot_tracked_points
        self.plot_tracked_points_path = plot_tracked_points_path
        self.box_available = box_available
        """If a box is available for the object."""

    @abstractmethod
    def execute(self,
                images: torch.Tensor,
                mask: torch.Tensor,
                depths: torch.Tensor,
                times: torch.Tensor,
                config: NAGConfig,
                dataset: NAGDataset,
                camera: TimedCameraSceneNode3D,
                mask_properties: BasicMaskProperties,
                plane_properties: Dict[str, Any],
                dtype: torch.dtype = torch.float32,
                device: torch.device = None,
                **kwargs
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the plane position for each object represented by the mask in the images.


        Parameters
        ----------
        images : torch.Tensor
            The images. Shape: (T, C_img, H, W)

        mask : torch.Tensor
            The mask. Shape: (T, 1, H, W)

        depths : torch.Tensor
            The depths. Shape: (T, 1, H, W)

        times : torch.Tensor
            The times. Shape: (T,)

        config : NAGConfig
            The config.

        dataset : NAGDataset
            The dataset.

        camera : TimedCameraSceneNode3D
            The camera.

        mask_properties : BasicMaskProperties
            The mask properties.

        plane_properties : Dict[str, Any]
            The plane properties.

        dtype : torch.dtype
            The dtype.

        device : torch.device
            The device.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            1. The plane positions as global position matrix. Shape: (T, 4, 4)
            2. The plane global center position. Shape: (T^, 3) (x, y, z)
            As the plane positions could be interpolated, this is in the non-interpolated form of having maybe a different length.
            3. The plane global unit quarternions for the orientation. Shape: (T^, 4) (x, y, z, w)
            4. The times of the plane positions. Shape: (T^,) If not interpolation is performed, this is the same as the input times.
            5. The plane scales. Shape: (T, 2) (x, y)
        """
    pass


@saveable()
def plot_proto_points(
        images: torch.Tensor,
        masks: torch.Tensor,
        proto_points: torch.Tensor,
        mask_center_point: torch.Tensor,
        resolution: Optional[torch.Tensor] = None,
        tight: bool = True,
        frame_numbers: bool = True,
) -> plt.Figure:
    """Plots the proto plane points and the mask center point on the image.

    Parameters
    ----------
    images : torch.Tensor
        Stack of images. Shape: (T, C, H, W)
    masks : torch.Tensor
        Stack of masks. Shape: (T, 1, H, W)
    proto_points : torch.Tensor
        Plane proto points in image coordinates. Shape: (T, 3, 2)
    mask_center_point : torch.Tensor
        Mask center point in image coordinates. Shape: (T, 1, 2)
    resolution : torch.Tensor
        Resolution of the mask and image. (H, W)
    frame_numbers : bool
        If to plot the frame numbers.

    Returns
    -------
    plt.Figure
        Figure with the plot.
    """

    pts = torch.cat([proto_points, mask_center_point], dim=1)
    colors = torch.tensor([[1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]], dtype=torch.float32)  # Red, Green, Blue, White
    colors = (colors.unsqueeze(0).expand(pts.shape[0], -1, -1))

    if resolution is not None:
        resolution = tensorify(
            resolution, dtype=torch.float32, device=pts.device)
        ratio = resolution / \
            torch.tensor(images.shape[2:],
                         dtype=torch.float32, device=pts.device)
        if not (ratio == 1).all():
            ratio_xy = ratio.flip(-1)
            images = F.interpolate(images, size=tuple(
                resolution.round().int().tolist()), mode='bilinear')
            masks = F.interpolate(masks.float(), size=tuple(
                resolution.round().int().tolist()), mode='nearest').bool()
            pts = pts * ratio_xy
    return plot_tracked_points(images, masks, pts, colors, tight=tight, frame_numbers=frame_numbers)


def compute_native_plane_scale(
    border_coords: torch.Tensor,
    resolution: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    times: torch.Tensor,
    global_plane_position: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the plane scale per timed plane.
    This is defined by the border coordinates of the mask and the global plane position.
    E.g. Assuming a tight bounding box on the mask, the scale is the difference between the min and max x and y coordinates
    of the intersection points of those outer rays with the plane.

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

    Returns
    -------
    torch.Tensor
        The plane scale. Shape (T, 2) (x, y)
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
    min_x = torch.amin(local_intersection_points[..., 0], dim=0)
    max_x = torch.amax(local_intersection_points[..., 0], dim=0)
    min_y = torch.amin(local_intersection_points[..., 1], dim=0)
    max_y = torch.amax(local_intersection_points[..., 1], dim=0)
    x_scale = max_x - min_x
    y_scale = max_y - min_y
    return torch.stack([x_scale, y_scale], dim=-1)


def compute_plane_scale(
    border_coords: torch.Tensor,
    resolution: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    times: torch.Tensor,
    global_plane_position: torch.Tensor,
    relative_plane_margin: torch.Tensor,
) -> torch.Tensor:
    """Compute the plane scale per timed plane.

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
        The plane scale. Shape (T, 2) (x, y)
    """
    native_scale = compute_native_plane_scale(
        border_coords, resolution, camera, times, global_plane_position)
    # Add the relative margin
    padded_scale = native_scale * (1 + relative_plane_margin)
    return padded_scale


def mask_to_camera_coordinates(uv: torch.Tensor,
                               uv_max: torch.Tensor,
                               camera_max: torch.Tensor) -> torch.Tensor:
    """Convert mask coordinates to camera coordinates.

    This will just linearly upscale the mask coordinates to the camera coordinates.
    In case the mask as a lower or higher resolution than the camera, this will be taken into account.

    Parameters
    ----------
    uv : torch.Tensor
        The mask coordinates. Shape: (B, 2) (y, x)

    uv_max : torch.Tensor
        The maximum mask coordinates. Shape: (2,) (y, x)

    camera_max : torch.Tensor
        The maximum camera coordinates. Shape: (2,) (y, x)

    Returns
    -------
    torch.Tensor
        The camera coordinates. Shape: (B, 2) (y, x)
    """
    uv, shape = flatten_batch_dims(uv, -2)
    B = uv.shape[0]
    uv_max = flatten_batch_dims(uv_max, -2)[0]
    camera_max = flatten_batch_dims(camera_max, -2)[0]
    if len(uv_max) == 1:
        uv_max = uv_max.repeat(B, 1)
    if len(camera_max) == 1:
        camera_max = camera_max.repeat(B, 1)
    re_uv = uv / uv_max
    return unflatten_batch_dims(re_uv * camera_max, shape)


def interpolate_plane_position(
    translation: torch.Tensor,
    orientation: torch.Tensor,
    times: torch.Tensor,
    missing_in_frame: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolates starting, ending and in between missing plane positions.

    E.g. if the object is not present in theses frames.

    Starting and ending missing positions will done in a linear fashion, based multiple past or future frames.

    In between missing positions will be interpolated based on the available frames, using a cubic spline.

    Parameters
    ----------
    translation : torch.Tensor
        The translation. Shape: (T, 3) (x, y, z)
    orientation : torch.Tensor
        The orientation. Shape: (T, 4)
        Unit quaternion, convention (x, y, z, w)
    times : torch.Tensor
        The times. Shape: (T,)

    missing_in_frame : torch.Tensor
        If the plane is missing in the frame. Shape: (T,)
        Defines which frames are missing / incomplete.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        1. The interpolated translation. Shape: (T, 3) (x, y, z)
        2. The interpolated orientation. Shape: (T, 4)
        Unit quaternion, convention (x, y, z, w)
    """

    from nag.model.timed_discrete_scene_node_3d import \
        get_translation_orientation
    from tools.transforms.geometric.quaternion import (quat_average, quat_composition,
                                                       quat_product, quat_product_scalar,
                                                       quat_subtraction)
    missing_t = times[missing_in_frame]
    still_missing = missing_in_frame.clone()
    vals, inverse, repeats = torch.unique_consecutive(
        missing_in_frame, return_inverse=True, return_counts=True)
    interpolated_translation = torch.zeros_like(translation)
    interpolated_orientation = torch.zeros_like(orientation)

    start_idx = 0
    interpolate_end = vals[-1]
    interpolate_start = vals[0]

    if interpolate_end:
        lookback_num_frames = repeats[-1]
        if repeats[-2] < lookback_num_frames:
            lookback_num_frames = repeats[-2]
        li = len(repeats)
        interp = (inverse == li - 1)
        missing_t = times[interp]
        start_idx = torch.sum(repeats[:-1])
        lookback = max(lookback_num_frames // 2, 2)
        idx = (~missing_in_frame).argwhere().squeeze(1)
        last_existing_idx = (idx == (start_idx - 1)).argwhere().squeeze(1)
        lookback_frame_indices = idx[(idx >= idx[max(
            (last_existing_idx - lookback), 0)]) & (idx <= idx[last_existing_idx])]
        past_idx = lookback_frame_indices

        end = len(past_idx) // 2
        p0_idx = past_idx[:end]
        p1_idx = past_idx[end:]

        if len(p0_idx) == 0 or len(p1_idx) == 0:
            # If no data is available, raise an error
            # If one is available, use the last one
            if len(p0_idx) == 0 and len(p1_idx) == 0:
                raise ValueError("No data available for interpolation.")
            if len(p0_idx) == 0:
                p0_idx = p1_idx
            else:
                p1_idx = p0_idx

        translation_past_0 = translation[p0_idx].mean(dim=0)
        translation_past_1 = translation[p1_idx].mean(dim=0)

        orient_past_0 = quat_average(orientation[p0_idx])
        orient_past_1 = quat_average(orientation[p1_idx])

        t0 = times[p0_idx].mean()
        t1 = times[p1_idx].mean()

        dt = t1 - t0
        dtranslation = translation_past_1 - translation_past_0
        dorientation = quat_subtraction(orient_past_1, orient_past_0)

        lasttranslation = translation[start_idx - 1]
        lasttime = times[start_idx - 1]
        lastorientation = orientation[start_idx - 1]

        virtualtranslation = lasttranslation + dtranslation
        virtualorientation = quat_product(lastorientation, dorientation)

        step = ((missing_t - lasttime) / dt)
        step[~torch.isfinite(step)] = 0.  # For no motion, set to 0
        interpolated_translation[interp], interpolated_orientation[interp] = _linear_interpolate_position_rotation(
            from_position=lasttranslation.unsqueeze(
                0).repeat(len(missing_t), 1),
            to_position=virtualtranslation.unsqueeze(
                0).repeat(len(missing_t), 1),
            from_quat=lastorientation.unsqueeze(0).repeat(len(missing_t), 1),
            to_quat=virtualorientation.unsqueeze(0).repeat(len(missing_t), 1),
            frac=step
        )
        still_missing[interp] = False

    if interpolate_start:
        lookback_num_frames = repeats[0]
        li = len(repeats)
        interp = (inverse == 0)
        missing_t = times[interp]
        start_idx = lookback_num_frames

        if repeats[1] < lookback_num_frames:
            # Dont look too far into the future, at max half of the existing frames
            lookback_num_frames = repeats[1]

        lookforward = max(lookback_num_frames // 2, 2)
        idx = (~missing_in_frame).argwhere().squeeze(1)
        lookback_frame_indices = idx[0]
        last_existing_idx = idx[(
            idx < (lookback_frame_indices + lookforward)) & (idx >= lookback_frame_indices)]
        future_idx = last_existing_idx

        end = len(future_idx) // 2
        p0_idx = future_idx[:end]
        p1_idx = future_idx[end:]

        if len(p0_idx) == 0 or len(p1_idx) == 0:
            # If no data is available, raise an error
            # If one is available, use the last one
            if len(p0_idx) == 0 and len(p1_idx) == 0:
                raise ValueError("No data available for interpolation.")
            if len(p0_idx) == 0:
                p0_idx = p1_idx
            else:
                p1_idx = p0_idx

        translation_future_0 = translation[p0_idx].mean(dim=0)
        translation_future_1 = translation[p1_idx].mean(dim=0)
        orient_future_0 = quat_average(orientation[p0_idx])
        orient_future_1 = quat_average(orientation[p1_idx])

        t0 = times[p0_idx].mean()
        t1 = times[p1_idx].mean()

        dt = t1 - t0
        dtranslation = translation_future_1 - translation_future_0
        dorientation = quat_subtraction(orient_future_1, orient_future_0)

        firsttranslation = translation[idx[0]]
        firsttime = times[idx[0]]
        firstorientation = orientation[idx[0]]

        virtualtranslation = firsttranslation - dtranslation
        virtualorientation = quat_subtraction(firstorientation, dorientation)

        step = torch.abs(((missing_t - firsttime) / dt))
        step[~torch.isfinite(step)] = 0.  # For no motion, set to 0
        interpolated_translation[interp], interpolated_orientation[interp] = _linear_interpolate_position_rotation(
            from_position=firsttranslation.unsqueeze(
                0).repeat(len(missing_t), 1),
            to_position=virtualtranslation.unsqueeze(
                0).repeat(len(missing_t), 1),
            from_quat=firstorientation.unsqueeze(
                0).repeat(len(missing_t), 1),
            to_quat=virtualorientation.unsqueeze(0).repeat(len(missing_t), 1),
            frac=step
        )
        still_missing[interp] = False

    if still_missing.sum() > 0:
        # Normalize the translation und orientations
        interpolated_translation[still_missing], interpolated_orientation[still_missing] = get_translation_orientation(
            translation=translation[~missing_in_frame], orientation=orientation[~missing_in_frame], times=times[~missing_in_frame], steps=times[still_missing], equidistant_times=False)
    return interpolated_translation[missing_in_frame], interpolated_orientation[missing_in_frame]


def compute_proto_plane_position_centeroid(
    images: torch.Tensor,
    mask: torch.Tensor,
    depths: torch.Tensor,
    times: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    mask_properties: BasicMaskProperties,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    depth_ray_thickness: float = 0.05,
    min_depth_ray_thickness: float = 10.,
    plot_tracked_points: bool = False,
    plot_tracked_points_path: Optional[str] = None,
) -> torch.Tensor:
    """Computes the proto plane position.

    These plane positions are just the raw plane approximations of the mask and depth within the scene.

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

    depth_ray_thickness : float
        The depth ray thickness.

    min_depth_ray_thickness : float
        The minimum depth ray thickness.

    Returns
    -------
    torch.Tensor
        The proto plane position. Shape: (T, 4, 4)
        As global position matrix.
        The position matrix may contain nan values if the mask is not present in the frame.
        These must be handled by the caller.

    """
    if device is None:
        device = images.device
    T, C_img, H, W = images.shape

    orig_bl = mask_properties.bottom_left
    orig_tr = mask_properties.top_right

    bl = mask_properties.padded_bottom_left
    tr = mask_properties.padded_top_right

    # Round to integer
    bl = bl.round().int()
    tr = tr.round().int()

    plane_center = bl + (tr - bl) / 2

    camera_plane_center = mask_to_camera_coordinates(plane_center, torch.tensor(
        [H, W], dtype=dtype, device=device), camera._image_resolution)
    camera_plane_center = torch.flip(
        camera_plane_center, dims=(-1,))  # Switch to xy

    # Compute the ray collisions with the mask border defined by bl and tr
    # Get also br and tl from orig_bl and orig_tr which are (y, x) coordinates

    orig_br = torch.stack([orig_bl[:, 0], orig_tr[:, 1]], dim=-1)
    orig_tl = torch.stack([orig_tr[:, 0], orig_bl[:, 1]], dim=-1)
    # Stack the border points
    border_points = torch.stack(
        [orig_bl, orig_br, orig_tr, orig_tl], dim=-3)

    missing_in_frame = mask_properties.missing_in_frame

    plane_support_points = get_plane_support_points(
        mask, depths, times, dtype, missing_in_frame, border_points,
        mask_center_of_mass=mask_properties.mask_center_of_mass,
        depth_ray_thickness=depth_ray_thickness,
        min_depth_ray_thickness=min_depth_ray_thickness
    )

    # Get the plane_support_points to world coordinates
    support_camera = mask_to_camera_coordinates(plane_support_points[..., :2], torch.tensor(
        [W, H], dtype=dtype, device=device), camera._image_resolution.flip(-1))

    ro, rd = camera.get_global_rays(
        uv=support_camera.swapaxes(0, 1), t=times, uv_includes_time=True)
    # global_plane_support_points = ro + rd * \
    #     plane_support_points[..., 2].unsqueeze(-1)

    if plot_tracked_points:
        plot_proto_points(
            images=images,
            masks=mask,
            proto_points=plane_support_points.swapaxes(0, 1)[..., :2],
            mask_center_point=plane_center[:, None, :2].flip(-1),  # Flip to xy
            resolution=torch.tensor(
                images.shape[2:], dtype=dtype, device=device) / 4,
            save=True,
            path=plot_tracked_points_path,
            override=True
        )

    # Compute the ray plane intersection for the predicted depth
    depth_plane_position = torch.eye(4, dtype=dtype, device=device).unsqueeze(
        0).unsqueeze(0).repeat(3, T, 1, 1)
    depth_plane_position[..., 2, 3] = plane_support_points[..., 2]
    plane_pos = torch.bmm(camera.get_global_position(t=times).repeat_interleave(
        3, dim=0), depth_plane_position.reshape(3 * T, 4, 4)).reshape(3, T, 4, 4)
    global_plane_support_points = compute_ray_plane_intersections_from_position_matrix(
        plane_pos, ro, rd)

    # TODO plane_support points can be nan if the object is not present at a timestamp
    r1 = global_plane_support_points[0, ...] - \
        global_plane_support_points[1, ...]
    r2 = global_plane_support_points[0, ...] - \
        global_plane_support_points[2, ...]
    r3 = global_plane_support_points[1, ...] - \
        global_plane_support_points[2, ...]

    normal = torch.cross(r1, r2, dim=-1)
    normal = normal / torch.norm(normal, dim=-1, keepdim=True)

    non_finited = torch.isfinite(normal)  # TODO FIx this
    # if non_finited.any():

    # Determine the orientation vector by getting the angle between the z forward facing vector and the normal
    z = torch.tensor([0, 0, 1], dtype=dtype,
                     device=device).unsqueeze(0).repeat(T, 1)

    rotmats = calculate_rotation_matrix(z, normal)

    quat = rotmat_to_unitquat(rotmats)
    proto_plane_position = torch.zeros(
        T, 4, 4, dtype=dtype, device=device)
    proto_plane_position[:, 3, 3] = 1
    proto_plane_position[:, :3, :3] = rotmats
    # Use one f the support points as the position

    proto_plane_position[:, :3,
                         3] = global_plane_support_points[0, ...]

    # Now get the the ray intersection of the mask center with the plane which we will define as new position

    ro, rd = camera.get_global_rays(
        camera_plane_center, t=times, uv_includes_time=True)
    plane_center = compute_ray_plane_intersections_from_position_matrix(
        proto_plane_position, ro, rd)

    global_plane_position = proto_plane_position.clone()
    global_plane_position[:, :3, 3] = plane_center

    return global_plane_position


def get_plane_support_points(
    mask: torch.Tensor,
    depths: torch.Tensor,
    times: torch.Tensor,
    dtype: torch.dtype,
    missing_in_frame: torch.Tensor,
    border_points: torch.Tensor,
    mask_center_of_mass: torch.Tensor,
    depth_ray_thickness: float,
    min_depth_ray_thickness: float,
    ray_angles: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the support points of the planes for each mask, based on a simple centeroid method.

    Parameters
    ----------
    mask : torch.Tensor
        The mask. Shape: (T, 1, H, W)

    depths : torch.Tensor
        The depths. Shape: (T, 1, H, W)

    times : torch.Tensor
        The times. Shape: (T,)

    dtype : torch.dtype
        The dtype.

    missing_in_frame : torch.Tensor
        If the plane is missing in the frame. Shape: (T,)

    border_points : torch.Tensor
        The border points. Shape: (4, T, 2) (y, x)

    mask_center_of_mass : torch.Tensor
        The center of mass of the mask in the image. Shape: (T, 2) (y, x)

    depth_ray_thickness : float
        Of thick the depth ray on the mask should be considered to find a depth.

    min_depth_ray_thickness : float
        The minimum thickness of the depth ray.

    ray_angles : Optional[torch.Tensor]
        The ray angles. Shape: (3,) for each ray. Should be in radians. Default is depending on the mask shape:
        if object height (MH) is dominand (MH >= MW / 4) its: rad([270, 30, 150]), if width dominand its: rad([0, 150, 210])
        These should be the angles w.r.t the x axis in clockwise order. E.g. 0 is x axis (right), 90 is -y axis (down), 180 is -x axis (left), 270 is y axis (up).

    Returns
    -------
    torch.Tensor
        The support points. Shape: (3, T, 3) (x, y, z)
    """

    bl, br, tr, tl = border_points
    device = mask.device
    T = len(times)
    D = 3
    if ray_angles is None:
        # Cast 3 Rays in mask space from the center of mask to the image border with 120 degrees between them
        # Counter clockwise (0 facing x axis positive). 270 is y up
        ray_angles = torch.deg2rad(torch.tensor([270, 30, 150]))

        # If the height of the mask is less than 1 / 4 of the mask with, we dont shoot 1 top two bottom, but 1 right end 2 left whereby left is only seperated by 60 degrees
        if ((tr[:, 0] - bl[:, 0]).float() < ((tr[:, 1] - bl[:, 1]).float() / 4)).any():
            ray_angles = torch.deg2rad(torch.tensor([0, 150, 210]))

    # Get the direction vectors
    directions = torch.stack([torch.sin(ray_angles), torch.cos(
        ray_angles)], dim=0).swapaxes(0, 1)  # Shape: (3, 2) (y, x)

    ret = compute_collisions(
        mask_center=mask_center_of_mass, directions=directions, border_points=border_points)
    ret[:, :, missing_in_frame] = float("nan")
    ret[0, :, missing_in_frame] = 0.

    invalid_collisions = torch.any(torch.isnan(ret), dim=-1)
    number_invalid = invalid_collisions.sum(dim=0)
    to_correct = torch.argwhere(number_invalid < 3)
    # Disable the invalid collisions, should be small number can only be the case if ray hits corner exactly
    for i, idx in enumerate(to_correct):
        valid_collisions = ~invalid_collisions[:, idx[0], idx[1]]
        vc_idx = torch.argwhere(valid_collisions).squeeze(1)[1:]
        msk = torch.zeros(4, 2, dtype=dtype, device=device)
        msk[vc_idx, :] = True
        ret[vc_idx, idx[0], idx[1]] = float("nan")

    invalid_collisions = torch.any(torch.isnan(ret), dim=-1)
    number_invalid = invalid_collisions.sum(dim=0)
    assert torch.all(number_invalid == 3)

    bounding_box_collisions = ret[~invalid_collisions].reshape(
        1, D, T, 2)
    # Create polygon from bounding box collision and mask center by going half thickness orthogonal to the mask center
    # Get the direction of the mask center
    mask_center_direction = mask_center_of_mass.unsqueeze(
        0) - bounding_box_collisions
    mask_center_direction_n = mask_center_direction / \
        torch.norm(mask_center_direction, dim=-1, keepdim=True)
    # Get the orthogonal direction
    orthogonal = torch.stack(
        [-mask_center_direction_n[..., 1], mask_center_direction_n[..., 0]], dim=-1)
    # Get the thickness
    mask_surface = mask[:, 0].sum(dim=(-1, -2))
    thickness = torch.sqrt(mask_surface) * depth_ray_thickness / 2
    thickness = torch.minimum(thickness, torch.tensor(
        min_depth_ray_thickness, dtype=thickness.dtype, device=thickness.device))
    # Get the points
    thickness = thickness.unsqueeze(0).unsqueeze(-1).repeat(1, D, 1, 2)
    # Get the points
    p_0 = bounding_box_collisions + orthogonal * thickness
    p_1 = bounding_box_collisions + orthogonal * thickness
    p_2 = bounding_box_collisions + orthogonal * -thickness
    p_3 = mask_center_of_mass.unsqueeze(0).unsqueeze(0).repeat(
        1, D, 1, 1) + orthogonal * -thickness
    p_4 = mask_center_of_mass.unsqueeze(0).unsqueeze(0).repeat(1, D, 1, 1)
    p_5 = mask_center_of_mass.unsqueeze(0).unsqueeze(0).repeat(
        1, D, 1, 1) + orthogonal * +thickness
    # stack the points
    # Swap the first and second dim, and flip y, x to x,y
    points = torch.cat([p_0, p_1, p_2, p_3, p_4, p_5],
                       dim=-4).swapaxes(0, 1).flip(-1)

    plane_support_points = torch.zeros(
        D, T, 3, dtype=dtype, device=device)  # x, y, z

    # As masks have different sizes, we neet to loop over the masks
    picked_random_points = torch.zeros_like(
        plane_support_points[0, :, 0], dtype=torch.bool)
    for i in range(T):
        if missing_in_frame[i]:
            plane_support_points[:, i, :] = 0.
            continue

        painted_mask = torch.zeros(mask[i, 0].shape, dtype=torch.bool, device=device).unsqueeze(
            0).unsqueeze(0).repeat(3, 1, 1, 1)
        painted_mask = draw_convex_polygon(painted_mask, points[:, :, i], colors=torch.tensor(
            1, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).repeat(3, 1)).bool()
        # Check if depeth select mask is part of the mask
        depth_select_mask = painted_mask & mask[i].unsqueeze(0)
        mask_depths = depths[i, 0, mask[i, 0]]
        try:
            for d in range(D):
                # Get the depth
                depth_select_mask_idx = depth_select_mask[d].argwhere()
                sel_d = depths[i, depth_select_mask[d]]
                if len(sel_d) == 0:
                    # Check if any of the other mask was hit
                    hits = [j for j in range(
                        D) if j != d and depth_select_mask[j].sum() > 0]
                    if len(hits) > 0:
                        # Merge masks and select again.
                        msk = torch.stack([depth_select_mask[j]
                                           for j in hits], dim=0).sum(dim=0) > 0
                        depth_select_mask[d] = msk
                        depth_select_mask_idx = depth_select_mask[d].argwhere(
                        )
                        sel_d = depths[i, depth_select_mask[d]]
                    else:
                        # Can Happen in case of weird mask shape where the center of mask is in a hole and mask is very deformed.
                        raise ValueError(
                            "No depth values found for mask. Try to to set depth_ray_thickness to a higher number.")
                m = torch.median(sel_d, -1)
                arg_sorted_seld = None
                itter = 0
                idx = m.indices
                # Ignore channel dim, flip indices from y, x to x, y
                dslidx = torch.flip(
                    depth_select_mask_idx[idx][1:], dims=(-1,))
                v = m.values

                # Check if dslidx is already withing the support points, if so we need to get another point
                # As we need 3 different points to define a plane
                while (d != 0 and
                        (plane_support_points[:, i, :2] == dslidx).all(dim=-1).any() and itter < len(sel_d)):
                    if arg_sorted_seld is None:
                        arg_sorted_seld = torch.argsort(sel_d)
                    nidx = (len(sel_d) // 2 + itter) % len(sel_d)
                    idx = arg_sorted_seld[nidx]
                    # Ignore channel dim, flip indices from y, x to x, y
                    dslidx = torch.flip(
                        depth_select_mask_idx[idx][1:], dims=(-1,))
                    v = sel_d[idx]
                    itter += 1

                if itter >= len(sel_d):
                    raise ValueError(
                        "Could not find a support point.")
                plane_support_points[d, i, :2] = dslidx
                plane_support_points[d, i, 2] = v
        except ValueError as e:
            # If we cannot find a support points by the proposed method, use just 3 random points
            mask_coords = torch.nonzero(mask[i, 0])
            if len(mask_coords) < 3:
                raise ValueError(
                    "Mask is too small to determine a plane.") from e
            random_points = torch.flip(
                mask_coords[torch.randint(0, len(mask_coords), (3,))], dims=(-1,))
            plane_support_points[:, i, :2] = random_points
            plane_support_points[:, i, 2] = depths[i, 0,
                                                   random_points[:, 1], random_points[:, 0]]
            picked_random_points[i] = True

    if picked_random_points.any():
        non_random = ~picked_random_points
        if non_random.sum() > 0:
            # If we have some non random points, we will interpolate the random points
            exist_t = times[non_random]
            random_t = times[picked_random_points]
            exists_p = plane_support_points[:, non_random, :]
            ET = exists_p.shape[1]
            interp = linear_interpolate_vector(exists_p.swapaxes(0, 1).reshape(
                ET, 9), exist_t, random_t, equidistant_times=False)
            plane_support_points[:, picked_random_points,
                                 :] = interp.reshape(-1, 3, 3).swapaxes(0, 1)
    return plane_support_points


def compute_collisions(mask_center: torch.Tensor, directions: torch.Tensor, border_points: torch.Tensor) -> torch.Tensor:
    """Compute the collisions of the rays with the border points / border lines.
    The rays are defined by the center of mass of the mask and the directions.
    The border points are the border points of the mask.

    Parameters
    ----------
    mask_center : torch.Tensor
        The rays. Shape: (T, 2) (y, x)
    directions : torch.Tensor
        The directions. Shape: (D, 2) (y, x)
    border_points : torch.Tensor
        The border points. Shape: (4, T, 2) (y, x)
        In counter clockwise order.
        bl, br, tr, tl

    Returns
    -------
    torch.Tensor
        The collisions. Shape: (4, D, T, 2) (y, x)

    """
    T = mask_center.shape[0]
    BP = border_points.shape[0]
    D = directions.shape[0]
    # # Stack the destinations of the border rays
    dest = torch.roll(border_points, -1, dims=0) - border_points

    # Reshape to (BP, D, T, 2)
    unsq_mask_center = mask_center.unsqueeze(
        0).unsqueeze(0).repeat(BP, D, 1, 1)
    mask_dirs = directions.unsqueeze(
        1).unsqueeze(0).repeat(BP, 1, T, 1)
    border_dest = dest.unsqueeze(1).repeat(1, D, 1, 1)
    border_src = border_points.unsqueeze(1).repeat(1, D, 1, 1)

    # Convert to 2D vecs for easier handling
    _as, shape = flatten_batch_dims(unsq_mask_center, -2)
    _ad, _ = flatten_batch_dims(mask_dirs, -2)
    _bd, _ = flatten_batch_dims(border_dest, -2)
    _bs, _ = flatten_batch_dims(border_src, -2)

    # Compute the collisions
    # Get the direction of the rays
    delta = _bs - _as
    det = _bd[..., 0] * _ad[..., 1] - _bd[..., 1] * _ad[..., 0]
    ret = torch.zeros_like(delta)
    ret.fill_(float("nan"))
    # For det approx 0, the rays are parallel
    parallel = torch.abs(det) < 1e-6
    # Compute the t values
    uv = torch.zeros_like(delta)
    uv[~parallel, 0] = (delta[~parallel, 1] * _bd[~parallel, 0] -
                        delta[~parallel, 0] * _bd[~parallel, 1]) / det[~parallel]
    uv[~parallel, 1] = (delta[~parallel, 1] * _ad[~parallel, 0] -
                        delta[~parallel, 0] * _ad[~parallel, 1]) / det[~parallel]

    hit = (uv > 0).all(dim=-1)
    ret[hit] = _as[hit] + uv[hit, 0].unsqueeze(-1) * _ad[hit]

    # Unflatten
    ret = unflatten_batch_dims(ret, shape)

    # Dont count the hits if they are outside the border
    min_x = torch.amin(border_points[..., 1], dim=0)
    max_x = torch.amax(border_points[..., 1], dim=0)
    min_y = torch.amin(border_points[..., 0], dim=0)
    max_y = torch.amax(border_points[..., 0], dim=0)
    eps = 1
    outside = ((ret[..., 0] + eps) < min_y) | (ret[..., 0] > (max_y + eps)) | (
        (ret[..., 1] + eps) < min_x) | (ret[..., 1] > (max_x + eps))
    ret[outside] = float("nan")
    ret[..., 0] = torch.clamp(ret[..., 0], min_y, max_y)
    ret[..., 1] = torch.clamp(ret[..., 1], min_x, max_x)
    return ret


def gaussian_smoothing(x: torch.Tensor, kernel_size: int = 5, sigma=1.0):
    """
    Smoothes a tensor along dimension 1 using a Gaussian kernel.

    Args:
        tensor: The input tensor of shape [B, N].
        kernel_size: The size of the Gaussian kernel.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        The smoothed tensor.
    """
    if kernel_size > x.shape[1]:
        kernel_size = x.shape[1]
    x, shape = flatten_batch_dims(x, -2)
    # Create a 1D Gaussian kernel
    indices = torch.linspace(-kernel_size / 2, kernel_size / 2,
                             kernel_size, dtype=x.dtype, device=x.device)
    pi2 = torch.tensor(2*torch.pi, dtype=x.dtype, device=x.device)
    kernel = 1 / (sigma * torch.sqrt(pi2)) * \
        torch.exp(-indices**2/(2*sigma**2))
    kernel /= kernel.sum()

    # Reshape the kernel to [1, 1, kernel_size]
    kernel = kernel.reshape(1, 1, kernel_size)

    # Same padding based on the kernel size
    padding = kernel_size // 2
    if kernel_size % 2 == 0:
        padding = (padding, padding - 1)
    else:
        padding = (padding, padding)

    x = F.pad(x, pad=padding, mode='reflect')

    # Apply the Gaussian kernel using convolution
    smoothed_tensor = F.conv1d(x.unsqueeze(1), kernel, padding=0).squeeze(1)

    # Reshape and extract the smoothed tensor
    return unflatten_batch_dims(smoothed_tensor, shape)


def average_pooling(x: torch.Tensor, kernel_size: int = 5):
    """
    Averages a tensor along dimension 1 using a kernel of ones.

    Args:
        tensor: The input tensor of shape [B, N].
        kernel_size: The size of the kernel.

    Returns:
        The smoothed tensor.
    """
    x, shape = flatten_batch_dims(x, -2)
    # Same padding based on the kernel size
    padding = kernel_size // 2
    if kernel_size % 2 == 0:
        padding = (padding, padding - 1)
    else:
        padding = (padding, padding)

    x = F.pad(x, pad=padding, mode='reflect')

    smoothed_tensor = F.avg_pool1d(
        x.unsqueeze(1), kernel_size, stride=1).squeeze(1)
    # Reshape and extract the smoothed tensor
    return unflatten_batch_dims(smoothed_tensor, shape)


def gaussian_smoothing_2d(x: torch.Tensor, kernel_size: int = 5, sigma=1.0):
    from torchvision.transforms import GaussianBlur
    x, shape = flatten_batch_dims(x, -4)
    g = GaussianBlur(kernel_size, sigma)
    x = g(x)
    return unflatten_batch_dims(x, shape)


def mask_to_plane_coordinates(uv: torch.Tensor,
                              uv_max: torch.Tensor,
                              t: torch.Tensor,
                              global_plane_position: torch.Tensor,
                              plane_scale: torch.Tensor,
                              camera: TimedCameraSceneNode3D,
                              ) -> torch.Tensor:
    """Convert the mask coordinates to plane local coordinates, by projecting the mask coordinates to the plane.

    uv : torch.Tensor
        The mask coordinates. Shape: (N, 2) (x, y)

    uv_max : torch.Tensor
        The maximum coordinates of the mask. Shape: (2, ) (x, y)

    t : torch.Tensor
        The current time of the mask. Shape: () 0d tensor

    global_plane_position : torch.Tensor
        The global plane position of timestamp t. Shape: (4, 4)
        As global position matrix.

    plane_scale : torch.Tensor
        The plane scale. Shape: (2, ) (x, y)

    camera : TimedCameraSceneNode3D
        The camera for the reprojection.

    Returns
    -------
    torch.Tensor
        The plane coordinates. Shape: (N, 2) (x, y)

    """
    dtype = uv.dtype
    device = uv.device
    cam_uv = mask_to_camera_coordinates(uv.flip(-1), uv_max=uv_max.flip(-1),
                                        camera_max=camera._image_resolution).flip(-1)  # Input as y,x -> Switch to xy

    # Get intersection of the rays with the plane
    ro, rd = camera.get_global_rays(cam_uv, t=t, uv_includes_time=False)
    global_intersections = compute_ray_plane_intersections_from_position_matrix(
        global_plane_position, ro[:, 0, :], rd[:, 0, :])
    # Convert global intersections to plane local
    local_intersections = global_to_local(
        global_plane_position, global_intersections, v_include_time=False)[..., :2]  # Ignoring z as z == 0
    # Get from plane locals to plane coordinates
    offset = default_plane_scale_offset(plane_scale.dtype)
    plane_coords = local_to_plane_coordinates(local_intersections, plane_scale, offset)[
        :, 0]  # Drop the time dimension
    return plane_coords


def plot_plane_position(global_plane_position,
                        times=None,
                        plane_scale=None,
                        camera=None,
                        **kwargs):
    from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D
    from nag.model.timed_plane_scene_node_3d import TimedPlaneSceneNode3D
    if times is None:
        times = torch.linspace(0, 1, global_plane_position.shape[0])
    world = TimedDiscreteSceneNode3D(name="World")
    plane = TimedPlaneSceneNode3D(name="Plane", position=global_plane_position,
                                  plane_scale=plane_scale,
                                  times=times)
    world.add_scene_children(plane)
    if camera is not None:
        world.add_scene_children(camera)
    return world.plot_scene(**kwargs)
