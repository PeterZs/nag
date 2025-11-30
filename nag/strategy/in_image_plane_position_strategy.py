from enum import Enum

import cv2
from matplotlib import pyplot as plt
from nag.homography.sequence_homography_finder import (SequenceHomographyFinder,
                                                       SequenceHomographyFinderConfig)
from nag.homography.homography_finder import is_coord_in_mask
from nag.strategy.plane_position_strategy import (PlanePositionStrategy, compute_plane_scale,
                                                  compute_proto_plane_position_centeroid,
                                                  gaussian_smoothing, get_plane_support_points,
                                                  interpolate_plane_position, mask_to_camera_coordinates,
                                                  plot_proto_points)
import torch
from typing import Any, Dict, Optional, Tuple
from nag.config.nag_config import NAGConfig
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
import torch
from tools.transforms.geometric.quaternion import quat_composition, quat_subtraction, quat_average, quat_product_scalar, quat_product
from nag.strategy.base_plane_initialization_strategy import BasicMaskProperties
from tools.transforms.geometric.transforms3d import flatten_batch_dims
from nag.model.timed_discrete_scene_node_3d import global_to_local
import torch
from tools.util.torch import flatten_batch_dims
from tools.transforms.geometric.mappings import rotmat_to_unitquat, rotvec_to_unitquat, unitquat_to_rotvec, unitquat_to_rotmat
from tools.transforms.to_tensor import tensorify
from tools.util.format import parse_enum
from tools.logger.logging import logger
from nag.transforms.transforms_timed_3d import align_rectangles
from nag.homography.sequence_homography_finder import SequenceHomographyFinder, plot_tracked_points
from nag.strategy.in_image_plane_position_strategy import mask_to_camera_coordinates, plot_proto_points
from tools.util.torch import flatten_batch_dims, tensorify, cummatmul
from tools.transforms.geometric.transforms3d import find_plane, compute_ray_plane_intersections
from tools.transforms.geometric.mappings import normal_to_rotmat
from nag.transforms.transforms_timed_3d import align_rectangles
from nag.model.timed_discrete_scene_node_3d import global_to_local
from tools.transforms.geometric.transforms3d import vector_angle, find_plane, plane_eval, assure_homogeneous_matrix, assure_homogeneous_vector, compute_ray_plane_intersections, compute_line_intersections
from tools.transforms.geometric.mappings import normal_to_rotmat, rotvec_to_rotmat, rotmat_to_rotvec, rotmat_to_normal
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D, get_local_rays, get_global_rays, local_to_image_coordinates
from tools.viz.matplotlib import saveable, get_mpl_figure
import numpy as np


class ProtoPlanePositionMethod(Enum):

    CENTEROID = "centeroid"

    HOMOGRAPHY = "homography"


class InImagePlanePositionStrategy(PlanePositionStrategy):

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
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        T, C_img, H, W = images.shape
        global_position, global_center, global_orientation, interp_times = compute_plane_position(
            images=images,
            mask=mask,
            depths=depths,
            times=times,
            camera=camera,
            mask_properties=mask_properties,
            object_index=plane_properties["index"],
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
            plot_tracked_points=self.plot_tracked_points,
            plot_tracked_points_path=self.plot_tracked_points_path,
            new_homography=True,
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


def get_plane_scale(
    mask_properties: BasicMaskProperties,
    global_plane_position: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    times: torch.Tensor,
    mask_resolution: torch.Tensor,
    relative_plane_margin: torch.Tensor,
) -> torch.Tensor:
    """Method to get the plane scale.

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
    plane_scale = compute_plane_scale(
        border_coords=mask_properties.border_points,
        resolution=mask_resolution,
        camera=camera,
        times=times,
        global_plane_position=global_plane_position,
        relative_plane_margin=tensorify(relative_plane_margin)
    )
    plane_surface = plane_scale.prod(dim=-1)
    # Select the largest mask
    max_idx = torch.argmax(plane_surface[~missing_in_frame])
    _plane_scale = plane_scale[~missing_in_frame][max_idx]
    return _plane_scale


def find_plane_rotation_matrix(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
):
    A, shp = flatten_batch_dims(A, -2)
    B, _ = flatten_batch_dims(B, -2)
    C, _ = flatten_batch_dims(C, -2)

    Bs, Cs = A.shape

    if A.shape != B.shape or A.shape != C.shape:
        raise ValueError("The input points must have the same shape.")

    AC = C - A
    AB = B - A

    # We norm the vectors as we dont estimate any scale.
    # We construct a rectangle by using the bisector of angle of AC and AB

    # Normalize the vectors
    nAC = AC / torch.norm(AC, dim=-1, keepdim=True)
    nAB = AB / torch.norm(AB, dim=-1, keepdim=True)

    BIS = nAC + nAB
    BIS = BIS / torch.norm(BIS, dim=-1, keepdim=True)

    # We generate 4 points using the bisector and the plane normal PN, to create points on the plane
    PN = torch.cross(AB, AC, dim=-1)
    PN = PN / torch.norm(PN, dim=-1, keepdim=True)

    # If PNs Z component is negative, we flip the sign of the normal, so it will point in the same direction as the camera
    # PN = torch.where(PN[..., 2:3] < 0, -PN, PN)

    ORTH = torch.cross(BIS, PN, dim=-1)
    ORTH = ORTH / torch.norm(ORTH, dim=-1, keepdim=True)

    # We use just half of the bisector and the orthogonal vector to create the rectangle of unit length
    R1 = A + BIS / 2 + ORTH / 2
    R2 = A + BIS / 2 - ORTH / 2
    R3 = A - BIS / 2 - ORTH / 2
    R4 = A - BIS / 2 + ORTH / 2
    target_rectangle = torch.stack([R1, R2, R3, R4], dim=-2)

    unit_rectangle = torch.tensor([
        [-0.5, -0.5, 0],  # Bottom left
        [0.5, -0.5, 0],  # Bottom right
        [0.5, 0.5, 0],  # Top right
        [-0.5, 0.5, 0]  # Top left
    ], device=A.device, dtype=A.dtype)
    unit_rectangle = unit_rectangle.unsqueeze(0).expand(Bs, -1, -1)
    # As the rectangle is defined by a affine transformation matrix which rotates and translates the unit rectangle
    # We can use the Procrustes analysis to find the optimal rotation and translation

    return align_rectangles(unit_rectangle, target_rectangle)


def _compute_homography_tracked_points_for_frame(
    images: torch.Tensor,
    mask: torch.Tensor,
    depths: torch.Tensor,
    times: torch.Tensor,
    mask_border_points: torch.Tensor,
    missing_in_frame: torch.Tensor,
    mask_center_of_mass: torch.Tensor,
    homography_finder: SequenceHomographyFinder,
    reference_frame_idx: int,
    object_index: int,
    custom_mask_center: Optional[torch.Tensor] = None,
    frame_index_offset: int = 0,
    ray_angles: Optional[torch.Tensor] = None,
    use_existing_points: Optional[torch.Tensor] = None,
    existing_points: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    depth_ray_thickness: float = 0.05,
    min_depth_ray_thickness: float = 10.,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """_summary_

    Parameters
    ----------
    images : torch.Tensor
        The images. Shape: (T, C, H, W)

    mask : torch.Tensor
        The masks. Shape: (T, 1, H, W)

    depths : torch.Tensor
        The depths. Shape: (T, 1, H, W)

    times : torch.Tensor
        The times. Shape: (T, )

    mask_border_points : torch.Tensor
        The border points of the masks. Shape: (4, T, 2) (y, x)
        Border points of the masks. The border points are ordered as bottom left, bottom right, top right, top left. (Counter clockwise from bottom left)

    missing_in_frame : torch.Tensor
        A boolean tensor indicating which masks are missing in the frame. Shape: (T, )

    mask_center_of_mass : torch.Tensor
        The center of mass of the masks. Shape: (T, 2) (y, x)

    homography_finder : SequenceHomographyFinder
        The homography finder.

    reference_frame_idx : int
        The reference frame index, i.e. the frame index of the mask that we want to track.

    object_index : int
        The object index of the current mask.

    custom_mask_center : Optional[torch.Tensor]
        The custom mask center. Shape: (2, ) (y, x)

    frame_index_offset : int
        The frame index offset. If the images are a subset of the full sequence, we need to adjust the frame indices.
        This should be the actual index of the first frame in the images tensor.

    ray_angles : Optional[torch.Tensor]
        The ray angles. Shape: (3, )
        The angles of the lookup rays. Will be used to sample representative points on the mask, from the mask_center
        in the direction of the ray_angles. Should be in radians.
        Should be given in a right-handed coordinate system, where the x-axis points to the right (0), the y-axis points down (90), etc.

    use_existing_points : Optional[torch.Tensor]
        If the provided existing points should be used instead of using the get_plane_support_points function.
        Shape: (3, ) boolean tensor.

    existing_points : Optional[torch.Tensor]
        Existing points to use instead of using the get_plane_support_points function.
        Shape: (3, 2) (x, y, z) in mask space.
        Only those marked as True in use_existing_points will be used.

    ...

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        1. projected_points_2d: The projected points in 2D image / mask space. Shape: (3, T, 2) (x, y)
        Each point represents a estimated and tracked point along time.
        2. projected_mask_center: The projected mask center in 2D image / mask space. Shape: (1, T, 2) (x, y)
        The center of the reference_frame_idx mask projected to all frames.
        3. missing_homographies: A boolean tensor indicating which homographies are missing or incorrect. Shape: (T, )
        A True value indicates that the homography is missing or incorrect, i.e the object is not visible in the frame
        or the feature points could not be reliably tracked.
    """
    if device is None:
        device = images.device
    T, C_img, H, W = images.shape

    # if the images are a subset of the full sequence, we need to adjust the frame indices
    frame_indices = torch.arange(
        frame_index_offset, T, device=device, dtype=torch.int32)

    if use_existing_points is None:
        use_existing_points = torch.zeros(3, dtype=torch.bool, device=device)

    # Shape (1, 2)
    if custom_mask_center is not None:
        mask_center = custom_mask_center[None, ...].flip(-1)  # Flip to (x, y)
    else:
        # Flip to (x, y
        mask_center = mask_center_of_mass[reference_frame_idx][None, ...].flip(
            -1)
    mask_center = flatten_batch_dims(mask_center, -2)[0]

    # Get some initial points for the largests masks
    initial_support_points = get_plane_support_points(
        mask[reference_frame_idx][None, ...],
        depths[reference_frame_idx][None, ...],
        times[reference_frame_idx][None, ...],
        dtype,
        missing_in_frame=missing_in_frame[reference_frame_idx][None, ...],
        border_points=mask_border_points[:,
                                         reference_frame_idx:reference_frame_idx+1],
        # Flip back (y, x) Shape (1, 2)
        mask_center_of_mass=mask_center.flip(-1),
        depth_ray_thickness=depth_ray_thickness,
        min_depth_ray_thickness=min_depth_ray_thickness,
        ray_angles=ray_angles
    )[:, :, :2]  # Shape (3, 1, 2) (x, y)

    if use_existing_points.any():
        initial_support_points[use_existing_points] = existing_points[use_existing_points].unsqueeze(
            1)

    homographies = torch.eye(3, dtype=dtype, device=device).unsqueeze(
        0).repeat(T, 1, 1)  # Shape (T, 3, 3)
    if reference_frame_idx < (T - 1):
        # If masks is smaller than last, we can track the homographies forward
        missing_frames_filtered = missing_in_frame[reference_frame_idx:]
        frame_indices_filtered = frame_indices[reference_frame_idx:]
        # We try to ignore frames where the object is missing
        homog = homography_finder.find_cumulative_homographies(
            images[reference_frame_idx:][~missing_frames_filtered],
            mask[reference_frame_idx:][~missing_frames_filtered],
            frame_indices=frame_indices_filtered[~missing_frames_filtered],
            object_indices=object_index
        )[:, 0]  # Only one object
        homographies[reference_frame_idx +
                     1:][~missing_frames_filtered[1:]] = homog.to(dtype)
    if reference_frame_idx > 0:
        # If the reference frame is not the first, we can track the homographies backwards
        missing_frames_filtered = missing_in_frame[:reference_frame_idx+1]
        missing_frames_filtered_ex_current = missing_in_frame[:reference_frame_idx]
        frame_indices_filtered = frame_indices[:reference_frame_idx+1]
        # We try to ignore frames where the object is missing
        h = homography_finder.find_cumulative_homographies(
            # Flip to reverse the frames
            images[:reference_frame_idx+1][~missing_frames_filtered].flip(0),
            mask[:reference_frame_idx+1][~missing_frames_filtered].flip(0),
            frame_indices=frame_indices_filtered[~missing_frames_filtered].flip(
                0),
            object_indices=object_index
        )[:, 0]
        homog = tensorify(h).flip(0)
        # Ignore the largest_mask_idx entry
        homographies[:reference_frame_idx][~missing_frames_filtered_ex_current] = homog.to(
            dtype)

    # If any homography contains nan, we declare it as missing for now, and replace these the eye
    missing_homographies = ~torch.isfinite(homographies).all(dim=(1, 2))
    # Logically homographies also missing if there is not mask
    missing_homographies |= missing_in_frame

    homographies[missing_homographies] = torch.eye(
        3, dtype=dtype, device=device)

    affine_initial = torch.cat([initial_support_points[..., :2],
                                # Shape (3, 1, 3)
                                torch.ones_like(initial_support_points[..., :1])], dim=-1)
    affine_initial = affine_initial.expand(3, T, 3)  # Shape (3, T, 3)

    affine_mask_center = torch.cat([mask_center, torch.ones_like(
        # Shape (T, 3)
        mask_center[..., :1])], dim=-1).unsqueeze(0).expand(T, -1, -1)

    projected_points_2d = torch.bmm(homographies.unsqueeze(0).expand(3, T, 3, 3).reshape(3 * T, 3, 3),
                                    affine_initial.reshape(3 * T, 3).unsqueeze(-1))[..., 0].reshape(3, T, 3)  # Shape (3, T, 3)
    projected_mask_center = torch.bmm(homographies.reshape(1 * T, 3, 3),
                                      affine_mask_center.reshape(1 * T, 3).unsqueeze(-1))[..., 0].reshape(1, T, 3)  # Shape (1, T, 3)
    # Back to affine
    projected_points_2d = (projected_points_2d /
                           projected_points_2d[..., 2:3])[..., :2]
    projected_mask_center = (projected_mask_center /
                             projected_mask_center[..., 2:3])[..., :2]

    return projected_points_2d[..., :2], projected_mask_center[..., :2], missing_homographies


def _compute_homography_tracked_points(
    images: torch.Tensor,
    mask: torch.Tensor,
    depths: torch.Tensor,
    times: torch.Tensor,
    mask_properties: BasicMaskProperties,
    object_index: int,
    homography_finder: SequenceHomographyFinder,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    depth_ray_thickness: float = 0.05,
    min_depth_ray_thickness: float = 10.,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    Computes points tracks based on homographies.
    Points are choosen at best, and resampled if needed, like if they drift out of the mask.

    Parameters
    ----------
    images : torch.Tensor
        The images. Shape: (T, C, H, W)
    mask : torch.Tensor
        The masks. Shape: (T, 1, H, W)
    depths : torch.Tensor
        The depths. Shape: (T, 1, H, W)
    times : torch.Tensor
        The times. Shape: (T, )
    mask_properties : BasicMaskProperties
        The mask properties.
    object_index : int
        The object index of the current mask.
    homography_finder : SequenceHomographyFinder
        The homography finder.
    dtype : torch.dtype
        The data type.
    device : torch.device
        The device.
    depth_ray_thickness : float
        The depth ray thickness.
    min_depth_ray_thickness : float
        The minimum depth ray thickness.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    1. projected_points_2d: The projected points in 2D image / mask space for a plane equation. Shape: (3, T, 2) (x, y)
    2. projected_mask_center: The projected mask center in 2D image / mask space. Shape: (1, T, 2) (x, y)
    3. missing_homographies: A boolean tensor indicating which homographies are missing or incorrect. Shape: (T, )
    """

    T, C_img, H, W = images.shape
    reference_frame_idx = None
    ray_angles = None
    all_points_on_mask = False
    done_points = torch.zeros(T, dtype=torch.bool, device=device)
    valid_done_points = torch.zeros(T, dtype=torch.bool, device=device)
    frame_indices = torch.arange(T, device=device, dtype=torch.int32)

    final_projected_points_2d = torch.zeros(
        3, T, 2, dtype=dtype, device=device)
    final_projected_mask_center = torch.zeros(
        1, T, 2, dtype=dtype, device=device)
    final_missing_homographies = torch.zeros(
        T, dtype=torch.bool, device=device)

    # Specinfy missing points as done for now
    done_points[mask_properties.missing_in_frame] = True
    centeroid = None
    _custom_next_frame_idx = None
    _next_ray_angles = None
    _next_custom_mask_center = None
    _next_use_existing_points = None
    _next_existing_points = None
    # We need to track the points on the mask to get the plane position
    # As the tracking may have errors, the points could move out of the mask which we need to handle
    total_iter = 0
    while not done_points.all():

        if reference_frame_idx is None and _custom_next_frame_idx is None:
            reference_frame_idx = mask_properties.largest_mask_idx
        elif _custom_next_frame_idx is not None:
            reference_frame_idx = _custom_next_frame_idx
            _custom_next_frame_idx = None

        # TODO
        frame_filter = ~done_points
        frame_filter_indices = frame_indices[frame_filter]
        first_invalid = frame_filter_indices.argmin()
        Ts = frame_filter.sum()

        if centeroid is None:
            mask_properties.mask_center_of_mass[frame_filter]

        projected_points_2d, projected_mask_center, missing_homographies = _compute_homography_tracked_points_for_frame(
            images=images[frame_filter],
            mask=mask[frame_filter],
            depths=depths[frame_filter],
            times=times[frame_filter],
            mask_border_points=mask_properties.border_points[:, frame_filter],
            missing_in_frame=mask_properties.missing_in_frame[frame_filter],
            object_index=object_index,
            mask_center_of_mass=mask_properties.mask_center_of_mass[frame_filter],
            homography_finder=homography_finder,
            reference_frame_idx=(frame_filter_indices ==
                                 reference_frame_idx).argwhere().item(),
            custom_mask_center=_next_custom_mask_center,
            frame_index_offset=first_invalid,
            ray_angles=_next_ray_angles,
            use_existing_points=_next_use_existing_points,
            existing_points=_next_existing_points,
            dtype=dtype,
            device=device,
            depth_ray_thickness=depth_ray_thickness,
            min_depth_ray_thickness=min_depth_ray_thickness,
        )
        # Check if all of the tracked points are still on the mask.
        on_the_mask = torch.zeros(Ts, dtype=torch.bool, device=device)
        batch_idx = torch.arange(Ts, device=device, dtype=torch.int32)
        # projected_points_2d: Shape (3, T, 2) (x, y)
        projected_points_2d_b = torch.cat(
            [
                batch_idx.unsqueeze(0).unsqueeze(-1).expand(3, Ts, 1),
                projected_points_2d.flip(-1)  # Flip to (y, x)
            ],
            dim=-1
        )
        projected_mask_center_b = torch.cat(
            [
                batch_idx.unsqueeze(0).unsqueeze(-1).expand(1, Ts, 1),
                projected_mask_center.flip(-1)  # Flip to (y, x)
            ],
            dim=-1
        )
        projected_combined = torch.cat(
            [projected_points_2d_b, projected_mask_center_b], dim=0)
        is_in_mask = is_coord_in_mask(projected_combined.reshape(
            4*Ts, 3).round().int(), mask[frame_filter, 0]).reshape(4, Ts)
        projected_points_2d_in_mask = is_in_mask[:3]  # Shape (3, Ts)
        projected_mask_center_in_mask = is_in_mask[3:4]  # Shape (1, Ts)

        invalid_points = ~projected_points_2d_in_mask.all(dim=0)
        invalid_center = ~projected_mask_center_in_mask.all(dim=0)

        successfull_points = ~invalid_points & ~invalid_center & ~missing_homographies
        successfull_points_all_mask = torch.zeros(
            T, dtype=torch.bool, device=device)
        successfull_points_all_mask[frame_filter_indices] = successfull_points

        done_points[frame_filter] = successfull_points
        valid_done_points[frame_filter] = successfull_points
        final_projected_points_2d[:,
                                  successfull_points_all_mask] = projected_points_2d[:, successfull_points]
        final_projected_mask_center[:,
                                    successfull_points_all_mask] = projected_mask_center[:, successfull_points]
        final_missing_homographies[frame_filter] = missing_homographies

        if not done_points.all():
            use_brute_force = False

            # Find first invalid point
            for i in range((~done_points).sum()):
                current_first_invalid = (~done_points).argwhere().min()
                local_first_invalid = (
                    frame_filter_indices == current_first_invalid)
                global_first_invalid = frame_filter.argwhere()[
                    local_first_invalid].squeeze()

                # If invalid due to homography, we have no option to fix it, so ignore
                if missing_homographies[local_first_invalid]:
                    done_points[global_first_invalid] = True
                    continue

                # If the first invalid point is the reference frame, its corrupt, so mark it as done
                if current_first_invalid == reference_frame_idx:
                    # Mark as done
                    logger.warning(
                        f"Reference frame {current_first_invalid} is still invalid, resetting center.")
                    use_brute_force = True
                    continue
                break
            local_first_invalid = (
                frame_filter_indices == current_first_invalid)
            last_valid_frames = valid_done_points.argwhere().squeeze(-1)
            last_valid_frames_filter = last_valid_frames < current_first_invalid

            if last_valid_frames_filter.sum() == 0:
                last_valid_frame = None
            else:
                last_valid_frame = last_valid_frames[last_valid_frames_filter].max(
                )

            # If the camera center is valid, but one of the points, restart from the failing
            # And keeping the center
            if (not use_brute_force) and invalid_points[local_first_invalid] and ~invalid_center[local_first_invalid]:
                # Get last valid points and center
                current_invalid_points = projected_points_2d[:,
                                                             local_first_invalid]
                current_invalid_center = projected_mask_center[:,
                                                               local_first_invalid]

                _custom_next_frame_idx = current_first_invalid
                _next_ray_angles = _compute_angles(
                    center=current_invalid_center[:, 0].expand(3, 2),
                    points=current_invalid_points[:, 0]
                )
                # Flip to (y, x)
                _next_custom_mask_center = current_invalid_center[:, 0].flip(
                    -1)
                # Use the existing points which are valid
                _next_use_existing_points = projected_points_2d_in_mask[:, local_first_invalid].squeeze(
                    1)
                _next_existing_points = projected_points_2d[:,
                                                            local_first_invalid][:, 0]

            else:
                # If the center is invalid, we can probably keep the rotation so so reusing the angles from the last valid frame

                if last_valid_frame is not None:
                    last_valid_points = final_projected_points_2d[:,
                                                                  last_valid_frame]
                    last_valid_center = final_projected_mask_center[:,
                                                                    last_valid_frame]
                    _next_ray_angles = _compute_angles(
                        center=last_valid_center.expand(3, 2),
                        points=last_valid_points
                    )
                else:
                    # In case the first frame of the sequence is invalid, we have no choice but to use brute force and sample again
                    # Techniqually if we would consider the points from the reference frame as well, this will not happen, but i am lazy
                    _next_ray_angles = None
                _custom_next_frame_idx = current_first_invalid
                _next_custom_mask_center = None
                # Use the existing points which are valid
                _next_use_existing_points = None
                _next_existing_points = None

    return final_projected_points_2d, final_projected_mask_center, final_missing_homographies


def _compute_angles(
        center: torch.Tensor,
        points: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the angles of the points relative to the center point.

    Parameters
    ----------
    center : torch.Tensor
        The center point. Shape: (N, 2)

    points : torch.Tensor
        The points. Shape: (N, 2)

    Returns
    -------
    torch.Tensor
        The angles of the points relative to the center point. Shape: (N, )
        Angles are in radians from -pi to pi.
    """
    center, shp = flatten_batch_dims(center, -2)
    points, _ = flatten_batch_dims(points, -2)

    N, _ = points.shape
    if center.shape != points.shape:
        raise ValueError(
            "The center point must have the same shape as the points.")

    # Calculate the vectors from the center to the points
    vecs = points - center
    vecs = vecs / torch.norm(vecs, dim=-1, keepdim=True)

    # Calculate the angles
    angles = torch.atan2(vecs[..., 1], vecs[..., 0])

    return angles


def compute_proto_plane_position_homography(
    images: torch.Tensor,
    mask: torch.Tensor,
    depths: torch.Tensor,
    times: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    mask_properties: BasicMaskProperties,
    object_index: int,
    homography_finder: SequenceHomographyFinder,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    depth_ray_thickness: float = 0.05,
    min_depth_ray_thickness: float = 10.,
    plot_tracked_points: bool = False,
    plot_tracked_points_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if device is None:
        device = images.device
    T, C_img, H, W = images.shape

    missing_in_frame = mask_properties.missing_in_frame
    largest_mask_idx = mask_properties.largest_mask_idx

    frame_indices = torch.arange(T, device=device, dtype=torch.int32)

    projected_points_2d, projected_mask_center, missing_homographies = _compute_homography_tracked_points(
        images=images,
        mask=mask,
        depths=depths,
        times=times,
        mask_properties=mask_properties,
        object_index=object_index,
        homography_finder=homography_finder,
        dtype=dtype,
        device=device,
        depth_ray_thickness=depth_ray_thickness,
        min_depth_ray_thickness=min_depth_ray_thickness
    )

    if plot_tracked_points:
        plot_proto_points(
            images=images,
            masks=mask,
            proto_points=projected_points_2d[..., :2].swapaxes(0, 1),
            mask_center_point=projected_mask_center[..., :2].swapaxes(0, 1),
            resolution=torch.tensor(
                images.shape[2:], dtype=dtype, device=device) / 4,
            save=True,
            path=plot_tracked_points_path,
            override=True
        )

    # Add a time dimension
    projected_points_2d = projected_points_2d.permute(
        1, 0, 2)  # Shape (T, 3, 2)
    projected_points_2d_lookup = torch.cat(
        [torch.arange(T, device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1).expand(T, 3, 1),
         projected_points_2d.flip(-1)  # Flip to (y, x)
         ], dim=-1).round().int().reshape(T * 3, 3)  # Shape (T * 3, 3)

    # Depth is taken by nearest neighbor
    depth_points = depths[projected_points_2d_lookup[..., 0], :,
                          # Shape (T * 3, 1)
                          projected_points_2d_lookup[..., 1], projected_points_2d_lookup[..., 2]].reshape(T, 3, 1)

    projected_points_3d = torch.cat([
        projected_points_2d,
        depth_points.reshape(T, 3, 1)
    ], dim=-1).reshape(T, 3, 3).permute(1, 0, 2)  # Shape (3, T, 3)

    # Get the plane_support_points to world coordinates
    support_camera = mask_to_camera_coordinates(projected_points_3d[..., :2].reshape(3 * T, 2), torch.tensor(
        [W, H], dtype=dtype, device=device), camera._image_resolution.flip(-1)).reshape(3, T, 2)

    mask_center_camera = mask_to_camera_coordinates(projected_mask_center[..., :2].reshape(1 * T, 2), torch.tensor(
        [W, H], dtype=dtype, device=device), camera._image_resolution.flip(-1)).reshape(1, T, 2)

    ro, rd = camera.get_global_rays(
        uv=support_camera.swapaxes(0, 1), t=times, uv_includes_time=True)

    global_plane_support_points = ro + rd * \
        depth_points.permute(1, 0, 2).expand(-1, -1, 3)

    # global_plane_support_points = ray_position_world

    proto_plane_position = torch.eye(
        4, dtype=dtype, device=device).unsqueeze(0).repeat(T, 1, 1)

    cant_get_proto = ~missing_in_frame & ~missing_homographies

    proto_plane_position[cant_get_proto] = find_plane_rotation_matrix(
        global_plane_support_points[0, cant_get_proto],
        global_plane_support_points[1, cant_get_proto],
        global_plane_support_points[2, cant_get_proto]
    )

    # Now get the the ray intersection of the mask center of the largest_idx_frame with the plane which we will define as new position
    # ro, rd = camera.get_global_rays(
    #     mask_center_camera.swapaxes(0, 1), t=times, uv_includes_time=True)
    # plane_center = compute_ray_plane_intersections_from_position_matrix(
    #     proto_plane_position, ro, rd)

    # global_plane_position = proto_plane_position.clone()
    # global_plane_position[:, :3, 3] = plane_center

    global_plane_position = proto_plane_position.clone()

    # plot_plane_points(
    #         global_plane_support_points[0, ...],
    #         global_plane_support_points[1, ...],
    #         global_plane_support_points[2, ...],
    #         proto_plane_position
    # )

    if ~torch.isfinite(global_plane_position).all():
        raise ValueError("Global plane position contains nan.")
    return global_plane_position, missing_homographies


def get_full_plane_support_points(
    mask: torch.Tensor,
    depths: torch.Tensor,
    times: torch.Tensor,
    dtype: torch.dtype,
    missing_in_frame: torch.Tensor,
) -> torch.Tensor:
    """
    Get the support points of the planes for each mask, defined as all points within the mask and corresponding depth.

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

    Returns
    -------
    torch.Tensor
        The support points. Shape: (MP, 4) (x, y, z, t)
    """
    device = mask.device
    T = len(times)
    D = 3

    depth_points = depths[mask].reshape(-1, 1)
    coords_tyx = mask.squeeze(1).argwhere().float()
    coords_t = coords_tyx[:, 0:1]  # Shape (MP, )
    coords_xy = coords_tyx[:, 1:].flip(-1)  # Shape (MP, 2) (x, y)
    stacked_coords = torch.cat([coords_xy, depth_points, coords_t], dim=-1)
    return stacked_coords


def compute_proto_plane_position_homography_new(
    images: torch.Tensor,
    mask: torch.Tensor,
    depths: torch.Tensor,
    times: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    mask_properties: BasicMaskProperties,
    homography_finder: SequenceHomographyFinder,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Parameters
    ----------
    images : torch.Tensor
        The images. Shape: (T, C, H, W)

    mask : torch.Tensor
        The masks. Shape: (T, 1, H, W)

    depths : torch.Tensor
        The depths. Shape: (T, 1, H, W)

    times : torch.Tensor
        The times. Shape: (T, )

    mask_properties : BasicMaskProperties
        The mask properties.

    homography_finder : SequenceHomographyFinder
        The homography finder.

    """
    if device is None:
        device = images.device
    T, C_img, _H, _W = images.shape

    missing_in_frame = mask_properties.missing_in_frame
    reference_frame_idx = mask_properties.largest_mask_idx

    # Get some initial points for the largests masks
    initial_support_points = get_full_plane_support_points(
        mask[reference_frame_idx][None, ...],
        depths[reference_frame_idx][None, ...],
        times[reference_frame_idx][None, ...],
        dtype,
        missing_in_frame=missing_in_frame[reference_frame_idx][None, ...],
    )  # Shape (MP, 4) (x, y, z, t)

    # scaled_support_points = initial_support_points * 1.0 / min_scale
    H = torch.eye(3, dtype=dtype, device=device).unsqueeze(
        0).repeat(T, 1, 1)  # Shape (T, 3, 3)

    post_filter = (~missing_in_frame)
    post_filter[torch.arange(T) < reference_frame_idx] = False
    pre_filter = (~missing_in_frame)
    pre_filter[torch.arange(T) > reference_frame_idx] = False

    keypoints = [list()] * T
    used_keypoints = [list()] * T

    if post_filter.sum() > 1:
        filtered_images = images[post_filter]
        filtered_masks = mask[post_filter]
        post, ctx = homography_finder.find_homography_mask_wise(
            filtered_images[:-1], filtered_images[1:], filtered_masks[:-1], filtered_masks[1:], return_keypoints=True, return_used_points=True)
        k = ctx.get("keypoints")
        used_points = ctx.get("used_points")

        post_filter_wo_ref = post_filter.clone()
        post_filter_wo_ref[reference_frame_idx] = False

        for entry, valid_idx in enumerate(post_filter_wo_ref.argwhere().squeeze(-1)):
            keypoints[valid_idx] = k[entry][0]
            used_keypoints[valid_idx] = used_points[entry][0]

        H[post_filter_wo_ref] = tensorify(
            post[:, 0], dtype=dtype, device=device)

    if pre_filter.sum() > 1:
        filtered_images = images[pre_filter].flip(
            0)  # Flip the images and masks
        filtered_masks = mask[pre_filter].flip(0)
        # By inverting the order of the images, we can track the homographies backwards, which we need to undo afterwards.
        pre, ctx = homography_finder.find_homography_mask_wise(
            filtered_images[:-1], filtered_images[1:], filtered_masks[:-1], filtered_masks[1:], return_keypoints=True, return_used_points=True)
        # Reverse the keypoints to undo the flip
        k = ctx.get("keypoints")[::-1]
        # Reverse the used points to undo the flip
        used_points = ctx.get("used_points")[::-1]

        pre_filter_wo_ref = pre_filter.clone()
        pre_filter_wo_ref[reference_frame_idx] = False

        for entry, valid_idx in enumerate(pre_filter_wo_ref.argwhere().squeeze(-1)):
            # Reverse the keypoints so it looks like homography would go from 1 to 2.
            keypoints[valid_idx] = k[entry][0][::-1]
            used_keypoints[valid_idx] = used_points[entry][0][::-1]

        H[pre_filter_wo_ref] = tensorify(pre[:, 0], dtype=dtype, device=device).flip(
            0).inverse()  # Flip back the homographies and invert so we mimic the same as the post filter

    invalid_homographies = ~torch.isfinite(H).any(dim=(-1, -2))
    H[invalid_homographies] = torch.eye(
        3, dtype=dtype, device=device).unsqueeze(0)
    invalid_homographies = invalid_homographies | missing_in_frame

    def mask_to_global(
            points: torch.Tensor,
            distance: torch.Tensor,
            camera: TimedCameraSceneNode3D,
            times: torch.Tensor,
            mask_resolution: torch.Tensor,
    ) -> torch.Tensor:
        init_points_cam = mask_to_camera_coordinates(
            points[..., :2], mask_resolution, camera._image_resolution.flip(-1).to(points.device))
        ro, rd = camera.get_global_rays(
            uv=init_points_cam, t=times, uv_includes_time=False)
        glob = (ro + rd * distance.expand(-1, 3).unsqueeze(1))
        return glob

    mask_resolution = torch.tensor([_W, _H], dtype=dtype, device=device)
    glob_initial_points = mask_to_global(initial_support_points[:, :2], initial_support_points[:, 2:3],
                                         camera, times[reference_frame_idx:reference_frame_idx+1], mask_resolution=mask_resolution).squeeze(1)

    # Fit a plane in the initial support points
    glob_centeroid, glob_normal = find_plane(
        glob_initial_points[:, :3])  # Shape (3, )
    # if normal[-1] < 0:
    #     normal = -normal # Flip normal so its pointing in the same direction as the camera
    points_affine = torch.cat([initial_support_points[:, :2], torch.ones_like(
        initial_support_points[:, :1][:, :1])], dim=-1)

    N = min(4, len(points_affine))
    selected_point_idx = torch.linspace(
        0, len(points_affine) - 1, N, device=device).round().int()
    selected_points = points_affine[selected_point_idx]  # Shape (N, 3)

    init_points_cam = mask_to_camera_coordinates(
        selected_points[..., :2], mask_resolution, camera._image_resolution.flip(-1))
    init_ro, init_rd = camera.get_global_rays(
        uv=init_points_cam, t=times[reference_frame_idx:reference_frame_idx+1], uv_includes_time=False)
    glob_plane_points = compute_ray_plane_intersections(glob_centeroid.unsqueeze(
        0).expand_as(init_ro), glob_normal.unsqueeze(0).expand_as(init_ro), init_ro, init_rd)

    plane_G = torch.eye(4, dtype=dtype, device=device).unsqueeze(
        0).repeat(T, 1, 1)  # Shape (T, 4, 4)
    ref_Rot_glob = normal_to_rotmat(glob_normal)

    plane_G[reference_frame_idx, :3, :3] = ref_Rot_glob
    plane_G[reference_frame_idx, :3, 3] = glob_centeroid

    tracked_points = torch.zeros_like(selected_points).unsqueeze(
        0).repeat(T, 1, 1)  # Shape (T, N, 3)
    tracked_points[reference_frame_idx] = selected_points

    tracked_global_points = torch.zeros_like(
        selected_points).unsqueeze(0).repeat(T, 1, 1)  # Shape (T, P, 3)
    tracked_global_points[reference_frame_idx] = glob_plane_points[:, 0]

    # Apply Homography to the points and compute transformation delta
    # Get following points
    for i in range(reference_frame_idx + 1, T):
        old_plane_estimate = plane_G[i - 1].clone()
        # Get old plane estimate to local camera coordinates
        old_plane_estimate_local = camera.get_global_position(
            t=times[i - 1]).inverse() @ old_plane_estimate

        if invalid_homographies[i]:
            tracked_points[i] = tracked_points[i - 1]
            plane_G[i] = camera.get_global_position(
                t=times[i]) @ old_plane_estimate_local
            continue
        # Apply homography to the points on the plane
        tracked_points[i] = torch.bmm(H[i].unsqueeze(0).expand(
            N, -1, -1), tracked_points[i - 1].unsqueeze(-1)).squeeze(-1)

        # Take the inverse of the homography, so rot, can etc. describe the transformation from image 2 to image 1
        num_solutions, cam_rot_cand, cam_trans_cand, normal_cand = cv2.decomposeHomographyMat(
            H[i].inverse().cpu().numpy(), camera.get_intrinsics(t=times[i]).squeeze(0).cpu().numpy())
        start_ref = keypoints[i][1]
        end_ref = keypoints[i][0]
        possible_sol = cv2.filterHomographyDecompByVisibleRefpoints(
            tuple((x.astype(np.float32) for x in cam_rot_cand)), tuple(
                (x.astype(np.float32) for x in normal_cand)),
            beforePoints=start_ref.astype(
                np.float32)[:, None], afterPoints=end_ref.astype(np.float32)[:, None],
            pointsMask=used_keypoints[i].astype(np.uint8))

        if possible_sol is None or len(possible_sol) == 0:
            logger.warning(
                f"Homography decomposition failed for frame {i}.")
            invalid_homographies[i] = True
            tracked_points[i] = tracked_points[i - 1]
            plane_G[i] = camera.get_global_position(
                t=times[i]) @ old_plane_estimate_local
            continue

        # Guessed valid rotations of to camera to from first image to second
        valid_rot = torch.tensor(np.array(cam_rot_cand), dtype=dtype)[
            possible_sol.squeeze(-1)]
        # Guessed valid translations of to camera to from first image to second
        valid_trans = torch.tensor(np.array(cam_trans_cand), dtype=dtype)[
            possible_sol.squeeze(-1)]
        # Guessed normal of the plane the points assumed to lie on.
        valid_normal = torch.tensor(np.array(normal_cand), dtype=dtype)[
            possible_sol.squeeze(-1)]
        valid_plane_rotmat = normal_to_rotmat(
            valid_normal.squeeze(-1))  # Shape (N, 3, 3)
        #

        used_kp1 = torch.tensor(keypoints[i][0][used_keypoints[i]]).to(dtype)
        used_kp2 = torch.tensor(keypoints[i][1][used_keypoints[i]]).to(dtype)

        angles = vector_angle(rotmat_to_normal(old_plane_estimate[:3, :3]).unsqueeze(
            0).expand(valid_normal.shape[0], -1), valid_normal.squeeze(-1), mode="tan2")
        min_idx = angles.argmin().squeeze(-1)

        obj_delta_local, err = compute_translation_delta_estimate(
            valid_rot[min_idx],
            valid_trans[min_idx],
            old_plane_estimate_local,
            used_kp1,
            used_kp2,
            times[i],
            camera,
        )
        local_obj_delta = obj_delta_local @ old_plane_estimate_local
        plane_G[i] = camera.get_global_position(t=times[i]) @ local_obj_delta

    for i in range(reference_frame_idx - 1, -1, -1):
        old_plane_estimate = plane_G[i + 1].clone()
        if not torch.isfinite(old_plane_estimate).all():
            logger.warning(
                f"Plane estimate for frame {i + 1} is not finite, skipping.")
            continue
        # Get old plane estimate to local camera coordinates
        old_plane_estimate_local = camera.get_global_position(
            t=times[i + 1]).inverse() @ old_plane_estimate

        if invalid_homographies[i]:
            tracked_points[i] = tracked_points[i + 1]
            plane_G[i] = camera.get_global_position(
                t=times[i]) @ old_plane_estimate_local
            continue
        # Apply homography to the points on the plane
        tracked_points[i] = torch.bmm(H[i].unsqueeze(0).expand(
            N, -1, -1), tracked_points[i + 1].unsqueeze(-1)).squeeze(-1)

        # Here, we dont take the inverse, as we want to compute the transformation from image 1 to image 2, E.g this corresponds to the object moving from image 2 to image 1.
        num_solutions, cam_rot_cand, cam_trans_cand, normal_cand = cv2.decomposeHomographyMat(
            H[i].cpu().numpy(), camera.get_intrinsics(t=times[i]).squeeze(0).cpu().numpy())
        start_ref = keypoints[i][0]  # No swap here
        end_ref = keypoints[i][1]
        possible_sol = cv2.filterHomographyDecompByVisibleRefpoints(
            tuple((x.astype(np.float32) for x in cam_rot_cand)), tuple(
                (x.astype(np.float32) for x in normal_cand)),
            beforePoints=start_ref.astype(
                np.float32)[:, None], afterPoints=end_ref.astype(np.float32)[:, None],
            pointsMask=used_keypoints[i].astype(np.uint8))

        if possible_sol is None or len(possible_sol) == 0:
            logger.warning(
                f"Homography decomposition failed for frame {i}.")
            invalid_homographies[i] = True
            tracked_points[i] = tracked_points[i + 1]
            plane_G[i] = camera.get_global_position(
                t=times[i]) @ old_plane_estimate_local
            continue

        # Guessed valid rotations of to camera to from first image to second
        valid_rot = torch.tensor(np.array(cam_rot_cand), dtype=dtype)[
            possible_sol.squeeze(-1)]
        # Guessed valid translations of to camera to from first image to second
        valid_trans = torch.tensor(np.array(cam_trans_cand), dtype=dtype)[
            possible_sol.squeeze(-1)]
        # Guessed normal of the plane the points assumed to lie on.
        valid_normal = torch.tensor(np.array(normal_cand), dtype=dtype)[
            possible_sol.squeeze(-1)]
        valid_plane_rotmat = normal_to_rotmat(
            valid_normal.squeeze(-1))  # Shape (N, 3, 3)

        used_kp1 = torch.tensor(keypoints[i][1][used_keypoints[i]]).to(
            dtype)  # Flip kp1 and kp2 here
        used_kp2 = torch.tensor(keypoints[i][0][used_keypoints[i]]).to(dtype)

        angles = vector_angle(rotmat_to_normal(old_plane_estimate[:3, :3]).unsqueeze(
            0).expand(valid_normal.shape[0], -1), valid_normal.squeeze(-1), mode="tan2")
        min_idx = angles.argmin().squeeze(-1)

        obj_delta_local, err = compute_translation_delta_estimate(
            valid_rot[min_idx],
            valid_trans[min_idx],
            old_plane_estimate_local,
            used_kp1,
            used_kp2,
            times[i],
            camera,
        )
        local_obj_delta = obj_delta_local @ old_plane_estimate_local
        plane_G[i] = camera.get_global_position(t=times[i]) @ local_obj_delta

    # norm_points = ((tracked_points) / tracked_points[:, :, 2:3]) * tracked_points[reference_frame_idx, :, 2:3]
    # cmap = plt.get_cmap("rainbow")
    # plot_tracked_points(images, masks, norm_points, colors=tensorify(cmap(torch.linspace(0, cmap.N, norm_points.shape[1]).round().int())[None, ...]).expand(T, -1, -1), open=True, save=True, path="temp/homography/in_homog.png", override=True)

    # Testing
    # init_points_loc = global_to_local(plane_G[reference_frame_idx], glob_plane_points[:, 0])[:, 0, :3]
    # check_glob = local_to_global(plane_G, init_points_loc, False)[..., :3]

    # img_coords = camera.global_to_image_coordinates(check_glob, t=times, v_includes_time=True)[..., :2].squeeze(-1).round().int()
    # mask_coords = img_coords / camera._image_resolution.flip(-1) * torch.tensor([_W, _H], dtype=dtype, device=device)

    # plot_tracked_points(images, masks, mask_coords.permute(1, 0, 2), colors=tensorify(cmap(torch.linspace(0, cmap.N, mask_coords.shape[0]).round().int())[None, ...]).expand(T, -1, -1), open=True, save=True, path="temp/homography/in_global1.png", override=True)
    # plot_position(plane_G, times=times, open=True)

    return plane_G, invalid_homographies


@saveable()
def plot_homography_decomposition(
        H: torch.Tensor,
        rotation_estimate: torch.Tensor,
        translation_estimate: torch.Tensor,
        plane_estimate: torch.Tensor,
        keypoints1: torch.Tensor,
        keypoints2: torch.Tensor,
        t: torch.Tensor,
        camera: TimedCameraSceneNode3D,
        move_camera: bool = True):
    dtype = rotation_estimate.dtype
    device = rotation_estimate.device

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig, ax = get_mpl_figure(1, 1, subplot_kw={'projection': '3d'})
    # Plot keypoints of first image

    # Plotting a plane of image size
    edge = torch.tensor(
        [[0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
         ], dtype=torch.float32) * torch.cat([camera._image_resolution.flip(-1).cpu(), torch.tensor([0])], dim=-1)

    loc_o, _ = camera.get_local_rays(
        edge[..., :2], t=torch.tensor(0.), uv_includes_time=False)

    # Plot the camera 0 plane
    ax.add_collection3d(
        Poly3DCollection(
            [loc_o[:, 0].numpy(
            )], facecolors="white", linewidths=1, edgecolors="black", alpha=.2))

    plane_target_gmat_cam1 = plane_estimate

    # Init for cam2 in case obj is moving
    plane_target_gmat_cam2 = plane_target_gmat_cam1.clone()

    # PLot the target plane (cam1)
    tilted_face = torch.bmm(plane_target_gmat_cam1.unsqueeze(0).expand(
        4, -1, -1).to(dtype=dtype), assure_homogeneous_vector(loc_o[:, 0] * 2).unsqueeze(-1)).squeeze(-1)[:, :3]
    ax.add_collection3d(
        Poly3DCollection(
            [tilted_face.numpy()], facecolors="white", linewidths=1, edgecolors="black", alpha=.2))

    # Compute error of the basic_homography
    pts = torch.bmm(H.unsqueeze(0).expand(keypoints1.shape[0], -1, -1), torch.cat([torch.tensor(
        keypoints1), torch.ones_like(torch.tensor(keypoints1)[:, 0:1])], dim=-1).unsqueeze(-1)).squeeze(-1)
    used_k1_homog_to_k2 = (pts / pts[:, 2:3])[:, :2]

    plain_homog_pixel_err = (used_k1_homog_to_k2 - keypoints2).norm(dim=-1)

    loc_kp1, loc_kp1_d = camera.get_local_rays(
        keypoints1[:, :2], t=t, uv_includes_time=False)
    # loc_kp2, loc_kp2_d = camera.get_local_rays(torch.tensor(used_kp2[:, :2]).to(dtype=dtype), t=times[i], uv_includes_time=False)
    loc_kp1 = loc_kp1[:, 0, :]

    intersec_k1 = compute_ray_plane_intersections(plane_target_gmat_cam1[:3, 3].unsqueeze(0).expand_as(loc_kp1), rotmat_to_normal(
        plane_target_gmat_cam1[:3, :3]).to(dtype=dtype).unsqueeze(0).expand_as(loc_kp1), loc_kp1, loc_kp1_d)

    # Draw the predicted frame
    # valid_rot, and valid_trans describe the transformation from image 2 to image 1
    crot = rotation_estimate.squeeze(-1).to(dtype=dtype)
    vtrans = translation_estimate.squeeze(-1).to(dtype=dtype)
    homo_estm_scale_inv = torch.eye(4, dtype=dtype, device=device)
    homo_estm_scale_inv[:3, :3] = crot
    homo_estm_scale_inv[:3, 3] = vtrans

    _, glob_kp2_d_guess = get_global_rays(keypoints2[:, :2],
                                          global_position=homo_estm_scale_inv.unsqueeze(
                                              0),
                                          inverse_intrinsics=camera.get_inverse_intrinsics(t=torch.tensor(0.)), lens_distortion=camera.get_lens_distortion(), focal_length=camera.focal_length, uv_includes_time=False)

    # We assume that we know the plane distance, we can compute the line intersection of the points on camera 1 with the shift estimated by the homography which is right up to a scale factor
    # And the ray coming from the intersection plane to camera 2.
    p1 = loc_kp1
    u1 = -1 * vtrans.expand_as(loc_kp1)  # Rays from cam1 to cam2
    p2 = intersec_k1
    u2 = -1 * glob_kp2_d_guess  # Rays from obj to cam2
    # May not be very intersecting, but we can use the closest point to the line as a good approximation of the intersection
    intersects, inter_ctx = compute_line_intersections(
        p1, u1, p2, u2, is_close_eps=1e-1)
    scales_cam_t = inter_ctx.get("r1")

    # Compute camera translation by the median of the scales
    selected_scale = torch.mean(scales_cam_t, dim=-1)

    v_scaled_trans = selected_scale.unsqueeze(0) * vtrans.squeeze(-1)
    g_pred_corrected = homo_estm_scale_inv.clone()
    # Invert the translation as v_scaled_trans points from cam1 to cam2, we want the translation from cam2 to cam1
    g_pred_corrected[:3, 3] = -1 * v_scaled_trans

    # Refine the position using Opencv generic Pnp
    distCoeff = camera.get_lens_distortion_opencv().cpu().clone().numpy()
    cam_mat = camera.get_intrinsics(t=t).squeeze(0).cpu().clone().numpy()
    in_tvec = g_pred_corrected.inverse(
    )[:3, 3].unsqueeze(-1).cpu().clone().numpy()
    # Add the camera focal length to the translation vector, as opencvs camera origin is not in the image plane, but in the focal length
    in_tvec[2] += camera.focal_length.item()
    in_rvec = rotmat_to_rotvec(g_pred_corrected.inverse()[
                               :3, :3]).unsqueeze(-1).clone().cpu().numpy()
    # Refine the position using Opencv generic Pnp

    nsol, rvec, tvec, repr_errors = cv2.solvePnPGeneric(
        objectPoints=intersec_k1.numpy(),
        imagePoints=keypoints2.numpy(),
        cameraMatrix=cam_mat,
        distCoeffs=distCoeff,
        tvec=in_tvec,
        rvec=in_rvec,
        useExtrinsicGuess=True,
    )
    rvec = rvec[0]
    tvec = tvec[0]
    repr_error = repr_errors[0]  # RMSE of the points

    # Refine the position using Opencv PnPRefineLM - works
    # rvec, tvec = cv2.solvePnPRefineLM(
    #     objectPoints=intersec_k1.numpy(),
    #     imagePoints=used_kp2.numpy(),
    #     cameraMatrix=cam_mat,
    #     distCoeffs=distCoeff,
    #     tvec=in_tvec,
    #     rvec=in_rvec,
    # )
    g_pred = torch.eye(4, dtype=dtype, device=device)
    g_pred[:3, :3] = rotvec_to_rotmat(
        torch.tensor(rvec).squeeze(1).to(dtype=dtype))
    g_pred[:3, 3] = torch.tensor(tvec).squeeze(1).to(
        dtype=dtype) - torch.tensor([0, 0, camera.focal_length]).to(dtype=dtype)

    if move_camera:
        # Take the inverse as its cam1 to cam2
        # Translation from cam2 to cam1 aka local to global
        g_pred_corrected = g_pred.inverse()
        g_pred_object = torch.eye(4, dtype=dtype, device=device)
    else:
        # cam1 to cam2, or current object to new position obj
        # Identity matrix, as we are not moving the camera
        g_pred_corrected = torch.eye(4, dtype=dtype, device=device)
        # Update the plane target matrix to the new position of the object
        plane_target_gmat_cam2 = g_pred @ plane_target_gmat_cam2

    # offset_to_image_plane = (g_pred_corrected.inverse() @ torch.tensor([0, 0, 1, 1]).to(dtype=dtype).unsqueeze(-1)).squeeze(-1)[:3] # Shape (3, )
    # g_pred_corrected[:3, 3] = g_pred_corrected[:3, 3] - offset_to_image_plane

    glob_kp2_pred, glob_kp2_d_pred = get_global_rays(keypoints2[:, :2],
                                                     global_position=g_pred_corrected.unsqueeze(
                                                         0),
                                                     inverse_intrinsics=camera.get_inverse_intrinsics(t=torch.tensor(0.)), lens_distortion=camera.get_lens_distortion(), focal_length=camera.focal_length, uv_includes_time=False)

    glob_cam2_pred, _ = get_global_rays(edge[..., :2],
                                        global_position=g_pred_corrected.unsqueeze(
                                            0),
                                        inverse_intrinsics=camera.get_inverse_intrinsics(t=torch.tensor(0.)), lens_distortion=camera.get_lens_distortion(), focal_length=camera.focal_length, uv_includes_time=False)

    if move_camera:
        # Plot the camera 2 plane
        ax.add_collection3d(
            Poly3DCollection(
                [glob_cam2_pred[:, 0].numpy(
                )], facecolors="white", linewidths=1, edgecolors="blue", alpha=.2))
    else:
        # Plot the object plane
        tilted_face2 = torch.bmm(plane_target_gmat_cam2.unsqueeze(0).expand(
            4, -1, -1).to(dtype=dtype), assure_homogeneous_vector(loc_o[:, 0] * 2).unsqueeze(-1)).squeeze(-1)[:, :3]
        ax.add_collection3d(
            Poly3DCollection(
                [tilted_face2.numpy()], facecolors="white", linewidths=1, edgecolors="blue", alpha=.2))

    from matplotlib.colors import to_rgba

    subsample_rays = 4

    # Plot the rays
    ax.quiver(*loc_kp1[::subsample_rays].detach().cpu().numpy().T, *loc_kp1_d[::subsample_rays].detach().cpu().numpy().T,
              length=1.,
              normalize=True,
              arrow_length_ratio=0,
              linewidths=.5,
              color=to_rgba("gray", alpha=0.4),
              label="Rays 1"
              )

    ax.quiver(*glob_kp2_pred[::subsample_rays].detach().cpu().numpy().T, *glob_kp2_d_pred[::subsample_rays].detach().cpu().numpy().T,
              length=1.,
              normalize=True,
              arrow_length_ratio=0,
              linewidths=.5,
              color=to_rgba("blue", alpha=0.4),
              label="Rays 2 pred"
              )

    glob_kp2_pred = glob_kp2_pred[:, 0, :]

    ax.scatter(loc_kp1[:, 0], loc_kp1[:, 1], np.zeros_like(
        loc_kp1[:, 1]), c='r', marker='o')
    ax.scatter(glob_kp2_pred[:, 0], glob_kp2_pred[:, 1],
               glob_kp2_pred[:, 2], c='b', marker='o')

    # Scatter intersection points
    intersec_k2 = compute_ray_plane_intersections(plane_target_gmat_cam2[:3, 3].unsqueeze(0).expand_as(glob_kp2_pred), rotmat_to_normal(
        plane_target_gmat_cam2[:3, :3]).unsqueeze(0).expand_as(glob_kp2_pred), glob_kp2_pred, glob_kp2_d_pred)

    if move_camera:
        # Object is static, intersection points of cam1 shall be the same as the intersection points of cam2
        intersect_k2_comp = intersec_k1
    else:
        # Object was moved, we take the intersection points of cam1, put them in the plane coordinate and project them back to the global coordinates of cam2
        plane_loc_intersec_k1 = global_to_local(
            plane_target_gmat_cam1, intersec_k1, False)[:, 0, :3]
        intersect_k2_comp = local_to_global(
            plane_target_gmat_cam2, plane_loc_intersec_k1, False)[:, 0, :3]

    err_dir = intersec_k2 - intersect_k2_comp
    intersection_error = err_dir[:, :2].norm(dim=-1)
    print("Mean Intersection Error at the object: ",
          intersection_error.mean().item())

    # color = torch.where(intersection_error < eps, color_right, color_wrong)
    ax.scatter(intersect_k2_comp[:, 0], intersect_k2_comp[:,
               1], intersect_k2_comp[:, 2], c='gray')
    ax.scatter(intersec_k2[:, 0], intersec_k2[:, 1],
               intersec_k2[:, 2], c='blue')

    from tools.transforms.min_max import MinMax
    cmap = plt.get_cmap("hot")
    norm = MinMax(new_min=0, new_max=cmap.N - 1, dim=(-1, ))

    error_norm = err_dir.norm(dim=-1)
    normalized_error = norm.fit_transform(error_norm)
    color = cmap(normalized_error.cpu().numpy().round().astype(int))

    ax.quiver(*intersect_k2_comp.detach().cpu().numpy().T, *err_dir.detach().cpu().numpy().T,
              colors=color,
              arrow_length_ratio=0,
              linewidths=.5,
              label="Intersection error"
              )

    # Project k1 intersection points back into the image domain of cam 2 using corrected camera pose and calculate the pixel error
    local_k1_points_in_cam2 = global_to_local(
        g_pred_corrected, intersect_k2_comp, False)
    image_k1_points_in_cam2 = local_to_image_coordinates(local_k1_points_in_cam2.permute(1, 0, 2), camera.get_intrinsics(
        t=t), focal_length=camera.focal_length, lens_distortion=camera.get_lens_distortion(), v_includes_time=True)[0]

    projected_geometric_pixel_err = (
        image_k1_points_in_cam2 - keypoints2).norm(dim=-1)

    print(
        f"Homography mean pixel error: {plain_homog_pixel_err.mean().item():.3f}")
    print(
        f"Projected geometric mean pixel error: {projected_geometric_pixel_err.mean().item():.3f}")

    # # Compute the camera estimation without rounding the scale and compute for every ray each.
    # g_pred_corrected_exact = homo_estm_scale_inv.clone().unsqueeze(0).repeat(scales_cam_t.shape[0], 1, 1)
    # g_pred_corrected_exact[:, :3, 3] = -1 * (scales_cam_t.unsqueeze(1).expand(-1, 3) * vtrans.unsqueeze(0)).squeeze(-1)

    # glob_kp2_pred_exact, glob_kp2_d_pred_exact = get_global_rays(used_kp2[:, :2].unsqueeze(1),
    #             global_position=g_pred_corrected_exact,
    #             inverse_intrinsics=camera.get_inverse_intrinsics(t=torch.tensor(0.)).expand(g_pred_corrected_exact.shape[0], -1, -1), lens_distortion=camera.get_lens_distortion(), focal_length=camera.focal_length, uv_includes_time=True)

    # glob_kp2_pred_exact = glob_kp2_pred_exact.squeeze(0)
    # glob_kp2_d_pred_exact = glob_kp2_d_pred_exact.squeeze(0)
    # intersec_k2_exact = compute_ray_plane_intersections(plane_target_gmat[:3, 3].unsqueeze(0).expand_as(glob_kp2_pred), valid_normal[solution_index].squeeze(-1).to(dtype=dtype).unsqueeze(0).expand_as(glob_kp2_pred_exact), glob_kp2_pred_exact, glob_kp2_d_pred_exact)
    # err_dir_exact = intersec_k2_exact - intersec_k1
    # intersection_error_exact = err_dir_exact[:, :2].norm(dim=-1)
    # print("Mean Intersection Error at the object (exact): ", intersection_error_exact.mean().item())
    # local_k1_points_in_cam2_exact = global_to_local(g_pred_corrected_exact, intersec_k1.unsqueeze(0), v_include_time=True)
    # image_k1_points_in_cam2_exact = local_to_image_coordinates(local_k1_points_in_cam2_exact, camera.get_intrinsics(t=times[0]), focal_length=camera.focal_length, lens_distortion=camera.get_lens_distortion(), v_includes_time=True)[0]
    # projected_geometric_pixel_err_exact = (image_k1_points_in_cam2_exact - used_kp2).norm(dim=-1)
    # print(f"Projected geometric mean pixel error (exact): {projected_geometric_pixel_err_exact.mean().item():.3f}")

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig


def compute_translation_delta_estimate_old(
        rotation_estimate: torch.Tensor,
        translation_estimate: torch.Tensor,
        plane_estimate: torch.Tensor,
        keypoints1: torch.Tensor,
        keypoints2: torch.Tensor,
        t: torch.Tensor,
        camera: TimedCameraSceneNode3D,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the translation delta of a homography decomposition.

    Eg. if the Homography describes the transformation from frame 1 to frame 2, the translation delta is geometric 3D translation from frame 1 to frame 2.

    Parameters
    ----------
    rotation_estimate : torch.Tensor
        The rotation estimate as rotation matrix from the homography decomposition. Shape: (3, 3)

    translation_estimate : torch.Tensor
        The translation estimate as translation vector from the homography decomposition. Shape: (3, )

    plane_estimate : torch.Tensor
        An enitial plane estimate. As this function is used to absolutly define the translation (has homogrophy decomposition does it just up to a translation scale),
        an initial guess for the plane must be provided by setting the translation vector to like (0, 0, 1) or a better estimate.
        The rotation of the plane may be aquired by the homography decomposition, or a better estimate.
        Homogenous transformation matrix of the plane. Shape: (4, 4)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
    1. The corrected transformation matrix (delta) from local cam 1 to (local) cam 2. Shape: (4, 4)
    Can also be used to transform the plane_estimate.

    2. The reprojection error (RMSE) of the points in the image plane of cam 2. Shape: (1, )
    """
    dtype = rotation_estimate.dtype
    device = rotation_estimate.device

    plane_target_gmat_cam1 = plane_estimate

    # Init for cam2 in case obj is moving
    plane_target_gmat_cam2 = plane_target_gmat_cam1.clone()

    loc_kp1, loc_kp1_d = camera.get_local_rays(
        keypoints1[:, :2], t=t, uv_includes_time=False)
    # loc_kp2, loc_kp2_d = camera.get_local_rays(torch.tensor(used_kp2[:, :2]).to(dtype=dtype), t=times[i], uv_includes_time=False)
    loc_kp1 = loc_kp1[:, 0, :]
    intersec_k1 = compute_ray_plane_intersections(plane_target_gmat_cam1[:3, 3].unsqueeze(0).expand_as(loc_kp1), rotmat_to_normal(
        plane_target_gmat_cam1[:3, :3]).to(dtype=dtype).unsqueeze(0).expand_as(loc_kp1), loc_kp1, loc_kp1_d)

    # valid_rot, and valid_trans describe the transformation from image 2 to image 1
    crot = rotation_estimate.squeeze(-1).to(dtype=dtype)
    vtrans = translation_estimate.squeeze(-1).to(dtype=dtype)
    homo_estm_scale_inv = torch.eye(4, dtype=dtype, device=device)
    homo_estm_scale_inv[:3, :3] = crot
    homo_estm_scale_inv[:3, 3] = vtrans

    _, glob_kp2_d_guess = get_global_rays(keypoints2[:, :2],
                                          global_position=homo_estm_scale_inv.unsqueeze(
                                              0),
                                          inverse_intrinsics=camera.get_inverse_intrinsics(t=torch.tensor(0.)), lens_distortion=camera.get_lens_distortion(), focal_length=camera.focal_length, uv_includes_time=False)

    # We assume that we know the plane distance, we can compute the line intersection of the points on camera 1 with the shift estimated by the homography which is right up to a scale factor
    # And the ray coming from the intersection plane to camera 2.
    p1 = loc_kp1
    u1 = -1 * vtrans.expand_as(loc_kp1)  # Rays from cam1 to cam2
    p2 = intersec_k1
    u2 = -1 * glob_kp2_d_guess  # Rays from obj to cam2
    # May not be very intersecting, but we can use the closest point to the line as a good approximation of the intersection
    _, inter_ctx = compute_line_intersections(
        p1, u1, p2, u2, is_close_eps=1e-1)
    scales_cam_t = inter_ctx.get("r1")

    # Compute camera translation by the median of the scales
    selected_scale = torch.nanmean(scales_cam_t, dim=-1)

    v_scaled_trans = selected_scale.unsqueeze(0) * vtrans.squeeze(-1)
    g_pred_corrected = homo_estm_scale_inv.clone()
    # Invert the translation as v_scaled_trans points from cam1 to cam2, we want the translation from cam2 to cam1
    g_pred_corrected[:3, 3] = -1 * v_scaled_trans

    # Refine the position using Opencv generic Pnp
    distCoeff = camera.get_lens_distortion_opencv().cpu().clone().numpy()
    cam_mat = camera.get_intrinsics(t=t).squeeze(0).cpu().clone().numpy()
    in_tvec = g_pred_corrected.inverse(
    )[:3, 3].unsqueeze(-1).cpu().clone().numpy()
    # Add the camera focal length to the translation vector, as opencvs camera origin is not in the image plane, but in the focal length
    in_tvec[2] += camera.focal_length.item()
    in_rvec = rotmat_to_rotvec(g_pred_corrected.inverse()[
                               :3, :3]).unsqueeze(-1).clone().cpu().numpy()

    # Refine the position using Opencv generic Pnp
    nsol, rvec, tvec, repr_errors = cv2.solvePnPGeneric(
        objectPoints=intersec_k1.numpy(),
        imagePoints=keypoints2.numpy(),
        cameraMatrix=cam_mat,
        distCoeffs=distCoeff,
        tvec=in_tvec,
        rvec=in_rvec,
        useExtrinsicGuess=True,
    )
    rvec = rvec[0]
    tvec = tvec[0]
    repr_error = torch.tensor(
        repr_errors[0], dtype=dtype)  # RMSE of the points

    g_pred = torch.eye(4, dtype=dtype, device=device)
    g_pred[:3, :3] = rotvec_to_rotmat(
        torch.tensor(rvec).squeeze(1).to(dtype=dtype))
    # Add focal length again.
    g_pred[:3, 3] = torch.tensor(tvec).squeeze(1).to(
        dtype=dtype) - torch.tensor([0, 0, camera.focal_length]).to(dtype=dtype)

    return g_pred, repr_error


def compute_translation_delta_estimate(
        rotation_estimate: torch.Tensor,
        translation_estimate: torch.Tensor,
        plane_estimate: torch.Tensor,
        keypoints1: torch.Tensor,
        keypoints2: torch.Tensor,
        t: torch.Tensor,
        camera: TimedCameraSceneNode3D,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the translation delta of a homography decomposition.

    Eg. if the Homography describes the transformation from frame 1 to frame 2, the translation delta is geometric 3D translation from frame 1 to frame 2.

    Parameters
    ----------
    rotation_estimate : torch.Tensor
        The rotation estimate as rotation matrix from the homography decomposition. Shape: (3, 3)

    translation_estimate : torch.Tensor
        The translation estimate as translation vector from the homography decomposition. Shape: (3, )

    plane_estimate : torch.Tensor
        An enitial plane estimate. As this function is used to absolutly define the translation (has homogrophy decomposition does it just up to a translation scale),
        an initial guess for the plane must be provided by setting the translation vector to like (0, 0, 1) or a better estimate.
        The rotation of the plane may be aquired by the homography decomposition, or a better estimate.
        Homogenous transformation matrix of the plane. Shape: (4, 4)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
    1. The corrected transformation matrix (delta) from local cam 1 to (local) cam 2. Shape: (4, 4)
    Can also be used to transform the plane_estimate.

    2. The reprojection error (RMSE) of the points in the image plane of cam 2. Shape: (1, )
    """
    dtype = rotation_estimate.dtype
    device = rotation_estimate.device

    plane_target_gmat_cam1 = plane_estimate

    # Init for cam2 in case obj is moving
    plane_target_gmat_cam2 = plane_target_gmat_cam1.clone()

    loc_kp1, loc_kp1_d = camera.get_local_rays(
        keypoints1[:, :2], t=t, uv_includes_time=False)
    # loc_kp2, loc_kp2_d = camera.get_local_rays(torch.tensor(used_kp2[:, :2]).to(dtype=dtype), t=times[i], uv_includes_time=False)
    loc_kp1 = loc_kp1[:, 0, :]
    intersec_k1 = compute_ray_plane_intersections(plane_target_gmat_cam1[:3, 3].unsqueeze(0).expand_as(loc_kp1), rotmat_to_normal(
        plane_target_gmat_cam1[:3, :3]).to(dtype=dtype).unsqueeze(0).expand_as(loc_kp1), loc_kp1, loc_kp1_d)

    # valid_rot, and valid_trans describe the transformation from image 2 to image 1
    crot = rotation_estimate.squeeze(-1).to(dtype=dtype)
    vtrans = translation_estimate.squeeze(-1).to(dtype=dtype)
    homo_estm_scale_inv = torch.eye(4, dtype=dtype, device=device)
    homo_estm_scale_inv[:3, :3] = crot
    homo_estm_scale_inv[:3, 3] = vtrans

    _, glob_kp2_d_guess = get_global_rays(keypoints2[:, :2],
                                          global_position=homo_estm_scale_inv.unsqueeze(
                                              0),
                                          inverse_intrinsics=camera.get_inverse_intrinsics(t=torch.tensor(0.)), lens_distortion=camera.get_lens_distortion(), focal_length=camera.focal_length, uv_includes_time=False)
    finite_intersect = torch.isfinite(intersec_k1).all(dim=-1)
    # We assume that we know the plane distance, we can compute the line intersection of the points on camera 1 with the shift estimated by the homography which is right up to a scale factor
    # And the ray coming from the intersection plane to camera 2.
    p1 = loc_kp1[finite_intersect]
    u1 = -1 * vtrans.expand_as(p1)  # Rays from cam1 to cam2
    p2 = intersec_k1[finite_intersect]
    u2 = -1 * glob_kp2_d_guess[finite_intersect]  # Rays from obj to cam2
    # May not be very intersecting, but we can use the closest point to the line as a good approximation of the intersection
    _, inter_ctx = compute_line_intersections(
        p1, u1, p2, u2, is_close_eps=1e-1)
    scales_cam_t = inter_ctx.get("r1")

    # Compute camera translation by the median of the scales
    selected_scale = torch.nanmean(scales_cam_t, dim=-1)

    v_scaled_trans = selected_scale.unsqueeze(0) * vtrans.squeeze(-1)
    g_pred_corrected = homo_estm_scale_inv.clone()
    # Invert the translation as v_scaled_trans points from cam1 to cam2, we want the translation from cam2 to cam1
    g_pred_corrected[:3, 3] = -1 * v_scaled_trans

    # Refine the position using Opencv generic Pnp
    distCoeff = camera.get_lens_distortion_opencv().cpu().clone().numpy()
    cam_mat = camera.get_intrinsics(t=t).squeeze(0).cpu().clone().numpy()
    in_tvec = g_pred_corrected.inverse(
    )[:3, 3].unsqueeze(-1).cpu().clone().numpy()
    # Add the camera focal length to the translation vector, as opencvs camera origin is not in the image plane, but in the focal length
    in_tvec[2] += camera.focal_length.item()
    in_rvec = rotmat_to_rotvec(g_pred_corrected.inverse()[
                               :3, :3]).unsqueeze(-1).clone().cpu().numpy()

    # Refine the position using Opencv generic Pnp
    nsol, rvec, tvec, repr_errors = cv2.solvePnPGeneric(
        objectPoints=intersec_k1[finite_intersect].numpy(),
        imagePoints=keypoints2[finite_intersect].numpy(),
        cameraMatrix=cam_mat,
        distCoeffs=distCoeff,
        tvec=in_tvec,
        rvec=in_rvec,
        useExtrinsicGuess=True,
    )
    rvec = rvec[0]
    tvec = tvec[0]
    repr_error = torch.tensor(
        repr_errors[0], dtype=dtype)  # RMSE of the points

    g_pred = torch.eye(4, dtype=dtype, device=device)
    g_pred[:3, :3] = rotvec_to_rotmat(
        torch.tensor(rvec).squeeze(1).to(dtype=dtype))
    # Add focal length again.
    g_pred[:3, 3] = torch.tensor(tvec).squeeze(1).to(
        dtype=dtype) - torch.tensor([0, 0, camera.focal_length]).to(dtype=dtype)

    return g_pred, repr_error


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    import numpy as np
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_plane_points(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        global_plane_position: Optional[torch.Tensor] = None
):
    fig, ax = plt.subplots(1, 1, figsize=(
        10, 10), subplot_kw={'projection': '3d'})

    T, _ = A.shape

    linspace = torch.linspace(0, 1, T)
    color_a = torch.tensor([1, 0, 0, 1], dtype=torch.float32).unsqueeze(
        0).expand(T, -1) * linspace.unsqueeze(-1).repeat(1, 4)
    color_b = torch.tensor([0, 1, 0, 1], dtype=torch.float32).unsqueeze(
        0).expand(T, -1) * linspace.unsqueeze(-1).repeat(1, 4)
    color_c = torch.tensor([0, 0, 1, 1], dtype=torch.float32).unsqueeze(
        0).expand(T, -1) * linspace.unsqueeze(-1).repeat(1, 4)
    color_a[:, 3] = 1
    color_b[:, 3] = 1
    color_c[:, 3] = 1

    if global_plane_position is not None:
        z = torch.zeros(T, 4, 1, dtype=torch.float32)
        z[:, 2:] = 1
        normal_end = torch.bmm(global_plane_position, z)[..., :3, 0]
        start = global_plane_position[:, :3, 3]

        ax.scatter(start[..., 0], start[..., 1], start[..., 2],
                   label='Plane center', c='black')
        ax.quiver(start[..., 0], start[..., 1], start[..., 2],
                  normal_end[..., 0], normal_end[..., 1], normal_end[..., 2],
                  length=0.1,
                  normalize=True,
                  color='black')

    ax.scatter(A[..., 0], A[..., 1], A[..., 2], label='A', c=color_a)
    ax.scatter(B[..., 0], B[..., 1], B[..., 2], label='B', c=color_b)
    ax.scatter(C[..., 0], C[..., 1], C[..., 2], label='C', c=color_c)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_box_aspect([1.0, 1.0, 1.0])
    set_axes_equal(ax)
    ax.view_init(elev=-45, azim=-90)
    return fig


def compute_plane_position(
    images: torch.Tensor,
    mask: torch.Tensor,
    depths: torch.Tensor,
    times: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    mask_properties: BasicMaskProperties,
    object_index: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    proto_plane_position_method: ProtoPlanePositionMethod = ProtoPlanePositionMethod.HOMOGRAPHY,
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
    new_homography: bool = True,
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

    object_index : int
        Object index of the current mask.

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
    missing_in_frame = mask_properties.missing_in_frame
    proto_position_error = torch.zeros_like(missing_in_frame)

    if callable(proto_plane_position_method):
        global_plane_position = proto_plane_position_method(
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
    else:
        proto_plane_position_method = parse_enum(
            ProtoPlanePositionMethod, proto_plane_position_method)
        got_positions = False
        while not got_positions:
            if proto_plane_position_method == ProtoPlanePositionMethod.CENTEROID:
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
                    plot_tracked_points=plot_tracked_points,
                    plot_tracked_points_path=plot_tracked_points_path,
                )
                got_positions = True
            elif proto_plane_position_method == ProtoPlanePositionMethod.HOMOGRAPHY:
                finder = SequenceHomographyFinder(
                    SequenceHomographyFinderConfig())
                if new_homography:
                    global_plane_position, missing_homographies = compute_proto_plane_position_homography_new(
                        images=images,
                        mask=mask,
                        depths=depths,
                        times=times,
                        camera=camera,
                        mask_properties=mask_properties,
                        object_index=object_index,
                        homography_finder=finder,
                        dtype=dtype,
                        device=device,
                    )
                else:
                    global_plane_position, missing_homographies = compute_proto_plane_position_homography(
                        images=images,
                        mask=mask,
                        depths=depths,
                        times=times,
                        camera=camera,
                        homography_finder=finder,
                        object_index=object_index,
                        mask_properties=mask_properties,
                        dtype=dtype,
                        device=device,
                        depth_ray_thickness=depth_ray_thickness,
                        min_depth_ray_thickness=min_depth_ray_thickness,
                        plot_tracked_points=plot_tracked_points,
                        plot_tracked_points_path=plot_tracked_points_path,
                    )
                proto_position_error |= missing_homographies
                got_positions = True

                # If the number of found homographies is less than 2, we use the centeroid method
                if (~(missing_in_frame | proto_position_error)).sum() < 2 and (~missing_in_frame).sum() > 2:
                    proto_plane_position_method = ProtoPlanePositionMethod.CENTEROID
                    got_positions = False
            else:
                raise NotImplementedError(
                    f"Proto plane position method {proto_plane_position_method} is not implemented.")

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
        smoothing_sigma=smoothing_sigma
    )


def refine_plane_position(
    global_plane_position: torch.Tensor,
    proto_position_error: torch.Tensor,
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
    only_correct_proto_error: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refines the plane position.

    Parameters
    ----------
    global_plane_position : torch.Tensor
        An existing global plane position to refine. Shape: (T, 4, 4)

    proto_position_error : torch.Tensor
        Timestamps in which the position may is incorrect. Shape: (T,)

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
    missing_in_frame = mask_properties.missing_in_frame

    plane_center = global_plane_position[:, :3, 3]
    quat = rotmat_to_unitquat(global_plane_position[:, :3, :3])

    if only_correct_proto_error:
        corrupted_frame = proto_position_error
    else:
        corrupted_frame = missing_in_frame | proto_position_error

    # If smoothing is enabled, apply Gaussian smoothing
    if translation_smoothing or orientation_smoothing or orientation_locking:
        camera_position = camera.get_global_position(t=times)
        if translation_smoothing:
            # Plane center is in global coordinates
            camera_rel_plane_center = global_to_local(
                camera_position, plane_center, v_include_time=True)[..., :3]

            smoothed_position = gaussian_smoothing(plane_center[~corrupted_frame].swapaxes(
                0, 1), kernel_size=smoothing_kernel_size, sigma=smoothing_sigma).swapaxes(0, 1)
            plane_center[~corrupted_frame] = smoothed_position
        if orientation_locking or orientation_smoothing:

            if orientation_locking:
                camera_global_rotation = rotmat_to_unitquat(
                    camera_position[:, :3, :3])
                local_rotation = quat_subtraction(
                    quat[~corrupted_frame], camera_global_rotation[~corrupted_frame])

                # Lock the orientation to the median orientation
                rotvec = unitquat_to_rotvec(local_rotation)
                median = torch.median(
                    rotvec, dim=0).values.unsqueeze(0)
                qmed = rotvec_to_unitquat(median).repeat(T, 1)
                # Go to global coordinates
                qmed = quat_composition(torch.stack(
                    [camera_global_rotation, qmed], dim=1))
                quat = qmed
            elif orientation_smoothing:
                # Convert the quaternion to rotation vector, apply smoothing and convert back to quaternion
                quat_before = quat.clone()
                to_smooth = quat[~corrupted_frame]

                # This code will not work for large rotations, as we dont have a wrap around.
                # Remove basis rotation
                avg = quat_average(to_smooth)
                tswa = quat_subtraction(
                    to_smooth, avg.unsqueeze(0).expand_as(to_smooth))
                # downscale rotation
                scale = 1 / 10
                tswa = quat_product_scalar(tswa, scale)

                rotvec = unitquat_to_rotvec(tswa)
                smoothvec = gaussian_smoothing(
                    rotvec.swapaxes(0, 1), kernel_size=smoothing_kernel_size, sigma=smoothing_sigma).swapaxes(0, 1)

                smquat = rotvec_to_unitquat(smoothvec)
                ups = quat_product_scalar(smquat, 1 / scale)
                ups_avg = quat_product(
                    avg.unsqueeze(0).expand_as(to_smooth), ups)

                quat[~corrupted_frame] = ups_avg

    # If a object is not present at a timestamp, we will need to interpolate the position and orientation
    if corrupted_frame.sum() > 0:
        plane_center[corrupted_frame], quat[corrupted_frame] = interpolate_plane_position(
            translation=plane_center, orientation=quat, times=times, missing_in_frame=corrupted_frame)

    global_plane_position[:, :3, 3] = plane_center
    global_plane_position[:, :3, :3] = unitquat_to_rotmat(quat)

    center_times = times

    if position_spline_fitting:
        # Approximate the position and orientation with a spline so its smooth
        from nag.model.timed_discrete_scene_node_3d import spline_approximation
        from nag.transforms.transforms_timed_3d import find_optimal_spline, hermite_catmull_rom_index, quat_hermite_catmull_rom_index
        K = position_spline_control_points if position_spline_control_points is not None else len(
            times) // 2

        new_plane_center, new_quat, _ctimes = spline_approximation(
            plane_center, quat, times, K)
        # Discard first and last point
        new_plane_center = new_plane_center[1:-1]
        new_quat = new_quat[1:-1]

        cp_times = torch.linspace(0, 1, K, dtype=dtype, device=device)

        interpolated_trans_fit = hermite_catmull_rom_index(
            new_plane_center, cp_times, times)
        interpolated_orient_fit = quat_hermite_catmull_rom_index(
            new_quat, cp_times, times)

        global_plane_position[:, :3, 3] = interpolated_trans_fit
        global_plane_position[:, :3, :3] = unitquat_to_rotmat(
            interpolated_orient_fit)

        plane_center = new_plane_center
        quat = new_quat
        center_times = cp_times

    return global_plane_position, plane_center, quat, center_times
