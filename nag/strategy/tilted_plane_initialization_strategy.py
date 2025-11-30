import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from matplotlib.axes import Axes
from tools.logger.logging import logger
from tools.transforms.min_max import MinMax
from tools.util.format import parse_type
from tools.util.torch import (flatten_batch_dims, tensorify,
                              unflatten_batch_dims)
from tools.util.typing import DEFAULT, VEC_TYPE
from tools.viz.matplotlib import saveable
from torch import Tensor

from nag.config.nag_config import NAGConfig
from nag.dataset.nag_dataset import NAGDataset
from nag.model.background_plane_scene_node_3d import compute_background_color
from nag.model.camera_scene_node_3d import plot_rays
from nag.model.discrete_plane_scene_node_3d import (DiscretePlaneSceneNode3D,
                                                    default_plane_scale_offset,
                                                    plane_coordinates_to_local)
from nag.model.learned_alpha_location_image_plane_scene_node_3d import \
    LearnedAlphaLocationImagePlaneSceneNode3D
from nag.model.learned_color_alpha_location_image_plane_scene_node_3d import \
    LearnedColorAlphaLocationImagePlaneSceneNode3D
from nag.model.learned_image_plane_scene_node_3d import \
    LearnedImagePlaneSceneNode3D
from nag.model.nag_model import NAGModel
from nag.model.timed_camera_scene_node_3d import (TimedCameraSceneNode3D,
                                                  local_to_image_coordinates)
from nag.model.timed_discrete_scene_node_3d import (global_to_local,
                                                    local_to_global)
from nag.model.view_dependent_spline_scene_node_3d import \
    ViewDependentSplineSceneNode3D
from nag.strategy.base_plane_initialization_strategy import BasicMaskProperties
from nag.strategy.in_image_plane_position_strategy import \
    InImagePlanePositionStrategy
from nag.strategy.plane_initialization_strategy import (
    PlaneInitializationStrategy, get_plane_config_properties)
from nag.strategy.plane_position_strategy import PlanePositionStrategy
from nag.transforms.transforms_timed_3d import linear_interpolate_vector


@torch.jit.script
def compute_mask_properties(
        mask: torch.Tensor,
        depth: torch.Tensor,
        times: torch.Tensor,
        relative_plane_margin: torch.Tensor) -> BasicMaskProperties:
    """Compute the basic mask properties.
    If mask is not present on all the frames, it will be interpolated.

    Parameters
    ----------
    mask : torch.Tensor
        The mask of the object. Shape: (T, 1, H, W)
    depth : torch.Tensor
        The depths of the object. Shape: (T, H, W)
    times : torch.Tensor
        The times of used frames for the object. Shape: (T,)
    relative_plane_margin : torch.Tensor
        Relative plane margin, as a percentage of the mask size.
        Use the enlarged mask to determine the plane size.

    Returns
    -------
    BasicMaskProperties
        Mask properties.
    """
    # Giving the masks determine the bounding box of the object
    # Get the bounding box of the object
    T = len(times)

    bl = torch.zeros((T, 2))
    # Fill With nan
    bl.fill_(float("nan"))

    tr = torch.zeros((T, 2))
    tr.fill_(float("nan"))
    # bl tr in (y, x) format

    average_dist_per_time = torch.zeros((T,))
    # Fill with nan
    average_dist_per_time.fill_(float("nan"))

    # Get the largest mask of the object, use this as init for the base image and alpha and for getting the offset for plane borders
    mask_sizes = mask.sum(dim=(-2, -1))

    missing_in_frame = mask_sizes.squeeze(1) == 0

    largest_mask_idx = torch.argmax(mask_sizes)

    # Assume rectangular mask, and a percentage of its side length.
    absolute_mask_margin = relative_plane_margin * \
        torch.sqrt(mask_sizes[largest_mask_idx].squeeze())

    mask_center_of_mass = torch.zeros((T, 2))

    for i in range(T):
        mask_i = mask[i, 0]
        # Only if any mask
        if torch.any(mask_i):
            bl[i] = torch.stack([torch.min(torch.nonzero(mask_i)[..., 0]), torch.min(
                torch.nonzero(mask_i)[..., 1])], dim=-1)
            tr[i] = torch.stack([torch.max(torch.nonzero(mask_i)[..., 0]), torch.max(
                torch.nonzero(mask_i)[..., 1])], dim=-1)
            average_dist_per_time[i] = torch.mean(depth[i, 0, mask_i])
            mask_center_of_mass[i] = torch.mean(
                torch.nonzero(mask_i).float(), dim=0)

    # Interpolate NaN values in a linear fashion, so a partial hidded object can enter the scene
    if torch.any(torch.isnan(bl)):
        nan_mask = torch.any(torch.isnan(bl), dim=-1)
        exist_t = times[~nan_mask]
        bl = linear_interpolate_vector(bl[~nan_mask].unsqueeze(
            0), exist_t, times, equidistant_times=False).squeeze(0)
        tr = linear_interpolate_vector(tr[~nan_mask].unsqueeze(
            0), exist_t, times, equidistant_times=False).squeeze(0)
        average_dist_per_time = linear_interpolate_vector(average_dist_per_time[~nan_mask].unsqueeze(
            0).unsqueeze(-1), exist_t, times, equidistant_times=False).squeeze(0).squeeze(-1)

    # Add to bl and tr the margin, which we determined from the largest mask
    bl_to_tr = (tr[largest_mask_idx] - bl[largest_mask_idx])
    tr_to_bl = (bl[largest_mask_idx] - tr[largest_mask_idx])

    # Norm
    bl_to_tr = bl_to_tr / torch.norm(bl_to_tr, dim=-1, keepdim=True)
    tr_to_bl = tr_to_bl / torch.norm(tr_to_bl, dim=-1, keepdim=True)
    # Add margin
    padded_bl = bl + tr_to_bl * absolute_mask_margin
    padded_tr = tr + bl_to_tr * absolute_mask_margin

    br = torch.stack([bl[:, 0], tr[:, 1]], dim=-1)
    tl = torch.stack([tr[:, 0], bl[:, 1]], dim=-1)
    # Stack the border points
    border_points = torch.stack(
        [bl, br, tr, tl], dim=-3)

    return BasicMaskProperties(
        bottom_left=bl,
        top_right=tr,
        padded_bottom_left=padded_bl,
        padded_top_right=padded_tr,
        mask_sizes=mask_sizes,
        average_dist_per_time=average_dist_per_time,
        largest_mask_idx=largest_mask_idx,
        mask_center_of_mass=mask_center_of_mass,
        missing_in_frame=missing_in_frame,
        border_points=border_points
    )


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


class TiltedPlaneInitializationStrategy(PlaneInitializationStrategy):
    """Simple plane initialization strategy:
    1. Planes have a rotation, depending on the depth mask.
    2. Planes have a fixed scale depending on the largest mask extent.
    3. If the mask is zero for some planes, position will be linearly interpolated until it enters the scene.
    """

    def __init__(self,
                 plane_position_strategy: Optional[PlanePositionStrategy] = None,
                 depth_ray_thickness: float = 0.05,
                 min_depth_ray_thickness: float = 10.,
                 translation_smoothing: bool = True,
                 orientation_smoothing: bool = True,
                 orientation_locking: bool = False,
                 position_spline_fitting: bool = False,
                 smoothing_kernel_size: int = 7,
                 smoothing_sigma: float = 5.0,
                 alpha_mask_resolution: Optional[VEC_TYPE] = None,
                 alpha_mask_smoothing: bool = True,
                 alpha_mask_smoothing_kernel_size: int = 5,
                 alpha_mask_smoothing_sigma: float = 1.0,
                 alpha_min_max_range: Tuple[float, float] = (0.1, 0.9),
                 temporal_alpha_consistency: bool = False,
                 temporal_alpha_consistency_fnc: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = None,
                 color_mask_resolution: Optional[VEC_TYPE] = None,
                 color_mask_smoothing: bool = True,
                 color_mask_smoothing_kernel_size: int = 5,
                 color_mask_smoothing_sigma: float = 1.0,
                 temporal_color_consistency: bool = False,
                 temporal_color_consistency_fnc: Optional[Callable[[
                     torch.Tensor], torch.Tensor]] = None,
                 correct_lens_distortion: bool = False,
                 plot_tracked_points: bool = False,
                 plot_tracked_points_path: str = None,
                 ):
        self.plane_position_strategy = parse_type(plane_position_strategy, PlanePositionStrategy,
                                                  variable_name="PlanePositionStrategy") if plane_position_strategy is not None else InImagePlanePositionStrategy
        self.depth_ray_thickness = depth_ray_thickness
        self.min_depth_ray_thickness = min_depth_ray_thickness
        self.translation_smoothing = translation_smoothing
        self.orientation_smoothing = orientation_smoothing
        self.orientation_locking = orientation_locking
        self.smoothing_kernel_size = smoothing_kernel_size
        self.smoothing_sigma = smoothing_sigma
        self.alpha_mask_resolution = tensorify(alpha_mask_resolution) if (
            alpha_mask_resolution is not None and alpha_mask_resolution != DEFAULT) else DEFAULT
        self.alpha_mask_smoothing = alpha_mask_smoothing
        self.alpha_mask_smoothing_kernel_size = alpha_mask_smoothing_kernel_size
        self.alpha_mask_smoothing_sigma = alpha_mask_smoothing_sigma
        self.temporal_alpha_consistency = temporal_alpha_consistency
        self.temporal_alpha_consistency_fnc = temporal_alpha_consistency_fnc
        self.alpha_min_max_range = alpha_min_max_range
        self.color_mask_resolution = tensorify(color_mask_resolution) if (
            color_mask_resolution is not None and color_mask_resolution != DEFAULT) else DEFAULT
        self.color_mask_smoothing = color_mask_smoothing
        self.color_mask_smoothing_kernel_size = color_mask_smoothing_kernel_size
        self.color_mask_smoothing_sigma = color_mask_smoothing_sigma
        self.temporal_color_consistency = temporal_color_consistency
        self.temporal_color_consistency_fnc = temporal_color_consistency_fnc
        self.position_spline_fitting = position_spline_fitting
        self.correct_lens_distortion = correct_lens_distortion
        self.plot_tracked_points = plot_tracked_points
        self.plot_tracked_points_path = plot_tracked_points_path

    def execute(self,
                object_index: int,
                mask_index: int,
                images: Tensor,
                masks: Tensor,
                depths: Tensor,
                times: Tensor,
                camera: TimedCameraSceneNode3D,
                nag_model: NAGModel,
                dataset: NAGDataset,
                config: NAGConfig,
                name: str = None,
                **kwargs) -> Dict[str, Any]:
        with torch.no_grad():
            mask = masks[:, mask_index].unsqueeze(1)
            dtype = config.dtype
            device = mask.device
            if not torch.any(mask):
                raise ValueError("Mask is empty.")

            T, C_img, H, W = images.shape
            _, _, MH, MW = mask.shape
            # Compute the basic mask properties
            mask_properties: BasicMaskProperties = compute_mask_properties(
                mask=mask, depth=depths, times=times, relative_plane_margin=tensorify(config.relative_plane_margin))

            plane_properties = get_plane_config_properties(
                object_index=object_index, times=times, config=config, name=name)

            missing_in_frame = mask_properties.missing_in_frame
            largest_mask_idx = mask_properties.largest_mask_idx

            path = None if self.plot_tracked_points_path is None else os.path.join(
                self.plot_tracked_points_path, f"object_{object_index}.png")
            position_strategy = self.plane_position_strategy(
                relative_plane_margin=config.relative_plane_margin,
                position_spline_fitting=self.position_spline_fitting,
                position_spline_control_points=config.object_rigid_control_points if config.object_rigid_control_points is not None else int(
                    round(T * config.object_rigid_control_points_ratio)),
                translation_smoothing=self.translation_smoothing,
                orientation_smoothing=self.orientation_smoothing,
                orientation_locking=self.orientation_locking,
                smoothing_kernel_size=self.smoothing_kernel_size,
                smoothing_sigma=self.smoothing_sigma,
                depth_ray_thickness=self.depth_ray_thickness,
                min_depth_ray_thickness=self.min_depth_ray_thickness,
                plot_tracked_points=self.plot_tracked_points,
                plot_tracked_points_path=path,
                box_available=kwargs.get("box", False),
            )

            global_plane_position, global_plane_center, global_plane_quat, interp_times, plane_scale = position_strategy(images=images,
                                                                                                                         mask=mask,
                                                                                                                         depths=depths,
                                                                                                                         times=times,
                                                                                                                         config=config,
                                                                                                                         camera=camera,
                                                                                                                         mask_properties=mask_properties,
                                                                                                                         plane_properties=plane_properties,
                                                                                                                         dtype=dtype,
                                                                                                                         device=device,
                                                                                                                         **kwargs
                                                                                                                         )
            plane_properties["times"] = interp_times

            if not global_plane_center.isfinite().all() or not global_plane_quat.isfinite().all():
                breakpoint()
                # Need to check if entered.
                raise ValueError("Plane center or orientation is not finite.")

            # Update the properties
            plane_properties["translation"] = global_plane_center
            plane_properties["orientation"] = global_plane_quat
            plane_properties["plane_scale"] = plane_scale
            plane_properties["flow_reference_time"] = times[largest_mask_idx]

            plane_type = parse_type(
                config.plane_type, LearnedImagePlaneSceneNode3D)
            alpha_masks = None
            if issubclass(plane_type, LearnedAlphaLocationImagePlaneSceneNode3D):
                alpha_mask_res = self.alpha_mask_resolution
                if alpha_mask_res == DEFAULT:
                    # Get the max amount of image data from the mask which we can project on the plane.
                    mask_size = (
                        mask_properties.top_right[largest_mask_idx] -
                        mask_properties.bottom_left[largest_mask_idx]
                    ).round().int()  # y, x
                    alpha_mask_res = (
                        mask_size * (1 + tensorify(config.relative_plane_margin))).round().int()
                    logger.info(
                        f"Alpha mask resolution was set to default, using: {' x '.join([str(x.item()) for x in alpha_mask_res])} (H x W) for {name} as this should covers the object in the native camera resolution.")

                alpha_img, alpha_masks = reproject_alpha(
                    masks=mask,
                    global_plane_positions=global_plane_position,
                    plane_scale=plane_scale,
                    resolution=alpha_mask_res,
                    times=times,
                    camera=camera,
                    smooth=self.alpha_mask_smoothing,
                    kernel_size=self.alpha_mask_smoothing_kernel_size,
                    sigma=self.alpha_mask_smoothing_sigma,
                    largest_mask_idx=largest_mask_idx,
                    temporal_consistency=self.temporal_alpha_consistency,
                    correct_lens_distortion=self.correct_lens_distortion,
                )
                # Scale the reprojected alpha to the range of 0.1 to 0.9
                mm = MinMax(
                    self.alpha_min_max_range[0], self.alpha_min_max_range[1], dim=(-2, -1))
                alpha_img = mm.fit_transform(alpha_img)
                plane_properties["initial_alpha"] = alpha_img
            else:
                mean_overlap = masks[largest_mask_idx, :, mask[largest_mask_idx, 0]].sum(
                    dim=0).float().mean()
                alpha = mean_overlap
                alpha = torch.clamp(
                    alpha, self.alpha_min_max_range[0], self.alpha_min_max_range[1])
                plane_properties["initial_alpha"] = alpha

            if issubclass(plane_type, LearnedColorAlphaLocationImagePlaneSceneNode3D):
                color_mask_resolution = self.color_mask_resolution
                if color_mask_resolution == DEFAULT:
                    # Get the max amount of image data from the mask which we can project on the plane.
                    mask_size = (
                        mask_properties.top_right[largest_mask_idx] -
                        mask_properties.bottom_left[largest_mask_idx]
                    )  # y, x
                    if config.use_dataset_color_reprojection:
                        # Calculate max resolution based on the dataset resolution
                        ratio = (torch.tensor(
                            dataset.image_shape[1:3]) / torch.tensor(dataset.initial_image_shape)).max()
                        mask_size = (mask_size * ratio)

                    color_mask_resolution = (
                        mask_size * (1. + tensorify(config.relative_plane_margin))).round().int()
                    logger.info(
                        f"Color mask resolution was set to default, using: {' x '.join([str(x.item()) for x in color_mask_resolution])} (H x W) for {name} as this should covers the object in the native camera resolution.")

                if config.use_dataset_color_reprojection:
                    rgb_img = reproject_color_new(
                        images=images,
                        masks=mask,
                        projected_masks=alpha_masks,
                        global_plane_positions=global_plane_position,
                        plane_scale=plane_scale,
                        resolution=color_mask_resolution,
                        times=times,
                        camera=camera,
                        smooth=self.color_mask_smoothing,
                        kernel_size=self.color_mask_smoothing_kernel_size,
                        sigma=self.color_mask_smoothing_sigma,
                        largest_mask_idx=largest_mask_idx,
                        temporal_consistency=self.temporal_color_consistency,
                        temporal_consistency_fnc=self.temporal_color_consistency_fnc,
                        correct_lens_distortion=self.correct_lens_distortion,
                        use_dataset=True,
                        dataset=dataset
                    )
                else:
                    rgb_img = reproject_color(
                        images=images,
                        masks=mask,
                        projected_masks=alpha_masks,
                        global_plane_positions=global_plane_position,
                        plane_scale=plane_scale,
                        resolution=color_mask_resolution,
                        times=times,
                        camera=camera,
                        smooth=self.color_mask_smoothing,
                        kernel_size=self.color_mask_smoothing_kernel_size,
                        sigma=self.color_mask_smoothing_sigma,
                        largest_mask_idx=largest_mask_idx,
                        temporal_consistency=self.temporal_color_consistency,
                        temporal_consistency_fnc=self.temporal_color_consistency_fnc,
                        correct_lens_distortion=self.correct_lens_distortion,
                    )

                rgb_img = rgb_img.float()
                plane_properties["initial_rgb"] = rgb_img
            else:
                # This is deprecated
                mask_colors = images[largest_mask_idx,
                                     :, mask[largest_mask_idx, 0]]
                mean_color = mask_colors.mean(dim=-1)
                color_variance = mask_colors.var(dim=-1)
                color_variance[color_variance == 0.] = 1.
                if config.has_background_plane:
                    background_color = compute_background_color(images)
                    alpha = self.alpha_min_max_range[1]
                    delta_color = (
                        mean_color - ((1 - alpha) * background_color)) / alpha
                    init_rgb = delta_color
                else:
                    init_rgb = mean_color
                plane_properties["initial_rgb"] = init_rgb
                plane_properties["initial_rgb_variance"] = color_variance

            if issubclass(plane_type, ViewDependentSplineSceneNode3D):
                from nag.model.discrete_plane_scene_node_3d import \
                    compute_incline_angle
                from nag.model.nag_model import get_object_intersection_points

                # Compute min max view angle range
                # Use camera to sample rays for a timestamps
                X, Y = 100, 100
                uv = torch.stack(torch.meshgrid(torch.linspace(0, camera._image_resolution[1], X), torch.linspace(
                    0, camera._image_resolution[0], Y), indexing="xy"), dim=-1)
                global_ray_origin, global_ray_direction = camera.get_global_rays(
                    uv.reshape(X * Y, 2), t=times)  # Shape (B, T, 3) and (B, T, 3
                intersection_points, is_inside, _, _ = get_object_intersection_points(
                    global_plane_position[None, ...],
                    plane_scale[None, ...],
                    torch.tensor([[-0.5, -0.5]], dtype=global_plane_position.dtype,
                                 device=global_plane_position.device),
                    global_ray_origin, global_ray_direction)  # Shape (N, B, T, 3), (N, B, T)
                global_target = global_ray_origin + global_ray_direction

                index_tensor = torch.arange(
                    X * Y, device=global_plane_position.device).unsqueeze(-1).unsqueeze(-1).repeat(1, T, 1)
                t_index = torch.arange(T, device=global_plane_position.device).unsqueeze(
                    0).unsqueeze(-1).repeat(X * Y, 1, 1)
                index_tensor = torch.cat([index_tensor, t_index], dim=-1)

                # Convert to local coordinates
                local_origin = global_to_local(
                    global_plane_position, global_ray_origin, v_include_time=True)[..., :3]
                local_target = global_to_local(
                    global_plane_position, global_target, v_include_time=True)[..., :3]
                local_direction = local_target - local_origin

                # Get hits
                hit_origin = local_origin[is_inside[0]]  # Shape (H, 3)
                hit_target = local_target[is_inside[0]]  # Shape (H, 3)
                hit_index = index_tensor[is_inside[0]]  # Shape (H, 3)

                hit_direction = hit_target - hit_origin
                hit_direction = hit_direction / \
                    torch.norm(hit_direction, dim=-1, keepdim=True)

                # Compute the incline angles
                all_incline_angles = compute_incline_angle(local_direction, torch.tensor(
                    [0., 0., 1.], dtype=hit_direction.dtype, device=hit_direction.device))
                incline_angles = all_incline_angles[is_inside[0]]

                assert torch.allclose(incline_angles[:, 2], torch.zeros_like(
                    incline_angles[:, 2]), atol=1e-5), "Incline angle is not correct. z should be 0."

                # Check if any angle is anti-parallel
                antiparallel = torch.isclose(incline_angles[:, :2], torch.tensor(
                    torch.pi, dtype=incline_angles.dtype, device=incline_angles.device), atol=1e-6).any(dim=-1)
                parallel = torch.isclose(incline_angles[:, :2], torch.tensor(
                    0., dtype=incline_angles.dtype, device=incline_angles.device), atol=1e-6).all(dim=-1)

                min_angle = torch.amin(
                    incline_angles[~antiparallel, :2], dim=0)
                max_angle = torch.amax(
                    incline_angles[~antiparallel, :2], dim=0)

                delta_angle = max_angle - min_angle
                # Add some margin
                margin = 0.3
                min_angle = torch.clamp(
                    min_angle * (1 + margin), -torch.pi, torch.pi)
                max_angle = torch.clamp(
                    max_angle * (1 + margin), -torch.pi, torch.pi)

                plane_properties["view_dependent_data_range"] = torch.stack(
                    [min_angle, max_angle], dim=0)

                num_control_points = int(
                    T * config.plane_view_dependent_control_point_ratio)
                plane_properties["num_view_dependent_control_points"] = num_control_points

            plane_properties.pop("position_init_strategy", None)
            return plane_properties


def reproject_alpha(
        masks: torch.Tensor,
        global_plane_positions: torch.Tensor,
        plane_scale: torch.Tensor,
        resolution: torch.Tensor,
        times: torch.Tensor,
        camera: TimedCameraSceneNode3D,
        largest_mask_idx: int,
        smooth: bool = True,
        kernel_size: int = 5,
        sigma: float = 1.0,
        temporal_consistency: bool = False,
        temporal_consistency_fnc: Optional[Callable[[
            torch.Tensor], torch.Tensor]] = None,
        correct_lens_distortion: bool = True,

) -> torch.Tensor:
    """Reproject the alpha values of the mask to the plane.

    Parameters
    ----------
    masks : torch.Tensor
        The masks of the object at all timestamps. Shape: (T, 1, H, W)
    global_plane_position : torch.Tensor
        The global plane position. Shape: (T, 4, 4)
    plane_scale : torch.Tensor
        The plane scale. Shape: (2, ) (x, y)
    resolution : torch.Tensor
        The resolution which the alpha should be reprojected to. Shape: (2, ) (RH, RW)
    times : torch.Tensor
        The timestamps of the masks. Shape: (T, )
    camera : TimedCameraSceneNode3D
        The camera for the projection.
    largest_mask_idx : int
        The index of the largest mask.
        If not temporal consistency is used, this is the only mask that will be reprojected.
    smooth : bool, optional
        Whether to smooth the alpha values, by default True
        Smoothes them with a gaussian kernel.
    kernel_size : int, optional
        The kernel size for the smoothing, by default 5
    sigma : float, optional
        The sigma for the smoothing, by default 1.0
    temporal_consistency : bool, optional
        Whether to use temporal consistency, by default False
        If true, all masks will be reprojected and the median per pixel will be taken.
        If false, only the largest mask will be reprojected.
    temporal_consistency_fnc : Optional[Callable[[torch.Tensor], torch.Tensor]], optional
        The function to use for temporal consistency, by default None
        If None, the 95% quantile will be used.
    correct_lens_distortion : bool, optional
        Whether to correct for lens distortion on reprojection, by default True

    Returns
    -------
    torch.Tensor
        The projected alpha values from mask space to plane space. Shape: (RH, RW)

    """
    if temporal_consistency and temporal_consistency_fnc is None:
        def temporal_consistency_fnc(
            x): return torch.quantile(x.float(), 0.95, dim=0)
    T, C, H, W = masks.shape
    resolution = tuple(tensorify(resolution).round().int().tolist())
    out_alpha = torch.zeros(
        (T if temporal_consistency else 1, 1, *resolution))
    ts = len(times) if temporal_consistency else 1

    t_use = times
    masks_use = masks
    global_plane_positions_use = global_plane_positions
    intrinsics = camera.get_intrinsics(t=times)

    dtype = torch.float32

    # if not temporal_consistency:
    #     t_use = times[largest_mask_idx].unsqueeze(0)
    #     masks_use = masks[largest_mask_idx].unsqueeze(0)
    #     global_plane_positions_use = global_plane_positions[largest_mask_idx].unsqueeze(0)
    #     intrinsics = intrinsics[largest_mask_idx].unsqueeze(0)

    x = torch.linspace(0, 1, resolution[1], dtype=dtype)
    y = torch.linspace(0, 1, resolution[0], dtype=dtype)
    grid = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)

    grid, gs = flatten_batch_dims(grid, -2)
    local_coords = plane_coordinates_to_local(
        grid, plane_scale, plane_offset=default_plane_scale_offset(dtype))
    global_coords = local_to_global(
        global_plane_positions_use, local_coords)
    in_camera_coords = camera.global_to_local(
        global_coords, t=t_use, v_include_time=True)
    in_image_coords = local_to_image_coordinates(in_camera_coords.permute(1, 0, 2)[..., :3],
                                                 intrinsics,
                                                 camera.focal_length,
                                                 camera._lens_distortion if correct_lens_distortion else None,
                                                 v_includes_time=True)

    rel_coords = in_image_coords / camera._image_resolution.flip(-1)
    grid_sample_grid = unflatten_batch_dims((rel_coords.permute(
        1, 0, 2) - 0.5) * 2, gs).permute(2, 0, 1, 3)  # Permute time to first dim

    out_alpha = torch.nn.functional.grid_sample(masks_use.float(
    ), grid_sample_grid, mode='bilinear', padding_mode='border', align_corners=False)

    if temporal_consistency:
        consistent_alpha = temporal_consistency_fnc(out_alpha)
    else:
        consistent_alpha = out_alpha[largest_mask_idx].squeeze(0)

    if not consistent_alpha.isfinite().all() or not out_alpha.isfinite().all():
        breakpoint()
        raise ValueError("Alpha proj. produced NaN values.")

    if not smooth:
        return consistent_alpha, out_alpha
    sma = gaussian_smoothing_2d(
        consistent_alpha, kernel_size=kernel_size, sigma=sigma)
    return sma, out_alpha


def reproject_color(
        images: torch.Tensor,
        masks: torch.Tensor,
        projected_masks: torch.Tensor,
        global_plane_positions: torch.Tensor,
        plane_scale: torch.Tensor,
        resolution: torch.Tensor,
        times: torch.Tensor,
        camera: TimedCameraSceneNode3D,
        largest_mask_idx: int,
        smooth: bool = True,
        kernel_size: int = 5,
        sigma: float = 1.0,
        fill_masked_with_closests_frame: bool = True,
        smooth_projected_masks: bool = False,
        smooth_projected_masks_kernel_size: int = 15,
        smooth_projected_masks_sigma: float = 1.0,
        smooth_projected_masks_threshold: float = 0.9,
        temporal_consistency: bool = False,
        temporal_consistency_fnc: Optional[Callable[[
            torch.Tensor], torch.Tensor]] = None,
        correct_lens_distortion: bool = True,
) -> torch.Tensor:
    """Reproject the color values of the mask to the plane.

    Parameters
    ----------
    images : torch.Tensor
        The images of the object at all timestamps. Shape: (T, C, H, W)
    masks : torch.Tensor
        The masks of the object at all timestamps. Shape: (T, 1, H, W)
    projected_masks : torch.Tensor
        The projected masks to the objects plane. Shape: (T, 1, RH2, RW2)
        If RH2, RW2 is different from RH, RW this mask will be resized to RH, RW
    global_plane_position : torch.Tensor
        The global plane position. Shape: (T, 4, 4)
    plane_scale : torch.Tensor
        The plane scale. Shape: (2, ) (x, y)
    resolution : torch.Tensor
        The resolution which the color should be reprojected to. Shape: (2, ) (RH, RW)
    times : torch.Tensor
        The timestamps of the masks. Shape: (T, )
    camera : TimedCameraSceneNode3D
        The camera for the projection.
    largest_mask_idx : int
        The index of the largest mask.
        If not temporal consistency is used, this is the only mask that will be reprojected.
    smooth : bool, optional
        Whether to smooth the alpha values, by default True
        Smoothes them with a gaussian kernel.
    kernel_size : int, optional
        The kernel size for the smoothing, by default 5
    sigma : float, optional
        The sigma for the smoothing, by default 1.0
    temporal_consistency : bool, optional
        Whether to use temporal consistency, by default False
        If true, all masks will be reprojected and the median per pixel will be taken.
        If false, only the largest mask will be reprojected.
    temporal_consistency_fnc : Optional[Callable[[torch.Tensor], torch.Tensor]], optional
        The function to apply to the reprojected color values for temporal consistency, by default
        lambda x: torch.mean(x.float(), dim=0).round().to(torch.uint8)
        Will take the mean of the color values and round them.
    correct_lens_distortion : bool, optional
        Whether to correct for lens distortion on reprojection, by default True

    dataset : Optional[Any], optional
        The dataset in case images should be loaded in a higher resolution, by default None

    Returns
    -------
    torch.Tensor
        The reprojected color values from image . Shape: (3, RH, RW)

    """
    from tools.util.torch import batched_exec
    if temporal_consistency and temporal_consistency_fnc is None:
        def temporal_consistency_fnc(x): return torch.nanmean(x, dim=0)
    T, _, H, W = masks.shape
    resolution = tuple(tensorify(resolution).round().int().tolist())

    out_color = torch.zeros(
        (T if temporal_consistency else 1, 3, *resolution), dtype=torch.uint8)

    dtype = torch.float32

    t_use = times
    image_use = images
    global_plane_positions_use = global_plane_positions
    intrinsics = camera.get_intrinsics(t=times)

    # if not temporal_consistency:
    #     t_use = times[largest_mask_idx].unsqueeze(0)
    #     image_use = images[largest_mask_idx].unsqueeze(0)
    #     global_plane_positions_use = global_plane_positions[largest_mask_idx].unsqueeze(0)
    #     intrinsics = intrinsics[largest_mask_idx].unsqueeze(0)

    x = torch.linspace(0, 1, resolution[1], dtype=dtype)
    y = torch.linspace(0, 1, resolution[0], dtype=dtype)
    grid = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)

    if projected_masks.shape[-2:] != resolution:
        grid_sample_grid = (grid.unsqueeze(0) * 2 - 1).expand(T, -1, -1, -1)

        def _interpolate(m, gsg):
            return torch.nn.functional.grid_sample(m.float(), gsg, mode='bilinear', padding_mode='border', align_corners=True).round().bool()
        projected_masks = batched_exec(
            projected_masks, grid_sample_grid, func=_interpolate, batch_size=20)

    grid, gs = flatten_batch_dims(grid, -2)
    local_coords = plane_coordinates_to_local(
        grid, plane_scale, plane_offset=default_plane_scale_offset(dtype))
    global_coords = local_to_global(
        global_plane_positions_use, local_coords)
    in_camera_coords = camera.global_to_local(
        global_coords, t=t_use, v_include_time=True)
    in_image_coords = local_to_image_coordinates(in_camera_coords.permute(1, 0, 2)[..., :3],
                                                 intrinsics,
                                                 camera.focal_length,
                                                 camera._lens_distortion if correct_lens_distortion else None,
                                                 v_includes_time=True)

    rel_coords = in_image_coords / camera._image_resolution.flip(-1)
    grid_sample_grid = unflatten_batch_dims((rel_coords.permute(
        1, 0, 2) - 0.5) * 2, gs).permute(2, 0, 1, 3)  # Permute time to first dim

    out_color = torch.nn.functional.grid_sample(image_use.float(
    ), grid_sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
    # out_color = color.squeeze(0)

    # Replace outcolor with nans, if projected mask is smaller .5
    if projected_masks is not None:
        pm = projected_masks
        if smooth_projected_masks:
            pm = gaussian_smoothing_2d(projected_masks.float(
            ), kernel_size=smooth_projected_masks_kernel_size, sigma=smooth_projected_masks_sigma)
            pm = pm >= smooth_projected_masks_threshold

        proj_ext = pm.expand(
            -1, 3, -1, -1) if pm.shape[0] == T else pm.expand(T, 3, -1, -1)
        out_color[~proj_ext.bool()] = torch.nan

    # Replace outcolor with nans, if the projected pixel is outside the image
    oob = torch.any(grid_sample_grid < -1, dim=-1) | torch.any(
        grid_sample_grid > 1, dim=-1)

    out_color[oob.unsqueeze(1).expand(-1, 3, -1, -1)] = torch.nan

    if temporal_consistency:
        consistent_color = temporal_consistency_fnc(out_color)
    else:
        consistent_color = out_color[largest_mask_idx]
        nan_pixels = torch.isnan(consistent_color).any(dim=0)
        if nan_pixels.any() and fill_masked_with_closests_frame:
            # Filling empty / masked out pixels with the closests frame which is not masked out.
            def first_finite(x, axis=0):
                nonz = torch.isfinite(x)
                _any = nonz.any(axis)
                return (_any & nonz).max(axis).indices, _any

            not_largest = torch.ones(T, dtype=torch.bool)
            not_largest[largest_mask_idx] = False
            indices = torch.argwhere(not_largest).squeeze(1)
            if len(indices) > 0:
                dist = torch.abs(indices - largest_mask_idx)
                order = torch.argsort(dist)
                closests = indices[order]
                closests_color = out_color[closests][:, :, nan_pixels]
                idx, is_filled = first_finite(closests_color[:, 0], 0)
                nan_pixel_idx = torch.argwhere(nan_pixels).squeeze()
                nan_pixel_idx_f = nan_pixel_idx[is_filled]
                lookup = torch.gather(indices, 0, order)
                consistent_color[:, nan_pixel_idx_f[:, 0], nan_pixel_idx_f[:, 1]] = out_color[lookup[idx[is_filled]],
                                                                                              :, nan_pixel_idx_f[:, 0], nan_pixel_idx_f[:, 1]].permute(1, 0)

            # Debug plotting for the first frame introducing color to the region.
            # fix_indices = torch.full(consistent_color[0].shape, torch.nan)
            # fix_indices[nan_pixels] = -1

            #
            # fix_indices[nan_pixel_idx_f[:, 0], nan_pixel_idx_f[:, 1]] = lookup[idx[is_filled]].float()
            # plot_as_image(fix_indices, title="Fix Indices", cmap="viridis", colorbar=True)

    if torch.any(torch.isnan(consistent_color)):
        # If we still have nans = a point was not visible in any of the images we set it to the mean color
        selected_mask = masks.bool()[largest_mask_idx, 0]
        if selected_mask.shape[-2:] != images.shape[-2:]:
            selected_mask = torch.nn.functional.interpolate(selected_mask.float().unsqueeze(
                0).unsqueeze(0), size=images.shape[-2:], mode="bilinear").bool().squeeze(0).squeeze(0)
        mean_color = images[largest_mask_idx, :, selected_mask].permute(
            1, 0).mean(dim=0)
        consistent_color[:, torch.isnan(consistent_color).any(
            dim=0)] = mean_color.unsqueeze(-1)

    if not smooth:
        return consistent_color.float()
    sma = gaussian_smoothing_2d(
        consistent_color, kernel_size=kernel_size, sigma=sigma)
    return sma


def reproject_color_new(
    images: torch.Tensor,
    masks: torch.Tensor,
    projected_masks: torch.Tensor,
    global_plane_positions: torch.Tensor,
    plane_scale: torch.Tensor,
    resolution: torch.Tensor,
    times: torch.Tensor,
    camera: TimedCameraSceneNode3D,
    largest_mask_idx: int,
    smooth: bool = True,
    kernel_size: int = 5,
    sigma: float = 1.0,
    fill_masked_with_closests_frame: bool = True,
    smooth_projected_masks: bool = False,
    smooth_projected_masks_kernel_size: int = 15,
    smooth_projected_masks_sigma: float = 1.0,
    smooth_projected_masks_threshold: float = 0.9,
    temporal_consistency: bool = False,
    temporal_consistency_fnc: Optional[Callable[[
        torch.Tensor], torch.Tensor]] = None,
    correct_lens_distortion: bool = True,
    dataset: Optional[Any] = None,
    use_dataset: bool = False,
    align_corners: bool = True,
) -> torch.Tensor:
    """Reproject the color values of the mask to the plane.

    Parameters
    ----------
    images : torch.Tensor
        The images of the object at all timestamps. Shape: (T, C, H, W)
    masks : torch.Tensor
        The masks of the object at all timestamps. Shape: (T, 1, H, W)
    projected_masks : torch.Tensor
        The projected masks to the objects plane. Shape: (T, 1, RH2, RW2)
        If RH2, RW2 is different from RH, RW this mask will be resized to RH, RW
    global_plane_position : torch.Tensor
        The global plane position. Shape: (T, 4, 4)
    plane_scale : torch.Tensor
        The plane scale. Shape: (2, ) (x, y)
    resolution : torch.Tensor
        The resolution which the color should be reprojected to. Shape: (2, ) (RH, RW)
    times : torch.Tensor
        The timestamps of the masks. Shape: (T, )
    camera : TimedCameraSceneNode3D
        The camera for the projection.
    largest_mask_idx : int
        The index of the largest mask.
        If not temporal consistency is used, this is the only mask that will be reprojected.
    smooth : bool, optional
        Whether to smooth the alpha values, by default True
        Smoothes them with a gaussian kernel.
    kernel_size : int, optional
        The kernel size for the smoothing, by default 5
    sigma : float, optional
        The sigma for the smoothing, by default 1.0
    temporal_consistency : bool, optional
        Whether to use temporal consistency, by default False
        If true, all masks will be reprojected and the median per pixel will be taken.
        If false, only the largest mask will be reprojected.
    temporal_consistency_fnc : Optional[Callable[[torch.Tensor], torch.Tensor]], optional
        The function to apply to the reprojected color values for temporal consistency, by default
        lambda x: torch.mean(x.float(), dim=0).round().to(torch.uint8)
        Will take the mean of the color values and round them.
    correct_lens_distortion : bool, optional
        Whether to correct for lens distortion on reprojection, by default True

    dataset : Optional[Any], optional
        The dataset in case images should be loaded in a higher resolution, by default None

    Returns
    -------
    torch.Tensor
        The reprojected color values from image . Shape: (3, RH, RW)

    """
    from tools.util.torch import batched_exec

    from nag.dataset.nag_dataset import NAGDataset
    dataset: Optional[NAGDataset]
    if temporal_consistency and temporal_consistency_fnc is None:
        def temporal_consistency_fnc(x): return torch.nanmean(x, dim=0)
    T, _, H, W = masks.shape
    resolution = tuple(tensorify(resolution).round().int().tolist())

    out_color = torch.zeros(
        (T if temporal_consistency else 1, 3, *resolution), dtype=torch.uint8)

    dtype = torch.float32

    t_use = times
    image_use = images
    global_plane_positions_use = global_plane_positions
    intrinsics = camera.get_intrinsics(t=times)

    # if not temporal_consistency:
    #     t_use = times[largest_mask_idx].unsqueeze(0)
    #     image_use = images[largest_mask_idx].unsqueeze(0)
    #     global_plane_positions_use = global_plane_positions[largest_mask_idx].unsqueeze(0)
    #     intrinsics = intrinsics[largest_mask_idx].unsqueeze(0)

    if align_corners:
        x = torch.linspace(0, 1, resolution[1], dtype=dtype)
        y = torch.linspace(0, 1, resolution[0], dtype=dtype)
    else:
        x = torch.arange(
            0, 1, 1 / resolution[1], dtype=dtype) + 0.5 / resolution[1]
        y = torch.arange(
            0, 1, 1 / resolution[1], dtype=dtype) + 0.5 / resolution[0]
    grid = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)

    if projected_masks.shape[-2:] != resolution:
        grid_sample_grid = (grid.unsqueeze(0) * 2 - 1).expand(T, -1, -1, -1)

        def _interpolate(m, gsg):
            return torch.nn.functional.grid_sample(m.float(), gsg, mode='bilinear', padding_mode='border', align_corners=align_corners).round().bool()
        projected_masks = batched_exec(
            projected_masks, grid_sample_grid, func=_interpolate, batch_size=20).bool()

    # Replace outcolor with nans, if projected mask is smaller .5

    pm = projected_masks
    if smooth_projected_masks:
        pm = gaussian_smoothing_2d(projected_masks.float(
        ), kernel_size=smooth_projected_masks_kernel_size, sigma=smooth_projected_masks_sigma)
        pm = pm >= smooth_projected_masks_threshold

    projected_valid_mask = pm.expand(-1, -1, -1, -
                                     1) if pm.shape[0] == T else pm.expand(T, 1, -1, -1)

    # Getting the image sampling points
    grid, gs = flatten_batch_dims(grid, -2)
    local_coords = plane_coordinates_to_local(
        grid, plane_scale, plane_offset=default_plane_scale_offset(dtype))
    global_coords = local_to_global(global_plane_positions_use,
                                    local_coords)
    in_camera_coords = camera.global_to_local(
        global_coords, t=t_use, v_include_time=True)
    in_image_coords = local_to_image_coordinates(in_camera_coords.permute(1, 0, 2)[..., :3],
                                                 intrinsics,
                                                 camera.focal_length,
                                                 camera._lens_distortion if correct_lens_distortion else None,
                                                 v_includes_time=True)

    rel_coords = in_image_coords / camera._image_resolution.flip(-1)
    grid_sample_grid = unflatten_batch_dims((rel_coords.permute(
        1, 0, 2) - 0.5) * 2, gs).permute(2, 0, 1, 3)
    # Compute those pixels which are outside of the image domain for each timestamp
    eps = 1e-6
    oob = (torch.any((grid_sample_grid + eps) < -1, dim=-
                     1) | torch.any((grid_sample_grid - eps) > 1, dim=-1))
    projected_valid_mask = projected_valid_mask.bool() & ~oob.unsqueeze(1)

    # Determine the image indices which need to be gridsampled
    needed_images_indices = None
    missing_pixel_indices_filled = None
    missing_pixel_frame_indices = None
    if temporal_consistency:
        # All
        needed_images_indices = torch.arange(
            T, dtype=torch.long, device=times.device)
    else:
        consistent_mask = projected_valid_mask[largest_mask_idx]
        needed_images_indices = torch.tensor(
            [largest_mask_idx], dtype=torch.long, device=times.device)
        missing_pixels = (~consistent_mask).any(dim=0)
        if not consistent_mask.all() and fill_masked_with_closests_frame:
            def first_true(x, axis=0):
                nonz = x
                _any = nonz.any(axis)
                return (_any & nonz).max(axis).indices, _any

            not_largest = torch.ones(T, dtype=torch.bool)
            not_largest[largest_mask_idx] = False
            indices = torch.argwhere(not_largest).squeeze(1)
            if len(indices) > 0:
                dist = torch.abs(indices - largest_mask_idx)
                order = torch.argsort(dist)
                closests = indices[order]
                closests_color = projected_valid_mask[closests][:,
                                                                :, missing_pixels]
                idx, is_filled = first_true(closests_color[:, 0], 0)
                missing_pixel_indices = torch.argwhere(
                    missing_pixels).squeeze()
                missing_pixel_indices_filled = missing_pixel_indices[is_filled]
                lookup = torch.gather(indices, 0, order)
                missing_pixel_frame_indices = lookup[idx[is_filled]]
                needed_images_indices = torch.cat(
                    [needed_images_indices, missing_pixel_frame_indices.unique()], dim=0)

    if use_dataset:
        def _interpolate(indices, gsg):
            img = dataset.load_image(
                indices, init_size=False, native_size=True)
            return torch.nn.functional.grid_sample(img.float(), gsg, mode='bilinear', padding_mode='border', align_corners=align_corners)
        out_color = batched_exec(
            needed_images_indices, grid_sample_grid[needed_images_indices], func=_interpolate, batch_size=5)

    else:
        def _interpolate(images, gsg):
            return torch.nn.functional.grid_sample(images.float(), gsg, mode='bilinear', padding_mode='border', align_corners=align_corners)
        out_color = batched_exec(image_use[needed_images_indices].float(
        ), grid_sample_grid[needed_images_indices], func=_interpolate, batch_size=5)

    # Order in out color is based on the needed images indices
    out_color_largest_mask_idx = (
        needed_images_indices == largest_mask_idx).argwhere().squeeze(-1)
    out_color_missing_pixel_frame_indices = None
    if missing_pixel_indices_filled is not None:
        out_color_missing_pixel_frame_indices = torch.zeros_like(
            missing_pixel_frame_indices)
        for i, idx in enumerate(needed_images_indices):
            out_color_missing_pixel_frame_indices[missing_pixel_frame_indices == idx] = i

    # Replace outcolor with nans for invalid projected pixels (occlusions)
    out_color[~projected_valid_mask[needed_images_indices].bool(
    ).expand(-1, 3, -1, -1)] = torch.nan

    if temporal_consistency:
        consistent_color = temporal_consistency_fnc(out_color)
    else:
        consistent_color = out_color[out_color_largest_mask_idx].squeeze(0)
        if (missing_pixel_indices_filled is not None and len(missing_pixel_indices_filled) > 0) and fill_masked_with_closests_frame:
            # Filling empty / masked out pixels with the closests frame which is not masked out.
            consistent_color[:, missing_pixel_indices_filled[:, 0], missing_pixel_indices_filled[:, 1]] = out_color[out_color_missing_pixel_frame_indices,
                                                                                                                    :, missing_pixel_indices_filled[:, 0], missing_pixel_indices_filled[:, 1]].permute(1, 0)

    if torch.any(torch.isnan(consistent_color)):
        # If we still have nans = a point was not visible in any of the images we set it to the mean color
        selected_mask = masks.bool()[largest_mask_idx, 0]
        if selected_mask.shape[-2:] != images.shape[-2:]:
            selected_mask = torch.nn.functional.interpolate(selected_mask.float().unsqueeze(
                0).unsqueeze(0), size=images.shape[-2:], mode="bilinear").bool().squeeze(0).squeeze(0)
        mean_color = images[largest_mask_idx, :, selected_mask].permute(
            1, 0).mean(dim=0)
        consistent_color[:, torch.isnan(consistent_color).any(
            dim=0)] = mean_color.unsqueeze(-1)

    if not smooth:
        return consistent_color.float()
    sma = gaussian_smoothing_2d(
        consistent_color, kernel_size=kernel_size, sigma=sigma)
    return sma


def plot_together(
        xyz,
        camera,
):
    fig = plot_coords(xyz.reshape(200, 200, 1, 4)[
                      ::20, :: 20, 0, :3], label="Plane Global")
    camera.plot_object(ax=fig.axes[0])
    return fig


@saveable()
def plot_local_coords(
        xyz: torch.Tensor,
        global_position: torch.Tensor,
        **kwargs
) -> None:
    global_coords = local_to_global(global_position, xyz)
    return plot_coords(global_coords, **kwargs)


@saveable()
def plot_coords(
        xyz: torch.Tensor,
        label: Optional[str] = None,
        ax: Optional[Axes] = None,
        legend: bool = True,
) -> None:
    from tools.viz.matplotlib import get_mpl_figure, saveable
    xyz = flatten_batch_dims(xyz, -2)[0]
    if ax is None:
        fig, ax = get_mpl_figure(subplot_kw={'projection': '3d'})
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        fig = ax.get_figure()
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=label)
    if legend:
        ax.legend()
    return fig


def reproject_interpolate(
        values: torch.Tensor,
        plane_coords: torch.Tensor,
        resolution: torch.Tensor,
) -> torch.Tensor:
    ngy, ngx = resolution  # Resolution is in yx format
    x = torch.arange(0, 1 - 1 / ngx, 1 / ngx,
                     dtype=plane_coords.dtype, device=plane_coords.device)
    y = torch.arange(0, 1 - 1 / ngy, 1 / ngy,
                     dtype=plane_coords.dtype, device=plane_coords.device)
    grid = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)

    from scipy.interpolate import LinearNDInterpolator

    # Interpolate the alpha values
    val, sh = flatten_batch_dims(values, -1)
    val = val.unsqueeze(-1)
    interpolator = LinearNDInterpolator(plane_coords, val, fill_value=0.)
    fgrid, shape = flatten_batch_dims(grid, -2)
    alpha = torch.tensor(interpolator(
        fgrid), dtype=plane_coords.dtype, device=plane_coords.device)
    ralpha = unflatten_batch_dims(alpha[..., 0], shape)
    return ralpha


def reproject_poly(
        mask: torch.Tensor,
        coords: torch.Tensor,
        resolution: torch.Tensor,
) -> torch.Tensor:
    from nag.utils.contour_node import ContourNode
    ry, rx = resolution
    nodes = ContourNode.from_mask(mask.bool().numpy())
    select_coords = torch.tensor(
        ContourNode.get_recursive_points_list(nodes), device=coords.device).int()
    # coords are in xy format and resolution in yx format
    new_coords = resolution * torch.flip(coords, dims=(-1,))
    new_coords = new_coords.round().int()
    # Clamp the coordinates
    new_coords[..., 0] = torch.clamp(new_coords[..., 0], 0, ry - 1)
    new_coords[..., 1] = torch.clamp(new_coords[..., 1], 0, rx - 1)
    selected_coords = torch.flip(
        new_coords[select_coords[:, 1], select_coords[:, 0]], dims=(-1,)).numpy()
    ContourNode.set_recursive_points_list(nodes, selected_coords)
    new_mask = torch.tensor(ContourNode.to_mask(
        nodes, resolution.numpy(), 0) / 255, dtype=mask.dtype, device=mask.device)
    return new_mask


# @saveable()
# def plot_as_scatter(mask, ret):
#     import matplotlib.pyplot as plt
#     from tools.viz.matplotlib import plot_as_image, saveable
#     fig, ax = plt.subplots()
#     plot_as_image(mask, axes=ax)
#     ax.scatter(ret[..., 1].reshape(-1), ret[..., 0].reshape(-1), c='r', s=10)
#     return fig


def plot_plane_hits(
        global_plane_position,
        plane_scale,
        ray_origins,
        ray_directions,
        intersection_points,
        resolution,
        camera,
        n_rays=20
):
    plane = DiscretePlaneSceneNode3D(
        position=global_plane_position, plane_scale=plane_scale)
    fig = plane.plot_scene()
    ax = fig.gca()
    ro = ray_origins.reshape(*resolution, 1, 3)
    rd = ray_directions.reshape(*resolution, 1, 3)

    mask = torch.zeros(
        (*resolution, 1), dtype=ray_origins.dtype, device=ray_origins.device)
    mask[::n_rays, ::n_rays] = 1
    mask = mask.bool()
    ro = ro[mask]

    rd = rd[mask]
    camera.plot_object(ax=ax)

    ip = intersection_points.reshape(*resolution, 3)
    ip = ip[mask[..., 0]].detach().cpu().numpy()

    ax.scatter(ip[..., 0], ip[..., 1], ip[..., 2], c='r', s=10)

    plot_rays(ray_origins=ro, ray_directions=rd, ax=ax)
    return fig


def gaussian_smoothing_2d(x: torch.Tensor, kernel_size: int = 5, sigma=1.0):
    from torchvision.transforms import GaussianBlur
    x, shape = flatten_batch_dims(x, -4)
    g = GaussianBlur(kernel_size, sigma)
    x = g(x)
    return unflatten_batch_dims(x, shape)


def compute_native_resolution(
        resolution: torch.Tensor,
        relative_plane_margin: torch.Tensor
):
    return resolution
