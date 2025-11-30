from typing import Any, Dict, NamedTuple
import kornia
from tools.model.abstract_scene_node import AbstractSceneNode
from torch import Tensor
from nag.config.nag_config import NAGConfig
from nag.dataset.nag_dataset import NAGDataset
from nag.model.nag_model import NAGModel
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.strategy.plane_initialization_strategy import PlaneInitializationStrategy, get_plane_config_properties
import torch
from tools.transforms.geometric.transforms3d import rotmat_to_unitquat, unitquat_to_rotmat
import collections
from tools.transforms.geometric.mappings import rotvec_to_unitquat
from nag.transforms.transforms_timed_3d import linear_interpolate_vector
from tools.util.typing import DEFAULT
from tools.util.torch import tensorify
from nag.strategy.base_plane_initialization_strategy import BasePlaneInitializationStrategy
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims
from kornia.utils.draw import draw_convex_polygon
from tools.viz.matplotlib import plot_as_image
from tools.transforms.geometric.transforms3d import calculate_rotation_matrix, compute_ray_plane_intersections_from_position_matrix
from nag.model.timed_discrete_scene_node_3d import global_to_local, local_to_global
from fast_pytorch_kmeans import KMeans


class BasicMaskProperties(NamedTuple):

    bottom_left: torch.Tensor
    """Bottom left corner of the object. Will be interpolated if object is not present on all the frames. Shape: (T, 2) (y, x)"""

    top_right: torch.Tensor
    """Top right corner of the object. Will be interpolated if object is not present on all the frames. Shape: (T, 2) (y, x)"""

    padded_bottom_left: torch.Tensor
    """Bottom left corner with relative margin padded to it. Shape: (T, 2) (y, x)"""

    padded_top_right: torch.Tensor
    """Top right corner with relative margin padded to it. Shape: (T, 2) (y, x)"""

    mask_sizes: torch.Tensor
    """The size of the mask for each timestamp. Shape: (T, )"""

    average_dist_per_time: torch.Tensor
    """The average distance of the object from the camera for each timestamp. Shape: (T, )"""

    mask_center_of_mass: torch.Tensor
    """The center of mass of the mask. Shape: (T, 2) (y, x)"""

    largest_mask_idx: torch.Tensor
    """The index of the largest mask"""


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

    return BasicMaskProperties(
        bottom_left=bl,
        top_right=tr,
        padded_bottom_left=padded_bl,
        padded_top_right=padded_tr,
        mask_sizes=mask_sizes,
        average_dist_per_time=average_dist_per_time,
        largest_mask_idx=largest_mask_idx,
        mask_center_of_mass=mask_center_of_mass
    )


def mask_to_camera_coordinates(uv: torch.Tensor,
                               uv_max: torch.Tensor,
                               camera_max: torch.Tensor) -> torch.Tensor:
    """Convert mask coordinates to camera coordinates.

    Parameters
    ----------
    uv : torch.Tensor
        The mask coordinates. Shape: (T, 2) (y, x)

    Returns
    -------
    torch.Tensor
        The camera coordinates. Shape: (T, 2) (y, x)
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


class KmeansTiltedPlaneInitializationStrategy(PlaneInitializationStrategy):
    """Simple plane initialization strategy:
    1. Planes have a rotation, depending on the depth mask.
    2. Planes have a fixed scale depending on the largest mask extent.
    3. If the mask is zero for some planes, position will be linearly interpolated until it enters the scene.
    """

    def __init__(self,
                 depth_ray_thickness: float = 0.05,
                 min_depth_ray_thickness: float = 10.
                 ):
        self.depth_ray_thickness = depth_ray_thickness
        self.min_depth_ray_thickness = min_depth_ray_thickness

    def compute_plane_scale(self,
                            border_coords: torch.Tensor,
                            resolution: torch.Tensor,
                            camera: TimedCameraSceneNode3D,
                            times: torch.Tensor,
                            global_plane_position: torch.Tensor,
                            relative_plane_margin: torch.Tensor,
                            ) -> torch.Tensor:
        """Compute the plane scale.

        Parameters
        ----------
        border_coords: torch.Tensor
            The border coordinates. Shape: (4, T, 2) (y, x)
            In counter clockwise order.
            bl, br, tr, tl

        resolution : torch.Tensor
            Resolution of the image. (H, W)
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
        # Add the relative margin
        x_scale += x_scale * relative_plane_margin
        y_scale += y_scale * relative_plane_margin
        return torch.stack([x_scale, y_scale], dim=-1)

    def execute(self,
                object_index: int,
                mask_index: int,
                images: Tensor,
                masks: Tensor,
                depths: Tensor,
                camera: TimedCameraSceneNode3D,
                nag_model: NAGModel,
                dataset: NAGDataset,
                config: NAGConfig,
                name: str = None,
                **kwargs) -> Dict[str, Any]:
        with torch.no_grad():
            mask = masks[:, mask_index].unsqueeze(1)
            times = camera.get_times()
            dtype = config.dtype

            if not torch.any(mask):
                raise ValueError("Mask is empty.")

            T, C_img, H, W = images.shape
            device = mask.device
            # Compute the basic mask properties
            mask_properties = compute_mask_properties(
                mask=mask, depth=depths, times=times, relative_plane_margin=tensorify(config.relative_plane_margin))

            orig_bl = mask_properties.bottom_left
            orig_tr = mask_properties.top_right
            bl = mask_properties.padded_bottom_left
            tr = mask_properties.padded_top_right
            average_dist_per_time = mask_properties.average_dist_per_time
            largest_mask_idx = mask_properties.largest_mask_idx

            # Calculated the padded distances
            padded_bl = orig_bl[largest_mask_idx].round(
            ).int() - bl[largest_mask_idx].round().int()
            padded_tr = tr[largest_mask_idx].round().int(
            ) - orig_tr[largest_mask_idx].round().int()
            # Concatenate to bottom, left, top, right
            padded = torch.cat([padded_bl, padded_tr], dim=-1).round().int()

            # Round to integer
            bl = bl.round().int()
            tr = tr.round().int()

            plane_center = bl + (tr - bl) / 2

            camera_plane_center = mask_to_camera_coordinates(plane_center, torch.tensor(
                [H, W], dtype=dtype, device=mask.device), camera._image_resolution)
            camera_plane_center = torch.flip(
                camera_plane_center, dims=(-1,))  # Switch to xy

            # Get the center of mass of the mask
            # Shape: (T, 2) (y, x)
            mask_com = mask_properties.mask_center_of_mass

            D = 3

            # Compute the ray collisions with the mask border defined by bl and tr
            # Get also br and tl from orig_bl and orig_tr which are (y, x) coordinates
            orig_br = torch.stack([orig_bl[:, 0], orig_tr[:, 1]], dim=-1)
            orig_tl = torch.stack([orig_tr[:, 0], orig_bl[:, 1]], dim=-1)
            # Stack the border points
            border_points = torch.stack(
                [orig_bl, orig_br, orig_tr, orig_tl], dim=-3)

            plane_support_points = torch.zeros(
                D, T, 3, dtype=dtype, device=device)  # x, y, z

            missing_in_frame = mask[:, 0].sum(dim=(-1, -2)) == 0

            # As masks have different sizes, we neet to loop over the masks
            for i in range(T):
                if missing_in_frame[i]:
                    continue
                kmeans = KMeans(n_clusters=3)
                m = mask[i, 0]
                selected_depths = depths[i, 0, m]
                m_idx = torch.argwhere(m)
                kmeans.fit(selected_depths.unsqueeze(-1))
                # Centeroids are the plane support points
                distance = kmeans.centroids.squeeze(-1).unsqueeze(
                    0) - selected_depths.unsqueeze(-1).repeat(1, 3)
                closest_points = torch.argmin(torch.abs(distance), dim=0)
                plane_support_points[:, i, :2] = torch.flip(
                    m_idx[closest_points], dims=(-1,))
                plane_support_points[:, i, 2] = kmeans.centroids.squeeze(-1)

            # Get the plane_support_points to word coordinates
            support_camera = mask_to_camera_coordinates(plane_support_points[..., :2], torch.tensor(
                [W, H], dtype=dtype, device=device), camera._image_resolution.flip(-1))
            ro, rd = camera.get_global_rays(
                uv=support_camera.swapaxes(0, 1), t=times, uv_includes_time=True)
            global_plane_support_points = ro + rd * \
                plane_support_points[..., 2].unsqueeze(-1)

            # TODO plane_support points can be nan if the object is not present at a timestamp
            r1 = global_plane_support_points[0, ...] - \
                global_plane_support_points[1, ...]
            r2 = global_plane_support_points[0, ...] - \
                global_plane_support_points[2, ...]

            normal = torch.cross(r1, r2, dim=-1)
            normal = normal / torch.norm(normal, dim=-1, keepdim=True)

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

            plane_properties = get_plane_config_properties(
                object_index=object_index, times=times, config=config, name=name)

            # If a object is not present at a timestamp, we will need to interpolate the position and orientation
            if missing_in_frame.sum() > 0:
                from nag.model.timed_discrete_scene_node_3d import get_translation_orientation
                missing_t = times[missing_in_frame]
                translation, orientation = get_translation_orientation(
                    translation=plane_center[~missing_in_frame], orientation=quat[~missing_in_frame], times=times[~missing_in_frame], steps=missing_t, equidistant_times=False)
                plane_center[missing_in_frame] = translation
                quat[missing_in_frame] = orientation

            global_plane_position[:, :3, 3] = plane_center
            global_plane_position[:, :3, :3] = unitquat_to_rotmat(quat)

            plane_scale = self.compute_plane_scale(
                border_coords=border_points,
                resolution=torch.tensor(
                    [H, W], dtype=dtype, device=mask.device),
                camera=camera,
                times=times,
                global_plane_position=global_plane_position,
                relative_plane_margin=tensorify(config.relative_plane_margin)
            )
            plane_surface = plane_scale.prod(dim=-1)
            # Select the largest mask
            max_idx = torch.argmax(plane_surface[~missing_in_frame])
            _plane_scale = plane_scale[~missing_in_frame][max_idx]

            # Update the properties
            plane_properties["translation"] = plane_center
            plane_properties["orientation"] = quat
            plane_properties["plane_scale"] = _plane_scale
            # plane_properties["unpadded_plane_scale"] = original_plane_scale

            return plane_properties
