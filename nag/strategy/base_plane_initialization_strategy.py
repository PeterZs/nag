from typing import Any, Dict, NamedTuple, Tuple
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
from nag.transforms.transforms_timed_3d import linear_interpolate_vector
from tools.util.typing import DEFAULT
from tools.util.torch import tensorify
from nag.model.learned_alpha_location_image_plane_scene_node_3d import LearnedAlphaLocationImagePlaneSceneNode3D


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

    missing_in_frame: torch.Tensor
    """If the mask is missing in the frame e.g. not one pixel is true. Shape: (T,)"""

    average_dist_per_time: torch.Tensor
    """The average distance of the object from the camera for each timestamp. Shape: (T, )"""

    mask_center_of_mass: torch.Tensor
    """The center of mass of the mask. Shape: (T, 2) (y, x)"""

    largest_mask_idx: torch.Tensor
    """The index of the largest mask"""

    border_points: torch.Tensor
    """The border points of the mask. Shape: (4, T, 2) (y, x)
    Order is bottom left, bottom right, top right, top left.
    These are the corners of a the bounding box tightly around the mask.
    """


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


class BasePlaneInitializationStrategy(PlaneInitializationStrategy):
    """Simple plane initialization strategy:
    1. Planes have a fixed rotation, always facing the camera.
    2. Planes have a fixed scale depending on the largest mask extent.
    3. Planes are initialized at the mean depth mask position.
    4. If the mask is zero for some planes, position will be linearly interpolated until it enters the scene.
    """

    def compute_plane_scale(self,
                            image_bottom_left: torch.Tensor,
                            image_top_right: torch.Tensor,
                            resolution: torch.Tensor,
                            camera: TimedCameraSceneNode3D,
                            times: torch.Tensor,
                            depth: torch.Tensor) -> torch.Tensor:
        """Compute the plane scale.

        Parameters
        ----------
        image_bottom_left : torch.Tensor
            Image bottom left corner. Shape (T, 2) (y, x)
        image_top_right : torch.Tensor
            Image top right corner. Shape (T, 2) (y, x)
        resolution : torch.Tensor
            Resolution of the image. (H, W)
        camera : torch.Tensor
            The camera
        times : torch.Tensor
            The times. Shape (T,)
        depth : torch.Tensor
            The average distance of the object from the camera for each timestamp. Shape (T,)

        Returns
        -------
        torch.Tensor
            The plane scale. Shape (2,) (x, y)
        """
        uv_max = camera._image_resolution
        bl_cam = image_bottom_left / resolution * uv_max
        tr_cam = image_top_right / resolution * uv_max

        # Switch to xy
        bl_cam = torch.flip(bl_cam, dims=(-1,))
        tr_cam = torch.flip(tr_cam, dims=(-1,))

        # Stack on new Batch dim -2
        coords = torch.stack([bl_cam, tr_cam], dim=-2)
        coord_ro, coord_rd = camera.get_global_rays(
            uv=coords, t=times, uv_includes_time=True)
        bl_global_ro, bl_global_rd = coord_ro[0], coord_rd[0]
        tr_global_ro, tr_global_rd = coord_ro[1], coord_rd[1]

        global_bl_points = bl_global_ro + bl_global_rd * \
            depth.unsqueeze(-1)
        global_tr_points = tr_global_ro + tr_global_rd * \
            depth.unsqueeze(-1)

        global_plane_scale = torch.max(
            global_tr_points - global_bl_points, dim=0).values  # Shape: (3, )
        # As plane as unit length in z the distance between x, y is the scale
        global_plane_scale_xy = global_plane_scale[:2]
        return global_plane_scale_xy

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

            if not torch.any(mask):
                raise ValueError("Mask is empty.")

            T, C_img, H, W = images.shape

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

            # Convert plane center to camera coordinates - As images maybe downscaled
            re_plane_center = plane_center / torch.tensor([H, W])
            camera_plane_center = re_plane_center * camera._image_resolution

            # Round the plane center to an integer
            camera_plane_center = camera_plane_center.round()

            # Get these image coordinates in global / world coordinates
            cplane_xy = torch.flip(camera_plane_center, dims=(-1,))

            # Get the global rays for the center of the plane, Will have shape B, T, 3 and with B = 1
            global_plane_center_ray_origin, global_plane_center_ray_direction = camera.get_global_rays(
                uv=cplane_xy.unsqueeze(-2), t=times, uv_includes_time=True)
            # Squeeze B=1
            global_plane_center_ray_origin = global_plane_center_ray_origin.squeeze(
                0)
            global_plane_center_ray_direction = global_plane_center_ray_direction.squeeze(
                0)

            # Position of the plane is the average distance from the camera
            global_plane_position = global_plane_center_ray_origin + \
                global_plane_center_ray_direction * \
                average_dist_per_time.unsqueeze(-1)

            # Get the orientation of the plane, as we lock the plane rotation to the camera, it must be the same as the camera
            glob_pos = camera.get_global_position()

            # Extract rotation and convert to quaternion
            global_plane_orientation = rotmat_to_unitquat(
                glob_pos[..., :3, :3])

            resolution = torch.tensor([H, W], dtype=dtype)

            padded_plane_scale = self.compute_plane_scale(
                bl, tr, resolution, camera, times, average_dist_per_time)
            original_plane_scale = self.compute_plane_scale(
                orig_bl, orig_tr, resolution, camera, times, average_dist_per_time)

            # Gather all arguments for the plane

            plane_properties = get_plane_config_properties(
                object_index=object_index, times=times, config=config, name=name)
            # Update the properties

            plane_properties["translation"] = global_plane_position
            plane_properties["orientation"] = global_plane_orientation
            plane_properties["plane_scale"] = padded_plane_scale

            from nag.model.learned_alpha_location_image_plane_scene_node_3d import LearnedAlphaLocationImagePlaneSceneNode3D
            if issubclass(config.plane_type, LearnedAlphaLocationImagePlaneSceneNode3D):
                # Get
                bounds_bl = bl[largest_mask_idx].int()
                bounds_tr = tr[largest_mask_idx].int()

                # As bounds are maybe outside the image, we need to pad it if necessary
                pad_d = torch.tensor([0, 0, 0, 0], dtype=bounds_bl.dtype)
                if (bounds_bl < 0).any():
                    pad_d[:2][bounds_bl < 0] = bounds_bl[bounds_bl < 0] * -1
                if (bounds_tr >= torch.tensor([H, W])).any():
                    pad_d[2:][bounds_tr >= torch.tensor([H, W])] = (bounds_tr[bounds_tr >= torch.tensor(
                        [H, W])] - torch.tensor([H, W], dtype=bounds_tr.dtype)[bounds_tr >= torch.tensor([H, W])])

                # Add from bounds_bl the padded values
                bounds_tr = bounds_tr + pad_d[:2]
                bounds_bl = bounds_bl + pad_d[:2]

                # bounds_tr = bounds_tr + pad_d[2:]
                img = images[largest_mask_idx]
                alpha = mask[largest_mask_idx, 0]

                if pad_d.sum() > 0:
                    # pad_d is in formax bl_y, bl_x, tr_y, tr_x => padding_bottom, padding_left, padding_top, padding_right
                    # For fpad need to (padding_left,padding_right, padding_top,padding_bottom)
                    pd = (pad_d[1], pad_d[3], pad_d[0], pad_d[2])
                    img = torch.nn.functional.pad(img, pd, mode="replicate")
                    alpha = torch.nn.functional.pad(
                        alpha, pd, mode="constant", value=0.)

                base_image = img[:,
                                 bounds_bl[0]:bounds_tr[0],
                                 bounds_bl[1]:bounds_tr[1]]

                base_alpha = alpha[
                    bounds_bl[0]:bounds_tr[0],
                    bounds_bl[1]:bounds_tr[1]]

                plane_properties["base_image"] = base_image
                plane_properties["base_alpha"] = base_alpha.to(dtype)

                # Reorder the padding to left, bottom, right, top
                plane_properties["base_padding"] = padded[[1, 0, 3, 2]]
                plane_properties["unpadded_plane_scale"] = original_plane_scale
            return plane_properties
