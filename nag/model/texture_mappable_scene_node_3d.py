from typing import Any, Optional
from tools.model.discrete_module_scene_node_3d import DiscreteModuleSceneNode3D
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
import torch
from tools.util.torch import tensorify
from nag.model.timed_discrete_scene_node_3d import global_to_local, local_to_global
from nag.model.timed_plane_scene_node_3d import TimedPlaneSceneNode3D
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims, compose_transformation_matrix
from nag.model.discrete_plane_scene_node_3d import local_to_plane_coordinates, plane_coordinates_to_local
from tools.transforms.to_tensor_image import ToTensorImage


class TextureMappableSceneNode3D(TimedPlaneSceneNode3D):
    """Class marking a scene node as texture mappable."""

    texture_map: Optional[torch.Tensor]
    """The texture map for the plane. Used when retexturing planes. Shape: (4, H, W)"""

    render_texture_map: bool
    """If the texture map should be rendered. If True, the texture map is rendered ontop of the predicted plane image."""

    def __init__(self,
                 texture_map: Optional[VEC_TYPE] = None,
                 render_texture_map: bool = False,
                 dtype: torch.dtype = torch.float32,
                 **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.texture_map = None
        tensorify_image = ToTensorImage(output_dtype=dtype)

        if texture_map is not None:
            del self.texture_map
            self.register_buffer("texture_map", tensorify_image(texture_map))

        self.render_texture_map = render_texture_map

    def create_texture_map(self,
                           image: torch.Tensor,
                           t: torch.Tensor,
                           camera: Any,
                           include_flow: bool = True
                           ):
        """Creates a texture in atlas space from the given image, by projecting it
        first onto plain space and then inverting the flow to render it the same way
        as the neural field.

        Will save the texture in the texture_map field.

        Parameters
        ----------
        image : torch.Tensor
            The desired texture.
        t : torch.Tensor
            Timestamp to project on.
        camera : Any
            The camera used for the projection.
        
        include_flow : bool, optional
            If flow inversion should take place, by default True
            If set to false, the flow will not inverted producing errors uppon rendering
            but could be helpfull for debugging.

        Raises
        ------
        ValueError
            On wrong channel format.
        """
        from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D, local_to_image_coordinates
        from tools.transforms.to_tensor_image import ToTensorImage
        from tools.transforms.geometric.transforms3d import compute_ray_plane_intersections_from_position_matrix

        with torch.no_grad():
            camera: TimedCameraSceneNode3D
            dtype = self._translation.dtype
            device = self._translation.device

            t = tensorify(t, dtype=dtype, device=device)
            global_plane_positions_use = self.get_global_position(t=t)

            tensorify_image = ToTensorImage(
                output_dtype=torch.float32, output_device=device)
            image = tensorify_image(image)

            C, H, W = image.shape

            if C not in [3, 4]:
                raise ValueError(
                    f"Image must have 3 or 4 channels, but has {C} channels.")

            # Check if camera resolution and image resolution match
            if camera._image_resolution[0] != H or camera._image_resolution[1] != W:
                self.logger.warning(
                    f"Camera resolution {camera._image_resolution} does not match image resolution {H, W}. Resizing image.")
                # Resize image to camera resolution
                image = torch.nn.functional.interpolate(image.unsqueeze(0), size=tuple(
                    camera._image_resolution.tolist()), mode='bilinear', align_corners=True).squeeze(0)
                C, H, W = image.shape

            def ignore_flow_proj(image):
                x = torch.linspace(0, 1, W, dtype=dtype, device=device)
                y = torch.linspace(0, 1, H, dtype=dtype, device=device)
                grid = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)

                grid, gs = flatten_batch_dims(grid, -2)

                local_coords = plane_coordinates_to_local(
                    grid, self._plane_scale, plane_offset=self._plane_scale_offset)

                global_coords = local_to_global(
                    global_plane_positions_use.detach(), local_coords)
                in_camera_coords = camera.global_to_local(
                    global_coords.to(camera._translation.device), t=t, v_include_time=True)

                in_image_coords = local_to_image_coordinates(in_camera_coords.permute(1, 0, 2)[..., :3],
                                                             camera.get_intrinsics(
                    t=t),
                    camera.focal_length,
                    camera._lens_distortion,
                    v_includes_time=True)

                rel_coords = in_image_coords / \
                    camera._image_resolution.flip(-1)
                grid_sample_grid = unflatten_batch_dims((rel_coords.permute(
                    1, 0, 2) - 0.5) * 2, gs).permute(2, 0, 1, 3)  # Permute time to first dim

                return torch.nn.functional.grid_sample(image[None], grid_sample_grid,
                                                       mode='bilinear', padding_mode='border',
                                                       align_corners=True)[0]

            def with_flow_proj(
                    image: torch.Tensor,
                    camera: TimedCameraSceneNode3D,
                    t: torch.Tensor,
                    plane_positions: torch.Tensor,
                    plane_scale: torch.Tensor,
                    plane_offset: torch.Tensor,
            ):
                from nag.model.nag_model import get_object_intersection_points
                from scipy.interpolate import LinearNDInterpolator
                # Add Alpha channel if not present
                if image.shape[0] == 3:
                    image = torch.cat(
                        [image, torch.ones_like(image[:1])], dim=0)

                # Shoot rays from the camera to the plane
                x = torch.linspace(0, W, W, dtype=dtype, device=device)
                y = torch.linspace(0, H, H, dtype=dtype, device=device)
                grid = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)

                ro, rd = camera.get_global_rays(
                    uv=grid.reshape(-1, 2), t=t, uv_includes_time=False)

                intersection_points, is_inside, _, _ = get_object_intersection_points(plane_positions[None, ...],
                                                                                      local_plane_scale=plane_scale[None, ...],
                                                                                      local_plane_scale_offset=plane_offset[
                                                                                          None, ...],
                                                                                      global_ray_origins=ro,
                                                                                      global_ray_directions=rd)

                local_intersection_points = global_to_local(
                    plane_positions, intersection_points, True)
                is_inside_pixels = image[:,
                                         is_inside.reshape(H, W)].permute(1, 0)

                inside_local_points = local_intersection_points[is_inside]
                plane_coordinates = local_to_plane_coordinates(
                    inside_local_points, plane_scale, plane_offset)

                def in_hull(p, hull):
                    """
                    Test if points in `p` are in `hull`

                    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
                    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
                    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
                    will be computed
                    """
                    from scipy.spatial import Delaunay
                    if not isinstance(hull, Delaunay):
                        hull = Delaunay(hull)
                    return hull.find_simplex(p) >= 0

                # Filtering out pixels wihout alpha
                inside_w_alpha = is_inside_pixels[:, -1] > 0

                in_h = torch.tensor(in_hull(plane_coordinates.cpu().numpy(), plane_coordinates[inside_w_alpha].cpu(
                ).numpy()), device=plane_coordinates.device, dtype=torch.bool)

                network_uv = plane_coordinates[in_h] - 0.5
                flow = self.get_flow(network_uv.unsqueeze(
                    1), t, sin_epoch=torch.tensor(1.)).squeeze(1)
                interp_coords = plane_coordinates[in_h] + flow

                is_inside_in_hull = is_inside_pixels[in_h]

                interp = LinearNDInterpolator(interp_coords.cpu().numpy(
                ), is_inside_in_hull.cpu().numpy(), fill_value=0)
                z = interp(
                    *(grid / torch.tensor([[W, H]], device=grid.device, dtype=grid.dtype)).reshape(-1, 2).cpu().numpy().T)
                tmap = torch.tensor(z, device=grid.device, dtype=image.dtype).reshape(
                    H, W, 4).permute(2, 0, 1)
                return tmap

            if not include_flow:
                projected_colors = ignore_flow_proj(image)
            else:
                projected_colors = with_flow_proj(
                    image,
                    camera,
                    t=t,
                    plane_positions=self.get_global_position(t=t),
                    plane_scale=self._plane_scale,
                    plane_offset=self._plane_scale_offset,
                )

            # Check if alpha channel is present, if not add it
            if C == 3 and projected_colors.shape[0] == 3:
                projected_colors = torch.cat(
                    [projected_colors, torch.ones_like(projected_colors[:1])], dim=0)

            self._create_plain_texture_map(projected_colors)

    def _create_plain_texture_map(self, image: VEC_TYPE):
        """
        Take the given image directly as texture map.

        Parameters
        ----------
        image : VEC_TYPE
            The image to be used as texture map. Shape: (C, H, W) if tensor or (H, W, C) if numpy array.

        Raises
        ------
        ValueError
            If the image does not have 3 or 4 channels.
        """
        from tools.transforms.to_tensor_image import ToTensorImage
        dtype = self._translation.dtype
        device = self._translation.device
        tensorify_image = ToTensorImage(
            output_dtype=torch.float32, output_device=device)

        image = tensorify_image(image).to(device=device, dtype=dtype)
        C, H, W = image.shape

        if C not in [3, 4]:
            raise ValueError(
                f"Image must have 3 or 4 channels, but has {C} channels.")

        if "texture_map" not in self._buffers:
            if "texture_map" in self.__dict__:
                del self.texture_map
            self.register_buffer("texture_map", image)
        else:
            self.texture_map = image

    def get_texture_map(self, uv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Gets the texture map RGBA values for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            Uv coordinates of the point.
            Shape: (B, 2) x, y should be in range [-0.5, 0.5]

        Returns
        -------
        torch.Tensor
            The rgb values in shape (B, 3)
        """
        if self.texture_map is None:
            raise ValueError(
                "Texture map not created. Call create_texture_map first.")

        grid = (uv * 2)[None, None, ...]  # (1, 1, B, 2) to match gridsample
        out = torch.nn.functional.grid_sample(self.texture_map.unsqueeze(
            0), grid, mode="bilinear", padding_mode="border", align_corners=True)[0].reshape(4, -1)
        # Swap channel dimension to the end
        out = out.permute(1, 0)
        return out

    def get_rendered_texture_map(self,
                                 uv: torch.Tensor,
                                 rgb: torch.Tensor,
                                 **kwargs) -> torch.Tensor:
        """Renders the texture map on the plane.

        Parameters
        ----------
        uv : torch.Tensor
            Uv coordinates of the point.
            Shape: (B, 2) x, y should be in range [-0.5, 0.5]

        rgb : torch.Tensor
            The rgb values of the plane. Shape: (B, 3)

        Returns
        -------
        torch.Tensor
            The rgb values in shape (B, 3)
        """
        if self.texture_map is None:
            raise ValueError(
                "Texture map not created. Call create_texture_map first.")

        texture = self.get_texture_map(uv)
        return rgb * (1 - texture[..., -1:]) + texture[..., :3] * texture[..., -1:]
