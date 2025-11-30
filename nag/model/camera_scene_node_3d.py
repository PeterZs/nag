from typing import Any, Iterable, Optional, Tuple, Union
from matplotlib.axes import Axes
from tools.model.module_scene_node_3d import ModuleSceneNode3D
import torch
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims, compose_transformation_matrix
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.util.numpy import numpyify


class CameraSceneNode3D(ModuleSceneNode3D):
    """A scene node representing a camera in 3D space. As a pinhole camera model."""

    _image_resolution: torch.Tensor
    """The resolution of the camera image as (height, width)."""

    _lens_distortion: torch.Tensor
    """Lens distortion parameters as (kappa_1, kappa_2, kappa_3, kappa_4, kappa_5) whereby 1-3 are radial and 4, 5 tangential correction coefficients.
    Further reading and source for android phones can be found here: https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics#LENS_DISTORTION
    """

    _intrinsics: torch.Tensor
    """The intrinsic camera calibration matrix as (3, 3) matrix.
        np.array([[fx,  s, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]])
        Further reading: https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics#LENS_INTRINSIC_CALIBRATION
    """

    _inverse_intrinsics: torch.Tensor
    """The inverse of the intrinsic camera calibration matrix as (3, 3) matrix."""

    def __init__(self,
                 image_resolution: VEC_TYPE,
                 lens_distortion: VEC_TYPE,
                 intrinsics: VEC_TYPE,
                 name: Optional[str] = None,
                 children: Optional[Iterable['AbstractSceneNode']] = None,
                 decoding: bool = False,
                 dtype: torch.dtype = torch.float32,
                 **kwargs
                 ):
        super().__init__(name=name, children=children,
                         decoding=decoding, dtype=dtype, **kwargs)
        if len(image_resolution) != 2:
            raise ValueError("Image resolution must be a 2D vector.")
        if lens_distortion.shape != (5, ):
            raise ValueError("Lens distortion must be a 5D vector.")
        if intrinsics.shape[-2:] != (3, 3):
            raise ValueError("Intrinsics must be a 3x3 matrix.")
        self.register_buffer("_image_resolution", tensorify(
            image_resolution, dtype=torch.int32))
        self.register_buffer("_lens_distortion", tensorify(
            lens_distortion, dtype=self.dtype))
        self.register_buffer("_intrinsics", torch.empty(0))
        self.register_buffer("_inverse_intrinsics", torch.empty(0))
        self.set_intrinsics(tensorify(intrinsics, dtype=self.dtype))

    @property
    def focal_length(self) -> torch.Tensor:
        return (self._intrinsics[..., 0, 0] / self._image_resolution[-1]).squeeze()

    def get_intrinsics(self) -> torch.Tensor:
        """Return the intrinsics matrix of the camera."""
        return self._intrinsics

    def set_intrinsics(self, intrinsics: torch.Tensor) -> None:
        """Set the intrinsics matrix of the camera."""
        if len(intrinsics.shape) != 2 and intrinsics.shape[-3] != 1:
            # Print warning
            self.logger.warning(
                f"Invalid intrinsics shape {intrinsics.shape}. Expected (3, 3), will only use the first element.")
            intrinsics = intrinsics[0]
        self._intrinsics = intrinsics
        self._inverse_intrinsics = torch.inverse(intrinsics)

    def get_focal_length(self) -> torch.Tensor:
        """Gets the focal length of the camera as (fy, fx).

        Returns
        -------
        torch.Tensor
            The focal length. (..., 2) tensor (fy, fx).
        """
        return torch.flip(self._intrinsics[..., :2, :2].diagonal(), dims=(-1,))  # Flip to (fy, fx) order

    def get_optical_axis(self) -> torch.Tensor:
        """Gets the optical axis of the camera as (cy, cx).

        Returns
        -------
        torch.Tensor
            The optical axis. (..., 2) tensor (cy, cx).
        """
        return torch.flip(self._intrinsics[..., :2, 2], dims=(-1,))  # Flip to (cy, cx) order

    def get_lens_distortion(self) -> torch.Tensor:
        """Gets the lens distortion parameters as Lens distortion parameters (k1, k2, k3, t1, t2).

        3 radial and 2 tangential distortion coefficients.

        Returns
        -------
        torch.Tensor
            The lens distortion parameters. (..., 5) tensor.
        """
        return self._lens_distortion

    def get_lens_distortion_opencv(self) -> torch.Tensor:
        """Gets the lens distortion parameters as Lens distortion parameters (k1, k2, t1, t1, k3).

        2 radial and 2 tangential distortion coefficients.

        Returns
        -------
        torch.Tensor
            The lens distortion parameters. (..., 5) tensor.
        """
        return torch.cat([self._lens_distortion[..., :2], self._lens_distortion[..., 3:5], self._lens_distortion[..., 2:3]], dim=-1)

    def get_inverse_intrinsics(self) -> torch.Tensor:
        """Gets the inverse of the intrinsic camera calibration matrix.

        Returns
        -------
        torch.Tensor
            The inverse of the intrinsic camera calibration matrix.
        """
        return self._inverse_intrinsics

    def _sample_pixel_grid(self,
                           subsample: Union[int, torch.Tensor] = 1,
                           subsample_offset: Union[int, torch.Tensor] = 0,
                           include_borders: bool = False,
                           resolution: Optional[torch.Tensor] = None
                           ) -> torch.Tensor:
        """Samples pixel grid for the camera image plane.

        Parameters
        ----------
        subsample : int, optional
            Subsample factor for the rays, by default 1.
            If subsample is 1, the grid will be the same size as the image resolution.

        subsample_offset : int, optional
            Offset for the subsample, by default 0
            Will start the subsample at the given offset.

        include_borders : bool, optional
            If True, the grid will include the borders of the image plane, by default False
            Only valid if subsample is > 1. If false, the last sample step will be smaller.

        resolution : Optional[torch.Tensor], optional
            Resolution of the image plane, by default None
            Shape (2,) (x, y) (width, height) of the image plane.
            None will use the internal image resolution of the camera.
            Can be used in similar fashion to subsample and subsample_offset, but setting the resolution directly.
            Is mutally exclusive with subsample and subsample_offset.

        Returns
        -------
        torch.Tensor
            The dense rays as (height, width, 3) of (x, y, z) tensor with z = 0.
        """
        # Create a grid of pixel coordinates

        idx, idy = None, None
        if isinstance(subsample, int):
            subsample = torch.tensor(
                subsample, dtype=torch.int32).unsqueeze(0).repeat(2)
        if isinstance(subsample_offset, int):
            subsample_offset = torch.tensor(
                subsample_offset, dtype=torch.int32).unsqueeze(0).repeat(2)

        _sample_resolution = self._image_resolution

        if resolution is not None:
            if not isinstance(resolution, torch.Tensor):
                resolution = torch.tensor(resolution, dtype=torch.int32)
            if len(resolution) != 2:
                raise ValueError(
                    "Resolution must be a 2D vector. Stating the width and height of the image plane. (x, y)")
            # Flip to (height, width) order
            _sample_resolution = resolution.flip(-1).to(
                dtype=self._image_resolution.dtype, device=self._image_resolution.device)

        if (subsample == 1).all() or not include_borders:
            idx = torch.arange(
                subsample_offset[0], _sample_resolution[1], subsample[0], dtype=torch.int32, device=_sample_resolution.device)
            idy = torch.arange(
                subsample_offset[1], _sample_resolution[0], subsample[1], dtype=torch.int32, device=_sample_resolution.device)
        else:
            idx = torch.linspace(
                subsample_offset[0], _sample_resolution[1] - 1, _sample_resolution[1] // subsample[0], dtype=torch.int32, device=_sample_resolution.device)
            idy = torch.linspace(
                subsample_offset[1], _sample_resolution[0] - 1, _sample_resolution[0] // subsample[1], dtype=torch.int32, device=_sample_resolution.device)

        if resolution is not None:
            # Multiply by the ratio of the resolutions to query the overall image
            ratio = self._image_resolution / _sample_resolution
            idx = idx * ratio[1]
            idy = idy * ratio[0]

        # Coordinate order is (x, y), (height, width)
        grid = torch.stack(torch.meshgrid(
            idx, idy, indexing="xy"), dim=-1).float()

        # Add Z coordinate of 0 for the image plane
        xyz = torch.cat([grid, torch.zeros_like(grid[..., :1])], dim=-1)
        return xyz

    def _get_ray_direction(self, uv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the ray direction for the given pixel coordinates in the local camera coordinate system.


        Parameters
        ----------
        uv : Optional[torch.Tensor], optional
            Given pixel coordinates to get the ray origins for, by default None
            Pixel coordinates are assumed to be in the image plane in order (x, y). [0, width) and [0, height)

        Returns
        -------
        torch.Tensor
            The ray direction in the local camera coordinate system.
            As (..., 3) (x, y, z) tensor.
            Due to the definition of the camera coordinate system (homographic coordinates), the z component is 1 for a point on the camera / image plane.
            which is graphically represented as Z = 0 so one must subtract (0, 0, 1) to get the image plane.

        """
        if uv is None:
            uv = self._sample_pixel_grid()

        # Convert uv to homogeneous coordinates
        uvz_plane = torch.cat(
            [uv[..., :2], torch.ones_like(uv[..., :1])], dim=-1)
        # Flatten
        xyz, batch_dims = flatten_batch_dims(uvz_plane, 1)

        inverse_intrinsics = self.get_inverse_intrinsics()
        # Multiply by the intrinsics inverse
        xyz_cam = torch.bmm(inverse_intrinsics.unsqueeze(
            0).repeat(xyz.shape[0], 1, 1), xyz.unsqueeze(-1)).squeeze(-1)
        xy = xyz_cam[..., :2]

        # Korrect for lens and distortion
        r2 = torch.sum(xy**2, dim=1, keepdim=True)  # N x 1
        r4 = r2**2
        r6 = r2**3
        kappa1, kappa2, kappa3 = self.get_lens_distortion()[..., 0:3]

        xy = xy * (1 + kappa1*r2 + kappa2*r4 + kappa3*r6)

        # Optional TODO tangential distortion

        # Add Z coordinate of 1 for the image plane again
        xyz_cam = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
        return unflatten_batch_dims(xyz_cam, batch_dims)

    def _get_pixel_coordinates(self, pixel_size: Optional[torch.Tensor] = None, uv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the pixel coordinates in the camera image plane.

        Parameters
        ----------
        pixel_size : Optional[torch.Tensor], optional
            Assumption for the pixel size, if not specified inferred by using image resolution and assuming chip to be unit size., by default None
            Dont specify if you want to use the default pixel size (unit size)

        uv : Optional[torch.Tensor], optional
            Already sampled pixel grid in index coordinates, by default None
            Pixel coordinates (height, width, 2) or (height * width, 2) or some subset pixel subset in image plane (x, y), x in [0, width) and y in [0, height)
        Returns
        -------
        torch.Tensor
            The 2D pixel coordinates in the camera image plane. With the z component as 0.
        """
        if uv is None:
            uv = self._sample_pixel_grid()

        if pixel_size is None:
            pixel_size = (1 / self._image_resolution).max().repeat(2)

        # Subtract the optical axis to center the grid
        # Flip to xy order
        pixel_grid = uv[..., :2] - \
            torch.flip(self.get_optical_axis(), dims=(-1, )).unsqueeze(0)

        # Multiply by the pixel size to get the grid in pixel coordinates
        # Flip to xy order and add z.
        pixel_grid *= torch.flip(pixel_size, dims=(-1, )).unsqueeze(0)

        # Add Z = 0 to represent local camera image plane
        pixel_grid = torch.cat(
            [pixel_grid, torch.zeros_like(pixel_grid[..., :1])], dim=-1)
        return pixel_grid

    def _get_global_pixel_coordinates(self,
                                      pixel_size: Optional[torch.Tensor] = None,
                                      sample_grid: Optional[torch.Tensor] = None,
                                      **kwargs) -> torch.Tensor:
        if sample_grid is None:
            sample_grid = self._sample_pixel_grid(subsample=kwargs.get(
                "pixel_grid_subsample", 50), include_borders=kwargs.get("pixel_grid_borders", True))
        pixel_grid = self._get_pixel_coordinates(
            pixel_size=pixel_size, uv=sample_grid)
        # 2D Grid to Vectors
        pixels, batch_dims = flatten_batch_dims(pixel_grid, 1)
        global_pixels = (torch.bmm(self.get_global_position().unsqueeze(0).repeat(
            pixels.shape[0], 1, 1), compose_transformation_matrix(position=pixels)))[..., :3, 3]
        return unflatten_batch_dims(global_pixels, batch_dims)

    def get_camera_pixel_corner_coordinates(self) -> torch.Tensor:
        """Get the corner coordinates of the camera pixels in camera coordinates.

        Returns
        -------
        torch.Tensor
            Tensor of shape (4, 2) (y, x) containing the corner coordinates of the camera pixels in camera coordinates.
            The corners are ordered as bottom left, top left, top right, bottom right.
            # Assumes that the camera has unit length w.r.t x
        """
        pixel_size = (1 / self._image_resolution[-1]).repeat(2)

        oa = self.get_optical_axis()
        rel_oa = oa / self._image_resolution

        # Camera should be located [-0.5, 0.5] for x (IF optical axis is at (0, 0)), y adjust according to resolution
        # Get the camera image plane: Cameras optical axis is at (0, 0, 0), while its frame adjusts depending on the resolution

        bl = (-oa) * pixel_size
        min_y, min_x = bl
        tr = (self._image_resolution - oa) * pixel_size
        max_y, max_x = tr

        br = torch.tensor([min_y, max_x], device=self._image_resolution.device)
        tl = torch.tensor([max_y, min_x], device=self._image_resolution.device)

        # stack the corners
        corners = torch.stack([bl, tl, tr, br], dim=0)
        return corners

    # region Plotting

    def _plot_pixel_grid(self, ax: Axes, pixel_size: Optional[torch.Tensor] = None, sample_grid: Optional[torch.Tensor] = None, **kwargs):
        global_pixels = self._get_global_pixel_coordinates(
            pixel_size=pixel_size, sample_grid=sample_grid, **kwargs)
        # Flatten the batch dims
        global_pixels = flatten_batch_dims(global_pixels, -2)[0]
        ax.scatter(*global_pixels.detach().cpu().numpy().T, color='b', s=1)

    def _plot_camera_wireframe(self, ax: Axes, pixel_size: Optional[torch.Tensor] = None, **kwargs):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        if pixel_size is None:
            pixel_size = (1 / self._image_resolution).max().repeat(2)

        # Projection center
        # Get the min focal length as projection distance / center of projection
        f_x = self.focal_length
        proj_c = torch.tensor([0, 0, -f_x, 1], dtype=torch.float32)

        corners = self.get_camera_pixel_corner_coordinates()
        # Transpose corners as matplotlib uses xy instead of yx for pytorch
        corners = torch.flip(corners, dims=(-1,))

        # Add z=0 and scale = 1
        corners = torch.cat([corners, torch.zeros_like(
            corners[..., :1]), torch.ones_like(corners[..., :1])], dim=-1)
        # Add the camera position
        global_corners = (self.get_global_position() @ corners.T).T[..., :3]

        # Add some triangles to make it look like a camera
        # Get a point behind the camera plane
        global_projection_center = (
            self.get_global_position() @ proj_c.T).T[..., :3]

        triangles = torch.stack([global_projection_center.unsqueeze(0).repeat(
            global_corners.shape[0], 1), global_corners, torch.roll(global_corners, shifts=1, dims=0)], dim=1)

        # Plot the rectangle
        ax.add_collection3d(Poly3DCollection([global_corners.numpy(
        )], facecolors='grey', linewidths=1, edgecolors='k', alpha=0.5))

        # Plot wireframe indicating projection direction
        ax.add_collection3d(Poly3DCollection(
            [*triangles.detach().cpu().numpy()], linewidths=1, edgecolors='k', alpha=0.0))

    def get_global_rays(self, uv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the global ray origins and directions for the given pixel coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            Given pixel coordinates to get the ray origins for.
            Pixel coordinates are assumed to be in the image plane in order (x, y). [0, width) and [0, height)


        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The global ray origins and directions in the global coordinate system.
            As (..., 3) (x, y, z) tensor.
        """
        ray_directions = self._get_ray_direction(uv=uv)
        # Get endpoint of rays by multiplying with a distance
        ray_origins = ray_directions - \
            (torch.tensor([0, 0, 1], dtype=torch.float32))

        rd = (2 * ray_directions -
              (torch.tensor([0, 0, 1], dtype=torch.float32)))

        global_ray_origins = (torch.bmm(self.get_global_position().unsqueeze(0).repeat(
            ray_origins.shape[0], 1, 1), compose_transformation_matrix(position=ray_origins)))[..., :3, 3]

        global_ray_endpoints = (torch.bmm(self.get_global_position().unsqueeze(0).repeat(
            ray_directions.shape[0], 1, 1), compose_transformation_matrix(position=rd)))[..., :3, 3]
        global_ray_directions = global_ray_endpoints - global_ray_origins

        # Normalize the directions to unit length
        global_ray_directions = global_ray_directions / \
            torch.norm(global_ray_directions, dim=-1, keepdim=True)

        return global_ray_origins, global_ray_directions

    def _plot_ray_cast(self, ax: Axes,
                       sample_grid: Optional[torch.Tensor] = None,
                       **kwargs):
        if sample_grid is None:
            sample_grid = self._sample_pixel_grid(subsample=kwargs.get(
                "pixel_grid_subsample", 50), include_borders=kwargs.get("pixel_grid_borders", True))

        # ray_directions = flatten_batch_dims(
        #     self._get_ray_direction(uv=sample_grid), 1)[0]

        # # Get endpoint of rays by multiplying with a distance
        # ray_origins = ray_directions - \
        #     (torch.tensor([0, 0, 1], dtype=torch.float32))
        # global_ray_origins = (torch.bmm(self.get_global_position().unsqueeze(0).repeat(
        #     ray_origins.shape[0], 1, 1), compose_transformation_matrix(position=ray_origins)))[..., :3, 3]

        ray_distance = kwargs.get("ray_distance", 1)
        # rd = (2 * ray_directions -
        #       (torch.tensor([0, 0, 1], dtype=torch.float32)))

        # global_ray_endpoints = (torch.bmm(self.get_global_position().unsqueeze(0).repeat(
        #     ray_directions.shape[0], 1, 1), compose_transformation_matrix(position=rd)))[..., :3, 3]
        # global_ray_directions = global_ray_endpoints - global_ray_origins

        global_ray_origins, global_ray_directions = self.get_global_rays(
            sample_grid)

        self._plot_rays(global_ray_origins, global_ray_directions,
                        ax, ray_distance=ray_distance, **kwargs)

    def _plot_rays(self,
                   ray_origins: torch.Tensor,
                   ray_directions: torch.Tensor,
                   ax: Axes,
                   ray_distance: float = 1.5,
                   linewidth: float = 0.5,
                   arrow_length_ratio: float = 0,
                   color: Any = 'gray',
                   label: str = "Rays",
                   **kwargs):
        """Plot the given rays.

        Parameters
        ----------
        ray_origins : torch.Tensor
            The origins of the rays as ([... B,] 3) tensor. Should be in global coordinates (x, y, z).
        ray_directions : torch.Tensor
            The directions of the rays as ([... B,] 3) tensor. Should be in global coordinates (x, y, z).

        ax : Axes
            The axes to plot the rays on.

        ray_distance : float, optional
            The distance to plot the rays, by default 1.5

        linewidth : float, optional
            The width of the ray, by default 0.5

        arrow_length_ratio : float, optional
            The ratio of the arrow head wrt its tail, by default 0

        color : Any, optional
            The color of the rays, by default 'gray'
            Can be also of shape ([... B,] 4) (r, g, b, a) tensor.

        label : str, optional
            The label of the rays, by default "Rays"

        """
        import numpy as np
        from tools.util.numpy import flatten_batch_dims

        ray_origins, _ = flatten_batch_dims(ray_origins, -2)
        ray_directions, _ = flatten_batch_dims(ray_directions, -2)

        if isinstance(color, torch.Tensor):
            color = numpyify(color)
            color, _ = flatten_batch_dims(color, -2)

        # Assure that color is rgba
        if isinstance(color, np.ndarray) and color.shape[-1] == 3:
            color = np.concatenate(
                [color, np.ones_like(color[..., :1])], axis=-1)

        if isinstance(color, np.ndarray):
            color = color.clip(0, 1)
        ax.quiver(*ray_origins.detach().cpu().numpy().T, *ray_directions.detach().cpu().numpy().T,
                  length=ray_distance,
                  normalize=True,
                  arrow_length_ratio=arrow_length_ratio,
                  linewidths=linewidth,
                  color=color,
                  label=label
                  )

    @torch.no_grad()
    def plot_object(self, ax: Axes,
                    **kwargs):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        pixel_size = (1 / self._image_resolution).max().repeat(2)

        if not isinstance(pixel_size, torch.Tensor):
            pixel_size = torch.tensor(pixel_size, dtype=torch.float32)

        if kwargs.get("plot_coordinate_annotations", False) and self._name is not None:
            position = self.get_global_position_vector()
            if len(position.shape) == 1:
                ax.text(*position[:3], self._name,
                        horizontalalignment='center', verticalalignment='center')

        self._plot_camera_wireframe(ax, **kwargs)

        sample_grid = self._sample_pixel_grid(subsample=kwargs.get(
            "pixel_grid_subsample", 50), include_borders=kwargs.get("pixel_grid_borders", True))

        if kwargs.get("plot_pixel_grid", False):
            self._plot_pixel_grid(ax,
                                  sample_grid=sample_grid, **kwargs)

        if kwargs.get("plot_ray_cast", False):
            self._plot_ray_cast(ax, sample_grid=sample_grid, **kwargs)

        # endregion


def plot_rays(
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        ax: Axes,
        ray_distance: float = 1.5,
        linewidth: float = 0.5,
        arrow_length_ratio: float = 0,
        color: Any = 'gray',
        label: str = "Rays",
        **kwargs):
    """Plot the given rays.

    Parameters
    ----------
    ray_origins : torch.Tensor
        The origins of the rays as ([... B,] 3) tensor. Should be in global coordinates (x, y, z).
    ray_directions : torch.Tensor
        The directions of the rays as ([... B,] 3) tensor. Should be in global coordinates (x, y, z).

    ax : Axes
        The axes to plot the rays on.

    ray_distance : float, optional
        The distance to plot the rays, by default 1.5

    linewidth : float, optional
        The width of the ray, by default 0.5

    arrow_length_ratio : float, optional
        The ratio of the arrow head wrt its tail, by default 0

    color : Any, optional
        The color of the rays, by default 'gray'
        Can be also of shape ([... B,] 4) (r, g, b, a) tensor.

    label : str, optional
        The label of the rays, by default "Rays"

    """
    import numpy as np
    from tools.util.numpy import flatten_batch_dims

    ray_origins, _ = flatten_batch_dims(ray_origins, -2)
    ray_directions, _ = flatten_batch_dims(ray_directions, -2)

    if isinstance(color, torch.Tensor):
        color = numpyify(color)
        color, _ = flatten_batch_dims(color, -2)

    # Assure that color is rgba
    if isinstance(color, np.ndarray) and color.shape[-1] == 3:
        color = np.concatenate(
            [color, np.ones_like(color[..., :1])], axis=-1)

    if isinstance(color, np.ndarray):
        color = color.clip(0, 1)
    ax.quiver(*ray_origins.detach().cpu().numpy().T, *ray_directions.detach().cpu().numpy().T,
              length=ray_distance,
              normalize=True,
              arrow_length_ratio=arrow_length_ratio,
              linewidths=linewidth,
              color=color,
              label=label
              )
