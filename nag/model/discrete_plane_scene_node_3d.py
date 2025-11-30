from typing import Iterable, Optional

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from nag.model.camera_scene_node_3d import CameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D
from tools.model.discrete_module_scene_node_3d import DiscreteModuleSceneNode3D

import torch
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims, compose_transformation_matrix

from nag.transforms.transforms_timed_3d import interpolate_vector
from tools.transforms.geometric.transforms3d import vector_angle_3d, rotmat_from_vectors, _rotmat_from_vectors
from tools.transforms.geometric.mappings import rotmat_to_rotvec


@torch.jit.script
def local_to_plane_coordinates(
        coords: torch.Tensor,
        plane_scale: torch.Tensor,
        plane_offset: torch.Tensor
) -> torch.Tensor:
    """Converts local coordinates to plane coordinates.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates in object local space. Shape: ([... , B,] 2+)
        The last dimension is the coordinates in local space. (x, y, [z, ...])
    plane_scale : torch.Tensor
        The scale of the plane. (2, ) (x, y)
        Multiplier for x and y coordinates.

    plane_offset : torch.Tensor
        The offset of the plane. (2, ) (x, y)

    Returns
    -------
    torch.Tensor
        Coordinates in plane space. Shape: ([... , B,] 2)
    """
    # Remove batch dims
    coords, batch_shape = flatten_batch_dims(coords, -2)
    coords_xy = coords[..., :2]

    res = coords_xy / plane_scale - plane_offset

    # Restore batch dims
    return unflatten_batch_dims(res, batch_shape)


@torch.jit.script
def batched_local_to_plane_coordinates(
        coords: torch.Tensor,
        plane_scale: torch.Tensor,
        plane_offset: torch.Tensor
) -> torch.Tensor:
    """Converts local coordinates to plane coordinates.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates in object local space. Shape: ([... , B,] O, 2+)
        The last dimension is the coordinates in local space. (x, y, [z, ...])
    plane_scale : torch.Tensor
        The scale of the planes. ([O,] 2) (x, y)
        Multiplier for x and y coordinates.

    plane_offset : torch.Tensor
        The offset of the planes. ([O,] 2) (x, y)

    Returns
    -------
    torch.Tensor
        Coordinates in plane space. Shape: ([... , B,] 2)
    """
    # Remove batch dims
    coords, batch_shape = flatten_batch_dims(coords, -3)
    plane_scale = flatten_batch_dims(plane_scale, -2)[0]
    plane_offset = flatten_batch_dims(plane_offset, -2)[0]

    coords_xy = coords[..., :2]

    res = coords_xy / plane_scale - plane_offset

    # Restore batch dims
    return unflatten_batch_dims(res, batch_shape)


@torch.jit.script
def plane_coordinates_to_local(coords: torch.Tensor, plane_scale: torch.Tensor, plane_offset: torch.Tensor) -> torch.Tensor:
    """Transforms plane coordinates to local coordinates.

    Applies a linear transformation to the plane coordinates to get local coordinates.
    Will multiply the x and y coordinates with the plane scale and add the plane scale offset.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates in plane space.
        Plane space is defined in the range [0, 1] for x and y.
        Shape: ([... , B,] 2)

    plane_scale : torch.Tensor
        The scale of the plane. (2, ) (x, y)
        Multiplier for x and y coordinates.

    plane_offset : torch.Tensor
        The offset of the plane. (2, ) (x, y)

    Returns
    -------
    torch.Tensor
        Local coordinates in object space. Shape: ([... , B,] 3) (x, y, z) Where z is 0 (=On the plane).
    """
    # Remove batch dims
    coords, batch_shape = flatten_batch_dims(coords, -2)

    coords_xy = coords[..., :2]

    res = (coords_xy + plane_offset) * plane_scale

    # Add z=0
    res = torch.cat(
        [res, torch.zeros_like(res[..., :1])], dim=-1)

    # Restore batch dims
    return unflatten_batch_dims(res, batch_shape)


@torch.jit.script
def batched_plane_coordinates_to_local(coords: torch.Tensor, plane_scale: torch.Tensor, plane_offset: torch.Tensor) -> torch.Tensor:
    """Transforms plane coordinates to local coordinates. For O planes.

    Applies a linear transformation to the plane coordinates to get local coordinates.
    Will multiply the x and y coordinates with the plane scale and add the plane scale offset.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates in plane space.
        Plane space is defined in the range [0, 1] for x and y.
        Shape: ([... , B,] O, 2)

    plane_scale : torch.Tensor
        The scale of the planes. ([O,] 2) (x, y)
        Multiplier for x and y coordinates.

    plane_offset : torch.Tensor
        The offset of the planes. ([O,] 2) (x, y)

    Returns
    -------
    torch.Tensor
        Local coordinates in object space. Shape: ([... , B,] O, 3) (x, y, z) Where z is 0 (=On the plane).
    """
    # Remove batch dims
    coords, batch_shape = flatten_batch_dims(coords, -3)
    plane_scale = flatten_batch_dims(plane_scale, -2)[0]
    plane_offset = flatten_batch_dims(plane_offset, -2)[0]

    coords_xy = coords[..., :2]

    res = (coords_xy + plane_offset) * plane_scale

    # Add z=0
    res = torch.cat(
        [res, torch.zeros_like(res[..., :1])], dim=-1)

    # Restore batch dims
    return unflatten_batch_dims(res, batch_shape)


def default_plane_scale(dtype: torch.dtype) -> torch.Tensor:
    return torch.ones(2, dtype=dtype)


def default_plane_scale_offset(dtype: torch.dtype) -> torch.Tensor:
    xy = torch.zeros(2, dtype=dtype)
    xy[...] = -0.5
    return xy


@torch.jit.script
def compute_incline_angle(
    ray_directions: torch.Tensor,
    plane_normal: torch.Tensor
) -> torch.Tensor:
    """
    Computes the incline angle of the ray directions, w.r.t the plane normal definition.

    Both ray_directions and plane_normal should be in the same coordinate system.

    Parameters
    ----------
    ray_directions : torch.Tensor
        The ray directions. Shape: ([..., B,] 3)

    plane_normal : torch.Tensor
        The plane normal. Shape: ([..., B,] 3)

    Returns
    -------
    torch.Tensor
        The incline angles as rotation vectors. Shape: ([..., B,] 3)
        Values are in the range [-pi, pi]
        Wherby the last compontent is 0.
    """
    ray_directions, shp = flatten_batch_dims(ray_directions, -2)
    plane_normal, _ = flatten_batch_dims(plane_normal, -2)

    B, _ = ray_directions.shape
    if plane_normal.shape[0] != B:
        if plane_normal.shape[0] == 1:
            plane_normal = plane_normal.repeat(B, 1)
        else:
            raise ValueError(
                f"plane_normal must have shape (1, 3) or ({B}, 3), but got {plane_normal.shape}")

    # Normalize the ray directions
    ray_directions = ray_directions / \
        ray_directions.norm(p=2, dim=-1, keepdim=True)
    plane_normal = plane_normal / plane_normal.norm(p=2, dim=-1, keepdim=True)
    # Compute the angle between the ray directions and the plane normal
    R = _rotmat_from_vectors(plane_normal, ray_directions)
    rot_vecs = rotmat_to_rotvec(R)
    return unflatten_batch_dims(rot_vecs, shp)


class DiscretePlaneSceneNode3D(DiscreteModuleSceneNode3D):
    """A scene node representing a plane in 3D space with discrete positions, orientation and implicit plane / coordinate scaling.
    """

    _plane_scale: torch.Tensor
    """An optional scale vector (2, ) (xy) for the scale of the plane. Used to form a linear transformation for plane coordinates."""

    _plane_scale_offset: torch.Tensor
    """An optional scale offset vector (2, ) (xy) for the scale of the plane.
    Used to form a linear transformation for plane coordinates.
    Plane coordinates are defined in the range [0, 1] for x and y.
    Default is (-0.5, -0.5) so plane lower left will be 0, 0 instead of the plane center."""

    def __init__(
        self,
            plane_scale: Optional[VEC_TYPE] = None,
            plane_scale_offset: Optional[VEC_TYPE] = None,
            translation: Optional[VEC_TYPE] = None,
            orientation: Optional[VEC_TYPE] = None,
            position: Optional[torch.Tensor] = None,
            name: Optional[str] = None,
            children: Optional[Iterable['AbstractSceneNode']] = None,
            decoding: bool = False,
            dtype: torch.dtype = torch.float32,
            _plane_scale: Optional[torch.Tensor] = None,
            _plane_scale_offset: Optional[torch.Tensor] = None,
            **kwargs
    ):
        super().__init__(
            translation=translation,
            orientation=orientation,
            position=position,
            name=name,
            children=children,
            decoding=decoding,
            dtype=dtype,
            **kwargs)
        # Set plane scale and offset
        self.set_scale(plane_scale, plane_scale_offset, dtype,
                       _plane_scale, _plane_scale_offset)

    def set_scale(self,
                  plane_scale: Optional[VEC_TYPE] = None,
                  plane_scale_offset: Optional[VEC_TYPE] = None,
                  dtype: torch.dtype = torch.float32,
                  _plane_scale: Optional[torch.Tensor] = None,
                  _plane_scale_offset: Optional[torch.Tensor] = None,
                  ):
        if _plane_scale is not None:
            self.register_buffer(
                "_plane_scale", _plane_scale, persistent=False)
            self.register_buffer("_plane_scale_offset",
                                 _plane_scale_offset, persistent=False)
        else:
            if plane_scale is None:
                plane_scale = default_plane_scale(dtype)
            if plane_scale_offset is None:
                plane_scale_offset = default_plane_scale_offset(dtype)
            self.register_buffer(
                "_plane_scale", tensorify(plane_scale, dtype=dtype, device=self._translation.device))
            self.register_buffer("_plane_scale_offset", tensorify(
                plane_scale_offset, dtype=dtype, device=self._translation.device))

    def get_plane_scale(self) -> torch.Tensor:
        return self._plane_scale

    def get_plane_scale_offset(self) -> torch.Tensor:
        return self._plane_scale_offset

    def set_plane_scale(self, scale: VEC_TYPE):
        """Sets the scale of the plane.

        The scale is used to form a linear transformation for plane coordinates.

        Parameters
        ----------
        scale : VEC_TYPE
            Scale of the plane as a vector (2, ) (xy).

        Raises
        ------
        ValueError
            If the scale is not of shape (2, ).
        """
        if scale is None:
            scale = self._get_default_plane_scale(self._translation.dtype)
        # Ensure that the scale is a tensor
        scale = tensorify(scale, dtype=self._translation.dtype,
                          device=self._translation.device)
        # Ensure that the scale has the correct shape
        if scale.shape != (2, ):
            raise ValueError(
                f"Plane scale must be of shape (2, ) but got {scale.shape}")
        self._plane_scale = scale

    def set_plane_scale_offset(self, offset: VEC_TYPE):
        if offset is None:
            offset = self._get_default_plane_scale_offset(
                self._translation.dtype)
        # Ensure that the offset is a tensor
        offset = tensorify(offset, dtype=self._translation.dtype,
                           device=self._translation.device)
        # Ensure that the offset has the correct shape
        if offset.shape != (2, ):
            raise ValueError(
                f"Plane scale offset must be of shape (2, ) but got {offset.shape}")
        self._plane_scale_offset = offset

    def get_plane_surface(self) -> torch.Tensor:
        """Gets the surface size of the plane in local coordinates.

        Returns
        -------
        torch.Tensor
            The surface size of the plane in local coordinates.
            Shape: (1, )
        """
        return self._plane_scale.prod()

    def plane_coordinates_to_local(self, coords: torch.Tensor) -> torch.Tensor:
        """Transforms plane coordinates to local coordinates.

        Applies a linear transformation to the plane coordinates to get local coordinates.
        Will multiply the x and y coordinates with the plane scale and add the plane scale offset.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates in plane space.
            Plane space is defined in the range [0, 1] for x and y.
            Shape: ([... , B,] 2)

        Returns
        -------
        torch.Tensor
            Local coordinates.
            Shape: ([... , B,] 3)
        """
        # Get scale and offset
        xy_scale = self.get_plane_scale()
        xy_offset = self.get_plane_scale_offset()
        return plane_coordinates_to_local(coords, xy_scale, xy_offset)

    def local_to_plane_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """Transforms local coordinates to plane coordinates.

        Applies a linear transformation to the local coordinates to get plane coordinates.
        Will divide the x and y coordinates with the plane scale and subtract the plane scale offset.

        Parameters
        ----------
        coords : torch.Tensor
            Local coordinates.
            Shape: ([... , B,] 3)

        Returns
        -------
        torch.Tensor
            Coordinates in plane space.
            Plane space is defined in the range [0, 1] for x and y.
            Shape: ([... , B,] 2)
        """
        # Get scale and offset
        xy_scale = self.get_plane_scale()
        xy_offset = self.get_plane_scale_offset()
        return local_to_plane_coordinates(coords, xy_scale, xy_offset)

    def get_plane_coordinate_corners(self) -> torch.Tensor:
        """Gets the corners of the plane in plane coordinates.

        Returns
        -------
        torch.Tensor
            Corners of the plane in plane coordinates.
            Shape: (4, 2)
            Points are ordered as bottom left, bottom right, top right, top left. (Counter clockwise from bottom left)
        """
        bl = torch.tensor([0, 0], dtype=self._translation.dtype,
                          device=self._translation.device)
        br = torch.tensor([1, 0], dtype=self._translation.dtype,
                          device=self._translation.device)
        tr = torch.tensor([1, 1], dtype=self._translation.dtype,
                          device=self._translation.device)
        tl = torch.tensor([0, 1], dtype=self._translation.dtype,
                          device=self._translation.device)
        points = torch.stack([bl, br, tr, tl], dim=0)
        return points

    def get_plane_corners(self) -> torch.Tensor:
        """
        Gets the corners of the plane in local coordinates.

        Returns
        -------
        torch.Tensor
            Corners of the plane in local coordinates.
            Shape: (4, 4)
            Points are ordered as bottom left, bottom right, top right, top left. (Counter clockwise from bottom left)
        """
        points = self.get_plane_coordinate_corners()
        # Add z=0
        res = self.plane_coordinates_to_local(points)

        # Add w=1
        res = torch.cat(
            [res, torch.ones(4, 1, dtype=self._translation.dtype, device=self._translation.device)], dim=-1)
        return res

    def get_plane_edge_lines(self, num_points: int = 100) -> torch.Tensor:
        """Gets the edge lines of the plane in local coordinates.

        Parameters
        ----------
        num_points : int, optional
            Number of points to sample on each edge, by default 100

        Returns
        -------
        torch.Tensor
            Edge lines of the plane in local coordinates.
            Shape: (4, num_points, 4)
            Points are ordered as bottom left, bottom right, top right, top left. (Counter clockwise from bottom left)
            The line segments are:
            bl -> br
            br -> tr
            tr -> tl
            tl -> bl
            While the first point is the actual corner point, the rest of the points are sampled in between.
        """
        corners = self.get_plane_coordinate_corners()
        res = torch.zeros(
            (4, num_points, 2), dtype=self._translation.dtype, device=self._translation.device)
        dist = torch.arange(
            0, 1, 1/num_points, dtype=self._translation.dtype, device=self._translation.device)
        if torch.isclose(dist[-1], torch.tensor(1.)) and len(dist) == num_points + 1:
            # This is a weird bug.
            dist = dist[:-1]
        for i in range(4):
            start = i
            end = (i + 1) % 4
            start_point = corners[start]
            end_point = corners[end]
            vec = end_point - start_point
            res[i, :, :2] = start_point + vec * \
                dist[:, None].expand(num_points, 2)
        return res

    def get_local_plane_edge_lines(self, num_points: int = 100) -> torch.Tensor:
        lines = self.get_plane_edge_lines(num_points=num_points)
        # Add z=0
        res = self.plane_coordinates_to_local(lines)

        # Add w=1
        res = torch.cat(
            [res, torch.ones(4, num_points, 1, dtype=self._translation.dtype, device=self._translation.device)], dim=-1)
        return res

    def get_global_plane_edge_lines(self, num_points: int = 100) -> torch.Tensor:
        """Gets the edge lines of the plane in global coordinates.

        Parameters
        ----------
        num_points : int, optional
            Number of points to sample on each edge, by default 100

        Returns
        -------
        torch.Tensor
            Edge lines of the plane in global coordinates.
            Shape: (4, num_points, 4)
            Points are ordered as bottom left, bottom right, top right, top left. (Counter clockwise from bottom left)
            The line segments are:
            bl -> br
            br -> tr
            tr -> tl
            tl -> bl
            While the first point is the actual corner point, the rest of the points are sampled in between.
        """
        lines = self.get_local_plane_edge_lines(num_points=num_points)
        return self.local_to_global(lines)

    def get_global_plane_corners(self) -> torch.Tensor:
        """Gets the corners of the plane in global coordinates.

        Returns
        -------
        torch.Tensor
            Corners of the plane in global coordinates.
            Shape: (4, 4)
            Points are ordered as bottom left, bottom right, top right, top left. (Counter clockwise from bottom left)
        """
        corners = self.get_plane_corners()
        return self.local_to_global(corners)

    # region Plotting

    def _plot_plane(self,
                    ax: Axes,
                    facecolor: str = 'white',
                    edgecolor: str = 'black',
                    alpha: float = 0.3,
                    line_width: float = 1.0,
                    **kwargs):
        """Plots the corners of the plane on the given axis.

        Parameters
        ----------
        ax : Axes
            The axis to plot on.
        corners : torch.Tensor
            The corners of the plane to plot.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        global_corners = self.get_global_plane_corners()
        # Plot the corners
        ax.add_collection3d(
            Poly3DCollection(
                [global_corners[..., :3].numpy(
                )], facecolors=facecolor, linewidths=line_width, edgecolors=edgecolor, alpha=alpha))

    def plot_object(self, ax: Axes, **kwargs):
        fig = super().plot_object(ax, **kwargs)
        self._plot_plane(ax, **kwargs)

    # endregion
