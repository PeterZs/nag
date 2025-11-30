from typing import Iterable, List, Optional, Any, Union, Callable

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from nag.model.camera_scene_node_3d import CameraSceneNode3D
from nag.model.discrete_plane_scene_node_3d import DiscretePlaneSceneNode3D
from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D

import torch
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.viz.matplotlib import saveable
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims, compose_transformation_matrix

from nag.transforms.transforms_timed_3d import interpolate_vector
from matplotlib.colors import Colormap
from tools.util.typing import _DEFAULT, DEFAULT, REAL_TYPE


class TimedPlaneSceneNode3D(DiscretePlaneSceneNode3D, TimedDiscreteSceneNode3D):
    """A scene node representing a camera in 3D space with timed discrete position and orientation."""

    def __init__(
        self,
            plane_scale: Optional[VEC_TYPE] = None,
            plane_scale_offset: Optional[VEC_TYPE] = None,
            translation: Optional[VEC_TYPE] = None,
            orientation: Optional[VEC_TYPE] = None,
            position: Optional[torch.Tensor] = None,
            times: Optional[VEC_TYPE] = None,
            name: Optional[str] = None,
            children: Optional[Iterable['AbstractSceneNode']] = None,
            decoding: bool = False,
            dtype: torch.dtype = torch.float32,
            **kwargs
    ):
        super().__init__(
            plane_scale=plane_scale,
            plane_scale_offset=plane_scale_offset,
            translation=translation,
            orientation=orientation,
            position=position,
            times=times,
            name=name,
            children=children,
            decoding=decoding,
            dtype=dtype,
            **kwargs)

    def get_global_plane_corners(self,
                                 t: Optional[torch.Tensor] = None,
                                 **kwargs) -> torch.Tensor:
        """Gets the corners of the plane in global coordinates.

        Returns
        -------
        torch.Tensor
            Corners of the plane in global coordinates.
            Shape: (4, T, 4)
            Points are ordered as bottom left, bottom right, top right, top left. (Counter clockwise from bottom left)
            (x, y, z, 1) coordinates.
        """
        corners = self.get_plane_corners()
        return self.local_to_global(corners, t=t)

    def get_global_plane_edge_lines(self,
                                    num_points: int = 100,
                                    t: Optional[torch.Tensor] = None,
                                    **kwargs) -> torch.Tensor:
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
        return self.local_to_global(lines, t=t)

    def get_visible_plane_image_corners(self, camera: Any, t: VEC_TYPE) -> torch.Tensor:
        """
        Get the visible corners of the plane in image space. The corners are clamped to the image domain.
        The corners are ordered as [bl, br, tr, tl] unless the plane was rotated.


        Parameters
        ----------
        camera : Any
            The camera object used to project the corners into image space.
        t : VEC_TYPE
            The time at which to get the image corners.
            Shape (...)

        Returns
        -------
        torch.Tensor
            The image corners of the plane in image space.
            Shape (B, 3) of (T, x, y) where T indicates the time idx. For each time idx may return 4 to 8 points,
            depending on the visibility of plane corners, or plane intersection with image domain.
            The points are ordered as [bl, br, tr, tl], eventually filling intersection points in between.

            Returns a tensor of size 0 if no points are inside.

        """
        t = tensorify(t, dtype=self._translation.dtype,
                      device=self._translation.device)
        t, shp = flatten_batch_dims(t, -1)

        def cross_2d(v1, v2):
            """Computes the 2D cross product (scalar value) of two 2D vectors."""
            return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]

        def is_point_in_triangle(point, triangle, eps=1e-6):
            """
            Checks if a point lies inside a triangle using PyTorch.

            Args:
                point (torch.Tensor): A tensor of shape (2,) representing the (x, y) coordinates of the point.
                triangle (torch.Tensor): A tensor of shape (3, 2) representing the (x, y) coordinates of the triangle vertices.

            Returns:
                torch.Tensor: A boolean tensor of shape (1,) indicating whether the point is inside the triangle.
            """
            # Extract triangle vertices
            v0 = triangle[0]
            v1 = triangle[1]
            v2 = triangle[2]

            # Compute vectors
            v0p = point - v0
            v1p = point - v1
            v2p = point - v2

            # Compute cross products
            cp1 = cross_2d(v1 - v0, v0p)
            cp2 = cross_2d(v2 - v1, v1p)
            cp3 = cross_2d(v0 - v2, v2p)

            # Check if point is on the same side of all edges
            return ((cp1 >= (0 - eps) and cp2 >= (0 - eps) and cp3 >= (0 - eps)) or (cp1 <= (0 + eps) and cp2 <= (0 + eps) and cp3 <= (0 + eps))).unsqueeze(0)

        T = len(t)
        global_corners = self.get_global_plane_corners(
            t=t).detach()  # bl, br, tr, tl
        cam_image_corners, oks_all = camera.global_to_image_coordinates(
            global_corners, t=t, v_includes_time=True, return_ok=True, clamp=False)
        cam_image_corners = cam_image_corners[..., :2].detach().cpu()
        # If all are ok we are in the image complety
        if oks_all.all():
            cam_image_corners = cam_image_corners.permute(1, 0, 2)  # (T, 4, 2)
            batch_idx = torch.arange(cam_image_corners.shape[0], device=cam_image_corners.device).unsqueeze(
                -1).repeat(1, 4).unsqueeze(-1)  # (T, 4, 1)
            _cat = torch.cat([batch_idx, cam_image_corners],
                             dim=-1)  # (T, 4, 3)
            return _cat.reshape(T * 4, 3)

        # If not we need to find the crossing points of image domain and plane edges
        # Get ruff scale in image space
        x_max = cam_image_corners[..., 0].max()
        x_min = cam_image_corners[..., 0].min()
        y_max = cam_image_corners[..., 1].max()
        y_min = cam_image_corners[..., 1].min()
        max_scale = max((x_max - x_min).max(), (y_max - y_min).max())

        edge_lines = self.get_global_plane_edge_lines(
            t=t, num_points=int(max_scale * 2.2)).detach()
        cam_edge_lines_all, oks_lines_all = camera.global_to_image_coordinates(
            edge_lines, t=t, v_includes_time=True, return_ok=True)
        cam_edge_lines_all = cam_edge_lines_all[..., :2].detach().cpu()

        cam_image_corners = cam_image_corners.permute(1, 0, 2)

        cam_points = torch.zeros(
            4, 2, device=cam_image_corners.device, dtype=cam_image_corners.dtype)
        # cam_points[0] = cam_image_corners[0, 0]  # bl => 0, 0
        cam_points[1] = torch.tensor(
            [camera._image_resolution[1] - 1, 0])  # br => width - 1, 0
        # tr => width - 1, height - 1
        cam_points[2] = torch.tensor(
            [camera._image_resolution[1] - 1, camera._image_resolution[0] - 1])
        cam_points[3] = torch.tensor(
            [0, camera._image_resolution[0] - 1])  # tl => 0, height - 1

        contains_plane_corner = torch.zeros(
            (T, 4), device=cam_image_corners.device, dtype=torch.bool)

        bound_points = []
        for i in range(T):
            oks = oks_all[:, i]
            cam_edge_lines = cam_edge_lines_all[:, :, i]  # (4, N, 2)

            oks_lines = oks_lines_all[:, :, i]  # (4, N)

            # TODO this is missing the case where the plane forms a pentagon due to beeing tilted and partially visible within an edge

            bl_ok, br_ok, tr_ok, tl_ok = oks
            for j, edge_line_oks in enumerate(oks_lines):
                # Edge Lines are:
                # 0: bl => br
                # 1: br => tr
                # 2: tr => tl
                # 3: tl => bl
                # Get changes withing edgelkines ok, marking intersections with the image domain
                def test_corner_in_triangle():
                    p0 = cam_image_corners[i, j]
                    p1 = cam_image_corners[i, (j + 1) % 4]
                    p2 = cam_image_corners[i, (j + 2) % 4]
                    tri = torch.stack([p0, p1, p2], dim=0)
                    for k in range(4):
                        if contains_plane_corner[i, k]:
                            continue
                        else:
                            p = cam_points[k]
                            if is_point_in_triangle(p, tri):
                                contains_plane_corner[i, k] = True
                                bound_points.append(
                                    torch.cat([torch.tensor([i], dtype=p.dtype, device=p.device), p]))

                # If actual corner is in the image, we can use it directly
                if oks[j]:
                    coord = cam_image_corners[i, j]
                    bound_points.append(
                        torch.cat([torch.tensor([i], dtype=coord.dtype, device=coord.device), coord]))

                change_points = edge_line_oks.diff().argwhere().squeeze(-1)
                # Add all change points
                for k, pt in enumerate(change_points):
                    coord = cam_edge_lines[j, pt]
                    bound_points.append(
                        torch.cat([torch.tensor([i], dtype=coord.dtype, device=coord.device), coord]))

                if ~edge_line_oks.all():
                    # If not completely in check if we missing a corner
                    test_corner_in_triangle()

                    # is_exit_point = ~edge_line_oks[change_points + 1]
        if len(bound_points) == 0:
            # No intersection points found
            return torch.zeros(0, 3, dtype=t.dtype, device=t.device)
        stacked = torch.stack(bound_points, dim=-2)
        im_domain = torch.tensor([[0, 0], [camera._image_resolution[1] - 1,
                                 camera._image_resolution[0] - 1]]).to(stacked.device) + 0.5
        stacked[..., -2] = torch.clamp(
            stacked[..., -2], min=im_domain[0, 0], max=im_domain[1, 0])
        stacked[..., -1] = torch.clamp(
            stacked[..., -1], min=im_domain[0, 1], max=im_domain[1, 1])
        return stacked

    def get_occluded_image_space(self,
                                 camera: Any,
                                 t: VEC_TYPE) -> torch.Tensor:
        """Gets the occluded image space of the plane in image space.

        Returns a mask in image space of the visible area of the plane.

        Parameters
        ----------
        camera : Any
            Camera to project the plane on.
        t : VEC_TYPE
            Time to project the plane at. Shape (...)

        Returns
        -------
        torch.Tensor
            Occluded image space of the plane in image space.
            Shape: (..., 1, height, width)
            The mask is True for the occluded area and False for the visible area.
        """
        import cv2
        import numpy as np
        t = tensorify(t, dtype=self._translation.dtype,
                      device=self._translation.device)
        t, shp = flatten_batch_dims(t, -1)

        corners = self.get_visible_plane_image_corners(camera=camera, t=t)
        space = torch.zeros(
            (len(t), 1, camera._image_resolution[0], camera._image_resolution[1]), dtype=torch.bool, device=corners.device)
        for i in range(len(t)):
            corner = corners[corners[..., 0] == i][..., 1:].cpu().numpy()
            mask = cv2.fillPoly(space[i, 0].cpu().numpy().astype(
                np.float32), [corner.round().astype(int)], 1.) > 1 / 2
            space[i, 0] = torch.from_numpy(mask).to(corners.device)
        return unflatten_batch_dims(space, shp)

    def get_smoothed_occluded_image_space(self, cam: Any, t: VEC_TYPE, fnc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], is_close_bounds_threshold: Optional[int] = 5) -> torch.Tensor:
        """Gets the occluded image space and applies a smoothing function to it.

        The smoothing function is applied to the distance from each edge.
        And should accept 2 arguments, the distance to the edge and a boolean mask of the image edges.

        1. torch.Tensor
            Distance to the edge Shape (B, 4)
            Second dimension indicates edges:
            0. bl => br
            1. br => tr
            2. tr => tl
            3. tl => bl
        2. torch.Tensor
            If its a close to bounds edge Shape (4, )

        The function should return a tensor of the same shape as the input. Containing values between 0 and 1.

        Parameters
        ----------
        cam : Any
            Camera to be projected on.
        t : VEC_TYPE
            Time to project the plane at. Shape (...)
        fnc : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The function to smooth.

        is_close_bounds_threshold : int, optional
            From when on the image should be treated as close to an image border, by default 20
            Values are in pixels. None means no threshold.

        Returns
        -------
        torch.Tensor
            Image space of the occluded area with the smoothing function applied.
            Shape (..., 1, H, W)
        """
        from tools.transforms.geometric.transforms2d import orthogonal_vector, assure_homogeneous_vector
        t = tensorify(t, dtype=self._translation.dtype,
                      device=self._translation.device)
        t, shp = flatten_batch_dims(t, -1)

        corners = self.get_visible_plane_image_corners(
            cam, t=t)  # (T, 4, 2) bl, br, tr, tl

        space = self.get_occluded_image_space(cam, t=t)  # (T, 1, H, W)
        T, _, H, W = space.shape

        combined = torch.zeros_like(space).float()

        for tidx in range(T):

            cur_corners = corners[corners[..., 0] == 0][..., 1:]
            N, _ = cur_corners.shape
            edgeidx = torch.arange(
                len(cur_corners), device=space.device)  # (N)

            rvecs = torch.roll(cur_corners, shifts=-1,
                               dims=-2) - cur_corners  # (N, 2)
            rvecs /= torch.norm(rvecs, dim=-1, keepdim=True)  # (N, 2)

            # Get the orthogonal inwards normals
            # (N, 2, 2) # Both candidate normals
            normals = torch.zeros_like(
                cur_corners).unsqueeze(-2).repeat(1, 2, 1)
            normals[..., 0, :] = orthogonal_vector(rvecs)
            normals[..., 1, :] = -orthogonal_vector(rvecs)
            normals /= torch.norm(normals, dim=-1, keepdim=True)  # (N, 2, 2)

            # Correct normals shall point inwards, eg. same direction as the next line segment
            rolled_vecs = torch.roll(
                rvecs, shifts=-1, dims=-2).unsqueeze(-2).repeat(1, 2, 1)  # (N, 2, 2)
            rolled_vecs_picked = rolled_vecs.clone()

            # For the case where we have just a triangle, as its almost out of the image, we need to check the next line segment
            cur_dot = (
                rvecs * rolled_vecs_picked[:, :, 0]).sum(dim=-1)  # (N, )
            is_close = torch.isclose(
                cur_dot, torch.tensor([1.0]), atol=1e-3)  # (N,)
            is_close_temp = torch.zeros_like(is_close)
            is_close_idx = edgeidx[is_close]

            rolled = 1
            while is_close.any():
                rolled += 1
                close_rvecs = rvecs[is_close]  # (N, 2)
                rolled_again = torch.roll(
                    rvecs, shifts=-rolled, dims=-2)  # (N, 2)
                # (N, 2)
                dot = (close_rvecs * rolled_again[is_close]).sum(dim=-1)
                cc = torch.isclose(dot, torch.tensor([1.0]), atol=1e-3)  # (N)
                found = is_close_temp.clone()
                found_idx = is_close_idx[~cc]
                found[found_idx[..., 0], found_idx[..., 1]] = True
                rolled_vecs_picked[found] = rolled_again[found]  # (N, 2)
                is_close[found_idx[..., 0], found_idx[..., 1]] = False
                is_close_idx = edgeidx[is_close]  # (N, 2)

            # Compute dot product, if eq 1 then the normals are in the same direction
            dot = (normals * rolled_vecs_picked).sum(dim=-1)  # (N, 2)
            # Having the largest dot product
            select = dot.argmax(dim=-1, keepdim=True)

            sel_idx = torch.stack(
                [edgeidx, select.squeeze(-1)], dim=-1)  # (N, 2)

            # (N, 2, 2) -> (N, 2) # Select the correct normals
            normal_select = normals[sel_idx[..., 0],
                                    sel_idx[..., 1]]
            mat = torch.eye(3)[None].repeat(N, 1, 1)  # (N, 3, 3)
            mat[..., :2, 0] = rvecs
            mat[..., :2, 1] = normal_select
            mat[..., :2, 2] = cur_corners

            # Get if edgepoint is close to the image border
            if is_close_bounds_threshold is None:
                is_image_edge = torch.zeros_like(
                    cur_corners[..., 0], dtype=torch.bool)  # (N, )
            else:
                corners_small = ((cur_corners - is_close_bounds_threshold)
                                 < 0.)  # (N, 2)
                corners_large = ((cur_corners + is_close_bounds_threshold) >
                                 torch.tensor([W, H], device=cur_corners.device))  # (N, 2)
                corners_close_bounds = corners_small | corners_large  # (N, 2)
                corners_close_bounds_next = torch.roll(
                    corners_close_bounds, shifts=-1, dims=-2)  # (N, 2)
                # (N, )
                # (N, ) # If current and next points are close to the same image edge
                is_image_edge = (corners_close_bounds &
                                 corners_close_bounds_next).any(dim=-1)

            # Mat should get us into a coordinate system of orthogonal vectors w.r.t each line where we can apply the function
            aw = space[tidx, 0].argwhere().flip(-1)  # (H * W occl, 2)
            B, _ = aw.shape
            cmat = mat.inverse()  # (N, 3, 3)
            aw_affine = assure_homogeneous_vector(aw).to(cmat.dtype)
            coord_trans = torch.bmm(
                cmat.unsqueeze(0).repeat(B, 1, 1, 1).reshape(B * N, 3, 3),
                aw_affine.unsqueeze(-1).unsqueeze(1).repeat(1,
                                                            N, 1, 1).reshape(B * N, 3, 1)
            ).squeeze(-1).reshape(B, N, 3)[..., :2]  # (B, N, 2) # (x, y, 1) -> (x, y)
            # y cooresponds to the orthogonal vector of the line, x to the line itself
            # Call fnc with y-coordinates
            out_values = torch.zeros(N, H, W, device=space.device)
            out_values[:, aw[:, 1], aw[:, 0]] = fnc(
                coord_trans[..., 1], is_image_edge).permute(1, 0)  # B, N, 1 -> B, N, 1

            combined[tidx] = out_values.prod(
                dim=0, keepdim=True)  # (T, 1, H, W)
        return unflatten_batch_dims(combined, shp)

    # region Plotting

    @saveable()
    def plot_2d_projection(self,
                           camera: Any,
                           ax: Optional[Axes] = None,
                           t: Optional[NUMERICAL_TYPE] = None,
                           color: Union[List[Any], Any, _DEFAULT] = DEFAULT,
                           corner_size: float = 1.0,
                           image_resolution: Optional[VEC_TYPE] = None,
                           connect_corners: bool = True,
                           plot_name: bool = True,
                           zorder: Optional[int] = None,
                           cmap: Union[str, Colormap] = 'tab10',
                           **kwargs) -> plt.Figure:
        """Plots the plane in 2D projection.

        Parameters
        ----------
        ax : Axes
            Axes to plot on.

        camera : TimedCameraSceneNode3D
            Camera to project the box on.

        t : Optional[NUMERICAL_TYPE], optional
            Time to project the box at, by default None. If None, the last time is used.

        image_resolution : Optional[Tuple[int, int]], optional
            Resolution of the image to project on, by default None. If None, the camera resolution is used.
            Shape: (height, width)

        """
        from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
        from tools.util.numpy import numpyify
        from tools.viz.matplotlib import get_mpl_figure
        from matplotlib.colors import to_rgba
        if not isinstance(cmap, Colormap):
            cmap = plt.get_cmap(cmap)

        def get_color(color: Union[str, _DEFAULT], oid: Any, cmap: Colormap) -> str:
            if color is DEFAULT:
                if isinstance(oid, str):
                    if oid.isnumeric():
                        return int(oid)
                if not isinstance(oid, int):
                    self.logger.warning(
                        f"oid is not a number: {oid}. Using default id(self) instead.")
                    oid = id(oid)
                return to_rgba(cmap(oid % cmap.N))
            else:
                color = to_rgba(color)
            return color
        local_zorder = 0

        def get_zorder() -> Optional[int]:
            nonlocal local_zorder
            if zorder is None:
                return None
            ord = zorder + local_zorder
            local_zorder += 1
            return ord

        if t is None:
            t = self._times[-1]
        else:
            t = tensorify(t, dtype=self._times.dtype,
                          device=self._times.device)
        t, shp = flatten_batch_dims(t, -1)
        if image_resolution is None:
            image_resolution = camera._image_resolution
        else:
            image_resolution = tensorify(
                image_resolution, device=self._times.device)
        if len(image_resolution) != 2:
            raise ValueError(f"Invalid image resolution: {image_resolution}")
        color = get_color(DEFAULT if color is None else color,
                          self.get_index(), cmap)

        camera: TimedCameraSceneNode3D

        if ax is None:
            fig, ax = get_mpl_figure(
                1, 1, ratio_or_img=camera._image_resolution[0] / camera._image_resolution[1])
        else:
            fig = ax.figure

        cam_image_corners = self.get_visible_plane_image_corners(
            camera=camera, t=t)

        for i in range(len(t)):
            coords = cam_image_corners[cam_image_corners[..., 0] == i][..., 1:]
            # All corners lying on the same line
            not_visible = (coords.diff(dim=-2) == 0).all(dim=-2).any()
            if ~not_visible:
                # If image resolution != camera resolution, scale the image corners
                if (camera._image_resolution != image_resolution).all():
                    # Check if scale is uniform
                    scale = image_resolution / camera._image_resolution
                    coords = coords * \
                        scale.flip(-1)  # Flip to (width, height)
                ax.scatter(
                    numpyify(coords[..., 0]), numpyify(coords[..., 1]), s=corner_size, c=[color] * len(coords), zorder=get_zorder())

                if connect_corners:
                    plot_corners = coords[list(range(coords.shape[0])) + [0]]
                    ax.plot(numpyify(plot_corners[:, 0]), numpyify(
                        plot_corners[:, 1]), color=color, linewidth=corner_size * 0.5, zorder=get_zorder())

                if plot_name:
                    global_center = self.get_global_position(
                        t=t).detach()[..., :3, 3]
                    cam_center, oks = camera.global_to_image_coordinates(
                        global_center, t=t[i], v_includes_time=True, return_ok=True)
                    if oks:
                        cam_center = numpyify(cam_center)
                        ax.text(x=cam_center[..., 0], y=cam_center[..., 1], s=self.get_name(
                        ), fontsize=8, color=color, ha='center', va='center', zorder=get_zorder())
        return fig

    def plot_2d_point_trace(self,
                            points: torch.Tensor,
                            camera: Any,
                            ax: Axes,
                            t: torch.Tensor,
                            t_min: Optional[REAL_TYPE] = None,
                            t_max: Optional[REAL_TYPE] = None,
                            t_step: REAL_TYPE = 0.1,
                            t_frames: Optional[torch.Tensor] = None,
                            cmap: str = "tab10",
                            zorder: Optional[int] = None,
                            **kwargs
                            ):
        from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
        from tools.transforms.geometric.transforms3d import assure_homogeneous_vector
        from matplotlib.colors import to_rgba
        from tools.util.torch import index_of_first
        cmap = plt.get_cmap(cmap)
        camera: TimedCameraSceneNode3D
        points = tensorify(points, dtype=self._translation.dtype,
                           device=self._translation.device)
        points, _ = flatten_batch_dims(points, -2)
        points = assure_homogeneous_vector(
            points, dtype=self._translation.dtype)
        if t_max is None:
            t_max = self._times.max()
        t_max = tensorify(t_max).squeeze().item()
        if t_min is None:
            t_min = t
        t_min = tensorify(t_min).squeeze().item()
        t_step = tensorify(t_step)

        traj_t = torch.arange(
            t_min, t_max, t_step, dtype=self._times.dtype, device=self._times.device)
        if t_frames is not None:
            t_frames = tensorify(
                t_frames, dtype=self._times.dtype, device=self._times.device)
            traj_t = torch.cat([traj_t, t_frames], dim=0)
            traj_t = torch.unique(traj_t)
            traj_t = torch.sort(traj_t)[0]

        global_points = self.local_to_global(points, t=traj_t)
        image_points, oks = camera.global_to_image_coordinates(
            global_points, t=traj_t, clamp=False, v_includes_time=True, return_ok=True)

        for b in range(image_points.shape[0]):
            img_pts = image_points[b, oks[b]]
            image_traj_t = traj_t[oks[b]]

            t_frames_visible = None
            t_frames_visible_pos = None
            if t_frames is not None:
                t_frames_idx = index_of_first(image_traj_t, t_frames)
                existing = t_frames_idx >= 0
                t_frames_idx = t_frames_idx[existing]
                t_frames_visible = t_frames[existing]

                min_back = t_frames_visible.diff().min()
                t_frames_visible_pos = img_pts[t_frames_idx]

            colors = cmap(b % cmap.N)
            ax.plot(*img_pts[:, :2].T.detach().cpu(),
                    color=colors, zorder=zorder)
            if t_frames_visible is not None:
                if zorder is not None:
                    zorder = zorder + 1
                ax.scatter(
                    *t_frames_visible_pos[:, :2].T.detach().cpu(), color=colors, zorder=zorder)
        return ax.figure

    def _plot_plane(self,
                    ax: Axes,
                    t: Optional[torch.Tensor] = None,
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
        plot_plane_vertex = kwargs.get("plot_plane_vertex", True)

        if plot_plane_vertex:
            global_corners = self.get_global_plane_corners(t=t)
            # Plot the corners
            verts = [global_corners[:, i, :3].detach().cpu().numpy()
                     for i in range(global_corners.shape[1])]
            ax.add_collection3d(
                Poly3DCollection(
                    verts, facecolors=facecolor, linewidths=line_width, edgecolors=edgecolor, alpha=alpha))

    def plot_object(self,
                    ax: Axes,
                    **kwargs):
        fig = super().plot_object(ax, **kwargs)
        t_plane_pos = kwargs.get("t", self._times[-1])
        if len(t_plane_pos.shape) == 0:
            t_plane_pos = t_plane_pos.unsqueeze(0)
        self._plot_plane(ax, **kwargs)
        kwargs = dict(kwargs)
        if kwargs.get('plot_plane_traces', False):
            cmap = plt.get_cmap("tab10")
            corners = self.get_plane_corners()
            corner_colors = [cmap(x) for x in range(corners.shape[0])]
            t_max = kwargs.pop("t_max", None)
            if t_max is None:
                t_max = t_plane_pos
                if len(t_max) > 1:
                    t_max = t_max.max()
            t_min = kwargs.pop("t_min", None)
            if t_min is None:
                if len(t_plane_pos) > 1:
                    t_min = t_plane_pos.min()
                else:
                    t_min = self._times[0]
            self.plot_point_trace(
                points=corners[:, :3],
                ax=ax,
                colors=corner_colors,
                t_step=kwargs.get("t_trace_step", 0.001),
                t_max=t_max,
                t_min=t_min,
                **kwargs)
    # endregion


def get_linear_segmented_smoothing_fnc(
        threshold_lower: float,
        threshold_upper: float,
        slope: float):
    from functools import partial

    def smoothing_function(x: torch.Tensor, slope: float) -> torch.Tensor:
        return torch.tanh((x * slope) - 2) * 0.5 + 0.5

    partial_smoothing_function = partial(smoothing_function, slope=slope)

    def linear_segmented_smoothing(
        x: torch.Tensor,
        is_image_edge: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal threshold_lower, threshold_upper, partial_smoothing_function
        y = torch.zeros_like(x, dtype=torch.float32)
        # Nonlinear area
        mask_nonlinear = (x > threshold_lower) & (x < threshold_upper)
        y[mask_nonlinear] = partial_smoothing_function(x[mask_nonlinear])

        # Linearer Bereich für x <= threshold_lower
        val_lower = partial_smoothing_function(torch.tensor([threshold_lower]))
        val_upper = partial_smoothing_function(torch.tensor([threshold_upper]))

        mask_linear_lower = x <= threshold_lower
        with torch.set_grad_enabled(True):
            input_tensor = torch.tensor(
                [threshold_lower], dtype=torch.float32, requires_grad=True)
            slope_lower = torch.autograd.grad(partial_smoothing_function(
                input_tensor), input_tensor, create_graph=True)[0].detach()

        y[mask_linear_lower] = slope_lower * \
            (x[mask_linear_lower] - (threshold_lower)) + val_lower

        # Lin area x >= threshold_upper
        mask_linear_upper = x >= threshold_upper
        with torch.set_grad_enabled(True):
            input_tensor = torch.tensor(
                [threshold_upper], dtype=torch.float32, requires_grad=True)
            slope_upper = torch.autograd.grad(partial_smoothing_function(
                input_tensor), input_tensor, create_graph=True)[0].detach()

        y[mask_linear_upper] = slope_upper * \
            (x[mask_linear_upper] - threshold_upper) + val_upper

        # Set 1 for image edges
        y[:, is_image_edge] = 1.0
        return torch.clamp(y, 0.0, 1.0)
    return linear_segmented_smoothing
