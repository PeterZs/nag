from matplotlib.figure import Figure
import torch
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D
from tools.scene.box_node_3d import BoxNode3D
from matplotlib.axes import Axes
from tools.model.discrete_module_scene_node_3d import DiscreteModuleSceneNode3D
from tools.model.visual_node_3d import VisualNode3D
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.util.torch import tensorify
from tools.viz.matplotlib import saveable
from typing import Any, Iterable, List, Literal, Optional, Set, Tuple, Union
import torch
from tools.viz.matplotlib import parse_color_rgba
from tools.labels.timed_box_3d import TimedBox3D


class TimedBoxSceneNode3D(TimedDiscreteSceneNode3D, BoxNode3D):
    """Pytorch Module class for a 3D box with time dimension."""

    def __init__(self,
                 name: str,
                 size: torch.Tensor,
                 **kwargs
                 ):
        super().__init__(name=name, size=size, **kwargs)

    @classmethod
    def from_timed_box_3d(cls, timed_box: TimedBox3D, **kwargs) -> 'TimedBoxSceneNode3D':
        """Creates a BoxNode3D from a TimedBox3D label."""
        from tools.transforms.geometric.mappings import rotvec_to_rotmat

        size = torch.stack([tensorify(timed_box.width), tensorify(
            timed_box.height), tensorify(timed_box.depth)], dim=1)

        rotmat = rotvec_to_rotmat(tensorify(timed_box.heading))
        T, _, _ = rotmat.shape

        position = torch.eye(4, dtype=rotmat.dtype, device=rotmat.device)[
            None, ...].repeat(T, 1, 1)
        position[:, :3, :3] = rotmat
        position[:, :3, 3] = tensorify(timed_box.center)

        # Check if sizes differ along time axis, if so, take the largest and warn
        size_diff = torch.diff(size, dim=0)
        if torch.any(size_diff.abs() > 1e-6):
            cls.logger.warning(
                f"Box sizes differ along time axis: {size_diff}")
        size = size.max(dim=0).values

        name = str(timed_box.id[:4])
        if hasattr(timed_box, "object_id"):
            name = f"{timed_box.id[:4]} ({timed_box.object_id})"

        box = cls(
            name=name, size=size,
            position=position,
            times=tensorify(timed_box.frame_times))
        return box

    def get_local_corners(self, **kwargs) -> torch.Tensor:
        """Returns the corners of the box in local coordinates.

        When viewed in a right-handed coordinate system x-right, y-forward, z-up (matpotlib)
        from the top, the first 4 corners are the bottom face, the last 4 corners are the top face.
        Starting from the bottom left corner and going anti-clockwise.

        Returns
        -------
        torch.Tensor
            Corners of the box in local coordinates.
            Shape: (8, 3)
        """
        size = self.size
        half_size = size / 2
        corners = torch.tensor([
            # Bottom face
            [-1, 1, -1],
            [1, 1, -1],
            [1, 1,  1],
            [-1, 1,  1],
            # Top face
            [-1, -1, -1],
            [1, -1, -1],
            [1, -1,  1],
            [-1, -1,  1]], dtype=self.dtype, device=self._translation.device)
        corners = corners * half_size
        return corners

    @torch.no_grad()
    def plot_2d_projection(self,
                           ax: Axes,
                           camera: TimedCameraSceneNode3D,
                           t: Optional[NUMERICAL_TYPE] = None,
                           image_resolution: Optional[VEC_TYPE] = None,
                           box_color: Any = "yellow",
                           plot_box_edge_markers: bool = False,
                           bottom_start_corner_color: Any = "red",
                           top_start_corner_color: Any = "green",
                           plot_forward_face_markers: bool = True,
                           plot_coordinate_system: bool = False,
                           plot_name: bool = False,
                           ignore_invisible: bool = True,
                           **kwargs) -> Figure:
        """Projects the box on the 2D axes of the camera.

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

        Returns
        -------
        Figure
            Figure containing the plot.
        """
        if t is None:
            t = self._times[-1]
        else:
            t = tensorify(t)
        if image_resolution is None:
            image_resolution = camera._image_resolution
        else:
            image_resolution = tensorify(image_resolution)
        if len(image_resolution) != 2:
            raise ValueError(f"Invalid image resolution: {image_resolution}")

        corners = self.get_local_corners()
        image_corners = self.local_to_global(
            corners, t=t, v_include_time=False)[..., :3]

        cam_image_corners, oks = camera.global_to_image_coordinates(
            image_corners, t=t, v_includes_time=True, return_ok=True)
        if ignore_invisible:
            ignore_t = (~oks).all(dim=0)
            cam_image_corners = cam_image_corners[:, ~ignore_t, :]
            if cam_image_corners.shape[1] == 0:
                return ax.figure

        # If image resolution != camera resolution, scale the image corners
        if (camera._image_resolution != image_resolution).all():
            # Check if scale is uniform
            scale = image_resolution / camera._image_resolution
            cam_image_corners = cam_image_corners * \
                scale.flip(-1)  # Flip to (width, height)

        # Plot the box
        lines = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)  # Connections
        ]
        for ti in range(cam_image_corners.shape[1]):
            image_corners = cam_image_corners[:, ti, :]
            for line in lines:
                ax.plot(*image_corners[line, :].T, color=box_color)

            if plot_box_edge_markers:
                # Plot start corners
                ax.scatter(*image_corners[0, :3],
                           color=bottom_start_corner_color)
                ax.scatter(*image_corners[4, :3], color=top_start_corner_color)

                # Add arrow pointing to the next corner
                ax.quiver(*image_corners[0, :3], *((image_corners[1] - image_corners[0]) / 5),
                          color=bottom_start_corner_color)
                ax.quiver(*image_corners[4, :3], *((image_corners[5] - image_corners[4]) / 5),
                          color=top_start_corner_color)

            if plot_forward_face_markers:
                # Plot forward face markers
                ax.scatter(*image_corners[1], color="blue")
                ax.scatter(*image_corners[2], color="blue")
                ax.scatter(*image_corners[5], color="blue")
                ax.scatter(*image_corners[6], color="blue")

        if plot_coordinate_system:
            vecs = self._get_global_node_coordinate_system_vectors(
                t=t)  # Shape: (B, 3, 2, 3)
            start = vecs[:, :, 0, :]
            ends = vecs[:, :, 1, :]
            B, _, _, _ = vecs.shape
            flatted = vecs.reshape(-1, 3)
            vin = vecs.reshape(B, 3 * 2, 3).permute(1, 0, 2)
            cam_image_vs = camera.global_to_image_coordinates(
                vin, t=t, v_includes_time=True)

            # If image resolution != camera resolution, scale the image corners
            if (camera._image_resolution != image_resolution).all():
                # Check if scale is uniform
                scale = image_resolution / camera._image_resolution
                cam_image_vs = cam_image_vs * \
                    scale.flip(-1)  # Flip to (width, height)
            cam_image_vs = cam_image_vs.permute(1, 0, 2).reshape(B, 3, 2, 2)
            colors = ["red", "green", "blue"]
            for i in range(3):
                _dirs = cam_image_vs[:, i, 1, :] - cam_image_vs[:, i, 0, :]
                ax.quiver(*cam_image_vs[:, i, 0, :].T, *_dirs.T,
                          color=colors[i], headwidth=1, headlength=1)

        if plot_name:
            pos = self.get_global_position(t=t)[:, :3, 3]
            text_img, oks = camera.global_to_image_coordinates(
                pos, t=t, v_includes_time=True, return_ok=True)
            if oks.all():
                txt = ax.text(*text_img.T, self.get_name(), color="black")
                txt.set_bbox(
                    dict(facecolor='white', alpha=1., edgecolor='black'))

        return ax.figure

    def plot_corners(self,
                     ax: Axes,
                     box_color: Any = "yellow",
                     plot_box_edge_markers: bool = False,
                     bottom_start_corner_color: Any = "red",
                     top_start_corner_color: Any = "green",
                     plot_forward_face_markers: bool = True,
                     plot_edge_point_traces: bool = False,
                     **kwargs):
        from tools.util.numpy import numpyify
        t_plane_pos = kwargs.get("t", self._times[-1])

        local_corners = self.get_local_corners()
        global_timed_corners = self.local_to_global(local_corners,
                                                    t=t_plane_pos,
                                                    v_include_time=False
                                                    )[..., :3]  # Shape: (8, T, 3)
        box_color = parse_color_rgba(box_color)
        bottom_start_corner_color = parse_color_rgba(bottom_start_corner_color)
        top_start_corner_color = parse_color_rgba(top_start_corner_color)

        lines = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)  # Connections
        ]
        for t in range(global_timed_corners.shape[1]):
            global_corners = numpyify(global_timed_corners[:, t, :])
            for line in lines:
                ax.plot(*global_corners[line, :].T, color=box_color)

            if plot_box_edge_markers:
                # Plot start corners
                ax.scatter(*global_corners[0, :3],
                           color=bottom_start_corner_color)
                ax.scatter(*global_corners[4, :3],
                           color=top_start_corner_color)

                # Add arrow pointing to the next corner
                ax.quiver(*global_corners[0, :3], *((global_corners[1, :3] - global_corners[0, :3]) / 5),
                          color=bottom_start_corner_color)
                ax.quiver(*global_corners[4, :3], *((global_corners[5, :3] - global_corners[4, :3]) / 5),
                          color=top_start_corner_color)
            if plot_forward_face_markers:
                # Plot forward face markers
                ax.scatter(*global_corners[1, :3], color="blue")
                ax.scatter(*global_corners[2, :3], color="blue")
                ax.scatter(*global_corners[5, :3], color="blue")
                ax.scatter(*global_corners[6, :3], color="blue")

            if plot_edge_point_traces:
                args = dict(kwargs)
                if "t_max" not in args:
                    args["t_max"] = t_plane_pos
                self.plot_point_trace(local_corners, ax=ax, **args)
        return ax.figure

    def plot_object(self, ax: Axes, **kwargs):
        fig = super().plot_object(ax, **kwargs)
        self.plot_corners(ax, **kwargs)
