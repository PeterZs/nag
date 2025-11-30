from typing import Any, Dict, Iterable, Optional, Tuple

from matplotlib.axes import Axes
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D

import torch
from tools.util.typing import VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify

from nag.model.timed_plane_scene_node_3d import TimedPlaneSceneNode3D


def compute_background_color(
        images: torch.Tensor) -> torch.Tensor:
    return images.mean(dim=(0, -2, -1))


class BackgroundPlaneSceneNode3D(
    TimedPlaneSceneNode3D,
):
    """A scene node representing a static background plane in 3D space."""

    _background_color: torch.Tensor
    """The color of the background plane in RGB format. Shape (3,)"""

    _is_background_learnable: bool
    """Whether the background color is learnable"""

    _background_color_fadeout: bool
    """Whether the background color should fade out over time."""

    _started_background_color_fadeout: torch.Tensor
    """The epoch when the background color fadeout started."""

    _is_camera_attached: bool
    """Whether the background plane is attached to a camera."""

    def __init__(self,
                 background_color_fadeout: bool = False,
                 is_background_learnable: bool = True,
                 background_color: Optional[VEC_TYPE] = None,
                 is_camera_attached: bool = True,
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
                 declare_background_color: bool = True,
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
        self.register_buffer("_is_camera_attached", torch.tensor(
            is_camera_attached, dtype=torch.bool))
        if declare_background_color:
            self._background_color_fadeout = background_color_fadeout
            self.register_buffer("_started_background_color_fadeout",
                                 torch.tensor(-1, dtype=dtype))
            if background_color is None:
                background_color = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
            if len(background_color) != 3:
                raise ValueError("Background color must be of shape (3,)")
            self._is_background_learnable = is_background_learnable
            if is_background_learnable:
                self._background_color = torch.nn.Parameter(
                    background_color, requires_grad=True)
            else:
                self.register_buffer("_background_color", background_color)

    def compute_fadeout_scale(self, sin_epoch: torch.Tensor) -> torch.Tensor:
        if self._started_background_color_fadeout < 0:
            self._started_background_color_fadeout = sin_epoch

        v_min = self._started_background_color_fadeout
        # Fade the background color out over time so objects need to learn to cover it
        # scale = 1 - sin_epoch
        return (sin_epoch - v_min) / ((1 - sin_epoch) - v_min)

    def compute_background_color(self,
                                 sin_epoch: torch.Tensor,
                                 next_sin_epoch: Optional[torch.Tensor] = None,
                                 batch_idx: Optional[int] = None,
                                 max_batch_idx: Optional[int] = None,
                                 ) -> torch.Tensor:
        if not self._background_color_fadeout:
            return self._background_color
        # Fade the background color out over time so objects need to learn to cover it
        # rgb = (1 - sin_epoch) * rgb
        if not self.training or max_batch_idx is None or max_batch_idx <= 0.:
            return (1 - sin_epoch) * self._background_color

        current_sin_epoch_scale = self.compute_fadeout_scale(sin_epoch)
        next_sin_epoch_scale = self.compute_fadeout_scale(next_sin_epoch)

        frac = batch_idx / (max_batch_idx - 1)
        red = (1 - frac) * current_sin_epoch_scale + \
            frac * next_sin_epoch_scale
        return (1 - red) * self._background_color

    def forward(self,
                intersection_points: torch.Tensor,
                sin_epoch: torch.Tensor,
                next_sin_epoch: Optional[torch.Tensor] = None,
                batch_idx: Optional[int] = None,
                max_batch_idx: Optional[int] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the background plane.
        Will return the RGB and a constant alpha value of 1 for each intersection point.

        Parameters
        ----------
        intersection_points : torch.Tensor
            The intersection points of the rays with the plane.
            Shape: (B, T, 3)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            1. The RGB values of the plane. Shape: (B, T, 3)
            2. The alpha values of the plane. Shape: (B, T, 1)
        """
        B, T = intersection_points.shape[:2]

        rgb = self.compute_background_color(sin_epoch=sin_epoch,
                                            next_sin_epoch=next_sin_epoch,
                                            batch_idx=batch_idx,
                                            max_batch_idx=max_batch_idx
                                            ).unsqueeze(0).unsqueeze(0).repeat(B, T, 1)
        alpha = torch.ones(B, T, 1, dtype=intersection_points.dtype,
                           device=intersection_points.device)
        return rgb, alpha

    def forward_modality(
        self,
        uv_plane: torch.Tensor,
        t: torch.Tensor,
        sin_epoch: torch.Tensor,
        right_idx_flow: Optional[torch.Tensor] = None,
        rel_frac_flow: Optional[torch.Tensor] = None,
        next_sin_epoch: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        max_batch_idx: Optional[int] = None,
        query_alpha: bool = True,
        query_color: bool = True,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the plane.

        Parameters
        ----------
        uv_plane : torch.Tensor
            The intersection points / query points of the rays with the plane.
            In plane space. Plane space is in normal range [0, 1].
            Shape: (B, T, 2)

        t : torch.Tensor
            The time of the intersection.
            Shape: (T, )

        sin_epoch : torch.Tensor
            The sine of the epoch.

        right_idx_flow : Optional[torch.Tensor], optional
            Index for flow controlpoint interpolation for times t, by default None

        rel_frac_flow : Optional[torch.Tensor], optional
            Step for position interpolation, by default None

        next_sin_epoch : Optional[torch.Tensor], optional
            The sine of the next epoch, by default None

        batch_idx : Optional[int], optional
            The batch index, by default None

        max_batch_idx : Optional[int], optional
            The maximum batch index, by default None

        query_alpha : bool, optional
            If True, the alpha values are queried, by default True

        query_color : bool, optional
            If True, the color values are queried, by default True

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            1. The RGB values of the plane. Shape: (B, T, 3)
            2. The alpha values of the plane. Shape: (B, T, 1)
            3. The flow values of the plane. Shape: (B, T, 2)
        """

        B, T = uv_plane.shape[:2]

        if query_color:
            color = self.compute_background_color(sin_epoch=sin_epoch,
                                                  next_sin_epoch=next_sin_epoch,
                                                  batch_idx=batch_idx,
                                                  max_batch_idx=max_batch_idx
                                                  ).unsqueeze(0).unsqueeze(0).repeat(B, T, 1)
        else:
            color = torch.zeros(B, T, 3, dtype=self.dtype,
                                device=uv_plane.device)

        alpha = torch.ones(B, T, 1, dtype=uv_plane.dtype,
                           device=uv_plane.device)
        flow = torch.zeros(B, T, 2, dtype=uv_plane.dtype,
                           device=uv_plane.device)
        return color, alpha, flow

    @classmethod
    def estimate_background_scale(
        cls,
        camera: TimedCameraSceneNode3D,
        scene_cutoff_distance: float = 1.5,
        relative_scale_margin: float = 0.1,
    ) -> torch.Tensor:
        """Estimates the scale of the background plane based on the camera.
        The Background plane is assumed to be static and non moving.

        Parameters
        ----------
        camera : TimedCameraSceneNode3D
            The camera to estimate the scale from.

        scene_cutoff_distance : float, optional
            The distance to the camera, by default 1.5

        relative_scale_margin : float, optional
            The relative scale margin, by default 0.1

        Returns
        -------
        torch.Tensor
            The estimated scale of the background plane.
        """
        outer_rays = [
            [0, 0],  # Bottom left
            # Top Right (max_width, max_height)
            [camera._image_resolution[1], camera._image_resolution[0]],
        ]
        uv = torch.tensor(outer_rays, dtype=camera._times.dtype,
                          device=camera._image_resolution.device)

        ray_directions = camera._get_ray_direction(
            uv).swapaxes(0, 1)[..., :3]  # (B, T, 3)
        ray_directions = ray_directions * camera.focal_length
        ray_origins = ray_directions - \
            (torch.tensor([0, 0, camera.focal_length], dtype=ray_directions.dtype,
                          device=ray_directions.device))

        # Normalize the ray directions
        ray_directions = ray_directions / \
            torch.norm(ray_directions, dim=-1, keepdim=True)

        zhit = scene_cutoff_distance / ray_directions[..., 2:3]
        # Propagate rays so they hit the cutoff distance
        outer_points = ray_origins + zhit.repeat(1, 1, 3) * ray_directions

        bl = outer_points[0]
        tr = outer_points[1]

        scale_xy = (tr - bl).max(dim=0).values[:2]
        if relative_scale_margin > 0:
            scale_xy += scale_xy * relative_scale_margin
        return scale_xy

    @classmethod
    def for_camera(cls,
                   camera: TimedCameraSceneNode3D,
                   images: torch.Tensor,
                   masks: torch.Tensor,
                   depths: torch.Tensor,
                   times: torch.Tensor,
                   background_color_fadeout: bool = False,
                   is_background_learnable: bool = True,
                   scene_cutoff_distance: float = 1.5,
                   dtype: torch.dtype = torch.float32,
                   relative_scale_margin: float = 0.1,
                   world: Optional[AbstractSceneNode] = None,
                   name: Optional[str] = None,
                   index: Optional[int] = None,
                   nag_model: Optional["Any"] = None,
                   config: Any = None,
                   proxy_init: bool = False,
                   **kwargs):
        with torch.no_grad():
            """Creates a background plane from a camera and a world."""
            # Get the outer rays of the camera
            position = torch.zeros(
                3, dtype=dtype, device=camera._image_resolution.device)
            position[2] = scene_cutoff_distance

            plane_scale = cls.estimate_background_scale(
                camera=camera,
                scene_cutoff_distance=scene_cutoff_distance,
                relative_scale_margin=relative_scale_margin
            )

            background_color = compute_background_color(images)

            args = dict(
                is_camera_attached=True,
                is_background_learnable=is_background_learnable,
                background_color_fadeout=background_color_fadeout,
                background_color=tensorify(
                    background_color, dtype=dtype, device=plane_scale.device) if background_color is not None else None,
                plane_scale=plane_scale,
                translation=position.unsqueeze(0),
                dtype=dtype,
                name=name,
                index=index,
                **kwargs)

            if nag_model is not None:
                from nag.model.nag_functional_model import NAGFunctionalModel
                if isinstance(nag_model, NAGFunctionalModel):
                    args = nag_model.patch_background_args(args)

            # Create the background plane
            plane = cls(
                **args
            )
            camera.add_scene_children(plane)

            return plane

    def _plot_plane(self,
                    ax: Axes,
                    t: Optional[torch.Tensor] = None,
                    edgecolor: str = 'black',
                    alpha: float = 1.,
                    line_width: float = 1.0,
                    facecolor: Optional[Any] = None,
                    **kwargs):
        """Plots the corners of the plane on the given axis.

        Parameters
        ----------
        ax : Axes
            The axis to plot on.
        corners : torch.Tensor
            The corners of the plane to plot.
        """
        if facecolor is None:
            facecolor = self._background_color.detach().cpu().numpy()
            # Add alpha to the color
            facecolor = list(facecolor) + [alpha]
        super()._plot_plane(ax=ax, t=t, edgecolor=edgecolor, alpha=alpha,
                            line_width=line_width, facecolor=facecolor, **kwargs)
