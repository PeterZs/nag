import math
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from nag.model.camera_scene_node_3d import CameraSceneNode3D
from nag.model.learned_offset_scene_node_3d import LearnedOffsetSceneNode3D
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_plane_scene_node_3d import TimedPlaneSceneNode3D
from nag.model.timed_discrete_scene_node_3d import global_to_local, local_to_global

import torch
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims, compose_transformation_matrix
from nag.config.encoding_config import EncodingConfig
from nag.config.network_config import NetworkConfig
from nag.transforms.transforms_timed_3d import interpolate_vector, linear_interpolate_vector
from tools.viz.matplotlib import plot_mask, plot_as_image
from tools.util.numpy import numpyify
from nag.utils import utils
try:
    import tinycudann as tcnn
except (ModuleNotFoundError, OSError) as err:
    from tools.logger.logging import logger
    from tools.util.mock_import import MockImport
    if not TYPE_CHECKING:
        logger.warning(f"Could not import tinycudann: {err}")
        tcnn = MockImport(mocked_property="tcnn")

import pytorch_lightning as pl
from tools.transforms.geometric.transforms3d import rotmat_to_unitquat, unitquat_to_rotmat
from nag.model.discrete_plane_scene_node_3d import local_to_plane_coordinates, plane_coordinates_to_local
from tools.util.torch import shadow_ones, shadow_zeros, shadow_identity_2d
from tools.transforms.min_max import MinMax
from tools.transforms.mean_std import MeanStd
from tools.transforms.fittable_transform import FittableTransform
from tools.util.typing import DEFAULT, _DEFAULT


class LearnedAberrationPlaneSceneNode3D(
    TimedPlaneSceneNode3D
):
    """A scene node representing a image plane in 3D space with timed discrete position and orientation and learnable offsets.
    And an encoding and
    """
    color: torch.Tensor
    """The color of the plane. Shape (3,) / float."""

    encoding_alpha: tcnn.Encoding
    """The encoding for the alpha matting of the plane."""

    encoding_flow: tcnn.Encoding
    """The encoding for the flow of the plane."""

    rgb_weight: torch.Tensor
    """The weight factor for the RGB offset. Shape (1,) / float."""

    alpha_weight: torch.Tensor
    """The weight factor for the alpha offset. Shape (1,) / float."""

    network_alpha: tcnn.Network
    """The tcnn network for predicting the alpha of the plane."""

    network_dtype: torch.dtype
    """The dtype for the network."""

    def __init__(
        self,
            encoding_alpha_config: Union[EncodingConfig, str] = "tiny",
            network_alpha_config: Union[NetworkConfig, str] = "tiny",
            alpha_rescaling: bool = True,
            alpha_normalization: Optional[FittableTransform] = None,
            coarse_to_fine_alpha: bool = True,
            plane_scale: Optional[VEC_TYPE] = None,
            plane_scale_offset: Optional[VEC_TYPE] = None,
            translation: Optional[VEC_TYPE] = None,
            orientation: Optional[VEC_TYPE] = None,
            initial_rgb: Optional[torch.Tensor] = None,
            initial_alpha: Optional[torch.Tensor] = None,
            rgb_weight: Optional[torch.Tensor] = None,
            alpha_weight: Optional[torch.Tensor] = None,
            position: Optional[torch.Tensor] = None,
            times: Optional[VEC_TYPE] = None,
            name: Optional[str] = None,
            children: Optional[Iterable['AbstractSceneNode']] = None,
            decoding: bool = False,
            dtype: torch.dtype = torch.float32,
            network_dtype: torch.dtype = torch.float16,
            index: Optional[int] = None,
            visible: bool = True,
            # Proxy init is used if the class should be initialized, but values may be dummies as weights are loaded afterwards
            proxy_init: bool = False,
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
            index=index,
            **kwargs)
        self.ray_dependent = False
        """If the planes properties are considered ray dependent. If true, ray_origins and ray_directions must be supplied for forward pass."""
        self.network_dtype = network_dtype

        self.encoding_alpha = tcnn.Encoding(
            n_input_dims=2,
            encoding_config=EncodingConfig.parse(
                encoding_alpha_config).to_dict(),
            dtype=network_dtype)
        self.visible = visible
        """If the plane is visible."""

        color = tensorify(initial_rgb) if initial_rgb is not None else torch.tensor(
            [0.5, 0.5, 0.5], dtype=dtype)
        self.color = torch.nn.Parameter(color)

        self.coarse_to_fine_alpha = coarse_to_fine_alpha

        self.network_alpha = tcnn.Network(n_input_dims=self.encoding_alpha.n_output_dims,
                                          n_output_dims=1, network_config=NetworkConfig.parse(network_alpha_config).to_dict())

        self.alpha_rescaling = alpha_rescaling

        self.init_initial_alpha(self.dtype, initial_alpha)

        if alpha_weight is None:
            alpha_weight = torch.tensor(0.1, dtype=self.dtype)
        else:
            alpha_weight = tensorify(alpha_weight, dtype=self.dtype)

        self.register_buffer("alpha_weight", alpha_weight)

        if self.alpha_rescaling:
            if alpha_normalization is None:
                alpha_normalization = MeanStd(dim=0, mean=0, std=DEFAULT)
            self.alpha_normalization = alpha_normalization
            if not proxy_init:
                self.estimate_alpha_scaling(alpha_normalization)
        else:
            self.alpha_normalization = None

    def after_checkpoint_loaded(self, **kwargs):
        super().after_checkpoint_loaded(**kwargs)
        if self.alpha_rescaling and self.alpha_normalization is not None:
            self.alpha_normalization.fitted = True

    def init_initial_alpha(self,
                           dtype: torch.dtype,
                           initial_alpha: Optional[torch.Tensor] = None,
                           ):
        if initial_alpha is None:
            initial_alpha = torch.tensor([[0.01]], dtype=dtype)
        self.register_buffer("initial_alpha", initial_alpha)

    def get_initial_alpha(self, uv: torch.Tensor) -> torch.Tensor:
        return self.initial_alpha

    def estimate_alpha_scaling(self, normalization: FittableTransform):
        H, W = 1000, 1000
        with torch.no_grad():
            device = torch.device("cuda")
            old_device = self._translation.device
            if device != old_device:
                self.to(device)
            x = torch.linspace(0, 1, W, device=device,
                               dtype=self._translation.dtype)
            y = torch.linspace(0, 1, H, device=device,
                               dtype=self._translation.dtype)
            grid = (torch.stack(torch.meshgrid(
                x, y, indexing="xy"), dim=-1)) - 0.5
            alpha = self.get_alpha(grid.reshape(
                H * W, 2), torch.tensor(0, device=device, dtype=self._translation.dtype))
            alpha_norm = normalization.fit_transform(alpha)
            if device != old_device:
                self.to(old_device)

    @classmethod
    def for_camera(cls,
                   camera: TimedCameraSceneNode3D,
                   images: torch.Tensor,
                   masks: torch.Tensor,
                   depths: torch.Tensor,
                   times: torch.Tensor,
                   dataset: Any,
                   camera_distance: float = 1e-3,
                   dtype: torch.dtype = torch.float32,
                   world: Optional[AbstractSceneNode] = None,
                   name: Optional[str] = None,
                   index: Optional[int] = None,
                   nag_model: Optional["Any"] = None,
                   config: Any = None,
                   **kwargs):
        from nag.strategy.tilted_plane_initialization_strategy import reproject_alpha, reproject_color
        from nag.strategy.plane_position_strategy import compute_plane_scale

        with torch.no_grad():
            plane_pos = torch.eye(
                4, dtype=dtype, device=camera._translation.device)
            _local_translation = torch.tensor(
                [0, 0, camera_distance], dtype=dtype)
            plane_pos[:3, 3] = _local_translation

            cam_close = camera.get_global_position(t=camera._times)

            T = len(camera._times)
            global_plane_position = torch.bmm(
                cam_close, plane_pos.unsqueeze(0).expand(T, -1, -1))

            border_coords = torch.tensor([[0, 0],
                                          [0, 1],
                                          [1, 1],
                                          [1, 0]], dtype=dtype, device=camera._translation.device)  # bl, br, tr, tl (y, x)
            plane_scale = compute_plane_scale(
                border_coords=border_coords.unsqueeze(1).expand(-1, T, -1),
                resolution=torch.tensor(
                    [1., 1.], dtype=dtype, device=camera._translation.device),
                camera=camera, times=camera._times,
                global_plane_position=global_plane_position,
                relative_plane_margin=0.05)
            plane_scale = plane_scale.amax(dim=0)

            args = dict(kwargs)
            args.update(
                plane_scale=plane_scale,
                translation=_local_translation[None, :].expand(T, -1),
                times=camera._times,
                network_dtype=config.tinycudann_network_dtype,
                dtype=dtype,
                name=name,
                index=index)

            if nag_model is not None:
                from nag.model.nag_functional_model import NAGFunctionalModel
                if isinstance(nag_model, NAGFunctionalModel):
                    args = nag_model.patch_aberration_plane_args(args)

            # Create the background plane
            plane = cls(
                **args
            )
            camera.add_scene_children(plane)
            return plane

    # region Forward
    def compute_rgb_alpha(self,
                          uv: torch.Tensor,
                          t: torch.Tensor,
                          sin_epoch: torch.Tensor,
                          right_idx_flow: Optional[torch.Tensor] = None,
                          rel_frac_flow: Optional[torch.Tensor] = None,
                          ray_origins: Optional[torch.Tensor] = None,
                          ray_directions: Optional[torch.Tensor] = None,
                          context: Optional[Dict[str, Any]] = None,
                          **kwargs
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the RGB and alpha values for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            UV coordinates of the points within the image plane. Shape: (B, T, 2) (x, y) in range [-0.5, 0.5]
        t : torch.Tensor
            The times of the points. Shape: (T, )
        sin_epoch : torch.Tensor
            The sine of the epoch. Progress marker. Tends towards 1 at the end of training.
        right_idx_flow : Optional[torch.Tensor], optional
            Index for position interpolation, by default None
        rel_frac_flow : Optional[torch.Tensor], optional
            Step for position interpolation, by default None
        ray_origins : Optional[torch.Tensor], optional
            The ray origins of the rays. In local coordinates of the current node. Shape: (B, T, 3)
        ray_directions : Optional[torch.Tensor], optional
            The ray directions of the rays. In local coordinates of the current node. Shape: (B, T, 3)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            RGB values and alpha values for the given uv coordinates. Shapes: (B, T, 3), (B, T, 1)
            Whereby both are in range [0, 1].
        """

        B, T, _ = uv.shape
        # In local coordinates the plane center is at (0, 0, 0)

        if not self.visible:
            rgb = torch.zeros(B, T, 3, dtype=self.dtype, device=uv.device)
            alpha = torch.zeros(B, T, 1, dtype=self.dtype, device=uv.device)
            return rgb, alpha

        query_points = uv  # (B, T, 2)
        # We need to collapse the time dimension and treat is as normal point, as the image plane should be "timeless"
        query_points = query_points.reshape(-1, 2)  # (B*T, 2)

        # Get the RGB values
        rgb = self.get_rgb(query_points, sin_epoch)
        rgb = rgb.clamp(0, 1)
        rgb = rgb.reshape(B, T, 3)

        # Get the alpha values
        network_alpha = self.get_alpha(query_points, sin_epoch)
        initial_alpha = self.get_initial_alpha(
            query_points)
        alpha = initial_alpha + self.alpha_weight * network_alpha
        alpha = alpha.clamp(0, 1)
        alpha = alpha.reshape(B, T, 1)

        if context is not None:
            idx = self.get_index()
            if context.get("store_object_alpha", False):
                if "object_alpha" not in context:
                    context["object_alpha"] = dict()
                context["object_alpha"][idx] = alpha

        return rgb, alpha

    def forward(self,
                uv: torch.Tensor,
                ray_origins: torch.Tensor,
                ray_directions: torch.Tensor,
                t: torch.Tensor,
                sin_epoch: torch.Tensor,
                uv_in_plane_space: bool = False,
                global_position: Optional[torch.Tensor] = None,
                plane_scale: Optional[torch.Tensor] = None,
                plane_scale_offset: Optional[torch.Tensor] = None,
                right_idx_flow: Optional[torch.Tensor] = None,
                rel_frac_flow: Optional[torch.Tensor] = None,
                next_sin_epoch: Optional[torch.Tensor] = None,
                batch_idx: Optional[int] = None,
                max_batch_idx: Optional[int] = None,
                context: Optional[Dict[str, Any]] = None,
                **kwargs
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the plane.

        Parameters
        ----------
        uv : torch.Tensor
            The intersection points / query points of the rays with the plane.
            In global coordinates unless uv_in_plane_space is True.
            Shape: (B, T, 3)
        ray_origins : torch.Tensor
            The ray origins of the rays. In global coordinates unless uv_in_plane_space is True.
            Shape: (B, T, 3)
        ray_directions : torch.Tensor
            The ray directions of the rays. In global coordinates unless uv_in_plane_space is True.
            Shape: (B, T, 3)
        t : torch.Tensor
            The time of the intersection.
            Shape: (T, )
        sin_epoch : torch.Tensor
            The sine of the epoch.
            Tends towards 1 at the end of the epoch and towards -1 at the beginning of the epoch.

        uv_in_plane_space : bool
            If the uv coordinates are in plane space. If True, the uv coordinates are assumend in the range [0, 1].

        global_position : Optional[torch.Tensor]
            The global position of the plane.
            If None, the global position of the plane is computed from the position and orientation of the plane.
            Can be supplied to avoid recomputing the global position. Shape: (T, 4, 4)

        plane_scale : Optional[torch.Tensor]
            The scale of the plane. If None, the scale in the object is used.
            Can be supplied to reduce overhead. Shape: (2, ) (x, y)

        plane_scale_offset : Optional[torch.Tensor]
            The offset of the plane scale. If None, the offset in the object is used.
            Can be supplied to reduce overhead. Shape: (2, ) (x, y)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            1. The RGB values of the plane. Shape: (B, T, 3)
            2. The alpha values of the plane. Shape: (B, T, 1)
        """
        B, T = uv.shape[:2]

        if uv.numel() == 0:
            return torch.zeros(B, T, 3, dtype=self.dtype, device=uv.device), torch.zeros(B, T, 1, dtype=self.dtype, device=uv.device)

        if not uv_in_plane_space:
            if global_position is None:
                global_position = self.get_global_position(t=t)

            if plane_scale is None:
                plane_scale = self.get_plane_scale()

            if plane_scale_offset is None:
                plane_scale_offset = self.get_plane_scale_offset()

            # Convert the global intersection points to local coordinates
            # (B, T, 2) (x, y) Ignore z and w. x, y should be in local coordinate system now
            uv_local = global_to_local(
                global_position, uv, v_include_time=True)[..., :2]

            # Convert local to plane space
            uv_plane = local_to_plane_coordinates(
                uv_local, plane_scale, plane_scale_offset)  # Go from local to plane coordinate system

            if self.ray_dependent:
                # Convert ray directions to plane space
                ray_global_target = ray_origins + ray_directions
                ray_local_target = global_to_local(
                    global_position, ray_global_target, v_include_time=True)[..., :3]
                ray_local_origin = global_to_local(
                    global_position, ray_origins, v_include_time=True)[..., :3]
                ray_local_directions = ray_local_target - ray_local_origin
                ray_directions = ray_local_directions
                ray_origins = ray_local_origin
            else:
                ray_directions = None
                ray_origins = None
        else:
            uv_plane = uv[..., :2]

        # Plane_intersection_points are in range [0, 1] for x, y, we need to convert them to [-0.5, 0.5]
        uv_network = uv_plane - 0.5

        return self.compute_rgb_alpha(uv=uv_network,
                                      t=t,
                                      sin_epoch=sin_epoch,
                                      right_idx_flow=right_idx_flow,
                                      rel_frac_flow=rel_frac_flow,
                                      ray_directions=ray_directions,
                                      ray_origins=ray_origins,
                                      context=context
                                      )

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
        ray_origins: Optional[torch.Tensor] = None,
        ray_directions: Optional[torch.Tensor] = None,
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

        ray_origins : Optional[torch.Tensor], optional
            The ray origins of the rays. In local coordinates of the current node. Shape: (B, T, 3)

        ray_directions : Optional[torch.Tensor], optional
            The ray directions of the rays. In local coordinates of the current node. Shape: (B, T, 3)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            1. The RGB values of the plane. Shape: (B, T, 3)
            2. The alpha values of the plane. Shape: (B, T, 1)
            3. The flow values of the plane. Shape: (B, T, 2)
        """

        uv_network = uv_plane - 0.5

        B, T, _ = uv_network.shape
        # In local coordinates the plane center is at (0, 0, 0)
        flow = torch.zeros_like(uv_network)

        if not self.visible:
            query_color = False
            query_alpha = False

        query_points = uv_network  # (B, T, 2)
        # We need to collapse the time dimension and treat is as normal point, as the image plane should be "timeless"
        query_points = query_points.reshape(-1, 2)  # (B*T, 2)
        # query_points = shadow_zeros(query_points)

        if query_color:
            # Get the RGB values
            rgb = self.get_rgb(query_points, sin_epoch)
            rgb = rgb.clamp(0, 1)
            # Reshape to original shape
            rgb = rgb.reshape(B, T, 3)
        else:
            rgb = torch.zeros(B, T, 3, dtype=self.dtype,
                              device=uv_plane.device)

        if query_alpha:
            # Get the alpha values
            network_alpha = self.get_alpha(query_points, sin_epoch)
            initial_alpha = self.get_initial_alpha(
                query_points)

            alpha = initial_alpha + self.alpha_weight * network_alpha

            # alpha = 1 / 2 * torch.tanh((alpha * 0.5) ** 3 / 2)

            alpha = alpha.clamp(0, 1)
            alpha = alpha.reshape(B, T, 1)
        else:
            alpha = torch.zeros(B, T, 1, dtype=self.dtype,
                                device=uv_plane.device)

        return rgb, alpha, flow

    def get_rgb(self, uv: torch.Tensor, sin_epoch: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get the RGB values for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of the point.
            Shape: (B, 2) x, y should be in range [-0.5, 0.5]

        Returns
        -------
        torch.Tensor
            The RGB values for the given uv coordinates.
            Shape: (B, 3)
        """
        B, _ = uv.shape
        return self.color[None, :].expand(B, -1)

    def get_alpha(self,
                  uv: torch.Tensor,
                  sin_epoch: torch.Tensor,
                  **kwargs
                  ) -> torch.Tensor:
        """Get the alpha values for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of the point.
            Shape: (B, 2) x, y should be in range [-0.5, 0.5]

        Returns
        -------
        torch.Tensor
            The alpha values for the given uv coordinates.
            Shape: (B, 1)
        """
        query_points = uv
        with torch.autocast(device_type='cuda', dtype=self.network_dtype):
            enc = self.encoding_alpha(query_points)
            if self.coarse_to_fine_alpha:
                enc = utils.mask(enc, sin_epoch)
            network_alpha = self.network_alpha(enc)
        network_alpha = network_alpha.to(dtype=self.dtype)

        if self.alpha_rescaling and self.alpha_normalization.fitted:
            network_alpha = self.alpha_normalization(network_alpha)
        return network_alpha

    def _get_estimated_alpha(self, uv: torch.Tensor) -> torch.Tensor:
        uv, shape = flatten_batch_dims(uv, -2)
        network_alpha = self.get_alpha(uv, torch.tensor(
            1.0, device=uv.device, dtype=uv.dtype))
        initial_alpha = self.get_initial_alpha(uv)
        alpha = initial_alpha + self.alpha_weight * network_alpha
        alpha = alpha.clamp(0, 1)
        return unflatten_batch_dims(alpha, shape)

    def _get_estimated_rgb(self, uv: torch.Tensor) -> torch.Tensor:
        uv, shape = flatten_batch_dims(uv, -2)
        rgb = self.get_rgb(uv, torch.tensor(
            1.0, device=uv.device, dtype=uv.dtype))
        rgb = rgb.clamp(0, 1)
        return unflatten_batch_dims(rgb, shape)

    # endregion

    # region Plotting

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
        from matplotlib.colors import to_rgba, get_named_colors_mapping
        facec = facecolor
        plot_plane_alpha = kwargs.get('plot_plane_alpha', False)
        plot_plane_image = kwargs.get('plot_plane_image', False)

        if plot_plane_alpha or plot_plane_image:
            # Remove base facec
            facec = np.array([0, 0, 0, 0])
            alpha = 0.

        super()._plot_plane(ax=ax, t=t, facecolor=facec, edgecolor=edgecolor,
                            alpha=alpha, line_width=line_width, **kwargs)

        if plot_plane_alpha or plot_plane_image:
            sub_H, sub_W = 200, 200

            subsample = kwargs.get('plot_plane_image_subsample', 10)

            xx, yy = torch.meshgrid(torch.arange(
                0, 1, 1 / sub_W * subsample, device=self._translation.device), torch.arange(0, 1, 1 / sub_H * subsample, device=self._translation.device), indexing="xy")
            # Add half a pixel to get the center of the pixel
            xx = xx + 1 / (2 * sub_W * subsample)
            yy = yy + 1 / (2 * sub_H * subsample)

            xy = torch.stack([xx, yy], dim=-1)
            # XYZ to local coordinates
            xyz = self.plane_coordinates_to_local(xy)

            # Add w=1
            xyzw = torch.cat(
                [xyz, torch.ones_like(xyz[..., :1])], dim=-1)

            xyz_glob = self.local_to_global(
                xyzw, t=t)[..., 0, :]  # Ignore time

            faces = torch.zeros(4, math.ceil(
                sub_H / subsample), math.ceil(sub_W / subsample), dtype=torch.float32, device=self._translation.device)  # RGBA

            if plot_plane_alpha or plot_plane_image:
                try:
                    old_device = self._translation.device
                    device = torch.device("cuda")
                    # Move model to cuda if not already
                    if device != old_device:
                        self.to(device)
                    xy = xy.to(device)

                    uv = xy - 0.5
                    if plot_plane_alpha and not plot_plane_image:
                        faces[3] = self._get_estimated_alpha(
                            uv)[..., 0].detach().cpu()  # H x W x 1
                        faces[:3] = 1.0
                    else:
                        faces[3] = 1.0

                    if plot_plane_image:

                        if plot_plane_alpha:
                            faces[:3] = self._get_estimated_rgb(
                                uv).permute(2, 0, 1).detach().cpu()
                            faces[3] = self._get_estimated_alpha(
                                uv)[..., 0].detach().cpu()  # H x W x 1
                        else:
                            faces[:3] = self._get_estimated_rgb(
                                uv).permute(2, 0, 1).detach().cpu()
                finally:
                    self.to(old_device)
            else:
                fc = facecolor
                if isinstance(fc, str):
                    fc = torch.from_numpy(np.array(to_rgba(fc)))
                faces[:3] = fc[:3].unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                                      *faces.shape[-2:])
            # Plot the surface
            xyz_glob = numpyify(xyz_glob)
            ax.plot_surface(xyz_glob[..., 0], xyz_glob[..., 1], xyz_glob[..., 2], rstride=1,
                            cstride=1, facecolors=numpyify(faces.permute(1, 2, 0)), shade=False)

    # endregion
