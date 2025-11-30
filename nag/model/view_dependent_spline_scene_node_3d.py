import math
from nag.model.view_dependent_image_plane_scene_node_3d import ViewDependentImagePlaneSceneNode3D
import torch
from tools.util.torch import tensorify
from nag.model.learned_image_plane_scene_node_3d import default_control_point_times
from nag.transforms.transforms_timed_3d import interpolate_vector, _get_interpolate_index_and_distance
from nag.utils import utils
from tools.transforms.min_max import MinMax
from tools.transforms.fittable_transform import FittableTransform
from typing import Optional, Dict, Any
import torch.nn.functional as F


class ViewDependentSplineSceneNode3D(ViewDependentImagePlaneSceneNode3D):

    def __init__(self,
                 num_view_dependent_control_points: int,
                 view_dependent_data_range: torch.Tensor,
                 dtype: torch.dtype = torch.float32,
                 proxy_init: bool = False,
                 **kwargs):
        view_dependence_input_dims = 2
        # 2 * RGBA Splines, for both angles.
        view_dependence_output_dims = 2 * 4 * \
            (num_view_dependent_control_points + 2)
        self.num_view_dependent_control_points = num_view_dependent_control_points + 2

        data_range = tensorify(view_dependent_data_range, dtype=dtype)
        if data_range.shape != (2, 2):
            raise ValueError(
                "The view_dependent_data_range should be of shape (2, 2)")

        super().__init__(
            view_dependence_input_dims=view_dependence_input_dims,
            view_dependence_output_dims=view_dependence_output_dims,
            view_dependence_normalization_init=False,
            dtype=dtype,
            proxy_init=proxy_init,
            **kwargs)

        self.view_dependence_angle_normalization = MinMax(
            new_min=0., new_max=1.)
        self.view_dependence_angle_normalization.min = data_range[0]
        self.view_dependence_angle_normalization.max = data_range[1]
        self.view_dependence_angle_normalization.fitted = True

        self.register_buffer("view_dependent_timestamps", default_control_point_times(
            num_view_dependent_control_points, dtype=dtype))

        if not proxy_init:
            self.estimate_view_dependence_scaling(
                self.view_dependence_normalization)

    def _query_view_dependence(self,
                               uv: torch.Tensor,
                               sin_epoch: torch.Tensor, **kwargs) -> torch.Tensor:
        """Query the view dependence values for the given uv coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            UV coordinates of the points. Shape: (B, T, 2)
            In the range [-0.5, 0.5]
        sin_epoch : torch.Tensor
            Sine of the epoch. Shape: ()
            For coarse to fine training.

        Returns
        -------
        torch.Tensor
            View dependence values for the given uv coordinates. Shape: (B, T, VT, 8)
            2 VT angle splines, each with RGBA offsets.
        """
        input_coords = uv  # (B, T, 2)
        input_coords = input_coords+0.5
        B, T, _ = input_coords.shape
        VT = self.num_view_dependent_control_points
        input_coords = input_coords.reshape(B * T, 2)

        with torch.autocast(device_type='cuda', dtype=self.network_dtype):
            view_dependence = self.network_view_dependence(utils.mask(
                self.encoding_view_dependence(input_coords), sin_epoch))  # (B, 2)

        view_dependence = view_dependence.to(dtype=self.dtype)

        if self.view_dependence_rescaling and self.view_dependence_normalization.fitted:
            view_dependence = self.view_dependence_normalization(
                view_dependence)

        view_dependence = view_dependence.reshape(B, T, VT, 8)

        return view_dependence

    def get_view_dependence(self,
                            uv: torch.Tensor,
                            angle: torch.Tensor,
                            t: torch.Tensor,
                            sin_epoch: torch.Tensor,
                            context: Optional[Dict[str, Any]] = None,
                            is_inside: Optional[torch.Tensor] = None,
                            **kwargs) -> torch.Tensor:
        """
        Get the view dependence values for the given uv coordinates and incline angles.

        Note: Time is not considered in the view dependence.

        Parameters
        ----------
        uv : torch.Tensor
            The uv coordinates of the point and resp. time
            Shape: (B, T, 2) x, y should be in range [-0.5, 0.5]

        angle : torch.Tensor
            The incline angles of the uv (intersection) points at the resp. time.
            Should be the angle w.r.t the normal (z+) of the plane.
            Shape: (B, T, 2) angles should be in range [-pi, pi]

        t : torch.Tensor
            The times of the points. Shape: (T, )

        Returns
        -------
        torch.Tensor
            The view dependence values for the given uv coordinates and incline angles. Represents the color (RGB) and A offset, for the resp. position, angle and time.
            Shape: (B, T, 4)
        """
        B, T, _ = uv.shape
        VT = self.num_view_dependent_control_points
        BT = B * T

        if is_inside is not None:
            uv = uv[is_inside].unsqueeze(1)
            angle = angle[is_inside].unsqueeze(1)
            BT = uv.shape[0]

        if uv.numel() != 0:
            value_out = self._query_view_dependence(
                uv, sin_epoch).reshape(BT, VT, 8)  # (BT, VT, 8)

            cap_angles = torch.clamp(angle, self.view_dependence_angle_normalization.min,
                                     self.view_dependence_angle_normalization.max)
            norm_angles = self.view_dependence_angle_normalization(
                cap_angles).reshape(BT, 2)  # (BT, 2) in range [0, 1]

            values_one = value_out[:, :, :4]  # (BT, VT, 4)
            values_two = value_out[:, :, 4:]  # (BT, VT, 4)

            view_stamps = self.view_dependent_timestamps.unsqueeze(
                0).expand(BT, VT)  # (BT, VT)

            view_dependence_one = interpolate_vector(values_one,
                                                     view_stamps,
                                                     norm_angles[..., 0].unsqueeze(
                                                         1),
                                                     equidistant_times=True,
                                                     method="cubic")  # (B * T, 1, 4)
            view_dependence_two = interpolate_vector(values_two,
                                                     view_stamps,
                                                     norm_angles[..., 1].unsqueeze(
                                                         1),
                                                     equidistant_times=True,
                                                     method="cubic"
                                                     )  # (BT, 4)

            view_dependence = 0.5 * \
                (view_dependence_one + view_dependence_two)  # (BT, 1, 4)
        else:
            view_dependence = torch.zeros(
                BT, 1, 4, dtype=uv.dtype, device=uv.device)
        if context is not None:
            idx = self.get_index()
            if context.get("store_object_view_depedent_tv", False):
                if "object_view_depedent_tv" not in context:
                    context["object_view_depedent_tv"] = dict()
                vs = value_out.reshape(BT, VT, 8)
                # Compute TV along VT
                tv = torch.diff(vs, dim=-2).abs().sum(dim=-2)  # (BT, 8)
                mtv = tv.mean(dim=-1)  # (BT)
                if is_inside is not None:
                    ret = torch.zeros(B, T, dtype=mtv.dtype, device=mtv.device)
                    ret[...] = torch.nan
                    ret[is_inside] = mtv
                    mtv = ret
                else:
                    mtv = mtv.reshape(B, T)
                context["object_view_depedent_tv"][idx] = mtv
            if context.get("store_object_view_depedent_lap", False):
                if "object_view_depedent_lap" not in context:
                    context["object_view_depedent_lap"] = dict()
                vs = value_out.reshape(BT, VT, 8)
                # Compute Discrete Laplacian along VT
                laplacian = torch.tensor(
                    [1, -2, 1], device=vs.device, dtype=vs.dtype)[None, None]  # (1, 1, 3)
                # Permute vs to (BT, 8, VT) for convolution
                vs = vs.permute(0, 2, 1)  # (BT, 8, VT)
                # Flatten the first 3 dimensions to apply the convolution
                vs = vs.reshape(BT * 8, VT).unsqueeze(1)  # (BT * 8, 1, VT)

                lap = F.conv1d(vs, laplacian, padding=0,
                               bias=None)  # (BT * 8, 1, VT - 2)
                lap = lap.abs().sum(dim=-1).reshape(BT, 8)  # (BT, 8)
                lap = lap.mean(dim=-1)  # (BT)

                if is_inside is not None:
                    ret = torch.zeros(B, T, dtype=lap.dtype, device=lap.device)
                    ret[...] = torch.nan
                    ret[is_inside] = lap
                    lap = ret
                else:
                    lap = lap.reshape(B, T)

                context["object_view_depedent_lap"][idx] = lap

        if is_inside is None:
            return view_dependence.reshape(B, T, 4)
        else:
            ret = torch.zeros(B, T, 4, dtype=view_dependence.dtype,
                              device=view_dependence.device)
            ret[is_inside] = view_dependence[:, 0]
            return ret

    def estimate_view_dependence_scaling(self, normalization: FittableTransform):
        # Considering 1e6 points for the estimation, and 6 different view directions per axis
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
            view_field = self._query_view_dependence(
                grid.reshape(H * W, 2).unsqueeze(1),
                torch.tensor(0, device=device, dtype=self._translation.dtype))
            _ = normalization.fit_transform(view_field.reshape(H * W, -1))
            if device != old_device:
                self.to(old_device)
