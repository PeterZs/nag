from typing import Any, Iterable, List, Optional, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from nag.config.intrinsic_camera_config import IntrinsicCameraConfig
from nag.config.pinhole_camera_config import PinholeCameraConfig
from nag.model.camera_scene_node_3d import CameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D, compose_translation_orientation, local_to_global, spline_approximation

import torch
from tools.util.typing import VEC_TYPE
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.util.torch import tensorify
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims, rotmat_to_unitquat

from nag.transforms.transforms_timed_3d import interpolate_vector
from nag.utils import utils
from tools.transforms.min_max import MinMax
from tools.logger.logging import logger
from tools.util.format import raise_on_none
from tools.transforms.geometric.quaternion import quat_subtraction


@torch.jit.script
def distort(xy: torch.Tensor, lens_distortion: torch.Tensor) -> torch.Tensor:
    r2 = torch.sum(xy**2, dim=-1, keepdim=True)
    r4 = r2**2
    r6 = r2**3
    xy_distorted = xy * \
        (1 + lens_distortion[..., 0]*r2 +
         lens_distortion[..., 1]*r4 + lens_distortion[..., 2]*r6)
    return xy_distorted


@torch.jit.script
def distort_tang(xy: torch.Tensor, lens_distortion: torch.Tensor) -> torch.Tensor:
    r2 = torch.sum(xy**2, dim=-1)
    r4 = r2**2
    r6 = r2**3

    k1 = lens_distortion[0]
    k2 = lens_distortion[1]
    k3 = lens_distortion[2]
    p1 = lens_distortion[3]
    p2 = lens_distortion[4]

    x, y = xy[..., 0], xy[..., 1]

    x_distorted = x * (1 + k1*r2 + k2*r4 + k3*r6) + \
        2 * p1*x*y + p2*(r2 + 2*x**2)
    y_distorted = y * (1 + k1*r2 + k2*r4 + k3*r6) + p1*(r2 + 2*y**2) + 2*p2*x*y

    xy_distorted = torch.stack([x_distorted, y_distorted], dim=-1)
    return xy_distorted


def _undistort(
        xy_distorted: torch.Tensor,
        lens_distortion: torch.Tensor,
        max_iter: int = 100,
        tol: float = 6e-5,
        solved_atol: float = 1e-6
) -> torch.Tensor:

    xy_undistorted = xy_distorted.clone()
    xy_undistorted.requires_grad_(True)

    use_tang = lens_distortion[3:].abs().sum() > 0
    idx = torch.arange(
        0, xy_distorted.shape[0], dtype=torch.int64, device=xy_distorted.device)

    xy_is_solved = torch.zeros_like(xy_distorted[:, 0], dtype=torch.bool)
    xy_solved = torch.zeros_like(xy_distorted).detach()

    for _ in range(max_iter):
        if use_tang:
            xy_distorted_new = distort_tang(xy_undistorted, lens_distortion)
        else:
            xy_distorted_new = distort(xy_undistorted, lens_distortion)
        delta = xy_distorted_new - xy_distorted[~xy_is_solved]

        J = torch.zeros_like(xy_distorted_new).unsqueeze(-1).repeat(1, 1, 2)
        for i in range(2):
            J[:, :, i] = torch.autograd.grad(
                xy_distorted_new[:, i].sum(), xy_undistorted, create_graph=True)[0]

        delta_inv = torch.zeros_like(J)
        finite_J = (J.isfinite().all(dim=(-2, -1)))
        well_cond = (~torch.isclose(J[finite_J].det().abs(), torch.tensor(
            0, dtype=J.dtype, device=J.device), atol=1e-6))
        regular_J = finite_J.clone()
        regular_J[finite_J] = well_cond

        delta_inv[regular_J] = torch.linalg.inv(J[regular_J])
        if not well_cond.all():
            finite_non_well_cond = finite_J.clone()
            finite_non_well_cond[finite_J] = ~well_cond
            delta_inv[~finite_non_well_cond] = torch.linalg.pinv(
                J[~finite_non_well_cond])
        if not finite_J.all():
            # Diverged
            delta_inv[~finite_J] = torch.zeros_like(J[~finite_J])

        step = torch.einsum('ijk,ik->ij', delta_inv, delta).detach()
        xy_undistorted = xy_undistorted - step

        is_solved = (torch.isclose(torch.tensor(
            0.), delta, atol=solved_atol)).all(dim=-1)

        if is_solved.any():
            newly_solved_idx = idx[~xy_is_solved][is_solved]
            xy_solved[newly_solved_idx] = xy_undistorted[is_solved].detach()
            xy_is_solved[newly_solved_idx] = True
            xy_undistorted = xy_undistorted[~is_solved]

            if xy_is_solved.all():
                break

        if (torch.norm(delta[delta != 0.], dim=-1) < tol).all():
            # Converged
            newly_solved_idx = idx[~xy_is_solved]
            xy_solved[newly_solved_idx] = xy_undistorted.detach()
            xy_is_solved[newly_solved_idx] = True
            break

    if not xy_is_solved.all():
        logger.warning(f"Undistortion did not converge for some entries!")
        xy_solved[~xy_is_solved] = xy_distorted[~xy_is_solved]

    return xy_solved.detach()


def undistort(
        xy_distorted: torch.Tensor,
        lens_distortion: torch.Tensor,
        max_iter: int = 100,
        tol: float = 6e-5,
        batch_size: int = 10000000
) -> torch.Tensor:
    """Undistort the distorted xy coordinates using the lens distortion parameters and
    the maximum number of iterations and tolerance.

    Performs a Newton-Raphson iteration to find the undistorted xy coordinates.

    Parameters
    ----------
    xy_distorted : torch.Tensor
        Distorted xy coordinates to undistort. Shape (..., 2)
        Coordinates system assumes the lens to be at (0, 0).
    lens_distortion : torch.Tensor
        The lens distortion parameters. Shape (5, )
        k1, k2, k3, p1, p2
        First 3 are radial distortion coefficients and the last 2 are tangential distortion coefficients.
    max_iter : int, optional
        Maximum interations, by default 100
    tol : float, optional
        Tolerance to stop early, by default 6e-5

    Returns
    -------
    torch.Tensor
        The undistorted xy coordinates. Shape (..., 2)
    """
    from tools.util.torch import batched_exec
    if (lens_distortion == 0.).all():
        # No distortion, return the original coordinates
        return xy_distorted
    with torch.set_grad_enabled(True):
        if xy_distorted.shape[-2] > batch_size:
            xy_undistorted = batched_exec(xy_distorted, func=lambda x: _undistort(x,
                                                                                  lens_distortion=lens_distortion,
                                                                                  max_iter=max_iter, tol=tol), batch_size=batch_size, free_memory=True)
            return xy_undistorted
        else:
            return _undistort(xy_distorted, lens_distortion, max_iter, tol)


@torch.jit.script
def get_local_ray_direction(
    uv: torch.Tensor,
    inverse_intrinsics: torch.Tensor,
    lens_distortion: torch.Tensor,
    uv_includes_time: bool = False,
) -> torch.Tensor:
    """
    Get the ray direction for the given pixel coordinates in the local camera coordinate system.
    This function is time dependent as the inverse intrinsics matrix is time dependent.

    Parameters
    ----------
    uv : torch.Tensor, optional
        Given pixel coordinates to get the ray origins for, by default None
        Pixel coordinates are assumed to be in the image plane in order (x, y). [0, width) and [0, height)
        Shape ([..., B,] 3) for height, width, and z.
        If uv_includes_time is True, the first dimension is assumed to be the time dimension and therefore
        must be of shape (T, [..., B], 3) where T is the number of times.

    inverse_intrinsics : torch.Tensor
        The inverse intrinsics matrix of the camera. Shape (T, 3, 3)

    lens_distortion : torch.Tensor
        Lens distortion parameters as (kappa_1, kappa_2, kappa_3, kappa_4, kappa_5) whereby 1-3 are radial and 4, 5 tangential correction coefficients.
        Shape (5, ) but only the first 3 are used.

    uv_includes_time : bool, optional
        If the uv coordinates include the time, by default False
        Will interpret the first dimension of uv as the time dimension.
        If True, the uv tensor must be of shape (T, [..., B], 3) where T is the number of times.

    Returns
    -------
    torch.Tensor
        The ray direction in the local camera coordinate system.
        As (T, [... B], 3) (x, y, z) tensor.
        Due to the definition of the camera coordinate system (homographic coordinates), the z component is 1 for a point on the camera / image plane.
        which is graphically represented as Z = 0 so one must subtract (0, 0, 1) to get the image plane.

    """
    # Convert uv to homogeneous coordinates
    uvz_plane = torch.cat(
        [uv[..., :2], torch.ones_like(uv[..., :1])], dim=-1)
    # Flatten
    if not uv_includes_time:
        xyz, batch_dims = flatten_batch_dims(uvz_plane, -2)
    else:
        T = uvz_plane.shape[0]
        B = torch.tensor(uvz_plane.shape[1:-1], dtype=torch.int64)
        B_D = int(torch.prod(B)) if len(B) > 0 else 1
        C = uvz_plane.shape[-1]
        xyz = uvz_plane.reshape(T, B_D, C)
        batch_dims: List[int] = torch.cat(
            [torch.tensor([T], dtype=torch.int64), B], dim=0).to(torch.int64).tolist()

    B = xyz.shape[-2]

    T = inverse_intrinsics.shape[0]
    # Repeat xyz for times
    if not uv_includes_time:
        xyz = xyz.unsqueeze(0).repeat(T, 1, 1)
        bd: List[int] = torch.cat([torch.tensor([T], dtype=torch.int64), torch.tensor(
            batch_dims, dtype=torch.int64)], dim=0).to(torch.int64).tolist()
        batch_dims = bd

    inverse_intrinsics = inverse_intrinsics.unsqueeze(1).repeat(1, B, 1, 1)
    # Multiply by the intrinsics inverse
    xyz_cam = torch.bmm(
        inverse_intrinsics.reshape(T * B, 3, 3),
        xyz.reshape(T * B, 3, 1)
    ).squeeze(-1)

    xy = xyz_cam[..., :2]

    # Korrect for lens and distortion
    dist_xy = distort(xy, lens_distortion)

    # Add Z coordinate of 1 for the image plane again
    xyz_cam = torch.cat([dist_xy, torch.ones_like(dist_xy[..., :1])], dim=-1)

    return unflatten_batch_dims(xyz_cam, batch_dims)


def local_to_image_coordinates(
    v: torch.Tensor,
    intrinsics: torch.Tensor,
    focal_length: torch.Tensor,
    lens_distortion: Optional[torch.Tensor] = None,
    v_includes_time: bool = False,
) -> torch.Tensor:
    """Inverse of get_local_ray_direction. Converts the local coordinates to image coordinates.

    Parameters
    ----------
    v : torch.Tensor
        XYZ coordinates in the local camera coordinate system. Shape (T, [... B,] 3) (x, y, z)
        Inputs
    intrinsics : torch.Tensor
        The intrinsics matrix of the camera. Shape (T, 3, 3)

    focal_length : torch.Tensor
        The focal length of the camera. Shape () 0d tensor

    lens_distortion : Optional[torch.Tensor], optional
        The lens distortion parameters. Shape (5, )
        If None, it is assumed that there is no lens distortion, by default None
        If specified, it tries to undistort the xy coordinates.

    v_includes_time : bool, optional
        If the xy coordinates include time, by default False
        If True, the xy tensor must be of shape (T, [... B], 3) where T is the number of times.

    Returns
    -------
    torch.Tensor
        The image coordinates in the image plane. Shape (T, [... B,] 2) (x, y) / uv
    """
    v, batch_dims = flatten_batch_dims(v, -2 if not v_includes_time else -3)
    intrinsics, _ = flatten_batch_dims(intrinsics, -3)
    if not v_includes_time:
        T = intrinsics.shape[0]
        v = v.unsqueeze(0).repeat(T, 1, 1)
    else:
        if v.shape[0] != intrinsics.shape[0]:
            raise ValueError(
                "Time steps of xy must match the time steps of the intrinsics.")
    T, B, _ = v.shape

    # Add 1 in z to shift z back to z=1 for homography
    v = v[..., :3] + \
        torch.tensor([0., 0., focal_length], dtype=v.dtype, device=v.device)

    v = v * 1 / focal_length
    # Normalize to z=1
    v = v[..., :2] / v[..., 2:3]

    if lens_distortion is not None:
        v = undistort(v[..., :2].reshape(T * B, 2),
                      lens_distortion).reshape(T, B, 2)
    xyz = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
    intrinsics = intrinsics.unsqueeze(1).repeat(1, B, 1, 1)
    uv = torch.bmm(intrinsics.reshape(T * B, 3, 3),
                   xyz.reshape(T * B, 3, 1)).reshape(T, B, 3).squeeze(-1)
    uv = uv[..., :2]
    return uv


@torch.jit.script
def get_local_rays(
        uv: torch.Tensor,
        inverse_intrinsics: torch.Tensor,
        lens_distortion: torch.Tensor,
        focal_length: torch.Tensor,
        uv_includes_time: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the local ray directions and origins for the given pixel coordinates.

    Assumes rays start at z=0.

    Parameters
    ----------
    uv : torch.Tensor
        Given pixel coordinates to get the ray origins for.
        Pixel coordinates are assumed to be in the image plane in order (x, y). [0, width) and [0, height)
        Shape ([T], [..., B,] 2).

    inverse_intrinsics : torch.Tensor
        The inverse intrinsics matrix of the camera. Shape (T, 3, 3)

    lens_distortion : torch.Tensor
        Lens distortion parameters as (kappa_1, kappa_2, kappa_3, kappa_4, kappa_5) whereby 1-3 are radial and 4, 5 tangential correction coefficients.
        Shape (5, ) but only the first 3 are used.

    focal_length : torch.Tensor
        The focal length of the camera. Shape () 0d tensor

    uv_includes_time : bool, optional
        If the uv coordinates include time as the first dimension, by default False

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        1. The ray origins in the local camera coordinate system.
        As (..., 3) (x, y, z) tensor. Shape ([..., B,], T, 3)
        z = 0.
        2. The ray directions in the local camera coordinate system.
        As (..., 3) (x, y, z) tensor. Shape ([..., B,], T, 3)

    """
    ray_directions = get_local_ray_direction(uv=uv,
                                             inverse_intrinsics=inverse_intrinsics,
                                             lens_distortion=lens_distortion,
                                             uv_includes_time=uv_includes_time)
    # Multiply by the focal length
    ray_directions = ray_directions * focal_length

    # Get endpoint of rays by multiplying with a distance
    sub = torch.zeros(3, dtype=ray_directions.dtype,
                      device=ray_directions.device)
    sub[2] = focal_length
    ray_origins = ray_directions - sub

    # Permute time to Second to last
    ray_directions = torch.moveaxis(ray_directions, 0, -2)
    ray_origins = torch.moveaxis(ray_origins, 0, -2)

    return ray_origins, ray_directions


@torch.jit.script
def get_global_rays(
        uv: torch.Tensor,
        inverse_intrinsics: torch.Tensor,
        lens_distortion: torch.Tensor,
        global_position: torch.Tensor,
        focal_length: torch.Tensor,
        uv_includes_time: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the global ray origins and directions for the given pixel coordinates.

    Parameters
    ----------
    uv : torch.Tensor
        Given pixel coordinates to get the ray origins for.
        Pixel coordinates are assumed to be in the image plane in order (x, y). [0, width) and [0, height)
        Shape ([T], [..., B,] 2).

    inverse_intrinsics : torch.Tensor
        The inverse intrinsics matrix of the camera. Shape (T, 3, 3)

    lens_distortion : torch.Tensor
        Lens distortion parameters as (kappa_1, kappa_2, kappa_3, kappa_4, kappa_5) whereby 1-3 are radial and 4, 5 tangential correction coefficients.
        Shape (5, ) but only the first 3 are used.

    global_position : torch.Tensor
        Global position matrix. Shape (T, 4, 4) where T is the number of times.

    image_resolution : torch.Tensor
        The image resolution in pixels. Shape (2, ) (width, height) (x, y) in pixels.

    uv_includes_time : bool, optional
        If the uv coordinates include time as the first dimension, by default False
        If True, the uv tensor must be of shape (T, [..., B], 2) where T is the number of times.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The global ray origins and directions in the global coordinate system.
        As (..., 3) (x, y, z) tensor. Shape ([..., B,], T, 3) for origins and directions.
    """
    ray_directions = get_local_ray_direction(uv=uv,
                                             inverse_intrinsics=inverse_intrinsics,
                                             lens_distortion=lens_distortion,
                                             uv_includes_time=uv_includes_time)

    # focal_length = inverse_intrinsics[0, 0, 0] * image_resolution[0]

    # Multiply by the focal length
    ray_directions = ray_directions * focal_length

    # Get endpoint of rays by multiplying with a distance
    sub = torch.zeros(3, dtype=ray_directions.dtype,
                      device=ray_directions.device)
    sub[2] = focal_length
    ray_origins = ray_directions - sub

    # Permute time to Second to last
    ray_directions = torch.moveaxis(ray_directions, 0, -2)
    ray_origins = torch.moveaxis(ray_origins, 0, -2)

    # Add w=1
    ray_directions = torch.cat(
        [ray_directions, torch.ones_like(ray_directions[..., :1])], dim=-1)
    ray_origins = torch.cat(
        [ray_origins, torch.ones_like(ray_origins[..., :1])], dim=-1)

    global_ray_origins = local_to_global(
        global_position,
        ray_origins, v_include_time=True)[..., :3]

    rd = (2 * ray_directions[..., :3] - sub)

    # Add w=1
    rd = torch.cat([rd, torch.ones_like(rd[..., :1])], dim=-1)

    # One could also just use bmm with the rotation matrix only, but i am lazy
    global_ray_endpoints = local_to_global(
        global_position,
        rd,
        v_include_time=True)[..., :3]

    global_ray_directions = global_ray_endpoints - global_ray_origins

    # Normalize the directions
    global_ray_directions = global_ray_directions / \
        torch.norm(global_ray_directions, dim=-1, keepdim=True)

    return global_ray_origins, global_ray_directions


class TimedCameraSceneNode3D(CameraSceneNode3D, TimedDiscreteSceneNode3D):
    """A scene node representing a camera in 3D space with timed discrete position and orientation."""

    def __init__(
        self,
            image_resolution: VEC_TYPE,
            lens_distortion: VEC_TYPE,
            intrinsics: VEC_TYPE,
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
            image_resolution=image_resolution,
            lens_distortion=lens_distortion,
            intrinsics=intrinsics,
            translation=translation,
            orientation=orientation,
            position=position,
            times=times,
            name=name,
            children=children,
            decoding=decoding,
            dtype=dtype,
            **kwargs)

    def get_intrinsics(self) -> torch.Tensor:
        """Return the intrinsics matrix of the camera."""
        return self._intrinsics

    @property
    def focal_length(self) -> torch.Tensor:
        """Following the pinhole camera model, the focal length is the product of the first element of the intrinsics matrix with the resolution in x or eq. to 1 / pixel width."""
        return (self._intrinsics[0, 0, 0] / self._image_resolution[-1]).squeeze()

        # if self._image_resolution[-1] < self._image_resolution[-2]:
        #     # X < Y portrait mode - use X
        #     return (self._intrinsics[0, 0, 0] / self._image_resolution[-1]).squeeze()
        # else:
        #     # X > Y landscape mode - use Y
        #     return (self._intrinsics[0, 1, 1] / self._image_resolution[-2]).squeeze()

    @classmethod
    def from_bundle(cls,
                    bundle: dict,
                    image_resolution: Optional[VEC_TYPE] = None,
                    frame_indices_filter: Optional[Any] = None,
                    position_spline_approximation: bool = False,
                    position_spline_control_points: Optional[int] = None,
                    model: Any = None,
                    **kwargs) -> "TimedCameraSceneNode3D":
        with torch.no_grad():
            frame_timestamps = utils.get_frame_timestamps(
                bundle).to(torch.float32)
            motion = bundle["motion"]
            if frame_indices_filter is not None:
                frame_timestamps = frame_timestamps[frame_indices_filter]
            rotations = utils.motion_to_rotmat(motion, frame_timestamps)
            # Base Position of camera is 0,0,0 facing the z axis
            position = torch.zeros(1, 3, dtype=torch.float32)
            reference_rotation = rotations[0]
            normalized_rotations = reference_rotation.T @ rotations
            norm_quat = rotmat_to_unitquat(normalized_rotations)

            # Normalize the time stamps
            mm = MinMax(0, 1, dim=-1)
            normalized_frame_timestamps = mm.fit_transform(
                frame_timestamps).float()

            intrinsics = utils.get_intrinsics(bundle)
            lens_distortion = utils.get_lens_distortion(bundle)
            if frame_indices_filter is not None:
                intrinsics = intrinsics[frame_indices_filter]
            if image_resolution is None:
                image_resolution = torch.tensor(
                    [bundle['raw_0']['width'], bundle['raw_0']['height']], dtype=torch.float32)

            times = normalized_frame_timestamps

            if position_spline_approximation:
                from nag.transforms.transforms_timed_3d import find_optimal_spline
                K = position_spline_control_points if position_spline_control_points is not None else len(
                    frame_timestamps) // 2
                _, norm_quat = spline_approximation(
                    None, norm_quat, normalized_frame_timestamps, K)

                new_intrinsics = torch.zeros(K, 3, 3, dtype=intrinsics.dtype)
                new_intrinsics[...] = intrinsics[:K]
                # Interpolate the principal point - varies do to stabilization
                new_intrinsics[:, :2, 2] = find_optimal_spline(
                    intrinsics[:, :2, 2], times, K)[0]
                intrinsics = new_intrinsics
                times = torch.linspace(0, 1, K, dtype=position.dtype)

            args = dict(
                image_resolution=image_resolution,
                lens_distortion=lens_distortion,
                intrinsics=intrinsics,
                translation=position,
                orientation=norm_quat,
                times=times,
                **kwargs
            )

            if model is not None:
                from nag.model.nag_functional_model import NAGFunctionalModel
                if isinstance(model, NAGFunctionalModel):
                    # Patch the values in the args
                    args = model.patch_camera_args(args)
            return cls(
                **args
            )

    @classmethod
    def from_images(cls,
                    images: torch.Tensor,
                    frame_indices_filter: Optional[Any] = None,
                    model: Any = None,
                    camera_config: Optional[IntrinsicCameraConfig] = None,
                    dtype: torch.dtype = torch.float32,
                    position_spline_approximation: bool = False,
                    position_spline_control_points: Optional[int] = None,
                    resolution: Optional[Tuple[int, int]] = None,
                    need_image_filter: bool = True,
                    normalize_camera: bool = False,
                    **kwargs) -> "TimedCameraSceneNode3D":
        with torch.no_grad():
            if frame_indices_filter is not None and need_image_filter:
                images = images[frame_indices_filter]
            if camera_config is None:
                camera_config = IntrinsicCameraConfig()
            device = images.device
            if resolution is None:
                T, C, H, W = images.shape
            else:
                T, C, _, _ = images.shape
                H, W = resolution
            if not isinstance(camera_config, PinholeCameraConfig) and not str(camera_config).startswith("PinholeCameraConfig"):
                norm_quat = torch.zeros(T, 4, dtype=dtype)
                norm_quat[:, -1] = 1.
                position = torch.zeros(1, 3, dtype=dtype)
            else:
                p = tensorify(camera_config.position, dtype=dtype)

                if normalize_camera:
                    # P0 = p[0]
                    # P0 = torch.linalg.inv(P0)
                    # p = torch.bmm(P0.expand(p.shape[0], -1, -1), p)

                    translations = p[:, :3, 3]
                    quat = rotmat_to_unitquat(p[:, :3, :3])
                    norm_quat = quat_subtraction(
                        quat, quat[0].unsqueeze(0).expand_as(quat))
                    norm_translations = translations - \
                        translations[0].unsqueeze(0)
                    p = compose_translation_orientation(
                        norm_translations, norm_quat)

                norm_quat = rotmat_to_unitquat(p[:, :3, :3])
                position = p[:, :3, 3]
                if frame_indices_filter is not None:
                    position = position[frame_indices_filter]
                    norm_quat = norm_quat[frame_indices_filter]

            normalized_frame_timestamps = torch.linspace(0, 1, T, dtype=dtype)

            intrinsics = camera_config.get_intrinsics(
                (W, H), dtype=dtype, device=device).unsqueeze(0).repeat(T, 1, 1)
            lens_distortion = camera_config.get_lens_distortion(
                dtype=dtype, device=device)
            image_resolution = torch.tensor([H, W], dtype=torch.int32)
            times = normalized_frame_timestamps

            if position_spline_approximation:
                from nag.transforms.transforms_timed_3d import find_optimal_spline
                K = position_spline_control_points if position_spline_control_points is not None else len(
                    normalized_frame_timestamps) // 2

                # save_tensor(None, position, "position", time=None, path="temp/spline_approx_debug", index="camera")
                # save_tensor(None, norm_quat, "norm_quat", time=None, path="temp/spline_approx_debug", index="camera")
                # save_tensor(None, normalized_frame_timestamps, "normalized_frame_timestamps", time=None, path="temp/spline_approx_debug", index="camera")
                # save_tensor(None, torch.tensor(K), "k", time=None, path="temp/spline_approx_debug", index="camera")

                position, norm_quat, _ctimes = spline_approximation(
                    position, norm_quat, normalized_frame_timestamps, K)
                # Remove first and last time
                position = position[1:-1]
                norm_quat = norm_quat[1:-1]
                times = torch.linspace(0, 1, K, dtype=position.dtype)
                # Intrinsics are not time dependent
                intrinsics = intrinsics[:K]

            args = dict(
                image_resolution=image_resolution,
                lens_distortion=lens_distortion,
                intrinsics=intrinsics,
                translation=position,
                orientation=norm_quat,
                times=times,
                dtype=dtype,
                **kwargs
            )

            if model is not None:
                from nag.model.nag_functional_model import NAGFunctionalModel
                if isinstance(model, NAGFunctionalModel):
                    # Patch the values in the args
                    args = model.patch_camera_args(args)
            return cls(
                **args
            )

    @classmethod
    def from_camera_config(cls,
                           config: IntrinsicCameraConfig,
                           resolution: Tuple[int, int],
                           times: torch.Tensor,
                           **kwargs
                           ) -> "TimedCameraSceneNode3D":
        intrinsics = config.get_intrinsics(resolution).unsqueeze(0)
        lens_distortion = config.get_lens_distortion()
        image_resolution = torch.tensor(resolution, dtype=torch.int32)
        return cls(
            intrinsics=intrinsics,
            lens_distortion=lens_distortion,
            image_resolution=image_resolution,
            times=times,
            **kwargs
        )

    @classmethod
    def from_pinhole_camera_config(cls,
                                   config: PinholeCameraConfig,
                                   normalize: bool = True,
                                   **kwargs
                                   ) -> "TimedCameraSceneNode3D":
        intrinsics = config.get_intrinsics(config.resolution).unsqueeze(0)
        lens_distortion = config.get_lens_distortion()
        image_resolution = torch.tensor(config.resolution, dtype=torch.int32)
        positions = tensorify(
            config.position) if config.position is not None else None
        positions = flatten_batch_dims(
            positions, -3)[0] if positions is not None else None
        times = tensorify(config.times) if config.times is not None else None
        if times is None and positions is not None:
            times = torch.linspace(
                0, 1, positions.shape[0], dtype=positions.dtype)
        if normalize and positions is not None:
            translations = positions[:, :3, 3]
            quat = rotmat_to_unitquat(positions[:, :3, :3])
            norm_quat = quat_subtraction(
                quat, quat[0].unsqueeze(0).expand_as(quat))
            norm_translations = translations - translations[0].unsqueeze(0)
            positions = compose_translation_orientation(
                norm_translations, norm_quat)
        return cls(
            intrinsics=intrinsics,
            lens_distortion=lens_distortion,
            image_resolution=image_resolution.flip(-1),
            position=positions,
            times=times,
            **kwargs
        )

    @classmethod
    def from_resolution(cls,
                        resolution: Tuple[int, int],
                        num_times: int = 1,
                        model: Any = None,
                        camera_config: Optional[IntrinsicCameraConfig] = None,
                        dtype: torch.dtype = torch.float32,
                        translation: Optional[torch.Tensor] = None,
                        orientation: Optional[torch.Tensor] = None,
                        position_spline_approximation: bool = False,
                        position_spline_control_points: Optional[int] = None,
                        **kwargs) -> "TimedCameraSceneNode3D":
        with torch.no_grad():
            if camera_config is None:
                camera_config = IntrinsicCameraConfig()
            T = num_times
            W, H = resolution
            if orientation is None:
                orientation = torch.zeros(T, 4, dtype=dtype)
                orientation[:, 0] = 1.
            if translation is None:
                translation = torch.zeros(T, 3, dtype=dtype)

            normalized_frame_timestamps = torch.linspace(0, 1, T, dtype=dtype)

            intrinsics = torch.eye(3, dtype=dtype).unsqueeze(0).repeat(T, 1, 1)
            # Set optical axis / principal point to the center of the image
            if camera_config.principal_point is None:
                intrinsics[:, :2, 2] = (torch.tensor(
                    [W, H], dtype=dtype) / 2).unsqueeze(0)
            else:
                intrinsics[:, :2, 2] = torch.tensor(
                    camera_config.principal_point, dtype=dtype).unsqueeze(0)
            max_hw = min(H, W)
            intrinsics[:, 0, 0] = camera_config.focal_length * max_hw
            intrinsics[:, 1, 1] = camera_config.focal_length * max_hw
            intrinsics[:, 0, 1] = camera_config.skew

            lens_distortion = torch.zeros(5, dtype=dtype) if camera_config.lens_distortion is None else torch.tensor(
                camera_config.lens_distortion, dtype=dtype)

            image_resolution = torch.tensor([H, W], dtype=torch.int32)

            times = normalized_frame_timestamps

            if position_spline_approximation:
                from nag.transforms.transforms_timed_3d import find_optimal_spline
                K = position_spline_control_points if position_spline_control_points is not None else len(
                    normalized_frame_timestamps) // 2
                translation, orientation = spline_approximation(
                    translation, orientation, normalized_frame_timestamps, K)
                times = torch.linspace(0, 1, K, dtype=translation.dtype)
                # Intrinsics are not time dependent
                intrinsics = intrinsics[:K]

            args = dict(
                image_resolution=image_resolution,
                lens_distortion=lens_distortion,
                intrinsics=intrinsics,
                translation=translation,
                orientation=orientation,
                times=times,
                dtype=dtype,
                **kwargs
            )

            if model is not None:
                from nag.model.nag_functional_model import NAGFunctionalModel
                if isinstance(model, NAGFunctionalModel):
                    # Patch the values in the args
                    args = model.patch_camera_args(args)
            return cls(
                **args
            )

    def set_intrinsics(self, intrinsics: torch.Tensor) -> None:
        """Set the intrinsics matrix of the camera."""
        # Make sure intrinsics are tx3x3
        if len(intrinsics.shape) == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.shape[0] != 1 and intrinsics.shape[0] != self._times.shape[0]:
            raise ValueError("Intrinsics must have the same length as times.")
        if intrinsics.shape[0] == 1:
            intrinsics = intrinsics.repeat(self._times.shape[0], 1, 1)
        self._intrinsics = intrinsics
        self._inverse_intrinsics = torch.inverse(intrinsics)

    def get_inverse_intrinsics(self,
                               t: Optional[torch.Tensor] = None,
                               right_idx: Optional[torch.Tensor] = None,
                               rel_frac: Optional[torch.Tensor] = None
                               ) -> torch.Tensor:
        if t is None:
            return self._inverse_intrinsics
        else:
            t, batch_dims = flatten_batch_dims(t, -1)
            T, C, _ = self._inverse_intrinsics.shape
            S = t.shape[0]
            if len(self._times) == 1:
                return self._inverse_intrinsics.repeat(S, 1, 1)
            inp_int = interpolate_vector(self._inverse_intrinsics.reshape(T, C*C),
                                         self._times,
                                         steps=t,
                                         equidistant_times=self._equidistant_times,
                                         right_idx=right_idx, rel_frac=rel_frac,
                                         method=self._interpolation).reshape(S, C, C)
            return inp_int

    def get_intrinsics(self,
                       t: Optional[torch.Tensor] = None,
                       right_idx: Optional[torch.Tensor] = None,
                       rel_frac: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        if t is None:
            return self._intrinsics
        else:
            t, batch_dims = flatten_batch_dims(t, -1)
            T, C, _ = self._intrinsics.shape
            S = t.shape[0]
            if len(self._times) == 1:
                return self._intrinsics.repeat(S, 1, 1)
            inp_int = interpolate_vector(self._intrinsics.reshape(T, C*C),
                                         self._times,
                                         steps=t,
                                         equidistant_times=self._equidistant_times,
                                         right_idx=right_idx, rel_frac=rel_frac,
                                         method=self._interpolation).reshape(S, C, C)
            return inp_int

    def get_optical_axis(self) -> torch.Tensor:
        return super().get_optical_axis()[0, ...]

    def _get_ray_direction(self,
                           uv: Optional[torch.Tensor] = None,
                           t: Optional[torch.Tensor] = None,
                           uv_includes_time: bool = False
                           ) -> torch.Tensor:
        """
        Get the ray direction for the given pixel coordinates in the local camera coordinate system.


        Parameters
        ----------
        uv : Optional[torch.Tensor], optional
            Given pixel coordinates to get the ray origins for, by default None
            Pixel coordinates are assumed to be in the image plane in order (x, y). [0, width) and [0, height)
            Shape ([..., B,] 2) for width and height.
            If uv_includes_time is True, the first dimension is assumed to be the time dimension and therefore
            must be of shape (T, [..., B], 2) where T is the number of times.


        t : Optional[torch.Tensor], optional
            The time at which to get the ray direction, by default None
            None will get the ray direction for all times.
            Should be of shape (T, ) where T is the number of times.

        uv_includes_time : bool, optional
            If the uv coordinates include the time, by default False
            Will interpret the first dimension of uv as the time dimension.
            If True, the uv tensor must be of shape (T, [..., B], 3) where T is the number of times.

        Returns
        -------
        torch.Tensor
            The ray direction in the local camera coordinate system.
            As (T, [..., B], 3) (x, y, z) tensor.
            Due to the definition of the camera coordinate system (homographic coordinates), the z component is 1 for a point on the camera / image plane.
            which is graphically represented as Z = 0 so one must subtract (0, 0, 1) to get the image plane.

        """
        if uv is None:
            uv = self._sample_pixel_grid()
            if uv_includes_time:
                uv = uv.unsqueeze(0).repeat(self._times.shape[0], 1, 1)
        else:
            if uv_includes_time:
                # Check if the first dimension is the time dimension
                if uv.shape[0] != self._times.shape[0]:
                    raise ValueError(
                        "The first dimension of uv must be the time dimension if uv_includes_time is True.")
        if t is None:
            t = self._times

        t, t_batch_dims = flatten_batch_dims(t, -1)
        return get_local_ray_direction(uv=uv, inverse_intrinsics=self.get_inverse_intrinsics(
            t=t), lens_distortion=self.get_lens_distortion(), uv_includes_time=uv_includes_time)

    def get_local_rays(self, uv: torch.Tensor, t: Optional[torch.Tensor] = None, uv_includes_time: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the local ray origins and directions for the given pixel coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            UV coordinates of the pixel to get the rays for.
            Shape ([T], [..., B,] 2) for width and height.
        t : Optional[torch.Tensor], optional
            Time to get the rays for, by default None
        uv_includes_time : bool, optional
            if UV includes time as the first dimension., by default False

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            1. The ray origins in the local camera coordinate system.
            As (..., 3) (x, y, z) tensor. Shape ([..., B,], T, 3) z = 0.
            2. The ray directions in the local camera coordinate system.
            As (..., 3) (x, y, z) tensor. Shape ([..., B,], T, 3)
        """
        ray_origins, ray_directions = get_local_rays(uv=uv,
                                                     inverse_intrinsics=self.get_inverse_intrinsics(
                                                         t=t),
                                                     lens_distortion=self.get_lens_distortion(),
                                                     focal_length=self.focal_length,
                                                     uv_includes_time=uv_includes_time)
        return ray_origins, ray_directions

    def get_global_rays(self, uv: torch.Tensor, t: Optional[torch.Tensor] = None, uv_includes_time: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the global ray origins and directions for the given pixel coordinates.

        Parameters
        ----------
        uv : torch.Tensor
            Given pixel coordinates to get the ray origins for.
            Pixel coordinates are assumed to be in the image plane in order (x, y). [0, width) and [0, height)
            Shape ([T], [..., B,] 2).

        t : Optional[torch.Tensor], optional
            The time at which to get the ray direction, by default None
            None will get the ray direction for all times. Should be of shape (T, ) where T is the number of times.

        uv_includes_time : bool, optional
            If the uv coordinates include time as the first dimension, by default False
            If True, the uv tensor must be of shape (T, [..., B], 2) where T is the number of times.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The global ray origins and directions in the global coordinate system.
            As (..., 3) (x, y, z) tensor. Shape ([..., B,], T, 3) for origins and directions.
        """
        global_ray_origins, global_ray_directions = get_global_rays(uv=uv,
                                                                    inverse_intrinsics=self.get_inverse_intrinsics(
                                                                        t=t),
                                                                    lens_distortion=self.get_lens_distortion(),
                                                                    global_position=self.get_global_position(
                                                                        t=t),
                                                                    focal_length=self.focal_length,
                                                                    uv_includes_time=uv_includes_time)
        return global_ray_origins, global_ray_directions

    def _get_global_pixel_coordinates(self,
                                      pixel_size: Optional[torch.Tensor] = None,
                                      sample_grid: Optional[torch.Tensor] = None,
                                      t: Optional[torch.Tensor] = None,
                                      **kwargs) -> torch.Tensor:
        if sample_grid is None:
            sample_grid = self._sample_pixel_grid(subsample=kwargs.get(
                "pixel_grid_subsample", 50), include_borders=kwargs.get("pixel_grid_borders", True))
        pixel_grid = self._get_pixel_coordinates(
            pixel_size=pixel_size, uv=sample_grid)
        # 2D Grid to Vectors
        pixels, batch_dims = flatten_batch_dims(pixel_grid, 1)
        # Stack w=1
        pixels = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1)
        global_pixels = self.local_to_global(pixels, t=t)[..., :3]
        return unflatten_batch_dims(global_pixels, batch_dims)

    def local_to_image_coordinates(self,
                                   v: torch.Tensor,
                                   t: Optional[torch.Tensor] = None,
                                   v_includes_time: bool = False,
                                   return_ok: bool = False,
                                   ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Inverse of get_local_ray_direction. Converts the local coordinates to image coordinates.

        Parameters
        ----------
        v : torch.Tensor
            XYZ coordinates in the local camera coordinate system. Shape ([... B,], 3) (x, y, z) if v_includes_time is False
            else ([... B,] T, 3) (x, y, z) if v_includes_time is True
        t : Optional[torch.Tensor], optional
            Timeframe to plot, by default None
        v_includes_time : bool, optional
            If the v coordinates include time, by default False
            If True, the v tensor must be of shape ([... B,] T, 3) where T is the number of times.
        return_ok : bool, optional
            If True, returns a boolean tensor indicating if the point is in front of the camera, by default False

        Returns
        -------
        If return_ok is False:
        torch.Tensor
            The image coordinates in the image plane. Shape ([... B,], T, 2) (x, y) / uv
        If return_ok is True:
        Tuple[torch.Tensor, torch.Tensor]
            The image coordinates in the image plane and a boolean tensor indicating if the point is in front of the camera.
            Shapes ([... B,], T, 2) (x, y) / uv and ([... B,], T) respectively.


        """
        if v_includes_time:
            # Swap time to the first dimension
            v = v.permute(1, 0, 2)  # Convert to T, B, 3
        if t is None:
            t = self._times

        out = local_to_image_coordinates(v=v,
                                         intrinsics=self.get_intrinsics(t=t),
                                         focal_length=self.focal_length,
                                         lens_distortion=self.get_lens_distortion(),
                                         v_includes_time=v_includes_time)

        if return_ok:
            valid = (v[..., 2] + self.focal_length) > 0
            return out.permute(1, 0, 2)[:, :, :2], valid.permute(1, 0)
        else:
            # Convert output to B, T, 2
            return out.permute(1, 0, 2)[:, :, :2]

    def global_to_image_coordinates(self,
                                    v: torch.Tensor,
                                    t: Optional[torch.Tensor] = None,
                                    v_includes_time: bool = False,
                                    clamp: bool = True,
                                    return_ok: bool = False,
                                    ) -> torch.Tensor:
        """Converts the global coordinates to image coordinates.

        Parameters
        ----------
        v : torch.Tensor
            XYZ coordinates in the global coordinate system. Shape ([... B,], 3) (x, y, z) if v_includes_time is False
            else ([... B,] T, 3) (x, y, z) if v_includes_time is True
        t : Optional[torch.Tensor], optional
            Times of v. By default, self._times is used.
        v_includes_time : bool, optional
            If the v coordinates include time, by default False
            If True, the v tensor must be of shape ([... B,] T, 3) where T is the number of times.

        Returns
        -------
        torch.Tensor
            The image coordinates in the image plane. Shape ([... B,], T, 2) (x, y) / uv

        """
        if v_includes_time:
            v, shp = flatten_batch_dims(v, -3)
        else:
            v, shp = flatten_batch_dims(v, -2)
        local_v = self.global_to_local(v, t=t, v_include_time=v_includes_time)
        image_v = self.local_to_image_coordinates(
            local_v, t=t, v_includes_time=True, return_ok=return_ok)  # Shape: (B, T, 2)
        if return_ok:
            image_v, valid = image_v

            in_x_range = (image_v[..., 0] >= 0) & (
                image_v[..., 0] < self._image_resolution[1])
            in_y_range = (image_v[..., 1] >= 0) & (
                image_v[..., 1] < self._image_resolution[0])
            valid = valid & in_x_range & in_y_range

        # Clamp to image resolution
        if clamp:
            image_v[:, :, 0] = image_v[:, :, 0].clamp(
                0, self._image_resolution[1])
            image_v[:, :, 1] = image_v[:, :, 1].clamp(
                0, self._image_resolution[0])
        if not return_ok:
            return unflatten_batch_dims(image_v, shp)
        else:
            return unflatten_batch_dims(image_v, shp), unflatten_batch_dims(valid, shp)

    def _plot_pixel_grid(self,
                         ax: Axes,
                         pixel_size: Optional[torch.Tensor] = None,
                         sample_grid: Optional[torch.Tensor] = None,
                         **kwargs):
        t_camera_pos = tensorify(kwargs.get(
            't', self._times[-1]))

        args = dict(kwargs)
        args.pop("t")
        global_pixels = self._get_global_pixel_coordinates(
            pixel_size=pixel_size, sample_grid=sample_grid, t=t_camera_pos, **args)
        # Flatten the batch dims
        global_pixels = flatten_batch_dims(global_pixels, -2)[0]
        ax.scatter(*global_pixels.detach().cpu().numpy().T, color='b', s=1)

    def _plot_camera_wireframe(self,
                               ax: Axes,
                               pixel_size: Optional[torch.Tensor] = None,
                               **kwargs):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        if pixel_size is None:
            pixel_size = (1 / self._image_resolution).max().repeat(2)

        t_camera_pos = tensorify(kwargs.get(
            't', self._times[-1]))

        # Projection center
        # Get the min focal length as projection distance / center of projection
        # Would be suprised if the focal length is different over time :)
        f = self.focal_length

        corners = self.get_camera_pixel_corner_coordinates()
        # Transpose corners as matplotlib uses xy instead of yx for pytorch
        corners = torch.flip(corners, dims=(-1,))

        # Add z=0 and scale = 1
        corners = torch.cat([corners, torch.zeros_like(
            corners[..., :1]), torch.ones_like(corners[..., :1])], dim=-1)

        batched_t, b_dim = flatten_batch_dims(t_camera_pos, -1)

        # Add the camera position
        global_corners = self.local_to_global(corners, t=batched_t)

        if f <= 1:
            proj_c = torch.tensor(
                [0, 0, -f, 1], dtype=self._translation.dtype, device=self._translation.device)

            # Add some triangles to make it look like a camera
            # Get a point behind the camera plane
            global_projection_center = self.local_to_global(
                proj_c.unsqueeze(0), t=batched_t)

            triangles = torch.stack([global_projection_center.repeat(
                global_corners.shape[0], 1, 1), global_corners, torch.roll(global_corners, shifts=1, dims=0)], dim=-2)

            # Flatten the traingles for times
            triangles = triangles.reshape(4 * len(batched_t), 3, 4)
            # Plot wireframe indicating projection direction
            ax.add_collection3d(Poly3DCollection(
                [*triangles[..., :3].detach().cpu().numpy()], linewidths=1, edgecolors='k', alpha=0.0))
        else:
            gro, grd = get_global_rays(uv=(corners[..., :2] + 0.5) * self._image_resolution.flip((-1,)),
                                       inverse_intrinsics=self.get_inverse_intrinsics(
                                           t=batched_t),
                                       lens_distortion=self.get_lens_distortion(),
                                       global_position=self.get_global_position(
                                           t=batched_t),
                                       focal_length=self.focal_length, uv_includes_time=False)

            # rd = grd - gro
            end = gro - grd

            vo = torch.stack([
                torch.roll(end, shifts=1, dims=0), end,
                global_corners[..., :3], torch.roll(global_corners[..., :3], shifts=1, dims=0)], dim=-2)
            ax.add_collection3d(Poly3DCollection([x for x in vo.reshape(
                4 * len(batched_t), 4, 3).detach().cpu().numpy()], linewidths=1, edgecolors='k', alpha=0.0))

        verts = [global_corners[:, i, :3].detach().cpu().numpy()
                 for i in range(global_corners.shape[1])]
        # Plot the rectangle
        ax.add_collection3d(Poly3DCollection(
            verts, facecolors='grey', linewidths=1, edgecolors='k', alpha=0.5))

        if kwargs.get("plot_camera_traces", False):
            # Trace corners of camera
            cmap = plt.get_cmap("tab10")
            corner_colors = [cmap(x) for x in range(corners.shape[0])]
            for i in range(batched_t.shape[0]):
                self.plot_point_trace(points=corners,
                                      ax=ax,
                                      t_step=kwargs.get("t_trace_step", 0.001),
                                      t_max=batched_t[i],
                                      colors=corner_colors)

    def _plot_ray_intersections(self,
                                intersections: torch.Tensor,
                                is_intersection: torch.Tensor,
                                ax: Axes,
                                t: Optional[torch.Tensor] = None,
                                **kwargs
                                ):
        """_summary_

        Parameters
        ----------
        intersections : torch.Tensor
             Intersection points of the rays with the infinite planes in the scene. Shape (N, B, T, 3)
        is_intersection : torch.Tensor
            Boolean tensor indicating if the rays intersect with the plane within its bounds. Shape (N, B, T)
        ax : Axes
            Axis to plot on.
        t : Optional[torch.Tensor], optional
            Timeframe to plot, by default None
        """
        N, B, T, _ = intersections.shape
        if t is None:
            t = torch.arange(T, dtype=torch.int32)

        time_is_intersec = is_intersection[:, :, t]
        inter = intersections[:, :, t, :][time_is_intersec].detach().cpu()
        no_inter = intersections[:, :, t, :][~time_is_intersec].detach().cpu()
        if kwargs.get("plot_ray_intersection_hits", True):
            ax.scatter(inter[:, 0], inter[:, 1], inter[:, 2], c="green", s=1)
        if kwargs.get("plot_ray_intersection_misses", True):
            ax.scatter(no_inter[:, 0], no_inter[:, 1],
                       no_inter[:, 2], c="red", s=1)

    def _plot_ray_cast(self,
                       ax: Axes,
                       sample_grid: Optional[torch.Tensor] = None,
                       **kwargs):
        if sample_grid is None:
            sample_grid = self._sample_pixel_grid(subsample=kwargs.get(
                "pixel_grid_subsample", 50), include_borders=kwargs.get("pixel_grid_borders", True))

        t_camera_pos = flatten_batch_dims(tensorify(kwargs.get(
            't', self._times[-1]), dtype=self._times.dtype, device=self._times.device), -1)[0]

        gro, grd = get_global_rays(
            uv=sample_grid,
            inverse_intrinsics=self.get_inverse_intrinsics(t=t_camera_pos),
            lens_distortion=self.get_lens_distortion(),
            global_position=self.get_global_position(t=t_camera_pos),
            focal_length=self.focal_length,
            uv_includes_time=False)
        gro = flatten_batch_dims(gro, -2)[0]
        grd = flatten_batch_dims(grd, -2)[0]

        ray_distance = kwargs.get("ray_distance", 1)

        ax.quiver(*gro.detach().cpu().numpy().T, *grd.detach().cpu().numpy().T,
                  length=ray_distance,
                  normalize=True,
                  arrow_length_ratio=0,
                  linewidths=0.5,
                  color='gray',
                  label="Rays")

    @torch.no_grad()
    def plot_object(self, ax: Axes,
                    **kwargs):
        super().plot_object(ax, **kwargs)
        # plot ray intersections
        if kwargs.get("plot_ray_intersections", False):
            intersections = raise_on_none(
                kwargs.get("ray_intersections", None))
            is_intersection = raise_on_none(
                kwargs.get("ray_is_intersection", None))
            self._plot_ray_intersections(
                intersections, is_intersection, ax, **kwargs)
        if kwargs.get("plot_rays", False):
            ray_thickness = kwargs.pop("ray_thickness", 0.5)
            ray_color = kwargs.pop("ray_color", "gray")
            self._plot_rays(ax=ax, color=ray_color,
                            linewidth=ray_thickness, **kwargs)
