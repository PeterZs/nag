from typing import Tuple
from tools.transforms.geometric.transforms3d import *
from tools.transforms.geometric.transforms3d import (
    _split_transformation_matrix, _compose_transformation_matrix, rotmat_to_unitquat, unitquat_to_rotmat)
import roma
import torch

from tools.transforms.geometric.mappings import rotvec_to_unitquat, unitquat_to_rotvec
from nag.transforms.utils import quat_conjugation, quat_product
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims


@torch.jit.script
def shortest_angle_distance(from_: torch.Tensor, to: torch.Tensor) -> torch.Tensor:
    """Compute the shortest angle distance between two angles.
    Angles are assumed to be in radians and between 0 and 2*pi.

    Parameters
    ----------
    from_ : torch.Tensor
        from angle in radians of arbitrary size.
    to : torch.Tensor
        to angle in radians of arbitrary size.

    Returns
    -------
    torch.Tensor
        Distance between the two angles.
    """
    delta_angle = (to - from_) % (2 * torch.pi)
    return ((2 * delta_angle) % (2 * torch.pi)) - delta_angle


@torch.jit.script
def linear_angle_interpolation(from_: torch.Tensor, to: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
    """
    Interpolate between two angles by the fraction frac.
    This is a linear interpolation and will take the shortest path between the two angles.
    Angles are assumed to be in radians and between 0 and 2*pi.

    Parameters
    ----------
    from_ : torch.Tensor
        The starting angle.
    to : torch.Tensor
        The target angle.
    frac : torch.Tensor
        The fraction to interpolate between the two angles.

    Returns
    -------
    torch.Tensor
        The interpolated angle.
    """
    return from_ + shortest_angle_distance(from_, to) * frac


@torch.jit.script
def position_quaternion_to_affine_matrix(position: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """
    Create an affine matrix from a position and a quaternion.

    Parameters
    ----------
    position : torch.Tensor
        The position tensor of shape (..., 3).
    quaternion : torch.Tensor
        The quaternion tensor of shape (..., 4).

    Returns
    -------
    torch.Tensor
        The affine matrix of shape (..., 4, 4).
    """
    rotation_matrix = unitquat_to_rotmat(quaternion)
    return _compose_transformation_matrix(position, rotation_matrix[..., :3, :3])


@torch.jit.script
def unitquat_slerp(q0: torch.Tensor, q1: torch.Tensor, steps: torch.Tensor, shortest_arc: bool = True) -> torch.Tensor:
    """
    Spherical linear interpolation between two unit quaternions.

    Args:
        q0, q1 (A x 4 tensor): batch of unit quaternions (A may contain multiple dimensions).
        steps (tensor of shape B): interpolation steps, 0.0 corresponding to q0 and 1.0 to q1 (B may contain multiple dimensions).
        shortest_arc (boolean): if True, interpolation will be performed along the shortest arc on SO(3) from `q0` to `q1` or `-q1`.

    Parameters
    ------------
    q0 : torch.Tensor
        batch of unit quaternions (B x 4 tensor) in the format (x, y, z, w).

    q1 : torch.Tensor
        batch of unit quaternions (B x 4 tensor) in the format (x, y, z, w).

    steps : torch.Tensor
        Interpolation steps, 0.0 corresponding to q0 and 1.0 to q1 (B tensor).

    Returns
    -------
        batch of interpolated quaternions (B x 4 tensor).

    Note
    -------
        When considering quaternions as rotation representations,
        one should keep in mind that spherical interpolation is not necessarily performed along the shortest arc,
        depending on the sign of ``torch.sum(q0*q1,dim=-1)``.

        Behavior is undefined when using ``shortest_arc=False`` with antipodal quaternions.

    See Also
    --------
        Original version with step per quaternion:
        https://github.com/naver/roma/blob/master/roma/utils.py
    """
    # Relative rotation
    rel_q = quat_product(quat_conjugation(q0), q1)
    rel_rotvec = unitquat_to_rotvec(rel_q, shortest_arc=shortest_arc)
    # Relative rotations to apply
    rel_rotvecs = steps.reshape(steps.shape + (1,)) * rel_rotvec
    rots = rotvec_to_unitquat(rel_rotvecs).reshape(rel_rotvecs.shape[-2], 4)
    interpolated_q = quat_product(q0, rots)
    return interpolated_q


@torch.jit.script
def _linear_interpolate_vector(
        from_vector: torch.Tensor,
        to_vector: torch.Tensor,
        frac: torch.Tensor) -> torch.Tensor:

    from_vector, f_bd = flatten_batch_dims(from_vector, -2)
    to_vector, _ = flatten_batch_dims(to_vector, -2)
    frac, _ = flatten_batch_dims(frac, -1)

    B, C = from_vector.shape

    t_frac = frac.unsqueeze(-1).repeat(1, C)
    new_position = from_vector + (to_vector - from_vector) * t_frac
    return unflatten_batch_dims(new_position, f_bd)


@torch.jit.script
def _linear_interpolate_rotation_quaternion(
        from_quat: torch.Tensor,
        to_quat: torch.Tensor,
        frac: torch.Tensor) -> torch.Tensor:

    from_quat, f_bd = flatten_batch_dims(from_quat, -2)
    to_quat, _ = flatten_batch_dims(to_quat, -2)
    frac, _ = flatten_batch_dims(frac, -1)

    B, C = from_quat.shape
    return unflatten_batch_dims(unitquat_slerp(from_quat, to_quat, frac, shortest_arc=True), f_bd)


@torch.jit.script
def _linear_interpolate_position_rotation(
        from_position: torch.Tensor,
        to_position: torch.Tensor,
        from_quat: torch.Tensor,
        to_quat: torch.Tensor,
        frac: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    new_position = _linear_interpolate_vector(from_position, to_position, frac)
    interp_angles = _linear_interpolate_rotation_quaternion(
        from_quat, to_quat, frac)
    return new_position, interp_angles


@torch.jit.script
def _linear_interpolate_affine_matrix(from_: torch.Tensor, to: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
    # Extract the position and rotation components of the affine matrix
    from_position, from_rotation = _split_transformation_matrix(from_)
    to_position, to_rotation = _split_transformation_matrix(to)

    from_quat = rotmat_to_unitquat(from_rotation)
    to_quat = rotmat_to_unitquat(to_rotation)

    new_position = _linear_interpolate_vector(
        from_vector=from_position, to_vector=to_position, frac=frac)

    new_quaternion = _linear_interpolate_rotation_quaternion(
        from_quat=from_quat, to_quat=to_quat, frac=frac)

    new_rotation = unitquat_to_rotmat(new_quaternion)

    interpolated_matricies = _compose_transformation_matrix(
        position=new_position, orientation=new_rotation[..., :3, :3])
    return interpolated_matricies


def linear_interpolate_affine_matrix(from_: torch.Tensor, to: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
    """
    Interpolate between two affine matrices by the fraction frac.
    This is a linear interpolation and will take the shortest path between the two rotations and positions.

    Parameters
    ----------
    from_ : torch.Tensor
        The starting affine matrix.
    to : torch.Tensor
        The target affine matrix.
    frac : torch.Tensor
        The fraction to interpolate between the two affine matrices.
        Relative distance between the two affine matrices in the range [0, 1].

    Returns
    -------
    torch.Tensor
        The interpolated affine matrix.
    """
    if from_.shape != to.shape:
        raise ValueError(
            f"from_ and to must have the same shape, but got {from_.shape} and {to.shape}")
    if from_.shape[-2:] != (4, 4):
        raise ValueError(
            f"from_ and to must have shape (..., 4, 4), but got {from_.shape}")
    return _linear_interpolate_affine_matrix(from_, to, frac)
