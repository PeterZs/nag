import torch
from typing import Any, Dict, Literal, Optional, Tuple, Union
from tools.util.torch import tensorify

import numpy as np
from tools.util.typing import VEC_TYPE
from tools.transforms.geometric.quaternion import quat_product, quat_product_scalar, quat_subtraction
from nag.transforms.transforms3d import (
    _linear_interpolate_vector,
    _linear_interpolate_rotation_quaternion,
    _linear_interpolate_affine_matrix, _linear_interpolate_position_rotation)
from nag.transforms.utils import hermite_catmull_rom_position, quat_hermite_catmull_rom_position
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims


def assure_affine_time_matrix(_input: VEC_TYPE,
                              dtype: Optional[torch.dtype] = None,
                              device: Optional[torch.device] = None,
                              requires_grad: bool = False) -> torch.Tensor:
    """Assuring the _input matrix instance is an affine timed matrix (t, 3, 3).
    Converting it into tensor if nessesary.
    Adds 1 to the vector if its size is 3.

    Parameters
    ----------
    _input : Union[torch.Tensor, np.ndarray]
        Matrix of x / y shape 3 or 4.
    dtype : Optional[torch.dtype], optional
        The dtype of the tensor, by default None
    device : Optional[torch.device], optional
        Its device, by default None
    requires_grad : bool, optional
        Whether it requires grad and the input was numpy array, by default False

    Returns
    -------
    torch.Tensor
        The affine tensor.

    Raises
    ------
    ValueError
        If shape is wrong.
    """
    _input = tensorify(_input, dtype=dtype, device=device,
                       requires_grad=requires_grad)
    if len(_input.shape) == 2:
        _input = _input.unsqueeze(0)

    if len(_input.shape) != 3:
        raise ValueError(
            f"assure_homogeneous_matrix works only on 2d and 3d tensors!")

    if _input.shape[-2] > 4 or _input.shape[-2] < 3:
        raise ValueError(
            f"assure_homogeneous_matrix works only for tensors of length 3 or 4.")
    if _input.shape[-2] == 4:
        pass
    else:
        # Length of 3
        _input = torch.cat(
            [_input, torch.tensor(
                [[0., 0., 0.] + ([] if _input.shape[1] == 3 else [1.])],
                device=_input.device, dtype=_input.dtype, requires_grad=_input.requires_grad).unsqueeze(0).repeat(_input.shape[-3], 1, 1)],
            axis=-2)
    if _input.shape[-1] > 4 or _input.shape[-1] < 3:
        raise ValueError(
            f"assure_homogeneous_matrix works only for tensors of length 3 or 4.")
    if _input.shape[-1] == 4:
        pass
    else:
        # Length of 3
        _input = torch.cat([_input, torch.tensor(
            [[0., 0., 0., 1.]], device=_input.device, dtype=_input.dtype, requires_grad=_input.requires_grad).T.unsqueeze(0).repeat(_input.shape[-3], 1, 1)], axis=-1)
    return _input


@torch.jit.script
def _get_interpolate_index_and_distance(
        times: torch.Tensor,
        steps: torch.Tensor,
        equidistant: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the index and relative distance to interpolate from times to steps.

    Parameters
    ----------
    times : torch.Tensor
        The timestamps t in shape ([... B], t) in increasing order.

    steps : torch.Tensor
        The timestamps t to interpolate the vector at. In shape ([... B], S).
        S is the number of steps to interpolate the signal at.

    equidistant : bool, optional
        If the times are equidistant, by default False.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        1. The index of the right time to interpolate from in shape (B * S, 2). (Batch index, right_index for correspnding s in t)
        2. The relative distance to interpolate from right_idx -1 to right_idx in shape (B, S).
    """

    times, shape_times = flatten_batch_dims(times, -2)
    steps, shape_steps = flatten_batch_dims(steps, -2)

    if steps.shape[0] == 1 and times.shape[0] != 1:
        steps = steps.repeat(times.shape[0], 1)

    B, N = steps.shape

    batch_idx = torch.arange(B, dtype=torch.int64,
                             device=steps.device).repeat_interleave(N)

    if times.shape[-1] <= 1:
        # No interpolation needed
        idxr = batch_idx.unsqueeze(-1)
        idxr = torch.cat([idxr, torch.ones_like(
            idxr, dtype=torch.int64)], dim=-1)
        return idxr, torch.zeros((B, N), dtype=torch.float32, device=times.device)

    # Check if times and steps match, then return the index and 0
    if times.shape == steps.shape and torch.allclose(times, steps):
        rdx = torch.clamp(torch.arange(
            1, times.shape[-1] + 1, device=times.device, dtype=torch.int32), 1, times.shape[-1] - 1)
        frac = torch.zeros_like(rdx, dtype=torch.float32)
        # As the last index is clamped, we need to set the last frac to 1.
        frac[-1] = 1.
        brdx = rdx.repeat(B)
        brdx_f = torch.stack([batch_idx, brdx], dim=-1)
        brdx_f = brdx_f.reshape(B * N, 2)
        return brdx_f, frac.unsqueeze(0).repeat(B, 1)

    if equidistant:
        tmin, tmax = times[..., 0].unsqueeze(-1).repeat(
            1, N), times[..., -1].unsqueeze(-1).repeat(1, N)
        step_size = ((tmax - tmin) / (times.shape[-1] - 1))
        right_idx = (torch.floor((steps - tmin) /
                     step_size) + 1).to(torch.int64)
        right_idx = torch.clamp(right_idx, 1, times.shape[-1] - 1)

        idx = torch.stack([batch_idx, right_idx.reshape(-1)],
                          dim=-1).reshape((B * N, 2))
        selected_times = times[idx.T[0], idx.T[1]].reshape(B, N)

        rel_frac = ((steps - selected_times) / step_size) + 1
        # If value is negative and close to 0 force it to 0 #TODO Maybe there is a better way, but numerical issues can lead to negative values.
        rel_frac[(rel_frac < 0) & torch.isclose(
            rel_frac, torch.tensor(0.), atol=5e-6)] = 0.
        return idx, rel_frac
    else:
        right_idx = torch.searchsorted(times, steps, side="right")
        right_idx = torch.clamp(right_idx, 1, times.shape[-1] - 1)

        idx = torch.stack([batch_idx, right_idx.reshape(-1)],
                          dim=-1).reshape((B * N, 2))

        selected_times = times[idx.T[0], idx.T[1]].reshape(B, N)
        frac = (steps - selected_times)

        dist = times[..., 1:] - times[..., :-1]
        # Add 1 so rel_frac describes dist between from an to, where negative values are reverse dir.

        selected_dist = dist[idx.T[0], idx.T[1] - 1].reshape(B, N)
        rel_frac = (frac / selected_dist) + 1.
        return idx, rel_frac


def linear_interpolate_vector(
        v: torch.Tensor,
        times: torch.Tensor,
        steps: torch.Tensor,
        equidistant_times: bool = False,
        right_idx: Optional[torch.Tensor] = None,
        rel_frac: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Interpolates the values within an of an object at time t linearly.

    Parameters
    ----------
    v : torch.Tensor
        Position of the object at timestep t in shape (t, ...).

    times : torch.Tensor
        The timestamps t in shape (t) in increasing order of the vector v.

    steps : torch.Tensor
        The timestamps t to interpolate the vector at. In shape (s, ).

    equidistant_times : bool, optional
        If the times are equidistant, by default False.

    right_idx : Optional[torch.Tensor], optional
        The right index of v for interpolation, if precomputed, by default None.
        If None, it will be computed.

    rel_frac : Optional[torch.Tensor], optional
        The relative distance to interpolate from right_idx -1 to right_idx, if precomputed, by default None.

    Returns
    -------
    torch.Tensor
        The interpolated vectors (s, ...) at timestep steps s.
    """
    if right_idx is None or rel_frac is None:
        right_idx, rel_frac = _get_interpolate_index_and_distance(
            times, steps, equidistant=equidistant_times)
    v, v_batch_shape = flatten_batch_dims(v, -3)

    B, T, C = v.shape
    S = steps.shape[-1]

    if T == 1:
        # Only one value, return it for all steps
        return unflatten_batch_dims(v.repeat(1, S, 1), v_batch_shape)

    from_pos = v[right_idx[:, 0], right_idx[:, 1] - 1].reshape(B, S, C)
    to_pos = v[right_idx[:, 0], right_idx[:, 1]].reshape(B, S, C)

    # Collapse B and T to one dimension
    new_pos = _linear_interpolate_vector(from_vector=from_pos,
                                         to_vector=to_pos,
                                         frac=rel_frac)

    return unflatten_batch_dims(new_pos, v_batch_shape)


def linear_interpolate_quaternion(
        orientation: torch.Tensor,
        times: torch.Tensor,
        steps: torch.Tensor,
        equidistant_times: bool = False,
        right_idx: Optional[torch.Tensor] = None,
        rel_frac: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Interpolates the position and orientation of an object at time t linaerly.

    Parameters
    ----------
    orientation : torch.Tensor
        Orientation of the object at timestep t in shape (t, 4). As normalized quaternion. (x, y, z, w)

    times : torch.Tensor
        The timestamps t in shape (t) in increasing order of the quaternion.

    steps : torch.Tensor
        The timestamps t to interpolate the vector at. In shape (s, ).

    equidistant_times : bool, optional
        If the times are equidistant, by default False.

    right_idx : Optional[torch.Tensor], optional
        The right index of v for interpolation, if precomputed, by default None.
        If None, it will be computed.

    rel_frac : Optional[torch.Tensor], optional
        The relative distance to interpolate from right_idx -1 to right_idx, if precomputed, by default None.

    Returns
    -------
    torch.Tensor
        The interpolated orientations (s, 4) at the steps s.
    """
    if right_idx is None or rel_frac is None:
        right_idx, rel_frac = _get_interpolate_index_and_distance(
            times, steps, equidistant=equidistant_times)

    orientation, o_batch_shape = flatten_batch_dims(orientation, -3)

    B, T, C = orientation.shape
    S = steps.shape[-1]

    if orientation.shape[1] == 1:
        if ((right_idx[:, 1] == 0) & (rel_frac == 0)).any():
            raise ValueError(
                "If signal has only one value, the right_idx must be 1 and rel_frac must be 0.")
        return unflatten_batch_dims(orientation.repeat(1, S, 1), o_batch_shape)

    from_quat = orientation[right_idx[:, 0],
                            right_idx[:, 1] - 1].reshape(B, S, C)
    to_quat = orientation[right_idx[:, 0], right_idx[:, 1]].reshape(B, S, C)
    new_quat = _linear_interpolate_rotation_quaternion(from_quat=from_quat,
                                                       to_quat=to_quat,
                                                       frac=rel_frac)
    return unflatten_batch_dims(new_quat, o_batch_shape)


@torch.jit.script
def interpolate_inter_time_positions(
        positions: torch.Tensor,
        orientation: torch.Tensor,
        times: torch.Tensor,
        steps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolates the position and orientation of an object at time t linaerly.

    Parameters
    ----------
    positions : torch.Tensor
        Position of the object at timestep t in shape (t, 3).
    orientation : torch.Tensor
        Orientation of the object at timestep t in shape (t, 4). As normalized quaternion. (x, y, z, w)
    times : torch.Tensor
        The timestamps t in shape (t) in increasing order of the positions and orientations.
    steps : torch.Tensor
        The timestamps t to interpolate the positions and orientations at. In shape (s, ).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The interpolated positions (s, 3) and orientations (s, 4) at the steps s.
    """
    right_idx, rel_frac = _get_interpolate_index_and_distance(times, steps)

    from_pos = positions[right_idx - 1]
    to_pos = positions[right_idx]

    from_quat = orientation[right_idx - 1]
    to_quat = orientation[right_idx]

    new_pos, new_quat = _linear_interpolate_position_rotation(from_position=from_pos,
                                                              to_position=to_pos,
                                                              from_quat=from_quat,
                                                              to_quat=to_quat,
                                                              frac=rel_frac)
    # if left is s
    return new_pos, new_quat


@torch.jit.script
def sample_orientation(positions: torch.Tensor,
                       orientations: torch.Tensor,
                       times: torch.Tensor,
                       sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples the position and orientation of an object at time t.
    If the time t is not in times, it will interpolate the position and orientation linearly.

    Parameters
    ----------
    positions : torch.Tensor
        The positions of the object at the times in shape (t, 3).

    orientations : torch.Tensor
        The orientations of the object at the times in shape (t, 4). As normalized quaternion. (x, y, z, w)

    times : torch.Tensor
        The timestamps t in shape (t) in increasing order of the positions and orientations.

    sample : torch.Tensor
        The timestamps s to sample the positions and orientations at. In shape (s, ).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The sampled positions (s, 3) and orientations (s, 4) at the sample times s.
    """
    match = torch.isin(sample, times)
    tm = sample[match]
    match_pos = torch.zeros(
        (0, 3), dtype=positions.dtype, device=positions.device)

    match_orient = torch.zeros(
        (0, 4), dtype=orientations.dtype, device=orientations.device)

    if len(tm) > 0:
        match_idx = torch.argwhere(
            tm.unsqueeze(-1) == times)[:, -1]
        match_pos = positions[match_idx]
        match_orient = orientations[match_idx]

    no_match = sample[~match]
    non_match_pos = torch.zeros(
        (0, 3), dtype=positions.dtype, device=positions.device)
    non_match_orient = torch.zeros(
        (0, 4), dtype=orientations.dtype, device=orientations.device)

    # Get the position for non existing t via interpolation
    if len(no_match) > 0:
        non_match_pos, non_match_orient = interpolate_inter_time_positions(
            positions, orientations, times, no_match)

    ret_pos = torch.zeros(
        (sample.shape[0], 3), dtype=positions.dtype, device=positions.device)
    ret_orient = torch.zeros(
        (sample.shape[0], 4), dtype=orientations.dtype, device=orientations.device)

    ret_pos[match, ...] = match_pos
    ret_orient[match, ...] = match_orient
    ret_pos[~match, ...] = non_match_pos
    ret_orient[~match, ...] = non_match_orient
    return ret_pos, ret_orient


@torch.jit.script
def hermite_catmull_rom_index(
        v: torch.Tensor,
        times: torch.Tensor,
        steps: torch.Tensor,
        equidistant_times: bool = False,
        right_idx: Optional[torch.Tensor] = None,
        rel_frac: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute the Catmull-Rom cubic spline interpolation between points in v.

    Parameters
    ----------
    v : torch.Tensor
        Signal to interpolate. Shape ([... B,] T, C).
    times : torch.Tensor
        Time t of the signal. Shape ([... B,] T).
    steps : torch.Tensor
        Steps of the interpolation. Shape ([... B,] S).
    equidistant_times : bool, optional
        If the times can be estimated equidistant - this will save some compute, by default False

    right_idx : Optional[torch.Tensor], optional
        Pre-computed right indexes for interpolation, by default None
    rel_frac : Optional[torch.Tensor], optional
        Pre-computed relativ fractionals for steps. normally in range [0, 1] can be < 0 or larger 1 if extrapolation, by default None

    Returns
    -------
    torch.Tensor
        Interpolated points at the steps. Shape ([... B,] S, C).
    """

    if right_idx is None or rel_frac is None:
        right_idx, rel_frac = _get_interpolate_index_and_distance(
            times, steps, equidistant=equidistant_times)

    v, v_batch_shape = flatten_batch_dims(v, -3)
    times, _ = flatten_batch_dims(times, -2)
    steps, _ = flatten_batch_dims(steps, -2)

    if steps.shape[0] == 1 and times.shape[0] != 1:
        steps = steps.repeat(times.shape[0], 1)

    B, T, C = v.shape
    S = steps.shape[-1]

    if len(v.shape) != 1:
        C = v.shape[-1]
    shape = (B, S, C)

    p0 = torch.zeros(shape, dtype=v.dtype, device=v.device)
    p1 = torch.zeros(shape, dtype=v.dtype, device=v.device)
    p2 = torch.zeros(shape, dtype=v.dtype, device=v.device)
    p3 = torch.zeros(shape, dtype=v.dtype, device=v.device)

    # Normal case: right_idx is not at the boundary (1 or len(v) - 1) or if it is, rel_frac is not smaller 0 (left) or greater 1 (right)
    norm_cond = ((right_idx[:, -1] > 1) | (rel_frac >= 0).reshape(
        B * S)) & ((right_idx[:, -1] < (T - 1)) | (rel_frac <= 1).reshape(B * S))

    # Unsqueeze the right_idx and rel_frac to match the shape of the output
    # right_idx = right_ix.unsqueeze(-1).repeat(1, 1, C)
    # rel_frac = rel_frac.unsqueeze(-1).repeat(1, 1, C)

    norm_cond_idx = right_idx[norm_cond]
    p1[norm_cond.reshape(B, S)] = v[norm_cond_idx[:, 0],
                                    norm_cond_idx[:, 1] - 1]
    p2[norm_cond.reshape(B, S)] = v[norm_cond_idx[:, 0], norm_cond_idx[:, 1]]

    # Select p0 and p3 by considering the boundary conditions as linear extrapolation
    condl = (right_idx[norm_cond])[:, 1] == 1

    norm_condl = norm_cond.clone().to(dtype=torch.bool)
    norm_not_condl = norm_cond.clone().to(dtype=torch.bool)

    norm_condl[norm_cond] = condl
    norm_not_condl[norm_cond] = ~condl

    p0[norm_condl.reshape(B, S)] = 2 * p1[norm_condl.reshape(B, S)
                                          ] - p2[norm_condl.reshape(B, S)]
    norm_not_condl_idx = right_idx[norm_not_condl]
    p0[norm_not_condl.reshape(
        B, S)] = v[norm_not_condl_idx[:, 0], norm_not_condl_idx[:, 1] - 2]

    condr = (right_idx[norm_cond])[:, 1] == T - 1
    norm_condr = norm_cond.clone().to(dtype=torch.bool)
    norm_not_condr = norm_cond.clone().to(dtype=torch.bool)
    norm_condr[norm_cond] = condr
    norm_not_condr[norm_cond] = ~condr

    p3[norm_condr.reshape(B, S)] = 2 * p2[norm_condr.reshape(B, S)
                                          ] - p1[norm_condr.reshape(B, S)]

    norm_not_condr_idx = right_idx[norm_not_condr]
    p3[norm_not_condr.reshape(
        B, S)] = v[norm_not_condr_idx[:, 0], norm_not_condr_idx[:, 1] + 1]

    # TODO Need to look over the following code again. The dir interpolation does not make sense if we use linear interpolation for out-of-bound values.

    # If rel_frac is smaller 0, or larger 1, we need to extrapolate the position in a linear fashion
    # Left side first
    condel = rel_frac[~norm_cond.reshape(B, S)] < 0
    not_norm_condl = (~norm_cond).clone()
    not_norm_condl[~norm_cond] = condel

    not_norm_condl_idx = right_idx[not_norm_condl]
    p3[not_norm_condl.reshape(
        B, S)] = v[not_norm_condl_idx[:, 0], not_norm_condl_idx[:, 1]]
    p2[not_norm_condl.reshape(
        B, S)] = v[not_norm_condl_idx[:, 0], not_norm_condl_idx[:, 1] - 1]

    # Linear extrapolation for p1 and p0
    dir = (p2[not_norm_condl.reshape(B, S)] - p3[not_norm_condl.reshape(B, S)])
    p1[not_norm_condl.reshape(B, S)] = dir * torch.abs(
        rel_frac[not_norm_condl.reshape(B, S)]).unsqueeze(-1) + p2[not_norm_condl.reshape(B, S)]
    p0[not_norm_condl.reshape(B, S)] = dir * (1. + torch.abs(
        rel_frac[not_norm_condl.reshape(B, S)])).unsqueeze(-1) + p2[not_norm_condl.reshape(B, S)]

    # Right side
    conder = rel_frac[~norm_cond.reshape(B, S)] > 1
    not_norm_conder = (~norm_cond).clone()
    not_norm_conder[~norm_cond] = conder

    not_norm_conder_idx = right_idx[not_norm_conder]
    p0[not_norm_conder.reshape(
        B, S)] = v[not_norm_conder_idx[:, 0], not_norm_conder_idx[:, 1] - 1]
    p1[not_norm_conder.reshape(
        B, S)] = v[not_norm_conder_idx[:, 0], not_norm_conder_idx[:, 1]]

    # Linear extrapolation for p2 and p3
    dir = (p1[not_norm_conder.reshape(B, S)] -
           p0[not_norm_conder.reshape(B, S)])

    p2[not_norm_conder.reshape(B, S)] = dir * (rel_frac[not_norm_conder.reshape(
        B, S)] - 1).unsqueeze(-1) + p1[not_norm_conder.reshape(B, S)]
    p3[not_norm_conder.reshape(B, S)] = dir * (rel_frac[not_norm_conder.reshape(
        B, S)]).unsqueeze(-1) + p1[not_norm_conder.reshape(B, S)]

    extrapolate = (rel_frac < 0) | (rel_frac > 1)
    interpolate = ~extrapolate
    res = torch.zeros_like(p1)

    if torch.any(interpolate):
        res[interpolate] = hermite_catmull_rom_position(rel_frac.unsqueeze(-1)[interpolate],
                                                        p0[interpolate],
                                                        p1[interpolate],
                                                        p2[interpolate],
                                                        p3[interpolate])
    if torch.any(extrapolate):
        # Decide wether its left or right
        left_extrapolate = rel_frac < 0
        right_extrapolate = rel_frac > 1
        if torch.any(left_extrapolate):
            res[left_extrapolate] = _linear_interpolate_vector(
                from_vector=p2[left_extrapolate],
                to_vector=p3[left_extrapolate],
                frac=rel_frac[left_extrapolate])
        if torch.any(right_extrapolate):
            res[right_extrapolate] = _linear_interpolate_vector(
                from_vector=p0[right_extrapolate],
                to_vector=p1[right_extrapolate],
                frac=rel_frac[right_extrapolate])
    return unflatten_batch_dims(res, v_batch_shape)


# @torch.jit.script
# def hermite_catmull_rom_new(
#         v: torch.Tensor,
#         times: torch.Tensor,
#         steps: torch.Tensor,
#         gradient: Optional[torch.Tensor] = None,
#         equidistant_times: bool = False,
#         right_idx: Optional[torch.Tensor] = None,
#         rel_frac: Optional[torch.Tensor] = None
#         ) -> torch.Tensor:
#     """Compute the Catmull-Rom cubic spline interpolation between points in v.

#     Parameters
#     ----------
#     v : torch.Tensor
#         Signal to interpolate. Shape ([...,], B, T, C).
#     times : torch.Tensor
#         Time t of the signal. Shape ([...,] B, T).
#     steps : torch.Tensor
#         Steps of the interpolation. Shape ([...,] B, S).
#     equidistant_times : bool, optional
#         If the times can be estimated equidistant - this will save some compute, by default False

#     right_idx : Optional[torch.Tensor], optional
#         Pre-computed right indexes for interpolation, by default None
#     rel_frac : Optional[torch.Tensor], optional
#         Pre-computed relativ fractionals for steps. normally in range [0, 1] can be < 0 or larger 1 if extrapolation, by default None

#     Returns
#     -------
#     torch.Tensor
#         Interpolated points at the steps.
#     """

#     if right_idx is None or rel_frac is None:
#         right_idx, rel_frac = _get_interpolate_index_and_distance(
#             times, steps, equidistant=equidistant_times)

#     v, v_batch_shape = flatten_batch_dims(v, -3)
#     times, _ = flatten_batch_dims(times, -2)
#     steps, _ = flatten_batch_dims(steps, -2)

#     if gradient is not None:
#         gradient, _ = flatten_batch_dims(gradient, -3)


#     if steps.shape[0] == 1 and times.shape[0] != 1:
#         steps = steps.repeat(times.shape[0], 1)

#     B, T, C = v.shape
#     S = steps.shape[-1]

#     if len(v.shape) != 1:
#         C = v.shape[-1]
#     shape = (B, S, C)

#     p0 = torch.zeros(shape, dtype=v.dtype, device=v.device)
#     p1 = torch.zeros(shape, dtype=v.dtype, device=v.device)
#     p2 = torch.zeros(shape, dtype=v.dtype, device=v.device)
#     p3 = torch.zeros(shape, dtype=v.dtype, device=v.device)

#     # Normal case: right_idx is not at the boundary (1 or len(v) - 1) or if it is, rel_frac is not smaller 0 (left) or greater 1 (right)
#     norm_cond = ((right_idx[:, -1] > 1) | (((right_idx[:, -1] == 1) & (rel_frac >= 0))).reshape(
#         B * S)) & ((right_idx[:, -1] < (T - 1)) | ((right_idx[:, -1] == (T - 1)) & (rel_frac <= 1)).reshape(B * S))

#     # Unsqueeze the right_idx and rel_frac to match the shape of the output
#     # right_idx = right_ix.unsqueeze(-1).repeat(1, 1, C)
#     # rel_frac = rel_frac.unsqueeze(-1).repeat(1, 1, C)

#     norm_cond_idx = right_idx[norm_cond]
#     p1[norm_cond.reshape(B, S)] = v[norm_cond_idx[:, 0],
#                                     norm_cond_idx[:, 1] - 1]
#     p2[norm_cond.reshape(B, S)] = v[norm_cond_idx[:, 0], norm_cond_idx[:, 1]]

#     # Select p0 and p3 by considering the boundary conditions as linear extrapolation
#     condl = (right_idx[norm_cond])[:, 1] <= 1

#     norm_condl = norm_cond.clone().to(dtype=torch.bool)
#     norm_not_condl = norm_cond.clone().to(dtype=torch.bool)

#     norm_condl[norm_cond] = condl
#     norm_not_condl[norm_cond] = ~condl

#     p0[norm_condl.reshape(B, S)] = 2 * p1[norm_condl.reshape(B, S)
#                                           ] - p2[norm_condl.reshape(B, S)]
#     norm_not_condl_idx = right_idx[norm_not_condl]
#     p0[norm_not_condl.reshape(
#         B, S)] = v[norm_not_condl_idx[:, 0], norm_not_condl_idx[:, 1] - 2]

#     condr = (right_idx[norm_cond])[:, 1] == T - 1
#     norm_condr = norm_cond.clone().to(dtype=torch.bool)
#     norm_not_condr = norm_cond.clone().to(dtype=torch.bool)
#     norm_condr[norm_cond] = condr
#     norm_not_condr[norm_cond] = ~condr

#     p3[norm_condr.reshape(B, S)] = 2 * p2[norm_condr.reshape(B, S)
#                                           ] - p1[norm_condr.reshape(B, S)]

#     norm_not_condr_idx = right_idx[norm_not_condr]
#     p3[norm_not_condr.reshape(
#         B, S)] = v[norm_not_condr_idx[:, 0], norm_not_condr_idx[:, 1] + 1]

#     # TODO Need to look over the following code again. The dir interpolation does not make sense if we use linear interpolation for out-of-bound values.

#     # If rel_frac is smaller 0, or larger 1, we need to extrapolate the position in a linear fashion
#     # Left side first
#     condel = ((rel_frac[~norm_cond.reshape(B, S)] < 0) | (right_idx[~norm_cond, -1] < 1))
#     not_norm_condl = (~norm_cond).clone()
#     not_norm_condl[~norm_cond] = condel

#     not_norm_condl_idx = right_idx[not_norm_condl]
#     p3[not_norm_condl.reshape(
#         B, S)] = v[not_norm_condl_idx[:, 0], not_norm_condl_idx[:, 1]]
#     p2[not_norm_condl.reshape(
#         B, S)] = v[not_norm_condl_idx[:, 0], not_norm_condl_idx[:, 1] - 1]

#     # Linear extrapolation for p1 and p0
#     dir = (p2[not_norm_condl.reshape(B, S)] - p3[not_norm_condl.reshape(B, S)])
#     p1[not_norm_condl.reshape(B, S)] = dir * torch.abs(
#         rel_frac[not_norm_condl.reshape(B, S)]).unsqueeze(-1) + p2[not_norm_condl.reshape(B, S)]
#     p0[not_norm_condl.reshape(B, S)] = dir * (1. + torch.abs(
#         rel_frac[not_norm_condl.reshape(B, S)])).unsqueeze(-1) + p2[not_norm_condl.reshape(B, S)]

#     # Right side
#     conder = rel_frac[~norm_cond.reshape(B, S)] > 1
#     not_norm_conder = (~norm_cond).clone()
#     not_norm_conder[~norm_cond] = conder

#     not_norm_conder_idx = right_idx[not_norm_conder]
#     p0[not_norm_conder.reshape(
#         B, S)] = v[not_norm_conder_idx[:, 0], not_norm_conder_idx[:, 1] - 1]
#     p1[not_norm_conder.reshape(
#         B, S)] = v[not_norm_conder_idx[:, 0], not_norm_conder_idx[:, 1]]

#     # Linear extrapolation for p2 and p3
#     dir = (p1[not_norm_conder.reshape(B, S)] -
#            p0[not_norm_conder.reshape(B, S)])

#     p2[not_norm_conder.reshape(B, S)] = dir * (rel_frac[not_norm_conder.reshape(
#         B, S)] - 1).unsqueeze(-1) + p1[not_norm_conder.reshape(B, S)]
#     p3[not_norm_conder.reshape(B, S)] = dir * (rel_frac[not_norm_conder.reshape(
#         B, S)]).unsqueeze(-1) + p1[not_norm_conder.reshape(B, S)]

#     extrapolate = (rel_frac < 0) | (rel_frac > 1)
#     interpolate = ~extrapolate
#     res = torch.zeros_like(p1)

#     if torch.any(interpolate):
#         res[interpolate] = hermite_catmull_rom_position(rel_frac.unsqueeze(-1)[interpolate],
#                                                         p0[interpolate],
#                                                         p1[interpolate],
#                                                         p2[interpolate],
#                                                         p3[interpolate])
#     if torch.any(extrapolate):
#         # Decide wether its left or right
#         left_extrapolate = rel_frac < 0
#         right_extrapolate = rel_frac > 1
#         if torch.any(left_extrapolate):
#             res[left_extrapolate] = _linear_interpolate_vector(
#                 from_vector=p2[left_extrapolate],
#                 to_vector=p3[left_extrapolate],
#                 frac=rel_frac[left_extrapolate])
#         if torch.any(right_extrapolate):
#             res[right_extrapolate] = _linear_interpolate_vector(
#                 from_vector=p0[right_extrapolate],
#                 to_vector=p1[right_extrapolate],
#                 frac=rel_frac[right_extrapolate])
#     return unflatten_batch_dims(res, v_batch_shape)


def align_rectangles(rect1: torch.Tensor, rect2: torch.Tensor):
    """
    Aligns two rectangles using Procrustes analysis and returns the transformation matrix
    to transform the first rectangle to the second rectangle.

    Parameters
    ----------
    rect1 : torch.Tensor
        The first rectangle. Shape: (B, P, 3)

    rect2 : torch.Tensor
        The second rectangle. Shape: (B, P, 3)

    Returns
    -------
    torch.Tensor
        The transformation matrix. Shape: (B, 4, 4)
    """
    rect1, shp = flatten_batch_dims(rect1, -3)
    rect2, _ = flatten_batch_dims(rect2, -3)

    B, P, _ = rect1.shape
    if rect2.shape != (B, P, 3):
        raise ValueError("The input rectangles must have the same shape.")

    # Center the rectangles
    center1 = rect1.mean(dim=-2)
    center2 = rect2.mean(dim=-2)

    rect1_centered = rect1 - center1.unsqueeze(-2).expand_as(rect1)
    rect2_centered = rect2 - center2.unsqueeze(-2).expand_as(rect1)

    # Calculate the optimal rotation using Procrustes analysis
    H = torch.bmm(
        rect2_centered[:, :3].transpose(-2, -1),
        rect1_centered[:, :3]
    )

    U, S, Vh = torch.linalg.svd(H)
    V = Vh.mH

    R = torch.bmm(U, V.transpose(-2, -1))

    # Calculate the translation
    t = center2 - torch.bmm(R, center1.unsqueeze(-1))[..., 0]

    T = torch.eye(4, dtype=rect1.dtype, device=rect1.device)[
        None, ...].repeat(B, 1, 1)
    T[..., :3, :3] = R  # Rotation matrix
    T[..., :3, 3] = t  # Translation vector
    T[..., 3, 3] = 1

    # Sanity check
    # tf = torch.bmm(T.unsqueeze(1).expand(-1, P, -1, -1).reshape(B*P, 4, 4), torch.cat([rect1.reshape(B*P, 3).unsqueeze(-1), torch.ones((B * P, 1, 1))], dim=-2))[:, :3, 0].reshape(B, P, 3)
    # torch.allclose(rect2, tf, atol=1e-6)
    return T


@torch.jit.script
def quat_hermite_catmull_rom_index(
        v: torch.Tensor,
        times: torch.Tensor,
        steps: torch.Tensor,
        equidistant_times: bool = False,
        right_idx: Optional[torch.Tensor] = None,
        rel_frac: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute the Catmull-Rom cubic spline interpolation between quaternion rotations in v.

    Parameters
    ----------
    v : torch.Tensor
        Signal to interpolate. Shape ([..., B], T, 4).
        Quaternions are expected to be normalized and in the order (x, y, z, w).
    times : torch.Tensor
        Time t of the signal. Shape ([..., B], T).
    steps : torch.Tensor
        Steps of the interpolation. Shape ([..., B], S).
    equidistant_times : bool, optional
        If the times can be estimated equidistant - this will save some compute, by default False

    right_idx : Optional[torch.Tensor], optional
        Pre-computed right indexes for interpolation, by default None
    rel_frac : Optional[torch.Tensor], optional
        Pre-computed relativ fractionals for steps. normally in range [0, 1] can be < 0 or larger 1 if extrapolation, by default None

    Returns
    -------
    torch.Tensor
        Interpolated rotations at the steps.  Shape ([... B,] S, 4).
    """

    if right_idx is None or rel_frac is None:
        right_idx, rel_frac = _get_interpolate_index_and_distance(
            times, steps, equidistant=equidistant_times)

    v, v_batch_shape = flatten_batch_dims(v, -3)
    times, times_shape = flatten_batch_dims(times, -2)
    steps, steps_shape = flatten_batch_dims(steps, -2)

    B, T, C = v.shape

    if C != 4:
        raise ValueError(
            f"Expected quaternion with shape (..., 4), got {v.shape}")

    S = steps.shape[-1]

    if len(v.shape) != 1:
        C = v.shape[-1]
    shape = (B, S, C)

    r0 = torch.zeros(shape, dtype=v.dtype, device=v.device)
    r1 = torch.zeros(shape, dtype=v.dtype, device=v.device)
    r2 = torch.zeros(shape, dtype=v.dtype, device=v.device)
    r3 = torch.zeros(shape, dtype=v.dtype, device=v.device)
    tensor_two = torch.tensor(2., dtype=v.dtype, device=v.device)
    tensor_one = torch.tensor(1., dtype=v.dtype, device=v.device)

    # Normal case: right_idx is not at the boundary (1 or len(v) - 1) or if it is, rel_frac is not smaller 0 (left) or greater 1 (right)
    norm_cond = ((right_idx[:, 1] > 1) | (rel_frac >= 0).reshape(
        B * S)) & ((right_idx[:, 1] < (T - 1)) | (rel_frac <= 1).reshape(B * S)).to(dtype=torch.bool)

    if torch.any(norm_cond):
        norm_cond_idx = right_idx[norm_cond]
        r1[norm_cond.reshape(B, S)] = v[norm_cond_idx[:, 0],
                                        norm_cond_idx[:, 1] - 1]
        r2[norm_cond.reshape(B, S)] = v[norm_cond_idx[:, 0],
                                        norm_cond_idx[:, 1]]
    else:
        pass
    # Select p0 and p3 by considering the boundary conditions aslinear extrapolation
    condl = (right_idx[norm_cond])[:, 1] == 1

    norm_condl = norm_cond.clone()
    norm_not_condl = norm_cond.clone()

    norm_condl[norm_cond] = condl
    norm_not_condl[norm_cond] = ~condl

    if torch.any(norm_condl):
        r0[norm_condl.reshape(B, S)] = quat_subtraction(quat_product_scalar(
            r1[norm_condl.reshape(B, S)], tensor_two), r2[norm_condl.reshape(B, S)])  # 2 * r1 - r2

    if torch.any(norm_not_condl):
        norm_not_condl_idx = right_idx[norm_not_condl]
        r0[norm_not_condl.reshape(
            B, S)] = v[norm_not_condl_idx[:, 0], norm_not_condl_idx[:, 1] - 2]

    condr = (right_idx[norm_cond])[:, 1] == T - 1
    norm_condr = norm_cond.clone()
    norm_not_condr = norm_cond.clone()
    norm_condr[norm_cond] = condr
    norm_not_condr[norm_cond] = ~condr

    if torch.any(norm_condr):
        r3[norm_condr.reshape(B, S)] = quat_subtraction(quat_product_scalar(
            r2[norm_condr.reshape(B, S)], tensor_two), r1[norm_condr.reshape(B, S)])

    if torch.any(norm_not_condr):
        norm_not_condr_idx = right_idx[norm_not_condr]
        r3[norm_not_condr.reshape(
            B, S)] = v[norm_not_condr_idx[:, 0], norm_not_condr_idx[:, 1] + 1]

    # If rel_frac is smaller 0, or larger 1, we need to extrapolate the position in a linear fashion
    # Left side first
    condel = rel_frac[~norm_cond.reshape(B, S)] < 0
    not_norm_condl = (~norm_cond).clone()
    not_norm_condl[~norm_cond] = condel

    if torch.any(not_norm_condl):
        not_norm_condl_idx = right_idx[not_norm_condl]
        r3[not_norm_condl.reshape(
            B, S)] = v[not_norm_condl_idx[:, 0], not_norm_condl_idx[:, 1]]
        r2[not_norm_condl.reshape(
            B, S)] = v[not_norm_condl_idx[:, 0], not_norm_condl_idx[:, 1] - 1]
        # Linear extrapolation for p1 and p0

        dir = quat_subtraction(r2[not_norm_condl.reshape(
            B, S)], r3[not_norm_condl.reshape(B, S)])  # r2 - r3
        r1[not_norm_condl.reshape(B, S)] = quat_product(quat_product_scalar(dir, torch.abs(
            rel_frac[not_norm_condl.reshape(B, S)]).unsqueeze(-1)), r2[not_norm_condl.reshape(B, S)])  # dir * abs(rel_frac) + r2
        r0[not_norm_condl.reshape(B, S)] = quat_product(quat_product_scalar(
            dir, (tensor_one + torch.abs(rel_frac[not_norm_condl.reshape(B, S)])).unsqueeze(-1)), r2[not_norm_condl.reshape(B, S)])

    # Right side
    conder = rel_frac[~norm_cond.reshape(B, S)] > 1
    not_norm_conder = (~norm_cond).clone().to(dtype=torch.bool)
    not_norm_conder[~norm_cond] = conder

    if torch.any(not_norm_conder):
        not_norm_conder_idx = right_idx[not_norm_conder]
        r0[not_norm_conder.reshape(
            B, S)] = v[not_norm_conder_idx[:, 0], not_norm_conder_idx[:, 1] - 1]
        r1[not_norm_conder.reshape(
            B, S)] = v[not_norm_conder_idx[:, 0], not_norm_conder_idx[:, 1]]

        # Linear extrapolation for p2 and p3
        # dir = (r1[not_norm_conder] - r0[not_norm_conder])

        dir = quat_subtraction(r1[not_norm_conder.reshape(
            B, S)], r0[not_norm_conder.reshape(B, S)])  # r1 - r0
        if dir.dtype == torch.int32 or dir.dtype == torch.int64:
            print("dir", dir)
            print("r1", r1[not_norm_conder.reshape(B, S)])
            print("r0", r0[not_norm_conder.reshape(B, S)])
        r2[not_norm_conder.reshape(B, S)] = quat_product(quat_product_scalar(
            dir, (rel_frac[not_norm_conder.reshape(B, S)] - 1).unsqueeze(-1)), r1[not_norm_conder.reshape(B, S)])
        r3[not_norm_conder.reshape(B, S)] = quat_product(quat_product_scalar(
            dir, (rel_frac[not_norm_conder.reshape(B, S)]).unsqueeze(-1)), r1[not_norm_conder.reshape(B, S)])

    extrapolate = (rel_frac < 0) | (rel_frac > 1)
    interpolate = ~extrapolate
    res = torch.zeros_like(r1)

    if torch.any(interpolate):
        res[interpolate] = quat_hermite_catmull_rom_position(rel_frac.unsqueeze(-1)[interpolate],
                                                             r0[interpolate],
                                                             r1[interpolate],
                                                             r2[interpolate],
                                                             r3[interpolate])
    if torch.any(extrapolate):
        # Decide wether its left or right
        left_extrapolate = rel_frac < 0
        right_extrapolate = rel_frac > 1
        if torch.any(left_extrapolate):
            res[left_extrapolate] = _linear_interpolate_rotation_quaternion(
                from_quat=r2[left_extrapolate],
                to_quat=r3[left_extrapolate],
                frac=rel_frac[left_extrapolate])
        if torch.any(right_extrapolate):
            res[right_extrapolate] = _linear_interpolate_rotation_quaternion(
                from_quat=r0[right_extrapolate],
                to_quat=r1[right_extrapolate],
                frac=rel_frac[right_extrapolate])
    return unflatten_batch_dims(res, v_batch_shape)


@torch.jit.script
def interpolate_vector(
        v: torch.Tensor,
        times: torch.Tensor,
        steps: torch.Tensor,
        equidistant_times: bool = False,
        right_idx: Optional[torch.Tensor] = None,
        rel_frac: Optional[torch.Tensor] = None,
        method: str = "linear") -> torch.Tensor:
    """Interpolates a vector signal v recorded at times t at the steps s.

    Parameters
    ----------
    v : torch.Tensor
        Signal to interpolate. Shape ([..., B], T, C).

    times : torch.Tensor
        Times of the signal beeing recorded. Shape ([..., B], T).

    steps : torch.Tensor
        Steps of the interpolation. Shape ([..., B], S).

    equidistant_times : bool, optional
        If the times can be estimated equidistant - this will save some compute, by default False

    right_idx : Optional[torch.Tensor], optional
        Pre-computed right indexes for interpolation, by default None
        Shape (B * S, 2). First index is the collapsed batch index, second the right index in v.

    rel_frac : Optional[torch.Tensor], optional
        Pre-computed relativ fractionals for steps. normally in range [0, 1] can be < 0 or larger 1 if extrapolation, by default None
        Shape (B, S)

    method : Literal["linear", "cubic"], optional
        The interpolation method, by default "linear"
        Can be "linear" or "cubic" where cubic is the Catmull-Rom cubic hermite spline interpolation.

    Returns
    -------
    torch.Tensor
        Interpolated points at the steps, shape ([..., B], S, C]).
    """

    if method == "linear":
        return linear_interpolate_vector(v, times, steps, equidistant_times, right_idx, rel_frac)
    elif method == "cubic":
        return hermite_catmull_rom_index(v, times, steps, equidistant_times, right_idx, rel_frac)
    else:
        raise ValueError(f"Unknown interpolation method {method}")


@torch.jit.script
def interpolate_orientation(
        v: torch.Tensor,
        times: torch.Tensor,
        steps: torch.Tensor,
        equidistant_times: bool = False,
        right_idx: Optional[torch.Tensor] = None,
        rel_frac: Optional[torch.Tensor] = None,
        method: str = "linear") -> torch.Tensor:
    """Interpolates a vector signal v - a rotation quaternion recorded at times t at the steps s.

    Parameters
    ----------
    v : torch.Tensor
        Quaternions to interpolate. Shape ([..., B], T, 4).

    times : torch.Tensor
        Times of the signal beeing recorded. Shape ([..., B], T).

    steps : torch.Tensor
        Steps of the interpolation. Shape ([..., B], S).

    equidistant_times : bool, optional
        If the times can be estimated equidistant - this will save some compute, by default False

    right_idx : Optional[torch.Tensor], optional
        Pre-computed right indexes for interpolation, by default None
        Shape (B * S, 2). First index is the collapsed batch index, second the right index in v.

    rel_frac : Optional[torch.Tensor], optional
        Pre-computed relativ fractionals for steps. normally in range [0, 1] can be < 0 or larger 1 if extrapolation, by default None
        Shape (B, S)

    method : Literal["linear", "cubic"], optional
        The interpolation method, by default "linear"
        Can be "linear" or "cubic" where cubic is the Catmull-Rom cubic hermite spline interpolation.

    Returns
    -------
    torch.Tensor
        Interpolated rotations at the steps, shape ([..., B], S, C]).
    """
    if method == "linear":
        return linear_interpolate_quaternion(v, times, steps, equidistant_times, right_idx, rel_frac)
    elif method == "cubic":
        return quat_hermite_catmull_rom_index(v, times, steps, equidistant_times, right_idx, rel_frac)
    else:
        raise ValueError(f"Unknown interpolation method {method}")


def find_optimal_spline(
        signal: torch.Tensor,
        times: torch.Tensor,
        K: int,
        epochs: int = 10000,
        lr=0.4,
        tol=1e-6,
        info: Dict[str, Any] = None,
        min_lr: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a cubic Hermite spline to the given signal.

    Parameters
    ----------

    signal: torch.Tensor
        Signal which should be fitted shape ([B,] T, C).

    times: torch.Tensor
        Time values for the signal shape ([B,] T).

    K: int
        Number of control points. (K)
        To include the boundary conditions two additional control points are added.
        E.g. K = 4 results in 6 control points.

    epochs: int
        Number of optimization steps.
        This is the upper limit of the number of optimization steps.

    lr: float
        Learning rate for the optimizer.

    tol: float
        Tolerance for the relative change of the loss value.
        If the relative change is smaller than tol the optimization stops.

    info: Dict[str, Any]
        Dictionary to store information about the optimization process.
        The following keys are used:
        - 'converged': bool
            True if the optimization converged.
        - 'initial_loss': float
            Initial loss value.
        - 'final_loss': float
            Final loss value.
        - 'max_epoch': int
            Number of epochs until the optimization stopped.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
    control_points: torch.Tensor
        Control points for the spline shape ([B,] K + 2, C).

    control_point_times: torch.Tensor
        Time values of the control points shape ([B,] K + 2).
        Linearly and equidistant spaced values between 0 and 1.

    """
    from tools.transforms.mean_std import MeanStd
    signal, shp = flatten_batch_dims(signal, -3)
    times = flatten_batch_dims(times, -2)[0]
    B, T, C = signal.shape

    KN = K + 2  # Add two additional control points for the boundary conditions

    control_points_param = torch.nn.Parameter(
        torch.randn((B, KN, C), device=signal.device))

    if K > 1:
        t_step = 1 / (K - 1)
        control_point_times = torch.linspace(
            0 - t_step, 1 + t_step, KN).unsqueeze(0).repeat(B, 1).to(signal.device)
    else:
        control_point_times = torch.tensor(
            [-1, 0, 1], device=signal.device).unsqueeze(0).repeat(B, 1)

    with torch.no_grad():
        ms_norm = MeanStd(mean=signal.mean(dim=-2), std=signal.std(dim=-2))
        control_points = ms_norm.fit_transform(control_points_param)
        control_points_param.data = control_points.detach()

    right_idx, rel_frac = _get_interpolate_index_and_distance(
        control_point_times.detach().clone(), times, True)

    optimizer = torch.optim.Adam([control_points_param], lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=30)

    criterion = torch.nn.MSELoss()
    prev_loss = float('inf')

    if info is not None:
        info['converged'] = False
        info['initial_loss'] = None
        info['max_epoch'] = epochs - 1

    with torch.set_grad_enabled(True):
        for epoch in range(epochs):
            optimizer.zero_grad()
            spline = hermite_catmull_rom_index(
                control_points_param, control_point_times, times, True, right_idx, rel_frac)
            loss = criterion(spline, signal)
            loss_value = loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)

            if loss_value == 0.:
                rel_change = 0.
            else:
                rel_change = abs((loss_value - prev_loss) / (prev_loss))

            if epoch == 0:
                if info is not None:
                    info['initial_loss'] = loss_value
            elif rel_change < tol or optimizer.param_groups[0]['lr'] < min_lr:
                if info is not None:
                    info['converged'] = True
                    info['max_epoch'] = epoch
                break
            prev_loss = loss_value

    if info is not None:
        info['final_loss'] = loss_value
    return unflatten_batch_dims(control_points_param.data.detach(), shp), unflatten_batch_dims(control_point_times, shp)


def find_optimal_spline_quat(
        signal: torch.Tensor,
        times: torch.Tensor,
        K: int,
        epochs: int = 10000,
        lr=0.01,
        tol=1e-6,
        info: Dict[str, Any] = None,
        min_lr: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a cubic Hermite spline to the given unit quaternion signal.

    Parameters
    ----------

    signal: torch.Tensor
        Signal (rotation angles) which should be fitted shape ([B,] T, 4).

    times: torch.Tensor
        Time values for the signal shape ([B,] T).

    K: int
        Number of control points. (K)
        To include the boundary conditions two additional control points are added.
        E.g. K = 4 results in 6 control points.

    epochs: int
        Number of optimization steps.
        This is the upper limit of the number of optimization steps.

    lr: float
        Learning rate for the optimizer.

    tol: float
        Tolerance for the relative change of the loss value.
        If the relative change is smaller than tol the optimization stops.

    info: Dict[str, Any]
        Dictionary to store information about the optimization process.
        The following keys are used:
        - 'converged': bool
            True if the optimization converged.
        - 'initial_loss_mse': float
            Initial loss value.
        - 'final_loss_mse': float
            Final loss value.
        - 'max_epoch': int
            Number of epochs until the optimization stopped.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
    control_points: torch.Tensor
        Control points for the spline shape ([B,] K + 2, C).

    control_point_times: torch.Tensor
        Time values of the control points shape ([B,] K + 2).
        Linearly and equidistant spaced values between 0 and 1.

    """
    from tools.transforms.geometric.quaternion import quat_mean, quat_std, quat_inverse, quat_normalize, unitquat_to_rotvec, rotvec_to_unitquat
    signal, shp = flatten_batch_dims(signal, -3)
    from nag.model.timed_discrete_scene_node_3d import plot_position, linear_interpolate_quaternion
    from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D, compose_translation_orientation, local_to_global, spline_approximation

    times = flatten_batch_dims(times, -2)[0]
    B, T, C = signal.shape

    if B > 1:
        raise ValueError("Batch dimension not supported yet.")

    if C != 4:
        raise ValueError(
            f"Expected quaternion with shape (..., 4), got {signal.shape}")

    KN = K + 2  # Add two additional control points for the boundary conditions

    control_points_param = torch.nn.Parameter(
        (torch.randn((B, KN, 3), device=signal.device)))

    if K > 1:
        t_step = 1 / (K - 1)
        control_point_times = torch.linspace(
            0 - t_step, 1 + t_step, KN).unsqueeze(0).repeat(B, 1).to(signal.device)
    else:
        control_point_times = torch.tensor(
            [-1, 0, 1], device=signal.device).unsqueeze(0).repeat(B, 1)

    right_idx, rel_frac = _get_interpolate_index_and_distance(
        control_point_times.detach().clone(), times, True)

    with torch.no_grad():
        ip = linear_interpolate_quaternion(
            signal, times, control_point_times, True)
        control_points_param.data = unitquat_to_rotvec(quat_normalize(ip))

    # _pn = torch.zeros_like(control_points_param)[..., :3]
    # _gpos = compose_translation_orientation(_pn, rotvec_to_unitquat(control_points_param.detach()))
    # plot_position(_gpos[0], times=control_point_times[0], title="Init Spline Positions", open=True)

    optimizer = torch.optim.Adam([control_points_param], lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=30)

    def quat_err(x, y):
        qd = quat_subtraction(y, x)
        angle = 2 * torch.atan2(torch.norm(qd[..., :3], dim=-1), qd[..., 3])
        min_dist_angle = torch.min(angle, 2 * np.pi - angle)
        diff = (y[..., :3] - x[..., :3]).norm(dim=-1)
        return (min_dist_angle + diff).mean()

    criterion = quat_err

    prev_loss = float('inf')

    if info is not None:
        info['converged'] = False
        info['initial_loss'] = None
        info['max_epoch'] = epochs - 1

    losses = torch.zeros(epochs)

    with torch.set_grad_enabled(True):
        for epoch in range(epochs):
            optimizer.zero_grad()
            spline = quat_hermite_catmull_rom_index(
                rotvec_to_unitquat(control_points_param), control_point_times, times, True, right_idx, rel_frac)
            loss = criterion(spline, signal)
            loss_value = loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)
            losses[epoch] = loss_value

            if loss_value == 0.:
                rel_change = 0.
            else:
                rel_change = abs((loss_value - prev_loss) / (prev_loss))

            if epoch == 0:
                if info is not None:
                    info['initial_loss'] = loss_value
            elif rel_change < tol or optimizer.param_groups[0]['lr'] < min_lr:
                if info is not None:
                    info['converged'] = True
                    info['max_epoch'] = epoch
                break
            prev_loss = loss_value

    if info is not None:
        info['final_loss'] = loss_value
        info['losses'] = losses[:epoch + 1]

    return unflatten_batch_dims(rotvec_to_unitquat(control_points_param.data.detach()), shp), unflatten_batch_dims(control_point_times, shp)
