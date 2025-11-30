from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Any, Iterable, List, Optional, Tuple

from matplotlib.figure import Figure
import torch
from matplotlib.axes import Axes
from tools.model.discrete_module_scene_node_3d import DiscreteModuleSceneNode3D
from tools.model.module_scene_node_3d import ModuleSceneNode3D
from tools.model.abstract_scene_node import AbstractSceneNode
from tools.transforms.geometric.quaternion import quat_average, quat_product, quat_subtraction
from nag.transforms.transforms3d import (component_position_matrix,
                                         position_quaternion_to_affine_matrix,
                                         _split_transformation_matrix
                                         )
from nag.transforms.transforms_timed_3d import (
    _get_interpolate_index_and_distance, assure_affine_time_matrix, interpolate_orientation, interpolate_vector, linear_interpolate_quaternion, linear_interpolate_vector)

from tools.util.typing import VEC_TYPE, REAL_TYPE
from tools.transforms.geometric.transforms3d import flatten_batch_dims, unflatten_batch_dims
from tools.util.torch import tensorify
from tools.model.module_scene_parent import ModuleSceneParent
from tools.viz.matplotlib import saveable
from matplotlib.axes import Axes
from typing import Optional, Tuple
from tools.transforms.geometric.mappings import rotmat_to_rotvec, unitquat_to_euler, rotmat_to_unitquat, unitquat_to_rotmat, rotvec_to_unitquat, unitquat_to_rotvec
from matplotlib.figure import Figure
from tools.viz.matplotlib import saveable
from tools.transforms.geometric.transforms3d import assure_homogeneous_vector
from tools.util.numpy import numpyify
NON_EQUAL_TIME_STEPS_WARNED = False


def spline_approximation(
        translation: Optional[torch.Tensor],
        orientation: Optional[torch.Tensor],
        times: torch.Tensor,
        K: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate the given translation and orientation with a spline.

    Parameters
    ----------
    translation : Optional[torch.Tensor]
        The translation of the object at discrete times specified in times. Shape ([... B], t, 3) where t is the number of discrete points.

    orientation : Optional[torch.Tensor]
        The orientation of the object at discrete times specified in times. Shape ([... B], t, 4) where t is the number of discrete points.

    times : torch.Tensor
        The discrete times at which the translation and orientation are defined. Shape ([..., B], t).

    K : int
        The number of control points.
        To incorporate the start and end point, the number of control points is K + 2.

    Returns
    -------
    Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
        The approximated 1. translation, 2.orientation of the object. Shape ([... B], K + 2, 3) and ([... B], K + 2, 4). 3. The times of the control points. S

    """

    from nag.transforms.transforms_timed_3d import find_optimal_spline, find_optimal_spline_quat, hermite_catmull_rom_index, quat_hermite_catmull_rom_index
    info = dict()
    if translation is not None and orientation is not None:
        new_trans, cp_times = find_optimal_spline(translation,
                                                  times,
                                                  K=K,
                                                  epochs=10000, lr=0.4, tol=1e-9, info=info)

        new_quat, _ = find_optimal_spline_quat(orientation,
                                               times,
                                               K=K,
                                               epochs=10000, lr=0.4, tol=1e-9, info=info)
        return new_trans, new_quat, cp_times
    elif translation is not None:
        interp = translation
        cp_pos, cp_times = find_optimal_spline(translation,
                                               times,
                                               K=K,
                                               epochs=10000, lr=0.4, tol=1e-9, info=info)
        return cp_pos, None, cp_times
    elif orientation is not None:
        new_quat, cp_times = find_optimal_spline_quat(orientation,
                                                      times,
                                                      K=K,
                                                      epochs=10000, lr=0.4, tol=1e-9, info=info)
        return None, new_quat, cp_times
    return None, None, None


@torch.jit.script
def compose_translation_orientation(translations: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
    """Compose the translation and orientation to a 4x4 matrix.

    Parameters
    ----------
    translations : torch.Tensor
        The translations. Shape ([... B,] T, 3)
    orientations : torch.Tensor
        The orientations. Shape ([... B,] T, 4)

    Returns
    -------
    torch.Tensor
        The composed transformation matrix. Shape ([... B,] T, 4, 4)
    """
    translations, shp = flatten_batch_dims(translations, -3)
    orientations, _ = flatten_batch_dims(orientations, -3)
    N, T, _ = translations.shape
    global_positions = torch.zeros(
        (N, T, 4, 4), dtype=translations.dtype, device=translations.device)
    global_positions[..., :3, 3] = translations[..., :3]
    global_positions[..., :3, :3] = unitquat_to_rotmat(orientations)
    global_positions[..., 3, 3] = 1
    return unflatten_batch_dims(global_positions, shp)


@torch.jit.script
def global_to_local(
    global_position: torch.Tensor,
    v: torch.Tensor,
    v_include_time: bool = False,
) -> torch.Tensor:
    """Converts global vectors to local vectors.
    Will return the local vectors for all times steps in global_position.

    Parameters
    ----------
    global_position : torch.Tensor
        The global position of the object. Shape is (T, 4, 4).

    v : torch.Tensor
        Vectors of shape ([... B,], (3 | 4)) to convert.
        If v_include_time is True, shape is ([... B,] T, (3 | 4))

    v_include_time : bool, optional
        If True, the vectors v include the time as the second to last dimension, by default False

    Returns
    -------
    torch.Tensor
        Vectors in local coordinates. Shape is ([... B,], T, 4).

    """
    v, v_batch_shape = flatten_batch_dims(
        v, -2 if not v_include_time else -3)
    B = v.shape[0]
    glob_mat, _ = flatten_batch_dims(global_position, -3)
    T = glob_mat.shape[0]

    # Check v time consistency
    if v_include_time and v.shape[1] != T:
        raise ValueError(
            "Time steps of v must match the time steps of the object.")

    # Repeat allong new first time axes the v, and new batch axes for the global matrix
    if not v_include_time:
        v = v.unsqueeze(-2).repeat(1, T, 1)

    # Check if last dim = 3, if 3 add w
    if v.shape[-1] == 3:
        v = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)

    # Invert the global matrix
    glob_mat = torch.inverse(glob_mat)
    glob_mat = glob_mat.unsqueeze(0).repeat(B, 1, 1, 1)

    # Flatten both batch dimensions
    v = v.reshape(B * T, 4)
    glob_mat = glob_mat.reshape(B * T, 4, 4)
    res = torch.bmm(glob_mat, v.unsqueeze(-1)).squeeze(-1)
    res = res.reshape(B, T, 4)
    return unflatten_batch_dims(res, v_batch_shape)


@torch.jit.script
def global_to_local_mat(
    global_position: torch.Tensor,
    other: torch.Tensor,
    other_include_time: bool = False,
) -> torch.Tensor:
    """Converts global positions to local positions.
    Will return the local vectors for all times steps in global_position.

    Parameters
    ----------
    global_position : torch.Tensor
        The global position of the object. Shape is (T, 4, 4).

    other : torch.Tensor
        Other positions of shape ([... B,], 4, 4) to convert.
        If other_include_time is True, shape is ([... B,] T, 4, 4)

    other_include_time : bool, optional
        If True, the other positions include the time as the third to last dimension, by default False

    Returns
    -------
    torch.Tensor
        Positions in local coordinates. Shape is ([... B,], T, 4, 4).

    """
    other, other_batch_shape = flatten_batch_dims(
        other, -3 if not other_include_time else -4)
    B = other.shape[0]
    glob_mat, _ = flatten_batch_dims(global_position, -3)
    T = glob_mat.shape[0]

    # Check v time consistency
    if other_include_time and other.shape[1] != T:
        raise ValueError(
            "Time steps of v must match the time steps of the object.")

    # Repeat allong new first time axes the v, and new batch axes for the global matrix
    if not other_include_time:
        other = other.unsqueeze(-3).repeat(1, T, 1, 1)

    # Invert the global matrix
    glob_mat = torch.inverse(glob_mat)
    glob_mat = glob_mat.unsqueeze(0).repeat(B, 1, 1, 1)

    # Flatten both batch dimensions
    other = other.reshape(B * T, 4, 4)
    glob_mat = glob_mat.reshape(B * T, 4, 4)
    res = torch.bmm(glob_mat, other)
    res = res.reshape(B, T, 4, 4)
    return unflatten_batch_dims(res, other_batch_shape)


@torch.jit.script
def local_to_global(
    global_position: torch.Tensor,
    v: torch.Tensor,
    v_include_time: bool = False,
) -> torch.Tensor:
    """Converts local vectors to global vectors.

    Parameters
    ----------
    global_position : torch.Tensor
        The global position of the object. Shape is (T, 4, 4).

    v : torch.Tensor
        Vectors of shape ([... B,] [3 | 4]) to convert.

        If v_include_time is True, shape is ([... B,] T, [3 | 4])

    v_include_time : bool, optional
        If True, the vectors v include the time as the second to last dimension, by default False

    Returns
    -------
    torch.Tensor
        Vectors in global coordinates. Shape is ([... B,], T, 4).
    """
    v, v_batch_shape = flatten_batch_dims(
        v, -2 if not v_include_time else -3)
    B = v.shape[0]
    glob_mat, _ = flatten_batch_dims(global_position, -3)
    T = glob_mat.shape[0]
    if v.shape[-1] == 3:
        v = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
    # Check v time consistency
    if v_include_time and v.shape[1] != T:
        raise ValueError(
            "Time steps of v must match the time steps of the object.")

    # Repeat allong new first time axes the v, and new batch axes for the global matrix
    if not v_include_time:
        v = v.unsqueeze(1).repeat(1, T, 1)

    glob_mat = glob_mat.unsqueeze(0).repeat(B, 1, 1, 1)

    # Flatten both batch dimensions
    v = v.reshape(B * T, 4)
    glob_mat = glob_mat.reshape(B * T, 4, 4)
    res = torch.bmm(glob_mat, v.unsqueeze(-1)).squeeze(-1)
    res = res.reshape(B, T, 4)

    return unflatten_batch_dims(res, v_batch_shape)


@torch.jit.script
def local_to_global_mat(
    global_position: torch.Tensor,
    other: torch.Tensor,
    other_include_time: bool = False,
) -> torch.Tensor:
    """Converts local positions to global positions.

    Parameters
    ----------
    global_position : torch.Tensor
        The global position of the object. Shape is (T, 4, 4).

    other : torch.Tensor
        Position matricies of shape ([... B,] 4, 4) to convert.

        If other_include_time is True, shape is ([... B,] T, 4, 4)

    other_include_time : bool, optional
        If True, the other positions include the time as the third to last dimension, by default False

    Returns
    -------
    torch.Tensor
        Positions in global coordinates. Shape is ([... B,], T, 4, 4).
    """
    other, v_batch_shape = flatten_batch_dims(
        other, -3 if not other_include_time else -4)
    B = other.shape[0]
    glob_mat, _ = flatten_batch_dims(global_position, -3)
    T = glob_mat.shape[0]

    # Check v time consistency
    if other_include_time and other.shape[1] != T:
        raise ValueError(
            "Time steps of v must match the time steps of the object.")

    # Repeat allong new first time axes the v, and new batch axes for the global matrix
    if not other_include_time:
        other = other.unsqueeze(1).repeat(1, T, 1, 1)

    glob_mat = glob_mat.unsqueeze(0).repeat(B, 1, 1, 1)

    # Flatten both batch dimensions
    other = other.reshape(B * T, 4, 4)
    glob_mat = glob_mat.reshape(B * T, 4, 4)
    res = torch.bmm(glob_mat, other)
    res = res.reshape(B, T, 4, 4)
    return unflatten_batch_dims(res, v_batch_shape)


@torch.jit.script
def get_translation(
        translation: torch.Tensor,
        times: torch.Tensor,
        steps: Optional[torch.Tensor] = None,
        equidistant_times: bool = False,
        interpolation: str = "cubic",
        right_idx: Optional[torch.Tensor] = None,
        rel_frac: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Get the translation at times t by eventually interpolating these.

    Parameters
    ----------
    translation : torch.Tensor
        The actual timed translation points ([... B,] t, 3)
    times : torch.Tensor
        The times of the discrete translations in time in range [0, 1]. Shape ([... B,] t)
    steps : Optional[torch.Tensor], optional
        The times at which the the translations should be sampled. Shape ([... B,] S) , by default None
        If None, returns all the translations.
    equidistant_times : bool, optional
        Flag indicating if the time steps (times) are equally spaced., by default False
        If True, the interpolation is faster as no sorting is needed.
    interpolation : str, optional
        The interpolation method which should be used., by default "cubic"
    right_idx : Optional[torch.Tensor], optional
        Precombuted right_idx from sorting, by default None
        Will be computed if not provided
    rel_frac : Optional[torch.Tensor], optional
        Precombuted relative stepsize from sorting, by default None
        Will be computed if not provided

    Returns
    -------
    torch.Tensor
        The translation at times steps (S). Shape ([... B,] S, 3)
    """
    if steps is None:
        return translation
    else:
        # Interpolate
        # If len of t is 1, repeat the time steps
        if len(times) <= 1:
            ft, shape = flatten_batch_dims(translation, -3)
            steps, _ = flatten_batch_dims(steps, -1)
            rep = ft.repeat(1, steps.shape[0], 1)
            return unflatten_batch_dims(rep, shape)
        return interpolate_vector(translation, times, steps=steps, equidistant_times=equidistant_times, right_idx=right_idx, rel_frac=rel_frac, method=interpolation)


@torch.jit.script
def get_orientation(
        orientation: torch.Tensor,
        times: torch.Tensor,
        steps: Optional[torch.Tensor] = None,
        equidistant_times: bool = False,
        interpolation: str = "cubic",
        right_idx: Optional[torch.Tensor] = None,
        rel_frac: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Get the offset orientation at times t.

    Parameters
    ----------
    orientation : torch.Tensor
        The orientation as normalized control points on each discrete timestep ([... B,] t, 4)

    times : torch.Tensor
        The times of the orientations in range [0, 1]. Shape ([... B,] t)

    steps : Optional[torch.Tensor], optional
        The times at which the orientations should be sampled. Shape ([... B,] S) , by default None
        If None, returns all the orientations.

    equidistant_times : bool, optional
        Flag indicating if the time steps (times) are equally spaced., by default False
        If True, the interpolation is faster as no sorting is needed.
    interpolation : str, optional
        The interpolation method which should be used., by default "cubic"

    right_idx : Optional[torch.Tensor], optional
        Precombuted right_idx from sorting, by default None
        Will be computed if not provided

    rel_frac : Optional[torch.Tensor], optional
        Precombuted relative stepsize from sorting, by default None
        Will be computed if not provided

    Returns
    -------
    torch.Tensor
        The orientation at times in steps (S). Shape ([... B,] S, 4)
    """
    if steps is None:
        return orientation
    else:
        # Interpolate
        if len(times) <= 1:
            ft, shape = flatten_batch_dims(orientation, -3)
            steps, _ = flatten_batch_dims(steps, -1)
            rep = ft.repeat(1, steps.shape[0], 1)
            return unflatten_batch_dims(rep, shape)
        return interpolate_orientation(
            orientation,
            times,
            steps=steps,
            equidistant_times=equidistant_times,
            right_idx=right_idx,
            rel_frac=rel_frac,
            method=interpolation
        )


@torch.jit.script
def get_translation_orientation(
        translation: torch.Tensor,
        orientation: torch.Tensor,
        times: torch.Tensor,
        steps: Optional[torch.Tensor] = None,
        equidistant_times: bool = False,
        interpolation: str = "cubic",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the translation and orientation of the object at each time t in steps.

    If steps is None, the position is returned for all time steps.

    Parameters
    ----------
    translation : torch.Tensor
        The actual timed translation points ([... B,] t, 3)

    orientation : torch.Tensor
        The orientation as normalized quaternions on each discrete timestep ([... B,] t, 4)

    times : torch.Tensor
        The times of the translations and orientations in range [0, 1]. Shape ([... B,] t)

    steps : Optional[torch.Tensor], optional
        Timestamps to get the object position for Shape ([... B], S), by default None
        If positions are not available for the given time steps, they are interpolated linearly.

    equidistant_times : bool, optional
        Flag indicating if the time steps (times) are equally spaced., by default False
        If True, the interpolation is faster as no sorting is needed.

    interpolation : str, optional
        The interpolation method which should be used., by default "cubic"

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The translation and orientation of the object at time t.
    """
    right_idx: Optional[torch.Tensor] = None
    rel_frac: Optional[torch.Tensor] = None
    if steps is not None:
        steps, _ = flatten_batch_dims(steps, -1)
        right_idx, rel_frac = _get_interpolate_index_and_distance(
            times, steps, equidistant=equidistant_times)
    interp_translation = get_translation(
        translation=translation, times=times, steps=steps, equidistant_times=equidistant_times, interpolation=interpolation, right_idx=right_idx, rel_frac=rel_frac)
    interp_orientation = get_orientation(
        orientation=orientation, times=times, steps=steps, equidistant_times=equidistant_times, interpolation=interpolation, right_idx=right_idx, rel_frac=rel_frac)
    return interp_translation, interp_orientation


class TimedDiscreteSceneNode3D(DiscreteModuleSceneNode3D):
    """Pytorch Module class for which have 3D positions for discrete timestamps."""

    _translation: torch.Tensor
    """The objects relative translation w.r.t its parent
    in as a vector (t x 3) containing its position on discrete time steps t."""

    _orientation: torch.Tensor
    """The objects relative orientation w.r.t its parent as normalized quaternion (t, 4). (x, y, z, w) for each time step t."""

    _times: torch.Tensor
    """The time steps of the translations and orientations in range [0, 1]. Shape (t,)"""

    _equidistant_times: bool
    """Flag indicating if the time steps are equally spaced."""

    _interpolation: str
    """The interpolation method to use for the position and orientation."""

    def __init__(self,
                 translation: Optional[VEC_TYPE] = None,
                 orientation: Optional[VEC_TYPE] = None,
                 position: Optional[VEC_TYPE] = None,
                 times: Optional[VEC_TYPE] = None,
                 interpolation: str = "cubic",
                 name: Optional[str] = None,
                 children: Optional[Iterable['AbstractSceneNode']] = None,
                 decoding: bool = False,
                 dtype: torch.dtype = torch.float32,
                 _times: Optional[torch.Tensor] = None,
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

        if times is None:
            times = torch.linspace(0, 1, 1, dtype=dtype)
        self._equidistant_times = True
        self._interpolation = interpolation
        self._init_times(times, dtype=dtype, _times=_times)

    def _init_times(self, times: Optional[VEC_TYPE], dtype: torch.dtype, _times: Optional[torch.Tensor] = None):
        if _times is not None:
            # Reference init
            self.register_buffer("_times", _times, persistent=False)
        else:
            # Normal init.
            self.register_buffer("_times", torch.ones(1, dtype=dtype))
            self.set_times(times)

    def _get_default_translation(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((1, 3), dtype=dtype)

    def _get_default_orientation(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([0., 0., 0., 1.], dtype=dtype).unsqueeze(0)

    def get_times(self) -> torch.Tensor:
        return self._times

    def set_times(self, value: torch.Tensor):
        # Check if times is a buffer, if not it can not be set
        if "_times" not in self._buffers:
            raise ValueError(
                "Times was not initializes as a buffer but as a reference. It can not be set in the object!")
        if len(value.shape) != 1:
            raise ValueError("Times must be a 1D tensor.")
        if value.shape[0] != self._translation.shape[0] and self._translation.shape[0] != 1:
            raise ValueError("Positions must have the same length as times.")
        value = value.to(self._translation.dtype)
        if self._translation.shape[0] == 1:
            self._translation = self._translation.repeat(value.shape[0], 1)
        if self._orientation.shape[0] == 1:
            self._orientation = self._orientation.repeat(value.shape[0], 1)
        self._times = value
        # Check if times are equally spaced
        eq = self._equidistant_times
        delta_t = torch.diff(self._times, dim=-1)
        self._equidistant_times = torch.allclose(
            delta_t, self._times[..., 1] - self._times[..., 0], atol=1e-4) if len(self._times) > 1 else True
        # Warn if times are not equally spaced
        if eq != self._equidistant_times and not self._equidistant_times:
            global NON_EQUAL_TIME_STEPS_WARNED
            if not NON_EQUAL_TIME_STEPS_WARNED:
                long = delta_t.shape[0] > 10
                tc = delta_t[:10] if long else delta_t
                self.logger.warning(
                    f"Times are not equally spaced. This can slow down interpolation.\nΔt: [{', '.join(['{:.4f}'.format(t.item()) for t in tc]) + ', ...' if long else ''}]")
                NON_EQUAL_TIME_STEPS_WARNED = True

    def get_translation(self,
                        t: Optional[torch.Tensor] = None,
                        right_idx: Optional[torch.Tensor] = None,
                        rel_frac: Optional[torch.Tensor] = None) -> torch.Tensor:
        return get_translation(self._translation, self._times, steps=t, equidistant_times=self._equidistant_times, interpolation=self._interpolation, right_idx=right_idx, rel_frac=rel_frac)

    def get_orientation(self,
                        t: Optional[torch.Tensor] = None,
                        right_idx: Optional[torch.Tensor] = None,
                        rel_frac: Optional[torch.Tensor] = None) -> torch.Tensor:
        return get_orientation(self._orientation, self._times, steps=t, equidistant_times=self._equidistant_times, interpolation=self._interpolation, right_idx=right_idx, rel_frac=rel_frac)

    def get_translation_orientation(self, t: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the translation and orientation of the object at time t.

        If t is None, the position is returned for all time steps.

        Parameters
        ----------
        t : Optional[torch.Tensor], optional
            Timestamps to get the object position for, by default None
            If positions are not available for the given time steps, they are interpolated linearly.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The translation and orientation of the object at time t.
        """
        return get_translation_orientation(self._translation, self._orientation, self._times, steps=t, equidistant_times=self._equidistant_times, interpolation=self._interpolation)

    def get_position(self, t: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Get the position of the object at time t.

        If t is None, the position is returned for all time steps.

        Parameters
        ----------
        t : Optional[torch.Tensor], optional
            Timestamps to get the object position for, by default None
            If positions are not available for the given time steps, they are interpolated linearly.

        Returns
        -------
        torch.Tensor
            The position of the object at time t (t x 4 x 4).
            As an affine matrix.
        """
        p, o = self.get_translation_orientation(t=t, **kwargs)
        return position_quaternion_to_affine_matrix(position=p, quaternion=o)

    def set_position(self, value: VEC_TYPE):
        pos = assure_affine_time_matrix(value, dtype=self._translation.dtype)
        if pos.shape[0] != self._times.shape[0] and pos.shape[0] != 1:
            raise ValueError("Position must have the same length as times.")
        if pos.shape[0] == 1:
            pos = pos.repeat(self._times.shape[0], 1, 1)
        pos, quat = self._parse_position(value)
        self._translation = pos
        self._orientation = quat

    def local_to_global(self,
                        v: torch.Tensor,
                        t: Optional[torch.Tensor] = None,
                        v_include_time: bool = False,
                        **kwargs) -> torch.Tensor:
        """Converts local vectors to global vectors.

        Parameters
        ----------
        v : torch.Tensor
            Vectors of shape ([... B,] [3 | 4]) to convert.

            If v_include_time is True, shape is ([... B,] T, [3 | 4])

        t : Optional[torch.Tensor], optional
            The time steps to get the global position for, by default None
            None means all time steps. Shape (T,)

        v_include_time : bool, optional
            If True, the vectors v include the time as the second to last dimension, by default False

        Returns
        -------
        torch.Tensor
            Vectors in global coordinates. Shape is ([... B,], T, 4).
        """
        glob_mat = self.get_global_position(t=t, **kwargs)
        return local_to_global(glob_mat, v, v_include_time)

    def global_to_local(self,
                        v: torch.Tensor,
                        t: Optional[torch.Tensor] = None,
                        v_include_time: bool = False,
                        **kwargs) -> torch.Tensor:
        """Converts global vectors to local vectors.

        Parameters
        ----------
        v : torch.Tensor
            Vectors of shape ([... B,], (3 | 4)) to convert.

            If v_include_time is True, shape is ([... B,] T, (3 | 4))

        t : Optional[torch.Tensor], optional
            The time steps to get the global position for, by default None
            None means all time steps. Shape (T,)

        v_include_time : bool, optional
            If True, the vectors v include the time as the second to last dimension, by default False

        Returns
        -------
        torch.Tensor
            Vectors in local coordinates. Shape is ([... B,], T, 4).
        """
        glob_mat = self.get_global_position(t=t, **kwargs)
        return global_to_local(glob_mat, v, v_include_time)

    def get_global_position(self, t: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Return the global position of the scene object, taking into account the position of the parents.

        Parameters
        ----------
        t : Optional[torch.Tensor], optional
            The time steps to get the global position for, by default None
            None means all time steps.
            If t is not matching the time steps of the object, the position is interpolated.

        Returns
        -------
        torch.Tensor
            Matrix describing the global position. In shape [t, 4, 4]
        """
        t = tensorify(t, device=self._translation.device,
                      dtype=self._translation.dtype) if t is not None else None
        cpos = self.get_position(t, **kwargs)
        if self._parent is None:
            return cpos
        else:
            if t is not None:
                t, batch_dims = flatten_batch_dims(t, -1)
            else:
                batch_dims = [self._times.shape[0]]
            parent = self.get_parent()
            # Unpack proxy if needed
            if isinstance(parent, ModuleSceneParent):
                parent = parent._node

            if isinstance(parent, TimedDiscreteSceneNode3D):
                # Check if the parent has the same time steps
                # If not, ask specific time steps
                if t is None and (parent.get_times() != self.get_times()).any():
                    t = self.get_times()
                ppos = parent.get_global_position(t, **kwargs)
            else:
                ppos = parent.get_global_position(**kwargs)
                ppos = ppos.unsqueeze(0).repeat(cpos.shape[0], 1, 1)
            return unflatten_batch_dims(torch.bmm(ppos, cpos), batch_dims)

    # region Plotting

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
        from matplotlib.colors import to_rgba
        from tools.util.torch import index_of_first
        cmap = plt.get_cmap(cmap)
        camera: TimedCameraSceneNode3D
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
        t_frames_idx = None
        if t_frames is not None:
            traj_t = torch.cat([traj_t, t_frames], dim=0)
            traj_t = torch.unique(traj_t)
            traj_t = torch.sort(traj_t)[0]

        global_points = self.local_to_global(points, t=traj_t)
        image_points, oks = camera.global_to_image_coordinates(
            global_points, t=traj_t)
        image_points = image_points[oks]
        image_traj_t = traj_t[oks]
        if t_frames is not None:
            t_frames_idx = index_of_first(image_traj_t, t_frames)
            existing = t_frames_idx >= 0
            t_frames_idx = t_frames_idx[existing]
            t_frames_visible = t_frames[existing]

        colors = cmap(torch.arange(len(image_points)) % cmap.N)
        ax.plot(*image_points[:, :2].T, color=colors, zorder=zorder)
        return ax.figure

    def plot_point_trace(self,
                         points: torch.Tensor,
                         ax: Axes,
                         colors: Iterable[Any] = None,
                         t_min: Optional[REAL_TYPE] = None,
                         t_max: Optional[REAL_TYPE] = None,
                         t_step: REAL_TYPE = 0.1,
                         **kwargs):
        """Plots the position of points over time until t_max. Which creates a trace line.

        Parameters
        ----------
        points : torch.Tensor
            The points to plot over time in the local coordinate system.
            Shape: ([..., B], 3)

        ax : Axes
            The axes to plot on.

        colors : Iterable[Any], optional
            The colors to use for the points, by default None.

        t_max : REAL_TYPE, optional
            The maximum time to plot the points, by default None.
            If None, the maximum time of the object is used.

        t_step : REAL_TYPE, optional
            The time step to plot the points, by default 0.1.

        """
        points, _ = flatten_batch_dims(points, -2)

        # Add w to points
        if points.shape[-1] == 3:
            points = torch.cat([points, torch.ones(
                points.shape[:-1] + (1,), dtype=points.dtype, device=points.device)], dim=-1)

        if t_max is None:
            t_max = self._times.max()
        t_max = tensorify(t_max).squeeze().item()
        if t_min is None:
            t_min = self._times.min()
        t_min = tensorify(t_min).squeeze().item()
        t_step = tensorify(t_step)

        cmap = plt.get_cmap(kwargs.get("cmap", "tab10"))
        if t_min > t_max:
            return
        steps = torch.linspace(t_min, t_max, int(
            (t_max - t_min) / t_step) + 1, dtype=self._times.dtype, device=self._times.device)
        # Get the global position of the object

        global_points = self.local_to_global(points, t=steps)

        for b in range(points.shape[0]):
            # point_mat = points_mat[b].unsqueeze(0).repeat(pos.shape[0], 1, 1)
            # global_points = torch.bmm(pos, point_mat)
            global_points_b = global_points[b].detach().cpu()
            color = cmap(
                b % cmap.N) if colors is None else colors[b % len(colors)]
            ax.plot(*global_points_b[:, :3].T, color=color)

    @saveable()
    def plot_position(self,
                      ax: Optional[Tuple[Axes, Axes]] = None,
                      t: Optional[torch.Tensor] = None,
                      use_global_position: bool = False,
                      title: Optional[str] = None,
                      **kwargs):
        """Plots the translation of the object over time.

        Parameters
        ----------
        ax : Optional[Axes], optional
            The axes to plot on, by default None
            If None, a new figure is created.

        t : Optional[torch.Tensor], optional
            The time steps to plot the translation for, by default None
            If None, the translation is plotted for all time steps.

        """

        with torch.no_grad():
            if use_global_position:
                position = self.get_global_position(t).cpu()
            else:
                position = self.get_position(t=t, **kwargs).cpu()

        return plot_position(position, self._times, ax=ax, t=t, title=title, **kwargs)

    def plot_object(self, ax: Axes, t: Optional[torch.Tensor] = None, **kwargs):
        """Gets a projection 3D axis and is called during plot_coordinates
        function. Can be used to plot arbitrary objects.

        Parameters
        ----------
        ax : Axes
            The axes to plot on.

        **kwargs
            Additional arguments.
        """
        t = tensorify(t, device=self._translation.device,
                      dtype=self._translation.dtype) if t is not None else None
        if t is not None and len(t.shape) == 0:
            t = t.unsqueeze(0)
        if kwargs.get("plot_coordinate_annotations", False) and self._name is not None:
            position = self.get_global_position_vector(t=t)
            for b in range(position.shape[0]):
                if position.shape[0] > 1:
                    t_text = f" t:{t[b]:.2f}" if t is not None else ""
                else:
                    t_text = ""
                position = numpyify(position)
                ax.text(*position[b, :3], self._name +
                        t_text, horizontalalignment='center', verticalalignment='center')
        if kwargs.get("plot_coordinate_lines", False):
            coordinate_system_indicator_length = kwargs.get(
                "coordinate_system_indicator_length", 1.0)
            pos = self.get_global_position(t)
            # Get the coordinate system vectors which endpoints we try to connect and plot
            cord_vec_x = _split_transformation_matrix(torch.bmm(pos, component_position_matrix(
                x=coordinate_system_indicator_length, dtype=pos.dtype, device=pos.device).repeat(pos.shape[0], 1, 1)))[0]
            cord_vec_y = _split_transformation_matrix(torch.bmm(pos, component_position_matrix(
                y=coordinate_system_indicator_length, dtype=pos.dtype, device=pos.device).repeat(pos.shape[0], 1, 1)))[0]
            cord_vec_z = _split_transformation_matrix(torch.bmm(pos, component_position_matrix(
                z=coordinate_system_indicator_length, dtype=pos.dtype, device=pos.device).repeat(pos.shape[0], 1, 1)))[0]

            ax.plot(*cord_vec_x.T, color='r')
            ax.plot(*cord_vec_y.T, color='g')
            ax.plot(*cord_vec_z.T, color='b')

    @saveable(default_fps=10,
              default_dpi=150,
              default_ext="gif",
              is_animation=True)
    def plot_scene_animation(self,
                             t_step: Optional[torch.Tensor] = None,
                             world: Optional["ModuleSceneNode3D"] = None,
                             ax: Optional[Axes] = None,
                             x_lim: Optional[Tuple[REAL_TYPE,
                                                   REAL_TYPE]] = None,
                             y_lim: Optional[Tuple[REAL_TYPE,
                                                   REAL_TYPE]] = None,
                             z_lim: Optional[Tuple[REAL_TYPE,
                                                   REAL_TYPE]] = None,
                             supersample: int = 4,
                             enable_axis: bool = True,
                             zoom: float = 1.0,
                             **kwargs) -> Tuple[Figure, FuncAnimation]:
        import matplotlib.pyplot as plt

        from functools import partial

        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(projection='3d')
        else:
            fig = ax.figure

        if not enable_axis:
            ax.set_axis_off()
        ax.set_box_aspect(None, zoom=zoom)
        ax.set_aspect("equal")
        t = set()
        # Query all children and add timesteps if they are not present
        for child in self.query_children(include_self=True):
            t = t.union(set(child.get_times().tolist()))
        t = torch.tensor(sorted(list(t)))

        if world is None:
            world = self.get_root()

        if t_step is None:
            t_step = (t.roll(-1, -1) - t)[:-1].min()
            t_step = t_step / supersample
        t_min = t.min()
        t_max = t.max()
        times = torch.linspace(t_min, t_max, int((t_max - t_min) / t_step))

        args = dict(kwargs)
        args.pop("t", None)
        args.pop("ax", None)
        time_draw = partial(world.plot_scene,
                            **args,
                            ax=ax)

        def init(ax: Axes, fig: Figure):
            try:
                fig = time_draw(t=torch.tensor([0.]))
                ax = fig.axes[0]
                if x_lim is not None:
                    ax.set_xlim(x_lim)
                if y_lim is not None:
                    ax.set_ylim(y_lim)
                if z_lim is not None:
                    ax.set_zlim(z_lim)

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

                ax.set_aspect("equal")
                if not enable_axis:
                    ax.set_axis_off()
            except Exception as e:
                self.logger.exception("Err")
            return fig.artists

        def update(frame: torch.Tensor, ax: Axes, fig: Figure):
            try:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                zlim = ax.get_zlim()
                aspect = ax.get_aspect()

                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()
                zlabel = ax.get_zlabel()

                ax.clear()
                fig = time_draw(t=frame)

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_zlabel(zlabel)
                # aspect = torch.ones(3).numpy() if aspect == "equal" else aspect
                ax.set_aspect(aspect)
                if not enable_axis:
                    ax.set_axis_off()
            except Exception as e:
                self.logger.exception("Err")
            return fig.artists

        update_p = partial(update, ax=ax, fig=fig)
        init_p = partial(init, ax=ax, fig=fig)

        ani = FuncAnimation(
            fig, update_p,
            frames=times,
            init_func=init_p, blit=False)
        return fig, ani
# endregion

# region Misc

    def save_tensor(self,
                    value: torch.Tensor,
                    name: str,
                    time: Optional[torch.Tensor] = None,
                    path: str = "temp/",
                    index: Optional[Any] = None,
                    **kwargs):
        """Saves the tensor to the given path.

        Parameters
        ----------
        value : torch.Tensor
            The tensor to save.
        path : str
            The path to save the tensor to.
        """
        return save_tensor(value=value,
                           name=name,
                           index=index if index is not None else self.get_index(),
                           time=time,
                           path=path,
                           **kwargs)

    def load_tensor(self,
                    name: str,
                    time: Optional[torch.Tensor] = None,
                    path: str = "temp/",
                    index: Optional[Any] = None,
                    **kwargs) -> torch.Tensor:
        """
        Loads the tensor from the given path.

        Parameters
        ----------
        name : str
            The name of the tensor.
        time : Optional[torch.Tensor], optional
            Some timestep marker, by default None
        path : str
            The folder path to load the tensor from.
        index : Optional[Any], optional
            Index or reference of the object, holding the tensor.
            Default is None, which means the index of the object is used.


        Returns
        -------
        torch.Tensor
            The loaded tensor.
        """
        return load_tensor(name=name,
                           index=index if index is not None else self.get_index(),
                           time=time,
                           path=path,
                           **kwargs)

    # endregion

# Misc Saving


def save_tensor(
        value: torch.Tensor,
        name: str,
        index: Any,
        time: Optional[torch.Tensor] = None,
        path: str = "temp/",
        **kwargs):
    """Saves the tensor to the given path.

    Parameters
    ----------
    value : torch.Tensor
        The tensor to save.
    name : str
        The name of the tensor.
    index : Any
        Index or reference of the object, holding the tensor.
    time : Optional[torch.Tensor], optional
        Some timestep marker, by default None
    path : str
        The path to save the tensor to.
    """
    import os
    from tools.util.path_tools import replace_unallowed_chars
    name_pattern = f"{name}_{str(index)}"
    if time is not None:
        name_pattern += f"_t_{time.round(decimals=6).detach().item():.6f}"

    name_str = replace_unallowed_chars(name_pattern, allow_dot=False)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(value, os.path.join(path, f"{name_str}.pt"))


def load_tensor(name: str,
                index: Any,
                time: Optional[torch.Tensor] = None,
                path: str = "temp/",
                **kwargs) -> torch.Tensor:
    """Loads the tensor from the given path.

    Parameters
    ----------
    name : str
        The name of the tensor.
    index : Any
        Index or reference of the object, holding the tensor.
    time : Optional[torch.Tensor], optional
         Some timestep marker, by default None
    path : str
        The folder path to load the tensor from.

    Returns
    -------
    torch.Tensor
        The loaded tensor.
    """
    import os
    from tools.util.path_tools import replace_unallowed_chars
    name_pattern = f"{name}_{str(index)}"
    if time is not None:
        name_pattern += f"_t_{time.round(decimals=6).detach().item():.6f}"
    name_str = replace_unallowed_chars(
        name_pattern, allow_dot=False)
    return torch.load(os.path.join(path, f"{name_str}.pt"))

# region General Plotting


def plot_timed_vlines(
        times: torch.Tensor,
        ax: Axes,
        t: Optional[torch.Tensor],
        color: str = "gray",
        linestyle: str = "--",
        linewidth: float = 1.0,
        alpha: float = 0.5,
        label: Optional[str] = None,
        **kwargs) -> Figure:
    """
    Plots vertical lines at specified time steps on a given Axes object.

    Parameters
    ----------
    times : torch.Tensor
        A Tensor representing the time steps where vertical lines should be drawn. Shape is (T,).
    ax : Axes
        An Axes object from the Matplotlib library, representing the plot area where the lines will be added.
    t : Optional[torch.Tensor]
        An optional Tensor that can be used to specify the x-axis range for the plot. If not provided, it defaults to the `times` tensor.
    color : str, optional
        A string specifying the color of the vertical lines, defaulting to "gray".
    linestyle : str, optional
        A string specifying the line style of the vertical lines, defaulting to "--".
    alpha : float, optional
        A float value between 0 and 1 representing the opacity of the vertical lines, defaulting to 0.5.
    label : Optional[str], optional
        The label for the vertical lines, by default None
    Returns
    -------
    Figure
        The Matplotlib Figure object containing the Axes with the plotted vertical lines.
    """
    if t is None:
        t = times
    fig = ax.figure
    mint = t.min()
    maxt = t.max()

    t = numpyify(t)

    vlines = times[(times >= mint) & (times <= maxt)]
    vlines = numpyify(vlines)
    for vline in vlines:
        ax.axvline(vline, color=color, linestyle=linestyle,
                   linewidth=linewidth, alpha=alpha, label=label)
    return fig


@saveable()
def plot_position(
        position: torch.Tensor,
        times: torch.Tensor,
        ax: Optional[Tuple[Axes, Axes]] = None,
        t: Optional[torch.Tensor] = None,
        title: Optional[str] = None,
        y_lim_translation: Optional[Tuple[float, float]] = None,
        y_lim_orientation: Optional[Tuple[float, float]] = None,
        time_in_frames: bool = False,
        t_real: Optional[torch.Tensor] = None,
        translation_labels: Optional[List[str]] = None,
        orientation_labels: Optional[List[str]] = None,
        **kwargs):
    """Plots the translation of the object over time.

    Parameters
    ----------
    ax : Optional[Axes], optional
        The axes to plot on, by default None
        If None, a new figure is created.

    t : Optional[torch.Tensor], optional
        The time steps to plot the translation for, by default None
        If None, the translation is plotted for all time steps.

    """
    import matplotlib.pyplot as plt
    from tools.viz.matplotlib import get_mpl_figure
    from tools.transforms.numpy.min_max import MinMax

    ax_init = True
    if ax is None:
        fig, ax = get_mpl_figure(1, 2)
        ax_init = True

    if t is None:
        t = times

    if time_in_frames:
        t = (t * times.shape[0]).round().int()
        times = (times * times.shape[0]).round().int()

    rot = position[..., :3, :3]  # Shape: [T, 3, 3]
    euler = unitquat_to_euler("xyz", rotmat_to_unitquat(rot), degrees=True)
    pos = position[..., :3, 3]  # Shape: [T, 3]

    ax1 = ax[0]
    ax2 = ax[1]
    fig = ax1.figure

    if title is not None:
        gs = fig.add_gridspec(1, 2)
        ax_t = fig.add_subplot(gs[0, :])
        ax_t.axis("off")
        ax_t.text(0.5, 1.05, title,
                  ha="center", fontsize=12,
                  transform=ax_t.transAxes,
                  )

    channels = [
        "x", "y", "z"] if translation_labels is None else translation_labels
    if len(channels) != pos.shape[-1]:
        raise ValueError(
            f"Number of translation labels ({len(channels)}) does not match the number of translation channels ({pos.shape[-1]})")

    pos = numpyify(pos)
    euler = numpyify(euler)
    t = numpyify(t)

    # Plot the translation
    c_max = pos.shape[-1]
    for c in range(c_max):
        if len(t) > 1:
            ax1.plot(t, pos[..., c], label=channels[c])
        else:
            width = 0.05
            rel_space = 0.05
            max_width = (c_max - 1) * width + (c_max - 1) * (width * rel_space)
            xpos = (t + c * width + c * (width * rel_space)) - max_width / 2
            ax1.bar(xpos, pos[..., c], width=width, label=channels[c])

    if t_real is not None:
        from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
        trmin = t_real.min()
        trmax = t_real.max()
        if trmin == trmax:
            # Avoiding singular transformation
            trmax = trmin + 1

        xmm = MinMax(new_min=trmin, new_max=trmax)
        _ = xmm.fit_transform(t)
        if xmm.min == xmm.max:
            # Avoiding singular transformation
            xmm.max = xmm.min + 1
        ax1_t = ax1.secondary_xaxis(
            "top", functions=(xmm, xmm.inverse_transform))
        ax2_t = ax2.secondary_xaxis(
            "top", functions=(xmm, xmm.inverse_transform))

        ax1_t.xaxis.set_minor_locator(MultipleLocator(1))
        ax2_t.xaxis.set_minor_locator(MultipleLocator(1))

    plot_timed_vlines(times, ax1,  t, **kwargs)
    plot_timed_vlines(times, ax2,  t, **kwargs)

    # Plot the rotation
    angles = ["X", "Y", "Z"] if orientation_labels is None else orientation_labels

    if len(angles) != euler.shape[-1]:
        raise ValueError(
            f"Number of angle labels ({len(angles)}) does not match the number of euler angles ({euler.shape[-1]})")

    c_max = euler.shape[-1]
    for c in range(c_max):
        if len(t) > 1:
            ax2.plot(t, euler[..., c], label=angles[c])
        else:
            width = 0.05
            rel_space = 0.05
            max_width = (c_max - 1) * width + (c_max - 1) * (width * rel_space)
            xpos = (t + c * width + c * (width * rel_space)) - max_width / 2
            ax2.bar(xpos, euler[..., c], width=width, label=angles[c])

    if ax_init:
        ax1.legend()
        ax2.legend()
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Translation [U]")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("↻ - Axis Rotation [°]")
        ax1.set_aspect("auto")
        ax2.set_aspect("auto")

    if y_lim_translation is not None:
        ax1.set_ylim(y_lim_translation)

    if y_lim_orientation is not None:
        ax2.set_ylim(y_lim_orientation)

    return fig
    # endregion
