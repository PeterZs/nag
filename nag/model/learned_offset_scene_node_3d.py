from typing import Iterable, Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import torch
from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D, get_orientation, get_translation, plot_position as plot_position_base, plot_timed_vlines
from tools.model.abstract_scene_node import AbstractSceneNode
from nag.transforms.utils import quat_composition
from tools.transforms.geometric.mappings import rotvec_to_unitquat, unitquat_to_rotvec
from nag.transforms.transforms_timed_3d import (
    _get_interpolate_index_and_distance, interpolate_orientation, interpolate_vector)
from tools.viz.matplotlib import saveable
from tools.util.typing import VEC_TYPE


@torch.jit.script
def get_combined_translation(
    translation: torch.Tensor,
    offset_translation: torch.Tensor,
    translation_offset_weight: torch.Tensor,
    times: torch.Tensor,
    offset_times: torch.Tensor,
    steps: Optional[torch.Tensor] = None,
    interpolation: str = 'cubic',
    equidistant_times: bool = False,
    equidistant_offset_times: bool = True,
    right_idx: Optional[torch.Tensor] = None,
    rel_frac: Optional[torch.Tensor] = None,
    right_idx_offset: Optional[torch.Tensor] = None,
    rel_frac_offset: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Get the translation of an object at timesteps s.

    Parameters
    ----------
    translation : torch.Tensor
        The translation of the object at discrete times specified in times. Shape ([... B], t, 3) where t is the number of discrete points.
    offset_translation : torch.Tensor
        The offset translation vector of the object at discrete times specified in offset_times. Shape ([... B], tc, 3) where tc is the number of discrete points.
    translation_offset_weight : torch.Tensor
        The weight of the offset translation vector. Shape ([... B]).
    times : torch.Tensor
        The discrete times at which the translation is defined. Shape ([..., B], t).
    offset_times : torch.Tensor
        The discrete times at which the offset translation is defined. Shape ([..., B], tc).
    steps : Optional[torch.Tensor], optional
        The timesteps ([...B,], S) to get the position for, by default None
    interpolation : str, optional
        The interpolation method to use, by default 'cubic'
    equidistant_times : bool, optional
        If the times are equidistant, by default False
    equidistant_offset_times : bool, optional
        If the offset times are equidistant, by default True
    right_idx : Optional[torch.Tensor], optional
        Right index for faster interpolation, by default None
    rel_frac : Optional[torch.Tensor], optional
        Relative stepsize for faster interpolation, by default None
    right_idx_offset : Optional[torch.Tensor], optional
        Right index for faster interpolation, by default None
    rel_frac_offset : Optional[torch.Tensor], optional
        Relative stepsize for faster interpolation, by default None

    Returns
    -------
    torch.Tensor
        The translation of the object at timesteps s. Shape ([... B], S, 3).
    """
    base_translation = get_translation(translation, times=times, steps=steps, equidistant_times=equidistant_times,
                                       interpolation=interpolation, right_idx=right_idx, rel_frac=rel_frac)
    offset_translation = translation_offset_weight * get_translation(
        translation=offset_translation,
        times=offset_times,
        steps=steps,
        interpolation=interpolation,
        # Offset times are always equidistant see linspace in _default_offset_times
        equidistant_times=equidistant_offset_times,
        right_idx=right_idx_offset,
        rel_frac=rel_frac_offset
    )
    if offset_translation.numel() == 0:
        return base_translation
    return base_translation + offset_translation


@torch.jit.script
def get_combined_orientation(
    orientation: torch.Tensor,
    rotation_offset: torch.Tensor,
    rotation_offset_weight: torch.Tensor,
    times: torch.Tensor,
    offset_times: torch.Tensor,
    steps: Optional[torch.Tensor] = None,
    interpolation: str = 'cubic',
    equidistant_times: bool = False,
    equidistant_offset_times: bool = True,
    right_idx: Optional[torch.Tensor] = None,
    rel_frac: Optional[torch.Tensor] = None,
    right_idx_offset: Optional[torch.Tensor] = None,
    rel_frac_offset: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Get the orientation of an object at timesteps s.

    Parameters
    ----------
    orientation : torch.Tensor
        The orientation of the object at discrete times specified in times.
        Shape ([... B], t, 4) where t is the number of discrete points.

    rotation_offset : torch.Tensor
        The offset rotation vector of the object at discrete times specified in offset_times.
        Shape ([... B], tc, 3) where tc is the number of discrete points.

    rotation_offset_weight : torch.Tensor
        A scalar weight for the offset rotation vector.

    times : torch.Tensor
        The discrete times at which the orientation is defined. Shape ([..., B], t).

    offset_times : torch.Tensor
        The discrete times at which the offset orientation is defined. Shape ([..., B], tc).

    steps : Optional[torch.Tensor], optional
        The timesteps ([...B,], S) to get the position for, by default None
    interpolation : str, optional
        The interpolation method to use, by default 'cubic'

    equidistant_times : bool, optional
        If the times are equidistant, by default False

    equidistant_offset_times : bool, optional
        If the offset times are equidistant, by default True

    right_idx : Optional[torch.Tensor], optional
        Right index for faster interpolation, by default None

    rel_frac : Optional[torch.Tensor], optional
        Relative stepsize for faster interpolation, by default None

    right_idx_offset : Optional[torch.Tensor], optional
        Right index for faster interpolation, by default None

    rel_frac_offset : Optional[torch.Tensor], optional
        Relative stepsize for faster interpolation, by default None

    Returns
    -------
    torch.Tensor
        The orientation of the object at timesteps s. Shape ([... B], S, 4).
    """
    base_orientation = get_orientation(
        orientation,
        times=times,
        steps=steps,
        interpolation=interpolation,
        equidistant_times=equidistant_times,
        right_idx=right_idx,
        rel_frac=rel_frac
    )
    # Decrease the impact of the offset rotation by the weight, as its a angular representation we must du it beforehand.
    quat = rotvec_to_unitquat(rotation_offset_weight * rotation_offset)
    offset_orientation = get_orientation(
        orientation=quat,
        times=offset_times,
        steps=steps,
        interpolation=interpolation,
        # Offset times are always equidistant see linspace in _default_offset_times
        equidistant_times=equidistant_offset_times,
        right_idx=right_idx_offset,
        rel_frac=rel_frac_offset
    )
    if offset_orientation.numel() == 0:
        return base_orientation
    return quat_composition(torch.stack([base_orientation, offset_orientation], dim=-2), normalize=True)


@torch.jit.script
def get_combined_translation_orientation(
        translation: torch.Tensor,
        translation_offset: torch.Tensor,
        orientation: torch.Tensor,
        rotation_offset: torch.Tensor,
        translation_offset_weight: torch.Tensor,
        rotation_offset_weight: torch.Tensor,
        times: torch.Tensor,
        offset_times: torch.Tensor,
        steps: Optional[torch.Tensor],
        interpolation: str,
        equidistant_times: bool = False,
        equidistant_offset_times: bool = True,
        right_idx: Optional[torch.Tensor] = None,
        rel_frac: Optional[torch.Tensor] = None,
        right_idx_offset: Optional[torch.Tensor] = None,
        rel_frac_offset: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    osteps: Optional[torch.Tensor] = None
    if steps is None:
        # Set steps if offset times and times are not equal - Fallback is allways the times
        if len(times.shape) > len(offset_times.shape):
            osteps = times
        elif (times.shape != offset_times.shape) or not (times == offset_times).all():
            osteps = times
    else:
        osteps = steps

    if osteps is not None:
        if right_idx is None and steps is not None:
            right_idx, rel_frac = _get_interpolate_index_and_distance(
                times, steps=osteps, equidistant=equidistant_times)

        if right_idx_offset is None:
            right_idx_offset, rel_frac_offset = _get_interpolate_index_and_distance(
                offset_times, steps=osteps, equidistant=equidistant_offset_times)  # Offset times are always equidistant see linspace in _default_offset_times

    translation = get_combined_translation(
        translation=translation,
        offset_translation=translation_offset,
        translation_offset_weight=translation_offset_weight,
        times=times,
        offset_times=offset_times,
        steps=osteps,
        interpolation=interpolation,
        equidistant_times=equidistant_times,
        equidistant_offset_times=equidistant_offset_times,
        right_idx=right_idx,
        rel_frac=rel_frac,
        right_idx_offset=right_idx_offset,
        rel_frac_offset=rel_frac_offset)

    orientation = get_combined_orientation(
        orientation=orientation,
        rotation_offset=rotation_offset,
        rotation_offset_weight=rotation_offset_weight,
        times=times,
        offset_times=offset_times,
        steps=osteps,
        interpolation=interpolation,
        equidistant_times=equidistant_times,
        equidistant_offset_times=equidistant_offset_times,
        right_idx=right_idx,
        rel_frac=rel_frac,
        right_idx_offset=right_idx_offset,
        rel_frac_offset=rel_frac_offset)
    return translation, orientation


def default_offset_times(
        num_control_points: int,
        dtype: torch.dtype) -> torch.Tensor:
    if num_control_points > 1:
        t_step = 1. / (num_control_points - 1)
        return torch.linspace(0 - t_step, 1 + t_step, num_control_points + 2, dtype=dtype)
    else:
        return torch.tensor([-1, 0, 1], dtype=dtype)


class LearnedOffsetSceneNode3D(TimedDiscreteSceneNode3D):
    """Pytorch Module class for a timed scene node with offset learned discrete positions."""

    _offset_translation: torch.nn.Parameter
    """The offset translations which are learned for each control point timestep (tc x 3)"""

    _offset_rotation_vector: torch.nn.Parameter
    """The offset rotations which are learned for each control point timestep (tc x 3). These are not in euler angles."""

    _offset_times: torch.Tensor
    """The time steps of the _offset_translations and _offset_rotations in range [0, 1]. Shape (tc,)"""

    _translation_offset_weight: torch.Tensor
    """The weight of the translation offset in the forward function"""

    _rotation_offset_weight: torch.Tensor
    """The weight of the rotation offset in the forward function"""

    _is_translation_learnable: bool
    """Whether the translation offset is learnable"""

    _is_rotation_learnable: bool
    """Whether the rotation offset is learnable"""

    def __init__(self,
                 num_control_points: int,
                 translation_offset_weight: float = 0.03,
                 rotation_offset_weight: float = 0.03,
                 learnable_translation: bool = True,
                 learnable_rotation: bool = True,
                 translation: Optional[VEC_TYPE] = None,
                 orientation: Optional[VEC_TYPE] = None,
                 position: Optional[VEC_TYPE] = None,
                 times: Optional[VEC_TYPE] = None,
                 name: Optional[str] = None,
                 children: Optional[Iterable['AbstractSceneNode']] = None,
                 decoding: bool = False,
                 dtype: torch.dtype = torch.float32,
                 _offset_times: Optional[torch.Tensor] = None,
                 _offset_translation: Optional[torch.Tensor] = None,
                 _offset_rotation_vector: Optional[torch.Tensor] = None,
                 _translation_offset_weight: Optional[torch.Tensor] = None,
                 _rotation_offset_weight: Optional[torch.Tensor] = None,
                 **kwargs
                 ):
        super().__init__(
            translation=translation,
            orientation=orientation,
            position=position,
            times=times,
            name=name,
            children=children,
            decoding=decoding,
            dtype=dtype,
            **kwargs)
        self._init_offsets(
            num_control_points=num_control_points,
            translation_offset_weight=translation_offset_weight,
            rotation_offset_weight=rotation_offset_weight,
            learnable_translation=learnable_translation,
            learnable_rotation=learnable_rotation,
            dtype=self.dtype,
            _offset_times=_offset_times,
            _offset_translation=_offset_translation,
            _offset_rotation_vector=_offset_rotation_vector,
            _translation_offset_weight=_translation_offset_weight,
            _rotation_offset_weight=_rotation_offset_weight
        )

    def _init_offsets(self,
                      num_control_points: int,
                      translation_offset_weight: float,
                      rotation_offset_weight: float,
                      learnable_translation: bool,
                      learnable_rotation: bool,
                      dtype: torch.dtype,
                      _offset_times: Optional[torch.Tensor],
                      _offset_translation: Optional[torch.Tensor],
                      _offset_rotation_vector: Optional[torch.Tensor],
                      _translation_offset_weight: Optional[torch.Tensor],
                      _rotation_offset_weight: Optional[torch.Tensor],
                      ):

        if _offset_rotation_vector is not None:
            self.register_buffer(
                "_offset_times", _offset_times, persistent=False)
            self.register_buffer("_offset_translation",
                                 _offset_translation, persistent=False)
            self.register_buffer("_offset_rotation_vector",
                                 _offset_rotation_vector, persistent=False)
            self.register_buffer("_translation_offset_weight",
                                 _translation_offset_weight, persistent=False)
            self.register_buffer("_rotation_offset_weight",
                                 _rotation_offset_weight, persistent=False)
        else:
            self._offset_rotation_vector = torch.nn.Parameter(
                data=self._default_offset_rotations_euler(num_control_points, dtype=dtype), requires_grad=learnable_rotation)
            self._offset_translation = torch.nn.Parameter(
                data=self._default_offset_translations(num_control_points, dtype=dtype), requires_grad=learnable_translation)
            self.register_buffer("_offset_times", default_offset_times(
                num_control_points, dtype=dtype))
            self.register_buffer("_translation_offset_weight", torch.tensor(
                translation_offset_weight, dtype=dtype))
            self.register_buffer("_rotation_offset_weight", torch.tensor(
                rotation_offset_weight, dtype=dtype))

    @property
    def is_translation_learnable(self) -> bool:
        return self._offset_translation.requires_grad

    @property
    def is_rotation_learnable(self) -> bool:
        return self._offset_rotation_vector.requires_grad

    @is_translation_learnable.setter
    def is_translation_learnable(self, value: bool) -> None:
        self._offset_translation.requires_grad = value

    @is_rotation_learnable.setter
    def is_rotation_learnable(self, value: bool) -> None:
        self._offset_rotation_vector.requires_grad = value

    def _default_offset_translations(self, num_control_points: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(num_control_points + 2, 3, dtype=dtype)

    def _default_offset_rotations_euler(self, num_control_points: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(num_control_points + 2, 3, dtype=dtype)

    def get_offset_translation(self,
                               t: Optional[torch.Tensor] = None,
                               right_idx: Optional[torch.Tensor] = None,
                               rel_frac: Optional[torch.Tensor] = None
                               ) -> torch.Tensor:
        return self._translation_offset_weight * get_translation(
            self._offset_translation,
            self._offset_times,
            t=t,
            interpolation=self._interpolation,
            # Offset times are always equidistant see linspace in _default_offset_times
            equidistant_times=True,
            right_idx=right_idx,
            rel_frac=rel_frac
        )

    def get_offset_orientation(self,
                               t: Optional[torch.Tensor] = None,
                               right_idx: Optional[torch.Tensor] = None,
                               rel_frac: Optional[torch.Tensor] = None
                               ) -> torch.Tensor:
        quat = rotvec_to_unitquat(
            self._rotation_offset_weight * self._offset_rotation_vector)
        return get_orientation(
            quat,
            self._offset_times,
            steps=t,
            interpolation=self._interpolation,
            # Offset times are always equidistant see linspace in _default_offset_times
            equidistant_times=True,
            right_idx=right_idx,
            rel_frac=rel_frac
        )

    def get_translation(self,
                        t: Optional[torch.Tensor] = None,
                        right_idx: Optional[torch.Tensor] = None,
                        rel_frac: Optional[torch.Tensor] = None,
                        right_idx_offset: Optional[torch.Tensor] = None,
                        rel_frac_offset: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:
        base_translation = super().get_translation(
            t=t, right_idx=right_idx, rel_frac=rel_frac)
        offset_translation = self.get_offset_translation(
            t=t, right_idx=right_idx_offset, rel_frac=rel_frac_offset)
        return base_translation + offset_translation

    def get_orientation(self,
                        t: Optional[torch.Tensor] = None,
                        right_idx: Optional[torch.Tensor] = None,
                        rel_frac: Optional[torch.Tensor] = None,
                        right_idx_offset: Optional[torch.Tensor] = None,
                        rel_frac_offset: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:
        base_orientation = super().get_orientation(
            t=t, right_idx=right_idx, rel_frac=rel_frac)
        offset_orientation = self.get_offset_orientation(
            t=t, right_idx=right_idx_offset, rel_frac=rel_frac_offset)
        return quat_composition(torch.stack([base_orientation, offset_orientation], dim=1), normalize=True)

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
        return get_combined_translation_orientation(
            translation=self._translation,
            translation_offset=self._offset_translation,
            orientation=self._orientation,
            rotation_offset=self._offset_rotation_vector,
            translation_offset_weight=self._translation_offset_weight,
            rotation_offset_weight=self._rotation_offset_weight,
            times=self._times,
            offset_times=self._offset_times,
            steps=t,
            interpolation=self._interpolation,
            equidistant_times=self._equidistant_times,
            equidistant_offset_times=True,
            right_idx=None,
            rel_frac=None,
            right_idx_offset=None,
            rel_frac_offset=None
        )


# region Plotting

@saveable()
def plot_position(
    position: torch.Tensor,
    times: torch.Tensor,
    offset_times: torch.Tensor,
    ax: Optional[Tuple[Axes, Axes]] = None,
    t: Optional[torch.Tensor] = None,
    title: Optional[str] = None,
    y_lim_translation: Optional[Tuple[float, float]] = None,
    y_lim_orientation: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Figure:
    fig = plot_position_base(position=position, times=times, ax=ax, t=t, title=title,
                             y_lim_translation=y_lim_translation, y_lim_orientation=y_lim_orientation,
                             **kwargs)
    # Add offset times
    plot_timed_vlines(offset_times, ax=fig.axes[0], t=t,
                      color="red", zorder=0,
                      alpha=1.,
                      linestyle="dotted",
                      label="Offset Times")
    plot_timed_vlines(offset_times, ax=fig.axes[1], t=t,
                      alpha=1.,
                      color="red", zorder=0,
                      linestyle="dotted", label="Offset Times")
    return fig
# endregion
