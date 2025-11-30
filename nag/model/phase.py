from typing import Any, Dict, List, Optional, Tuple, Type, Union
from tools.serialization.json_convertible import JsonConvertible
from tools.logger.logging import logger
from tools.mixin.fast_repr_mixin import FastReprMixin
from tools.util.typing import _DEFAULT, DEFAULT
from tools.util.reflection import set_nested_value, get_nested_value


class Phase(JsonConvertible, FastReprMixin):

    name: str
    """Name of the phase for display purposes."""

    start_at: Optional[int]
    """The time where this phase should start. -1 is immediate / always. Will be set automatically."""

    length: int
    """Length of the phase in epochs. -1 Is infinite length."""

    is_color_alpha_learnable: bool
    """Whether color and alpha of the planes can be learned within this phase."""

    is_position_learnable: bool
    """Whether the position of the planes can be learned within this phase."""

    is_flow_learnable: bool
    """Whether the flow of the planes can be learned within this phase."""

    is_view_dependence_learnable: bool
    """Whether the view dependence of the planes can be learned within this phase. (View Dependent Plane must be used.)"""

    is_time_dependence_learnable: bool
    """Whether the time dependence of the planes can be learned within this phase. (Time Dependent Plane must be used.)"""

    is_position_gradient_rescaling_enabled: Union[_DEFAULT, bool]
    """Whether the position gradient rescaling is enabled.
    Can be used to toggle the rescaling of the position gradients.
    Hooks are set in the normal config.
    If not set, the value of the config will be used."""

    plane_position_rescaling_hook: Union[str, Type, callable, _DEFAULT]

    plane_position_rescaling_hook_kwargs: Union[_DEFAULT, Dict[str, Any]]

    plane_normal_rescaling_hook: Union[str, Type, callable, _DEFAULT]

    plane_normal_rescaling_hook_kwargs: Union[_DEFAULT, Dict[str, Any]]

    is_grad_scaler_enabled: Union[_DEFAULT, bool]
    """Whether the gradient scaler is enabled."""

    background_color_fadeout: Union[_DEFAULT, bool]
    """Whether the background color should fade out."""

    is_lr_scheduler_active: bool
    """Whether the learning rate scheduler is active."""

    dataset_kwargs: Optional[Dict[str, Any]] = None
    """Update the dataset attributes of the system with the given values.
    Keys can be property-paths to eventually nested properties.
    E.g. "my_property.sub_property".
    Values can be a value of any type. NOTSET will remove the property from the parent object.
    """

    loss_kwargs: Optional[Dict[str, Any]] = None
    """Update the loss attributes of the system with the given values.
    Keys can be property-paths to eventually nested properties."""

    def __init__(self,
                 name: str,
                 length: int,
                 is_color_alpha_learnable: bool = True,
                 is_flow_learnable: bool = True,
                 is_position_learnable: bool = True,
                 is_view_dependence_learnable: bool = False,
                 is_time_dependence_learnable: bool = False,
                 is_lr_scheduler_active: bool = True,
                 is_position_gradient_rescaling_enabled: Union[_DEFAULT,
                                                               bool] = DEFAULT,
                 plane_position_rescaling_hook: Union[str,
                                                      Type, callable, _DEFAULT] = DEFAULT,
                 plane_position_rescaling_hook_kwargs: Union[_DEFAULT,
                                                             Dict[str, Any]] = DEFAULT,
                 plane_normal_rescaling_hook: Union[str,
                                                    Type, callable, _DEFAULT] = DEFAULT,
                 plane_normal_rescaling_hook_kwargs: Union[_DEFAULT,
                                                           Dict[str, Any]] = DEFAULT,
                 start_at: Optional[int] = None,
                 is_grad_scaler_enabled: Union[_DEFAULT, bool] = DEFAULT,
                 background_color_fadeout: Union[_DEFAULT, bool] = DEFAULT,
                 dataset_kwargs: Optional[Dict[str, Any]] = None,
                 loss_kwargs: Optional[Dict[str, Any]] = None,
                 decoding: bool = False, **kwargs):
        super().__init__(decoding, **kwargs)
        self.name = name
        self.start_at = start_at
        self.length = length
        self.is_color_alpha_learnable = is_color_alpha_learnable
        self.is_position_learnable = is_position_learnable
        self.is_flow_learnable = is_flow_learnable
        self.is_view_dependence_learnable = is_view_dependence_learnable
        self.is_time_dependence_learnable = is_time_dependence_learnable
        self.is_position_gradient_rescaling_enabled = is_position_gradient_rescaling_enabled
        self.plane_position_rescaling_hook = plane_position_rescaling_hook
        self.plane_position_rescaling_hook_kwargs = plane_position_rescaling_hook_kwargs
        self.plane_normal_rescaling_hook = plane_normal_rescaling_hook
        self.plane_normal_rescaling_hook_kwargs = plane_normal_rescaling_hook_kwargs
        self.is_grad_scaler_enabled = is_grad_scaler_enabled
        self.background_color_fadeout = background_color_fadeout
        self.is_lr_scheduler_active = is_lr_scheduler_active
        self.dataset_kwargs = dataset_kwargs
        self.loss_kwargs = loss_kwargs

    def is_active(self, time: int) -> bool:
        if self.start_at == -1:
            return True
        return (time >= self.start_at) & ((time < (self.start_at + self.length)) | (self.length == -1))

    def is_past(self, time: int) -> bool:
        if self.start_at == -1:
            return True
        return time > (self.start_at + self.length)

    @classmethod
    def parse(cls, phases: Optional[List["Phase"]], max_time: int) -> List["Phase"]:
        if phases is None or len(phases) == 0:
            logger.warning("No phases defined. Using default phase.")
            return [Phase(name="default", start_at=-1, length=-1)]
        ret = []
        current_time = 0

        def set_time(p: Phase):
            nonlocal current_time
            if current_time != -1:
                p.start_at = current_time
                if p.length == -1:
                    current_time = -1
                else:
                    current_time += p.length

        for phase in phases:
            if isinstance(phase, Phase):
                set_time(phase)
                ret.append(phase)
            elif isinstance(phase, dict):
                p = Phase(**phase)
                set_time(p)
                ret.append(p)
            else:
                raise ValueError(f"Cannot parse phase {phase}")
        if current_time != -1 and current_time < (max_time - 1):
            # Log Warning
            logger.warning(
                f"Phases do not cover the full time range. Missing: {max_time - current_time}")
            ret.append(Phase(name="default", start_at=current_time, length=-1))
        return ret

    def change_phase(self,
                     old_phase: Optional["Phase"],
                     model: Any,
                     log: bool = True
                     ):
        from nag.model.nag_model import NAGModel
        from nag.model.nag_functional_model import NAGFunctionalModel
        if not isinstance(model, NAGModel):
            raise ValueError(f"Model {model} is not a NAGModel")

        if log:
            logger.info(
                f"Changing phase to: \n{self.to_yaml(toplevel_wrapping=False, no_uuid=True, no_large_data=True)}")

        if old_phase is None:
            # Set everything
            model.enable_color_alpha(self.is_color_alpha_learnable)
            model.enable_position_learning(self.is_position_learnable)
            model.enable_flow(self.is_flow_learnable)
            model.enable_view_dependence(self.is_view_dependence_learnable)
            model.enable_time_dependence(self.is_time_dependence_learnable)
            if isinstance(model, NAGFunctionalModel):
                if self.is_position_gradient_rescaling_enabled != DEFAULT:
                    model.set_position_gradient_rescaling(
                        self.is_position_gradient_rescaling_enabled, phase=self)
                if (self.is_grad_scaler_enabled != DEFAULT):
                    model.set_grad_scaler_enabled(
                        self.is_grad_scaler_enabled)
            if self.background_color_fadeout != DEFAULT:
                obj = model.get_background_plane()
                if obj is not None:
                    obj._background_color_fadeout = self.background_color_fadeout
            model.lr_scheduler_enabled = self.is_lr_scheduler_active
        else:
            if old_phase.is_color_alpha_learnable != self.is_color_alpha_learnable:
                model.enable_color_alpha(self.is_color_alpha_learnable)
            if old_phase.is_position_learnable != self.is_position_learnable:
                model.enable_position_learning(self.is_position_learnable)
            if old_phase.is_flow_learnable != self.is_flow_learnable:
                model.enable_flow(self.is_flow_learnable)
            if old_phase.is_view_dependence_learnable != self.is_view_dependence_learnable:
                model.enable_view_dependence(self.is_view_dependence_learnable)
            if old_phase.is_time_dependence_learnable != self.is_time_dependence_learnable:
                model.enable_time_dependence(self.is_time_dependence_learnable)
            if isinstance(model, NAGFunctionalModel):
                if (self.is_position_gradient_rescaling_enabled != DEFAULT):
                    model.set_position_gradient_rescaling(
                        self.is_position_gradient_rescaling_enabled, phase=self)
                if (self.is_grad_scaler_enabled != DEFAULT):
                    model.set_grad_scaler_enabled(
                        self.is_grad_scaler_enabled)
            if self.background_color_fadeout != DEFAULT:
                if old_phase.background_color_fadeout != self.background_color_fadeout:
                    obj = model.get_background_plane()
                    if obj is not None:
                        obj._background_color_fadeout = self.background_color_fadeout
            model.lr_scheduler_enabled = self.is_lr_scheduler_active
        self.update_arbitrary(model._trainer.train_dataloader.dataset,
                              self.dataset_kwargs, log=True, entry_object_name="dataset")
        self.update_arbitrary(model.loss, self.loss_kwargs,
                              log=True, entry_object_name="loss")

    def update_arbitrary(self,
                         entry_object: Any,
                         update_dict: Optional[Dict[str, Any]],
                         log: bool = True,
                         entry_object_name: Optional[str] = None
                         ) -> None:
        if update_dict is None or entry_object is None:
            return
        updates = dict()
        for key, value in update_dict.items():
            # Update the entry object
            o_val = set_nested_value(entry_object, key, value)
            if log:
                updates[key] = f"{str(o_val)} -> {str(value)}"
        if log:
            keys_text = '\n\t'.join([f'{k}: {v}' for k, v in updates.items()])
            logger.info(
                f"Updated {entry_object_name + ' of type ' if (entry_object_name is not None and len(entry_object_name) > 0) else ''}{type(entry_object).__name__} on keys: \n\t{keys_text}.")

    def is_phase_change(cls,
                        training_phases: List["Phase"],
                        active_index: int,
                        time: int) -> Tuple[bool, int]:
        # Check if the active phase is still active
        if training_phases[active_index].is_active(time):
            return False, active_index
        next_index = active_index + 1
        # Check if the next phase is active
        if not training_phases[next_index].is_active(time):
            # Misconfiguration
            raise ValueError(
                f"Phase {next_index} is not active at time {time}")
        return True, next_index
