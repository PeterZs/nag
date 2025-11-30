from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from nag.analytics.result_model_config import ResultModelConfig


def get_default_parse_result_patterns() -> Dict[str, Union[str, Dict[str, Any]]]:
    ret = dict()
    ret["final"] = dict(pattern=r"(?P<time>\d+)_t(_(?P<frame_index>\d+))?\.png",
                        parser=dict(time=int, frame_index=int))
    return ret


def get_default_parse_intermediate_result_patterns() -> Dict[str, Union[str, Dict[str, Any]]]:
    ret = dict()
    ret["in_training"] = dict(
        pattern=r"(?P<epoch>\d+)_epoch_(?P<time>\d+)_t(_(?P<kind>\w+))?.png", parser=dict(epoch=int, time=int, kind=str))
    return ret


def get_default_parse_checkpoint_patterns() -> Dict[str, Union[str, Dict[str, Any]]]:
    ret = dict()
    ret["checkpoint"] = dict(
        pattern=r"(final-)?(?P<epoch>[0-9]+)\.ckpt", parser=dict(epoch=int))
    return ret


@dataclass
class NAGResultModelConfig(ResultModelConfig):
    """Configuration for the NAGResultModel."""
    run_config_pattern: str = r"training_config(_)?(?P<config_name>[a-zA-Z0-9_\(\)\s]*)\.ya?ml"

    name: str = field(default=None)
    """Name of the model."""

    parsed_name: Optional[str] = field(default=None)

    number: Optional[int] = field(default=None)
    """A number which is assigned for identification."""

    parse_number_pattern: str = r"^(?P<number>[0-9]+).+$"
    """Pattern for parsing the number from the name string."""

    final_result_directories: List[Union[str, Dict[str, Any]]] = field(
        default_factory=lambda: [dict(pattern=r"final/((?P<object_composition_index>(\d+))(?P<name>[\w \*_\-\(\)\d]*))|complete|complete_no_view_dependency", parser=dict())])
    """Directories where the results are stored. Can be a string, which might be a regex pattern including path seperators, or a dictionary with a pattern (regex) and a parser to parse named groups."""

    intermediate_result_directories: List[Union[str, Dict[str, Any]]] = field(
        default_factory=lambda: [r"in_training"])

    epochs: List[int] = field(default_factory=list)
    """Epochs used within the model."""

    parse_result_patterns: Dict[str, str] = field(
        default_factory=get_default_parse_result_patterns)
    """Parse regex patterns for results which should be indexed in the result model."""

    parse_intermediate_result_patterns: Dict[str, str] = field(
        default_factory=get_default_parse_intermediate_result_patterns)

    ckeckpoint_directories: List[str] = field(
        default_factory=lambda: ["checkpoints"])
    """Directories where the checkpoints are stored. Can be a string, which might be a regex pattern including path seperators, or a dictionary with a pattern (regex) and a parser to parse named groups."""

    parse_checkpoint_patterns: Dict[str, Union[str, Dict[str, Any]]] = field(
        default_factory=get_default_parse_checkpoint_patterns)

    parse_name_pattern: str = r"^((?P<model_index>[0-9]+)_)?((?P<year>[0-9]{4})-(?P<month>[0-9]{1,2}))-(?P<day>[0-9]{1,2})_(?P<hour>[0-9]{1,2})-(?P<minute>[0-9]{1,2})-(?P<second>[0-9]{1,2})_(?P<model_name>[\w\s_\-]+)$"
