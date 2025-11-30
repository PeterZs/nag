from dataclasses import dataclass, field
from typing import Dict, List, Optional
from tools.serialization.json_convertible import JsonConvertible


def get_default_parse_patterns() -> Dict[str, str]:
    ret = dict()
    ret["alpha"] = r"alpha.png"
    ret["combined"] = r"combined.png"
    ret["obstruction"] = r"obstruction.png"
    ret["reference"] = r"reference.png"
    ret["transmission"] = r"transmission.png"
    ret["flow_obstruction"] = r"flow_obstruction.npy"
    ret["flow_transmission"] = r"flow_transmission.npy"
    return ret

def get_default_parse_checkpoint_patterns() -> Dict[str, str]:
    ret = dict()
    ret["checkpoint"] = r"(epoch_(?P<epoch>[0-9]+))|(last)\.ckpt"
    ret["bundle"] = r"bundle(_(?P<epoch>[0-9]+))?\.pkl"
    return ret

@dataclass
class ResultModelConfig(JsonConvertible):

    name: str = field(default=None)
    """Name of the model."""

    parsed_name: Optional[str] = field(default=None)

    number: Optional[int] = field(default=None)
    """A number which is assigned for identification."""

    parse_number_pattern: str = r"^(?P<number>[0-9]+).+$"
    """Pattern for parsing the number from the name string."""

    result_directories: List[str] = field(default_factory=lambda : [".", "final"])
    """Directories where the results are stored."""

    epochs: List[int] = field(default_factory=list)
    """Epochs used within the model."""

    parse_result_patterns: Dict[str, str] = field(default_factory=get_default_parse_patterns)
    """Parse regex patterns for results which should be indexed in the result model."""

    run_config_pattern: str = r"init_cfg_(?P<config_name>[a-zA-Z0-9_\(\)\s]+)\.ya?ml"

    checkpoint_pattern: str = r"(epoch_(?P<epoch>[0-9]+))|(last)\.ckpt"

    parse_checkpoint_patterns: Dict[str, str] = field(default_factory=get_default_parse_checkpoint_patterns)

    parse_name_pattern: str = r"^((?P<model_index>[0-9]+)_)?(?P<model_name>[\w\s_\-]+)(_(?P<year>[0-9]{4})_(?P<month>[0-9]{1,2}))_(?P<day>[0-9]{1,2})_(?P<hour>[0-9]{1,2})_(?P<minute>[0-9]{1,2})_(?P<second>[0-9]{1,2})$"