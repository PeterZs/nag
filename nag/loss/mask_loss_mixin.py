import torch
from typing import Any, Tuple
from tools.util.numpy import numpyify


class MaskLossMixin():
    """Mixin class for Mask Loss."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
