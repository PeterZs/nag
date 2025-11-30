from abc import ABC, abstractmethod
from typing import Any
from tools.serialization.json_convertible import JsonConvertible

class Strategy(JsonConvertible):
    """Strategy class. Simple wrapper for functions having behavior."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the strategy.

        Parameters
        ----------
        *args : Any
            Any arguments.

        **kwargs : Any
            Any keyword arguments.

        Returns
        -------
        Any
            The result of the strategy.
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """Call the execute method of the strategy.

        Returns
        -------
        Any
            The result of the execute method.
        """
        return self.execute(*args, **kwargs)