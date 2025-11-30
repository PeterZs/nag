from typing import Any


class ModelBuildException(ValueError):

    def __init__(self, message, data: Any = None):
        super().__init__(message)
        self.data = data
