from dataclasses import dataclass, field
from tools.serialization.json_convertible import JsonConvertible
from typing import Optional, Tuple
from abc import ABC, abstractmethod
import torch


@dataclass
class CameraConfig(JsonConvertible):
    """Abstract class to discribe a camera within a scene."""

    @abstractmethod
    def get_intrinsics(self,
                       resolution: Tuple[int, int],
                       dtype: torch.dtype = torch.float32,
                       device: torch.device = None) -> torch.Tensor:
        """Get the intrinsic matrix K for the camera.

        Parameters
        ----------

        resolution : Tuple[int, int]
            Resolution of the image (width, height).

        dtype : torch.dtype, optional
            Data type of the returned tensor.

        device : torch.device, optional
            Device of the returned tensor.

        Returns
        -------
        torch.Tensor
            3x3 intrinsic matrix K.
        """
        pass

    @abstractmethod
    def get_lens_distortion(self,
                            dtype: torch.dtype = torch.float32,
                            device: torch.device = None) -> torch.Tensor:
        """Get the lens distortion parameters for the camera.

        Parameters
        ----------
        dtype : torch.dtype, optional
            Data type of the returned tensor.

        device : torch.device, optional
            Device of the returned tensor.

        Returns
        -------
        torch.Tensor
            Lens distortion parameters. Shape: (5,)
            (r1, r2, r3, t1, r2)
        """
        pass
