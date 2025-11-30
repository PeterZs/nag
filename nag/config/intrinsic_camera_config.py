from dataclasses import dataclass, field
from tools.serialization.json_convertible import JsonConvertible
from typing import Optional, Tuple
from nag.config.camera_config import CameraConfig
import torch
from tools.util.typing import VEC_TYPE
import numpy as np


@dataclass
class IntrinsicCameraConfig(CameraConfig):
    """Configuration for a camera. Parameters will form the intrinsic matrix K."""

    principal_point: Optional[Tuple[float, float]] = field(default=None)
    """Principal point in pixels (x, y)."""

    focal_length: float = field(default=1.)
    """Relative focal length. Should be independent from the resolution."""

    skew: float = 0.
    """Skew parameter."""

    lens_distortion: Optional[Tuple[float, float,
                                    float, float, float]] = field(default=None)
    """Lens distortion parameters (k1, k2, k3, t1, t2)."""

    def fields_to_native(self):
        """Convert fields to lists."""
        if self.principal_point is not None and isinstance(self.principal_point, (np.ndarray, torch.Tensor)):
            self.principal_point = self.principal_point.tolist()
        if self.lens_distortion is not None and isinstance(self.lens_distortion, (np.ndarray, torch.Tensor)):
            self.lens_distortion = self.lens_distortion.tolist()

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
        W, H = resolution
        intrinsics = torch.eye(3, dtype=dtype, device=device)
        # Set optical axis / principal point to the center of the image
        if self.principal_point is None:
            intrinsics[:2, 2] = (torch.tensor(
                [W, H], dtype=dtype, device=device) / 2)
        else:
            intrinsics[:2, 2] = torch.tensor(
                self.principal_point, dtype=dtype, device=device)

        intrinsics[0, 0] = self.focal_length * W
        intrinsics[1, 1] = self.focal_length * H
        intrinsics[0, 1] = self.skew
        return intrinsics

    def get_lens_distortion(self,
                            dtype: torch.dtype = torch.float32,
                            device: torch.device = None) -> torch.Tensor:
        """Get the lens distortion parameters for the camera."""
        lens_distortion = torch.zeros(5, dtype=dtype, device=device) if self.lens_distortion is None else torch.tensor(
            self.lens_distortion, dtype=dtype, device=device)
        return lens_distortion
