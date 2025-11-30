from dataclasses import dataclass, field
from typing import Literal, Type, Union, Any, Dict, Tuple
from tools.serialization.json_convertible import JsonConvertible
from nag.homography.match_finder import MatchFinderConfig, MatchFinder
from tools.util.torch import tensorify, flatten_batch_dims
from tools.torch.parse_device import parse_device
from tools.util.typing import VEC_TYPE
import torch
from tools.transforms.to_tensor_image import ToTensorImage
import kornia as K
import kornia.feature as KF
from tools.context.temporary_device import TemporaryDevice


@dataclass
class LoftrFinderConfig(MatchFinderConfig):

    pretrained_model: Literal["outdoor", "indoor"] = field(default="outdoor")

    device: Union[str, torch.device] = field(default=None)

    def __post_init__(self):
        self.device = parse_device(self.device)


class LoftrMatchFinder(MatchFinder):

    def __init__(self, config: LoftrFinderConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.tensorify = ToTensorImage(
            output_dtype=torch.float32, output_device=self.config.device)
        self.loftr = KF.LoFTR(pretrained=self.config.pretrained_model)
        self.loftr.to("cpu")
        self.loftr.eval()

    @classmethod
    def config_type(cls) -> Type[LoftrFinderConfig]:
        return LoftrFinderConfig

    def find_matches(self,
                     image1: VEC_TYPE,
                     image2: VEC_TYPE,
                     is_batch_mode: bool = False
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        image1, shp = flatten_batch_dims(self.tensorify(image1), -4)
        image2, _ = flatten_batch_dims(self.tensorify(image2), -4)
        if image1.shape != image2.shape:
            raise ValueError(
                f"Image shapes must match. Got {image1.shape} and {image2.shape}")
        if image1.shape[1] != 1:
            image1 = K.color.rgb_to_grayscale(image1)
            image2 = K.color.rgb_to_grayscale(image2)
        with TemporaryDevice(self.loftr, self.config.device, keep_device=is_batch_mode, output_device="cpu"):
            rdict = self.loftr({"image0": image1, "image1": image2})
        # , {"confidence": rdict["confidence"]}
        return rdict["keypoints0"], rdict["keypoints1"]
