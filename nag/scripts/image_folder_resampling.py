import argparse
import os
from typing import Literal, Optional

import numpy as np
from tools.util.path_tools import format_os_independent
from tools.logger.logging import basic_config
from tools.util.package_tools import set_module_path
from tools.mixin.argparser_mixin import ArgparserMixin
import torch
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from nag.utils import utils
from tools.util.numpy import numpyify_image
from tools.viz.matplotlib import save_as_image, save_as_image_stack
from tools.util.format import parse_format_string
from tools.util.path_tools import read_directory
from tools.io.image import load_image, save_image
from torch.nn.functional import grid_sample
from tools.util.torch import tensorify_image
from tools.transforms.min_max import MinMax


def supported_extensions():
    return ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]


def default_image_file_pattern():
    return r"^(?P<filename>[\w,]+)\.(?P<extension>(" + "|".join(supported_extensions()) + r"))$"


@dataclass
class ImageResamplingConfig(ArgparserMixin):

    images_input_path: str
    """Path to the image folder to resample."""

    images_output_path: str = field()
    """Output path for the images. Need to be specified."""

    image_file_pattern: str = field(
        default_factory=lambda: default_image_file_pattern())
    """Regex pattern to match the image files. Default is the default_image_file_pattern."""

    interpolation_method: Literal["bilinear", "nearest", "subsample"] = field(
        default="bilinear")
    """Interpolation method for the resampling. Default is bilinear."""

    resampling_ratio: float = field(default=0.5)
    """Resampling ratio for the images. Default is 0.5."""

    override: bool = field(default=False)
    """If True, override the existing files. Default is False."""

    def prepare(self):
        self.images_input_path = os.path.normpath(self.images_input_path)
        self.images_input_path = format_os_independent(self.images_input_path)
        self.images_output_path = os.path.normpath(self.images_output_path)
        self.images_output_path = format_os_independent(
            self.images_output_path)
        if self.image_file_pattern is None:
            self.image_file_pattern = default_image_file_pattern()


def current_filename() -> str:
    return os.path.basename(__file__).split('.')[0]


def config():
    import nsf
    basic_config()
    set_module_path(os.path.dirname(nag.__file__))


def get_config() -> ImageResamplingConfig:
    parser = argparse.ArgumentParser(
        description='Resample images in a folder.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config: ImageResamplingConfig = ImageResamplingConfig.parse_args(
        parser, add_config_path=False, sep="-")
    config.prepare()
    return config


def resample(image: np.ndarray, resampling_ratio: float, mode: str) -> np.ndarray:
    if resampling_ratio <= 0:
        raise ValueError("Resampling ratio should be greater than 0.")
    if mode != "subsample":
        H, W, C = image.shape
        H_s, W_s = int(H * resampling_ratio), int(W * resampling_ratio)
        image_tensor = tensorify_image(image, dtype=torch.float32)
        grid = torch.nn.functional.affine_grid(torch.tensor(
            [[[1., 0, 0], [0, 1., 0]]]), (1, 3, H_s, W_s), align_corners=True)
        # Move to GPU if available
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            grid = grid.cuda()

        # Convert to 0-1 range
        image_tensor = image_tensor / 255.0
        resampled_image = grid_sample(image_tensor.unsqueeze(
            0), grid, mode=mode, align_corners=True)[0]
        # Convert back to 0-255 range
        resampled_image = (resampled_image * 255.0).to(torch.uint8)
        return numpyify_image(resampled_image)

    # Subsampling
    skips = (1 / resampling_ratio)
    # Check if close to integer
    if not np.isclose(skips, int(skips), rtol=0.05):
        raise ValueError(
            "Resampling ratio should be close to 1/n, where n > 0 is an integer.")
    skips = int(skips)
    return image[::skips, ::skips]


def main(cfg: ImageResamplingConfig):
    image_files = {x["path"] for x in read_directory(
        cfg.images_input_path, cfg.image_file_pattern)}
    # Create output directory
    if not os.path.exists(cfg.images_output_path):
        os.makedirs(cfg.images_output_path, exist_ok=True)

    for image_file in tqdm(image_files, desc="Resampling images"):
        output_path = os.path.join(
            cfg.images_output_path, os.path.basename(image_file))

        image, metadata = load_image(image_file, load_metadata=True)
        resampled_image = resample(
            image, cfg.resampling_ratio, mode=cfg.interpolation_method)
        save_image(resampled_image, output_path, mkdirs=False,
                   override=cfg.override, metadata=metadata)


if __name__ == '__main__':
    config()
    cfg = get_config()
    main(cfg)
