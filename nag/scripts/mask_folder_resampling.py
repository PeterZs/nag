import argparse
import os
from typing import Optional

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
from tools.util.format import parse_format_string
from tools.util.path_tools import read_directory
from nag.scripts.image_folder_resampling import ImageResamplingConfig, main as image_resampling_main


def supported_extensions():
    return ["png"]


def default_mask_file_pattern():
    return r"^(?P<filename>[\w,]+)\.(?P<extension>(" + "|".join(supported_extensions()) + r"))$"


@dataclass
class MaskResamplingConfig(ArgparserMixin):

    masks_input_path: str
    """Path to the mask folder to resample."""

    masks_output_path: str = field()
    """Output path for the masks. Need to be specified."""

    mask_object_folder_pattern: str = field(
        default=r"^(?P<object_folder>[\w,]+)$")
    """Regex pattern to match the object folders"""

    mask_file_pattern: str = field(
        default_factory=lambda: default_mask_file_pattern())
    """Regex pattern to match the mask files. Default is the default_mask_file_pattern."""

    resampling_ratio: float = field(default=0.5)
    """Resampling ratio for the masks. Default is 0.5."""

    override: bool = field(default=False)
    """If True, override the existing files. Default is False."""

    def prepare(self):
        self.masks_input_path = os.path.normpath(self.masks_input_path)
        self.masks_input_path = format_os_independent(self.masks_input_path)
        self.masks_output_path = os.path.normpath(self.masks_output_path)
        self.masks_output_path = format_os_independent(self.masks_output_path)
        if self.mask_file_pattern is None:
            self.mask_file_pattern = default_mask_file_pattern()


def current_filename() -> str:
    return os.path.basename(__file__).split('.')[0]


def config():
    import nsf
    basic_config()
    set_module_path(os.path.dirname(nag.__file__))


def get_config() -> MaskResamplingConfig:
    parser = argparse.ArgumentParser(
        description='Resample masks in a folder. Mask folder is expected to have a structure of object folders containing mask files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config: MaskResamplingConfig = MaskResamplingConfig.parse_args(
        parser, add_config_path=False, sep="-")
    config.prepare()
    return config


def main(cfg: MaskResamplingConfig):
    masks_object_paths = [x["path"] for x in read_directory(
        cfg.masks_input_path, cfg.mask_object_folder_pattern)]
    # Create output folder
    for mask_object_path in tqdm(masks_object_paths, desc="Resampling mask objects"):
        base_folder = format_os_independent(mask_object_path).split("/")[-1]
        output_folder = os.path.join(cfg.masks_output_path, base_folder)
        ifg = ImageResamplingConfig(
            images_input_path=mask_object_path,
            images_output_path=output_folder,
            resampling_ratio=cfg.resampling_ratio,
            image_file_pattern=cfg.mask_file_pattern,
            interpolation_method="subsample", override=cfg.override)
        ifg.prepare()
        image_resampling_main(ifg)


if __name__ == '__main__':
    config()
    cfg = get_config()
    main(cfg)
