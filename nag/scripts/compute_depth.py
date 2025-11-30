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
from tools.viz.matplotlib import save_as_image, save_as_image_stack
from tools.util.format import parse_format_string
from tools.util.format import parse_format_string
from tools.dataset.image_path_dataset import ImagePathDataset
from tools.run.dataset_processor import DatasetProcessor
from tools.transforms.to_numpy_image import numpyify_image
from tools.io.image import save_image
from PIL import Image
from tools.util.progress_factory import ProgressFactory
from tools.transforms.numpy.min_max import MinMax


@dataclass
class ComputeDepthConfig(ArgparserMixin):

    images_folder: str
    """Path to the folder containing the images."""

    image_filename_format: str = field(
        default=r"(?P<index>\d+)\.((png)|(jpg)|(jpeg))")
    """Filename format for the images. Default is {index}.png."""

    depth_output_folder: str = field(default="{images_folder}/../depth")
    """Output folder for the depth images. Default is {images_folder}/../depth."""

    depth_filename_format: str = field(default="{index}.tiff")
    """Filename format for the depth images. Default is {index}.tiff."""

    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")
    """Device to use for the depth estimation. Default is cuda."""

    progress_bar: bool = field(default=True)
    """Whether to show a progress bar. Default is True."""

    progress_factory: Optional[str] = field(default=None)
    """Progress factory for the dataset processor."""

    def prepare(self):
        self.images_folder = format_os_independent(
            os.path.normpath(self.images_folder))
        dof = parse_format_string(self.depth_output_folder, [self])[0]
        self.depth_output_folder = format_os_independent(os.path.normpath(dof))
        self.device = torch.device(self.device)
        if self.progress_factory is not None:
            self.progress_factory = ProgressFactory()


def current_filename() -> str:
    return os.path.basename(__file__).split('.')[0]


def config():
    import nsf
    basic_config()
    set_module_path(os.path.dirname(nag.__file__))


def get_pipeline(cfg) -> callable:
    from transformers import pipeline

    checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
    pipe = pipeline("depth-estimation", model=checkpoint, device=cfg.device)
    return pipe


def get_config() -> ComputeDepthConfig:
    parser = argparse.ArgumentParser(
        description='Computes depth images from a folder of images via Monocular Depth Estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config: ComputeDepthConfig = ComputeDepthConfig.parse_args(
        parser, add_config_path=False, sep="-")
    config.prepare()
    return config


def main(cfg: ComputeDepthConfig):
    os.makedirs(cfg.depth_output_folder, exist_ok=True)
    pipe = get_pipeline(cfg)
    im_max = 0.
    im_min = 1e12

    def get_depth_image(image: np.ndarray, index: int, depth_image_filename: str):
        nonlocal im_max
        nonlocal im_min
        depth = pipe(Image.fromarray(image))
        filename = parse_format_string(
            depth_image_filename, [dict()], index_offset=index)[0]
        dimg = depth["predicted_depth"]
        np_depth = numpyify_image(dimg)
        im_max = max(im_max, np.max(np_depth))
        im_min = min(im_min, np.min(np_depth))
        save_image(np_depth, os.path.join(
            cfg.depth_output_folder, filename), override=True)

    dataset = ImagePathDataset.from_folder(
        cfg.images_folder, filename_format=cfg.image_filename_format)
    proc = DatasetProcessor(dataset, get_depth_image,
                            progress_bar=cfg.progress_bar,
                            progress_factory=cfg.progress_factory,
                            depth_image_filename=cfg.depth_filename_format, process_text="Computing depth images")
    proc()

    depth_paths = ImagePathDataset.from_folder(
        cfg.depth_output_folder, filename_format=r"(?P<index>\d+).tiff")
    min_max = MinMax(new_min=0., new_max=1.)
    min_max.min = np.array(im_min)
    min_max.max = np.array(im_max)
    min_max.fitted = True

    def reset_values_to_dist_from_cam(image: np.ndarray, index: int, min_max: MinMax):
        path = depth_paths.paths[index]
        n_img = min_max(image)
        image = 1. - n_img
        save_image(image, path, override=True)

    proc_res = DatasetProcessor(depth_paths,
                                reset_values_to_dist_from_cam,
                                progress_bar=cfg.progress_bar,
                                progress_factory=cfg.progress_factory,
                                min_max=min_max,
                                process_text="Resetting values to distance from camera"
                                )
    proc_res()


if __name__ == '__main__':
    config()
    cfg = get_config()
    main(cfg)
