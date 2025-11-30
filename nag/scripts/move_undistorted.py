import argparse
import os
from typing import Optional

from tools.logger.logging import basic_config
from tools.mixin.argparser_mixin import ArgparserMixin
from dataclasses import dataclass, field
import os
from tools.logger.logging import basic_config
from tools.util.package_tools import set_exec_module, get_executed_module_root_path
import numpy as np
from dotenv import load_dotenv

import torch
from tools.util.path_tools import process_path
import pandas as pd
import pandas as pd
from tools.util.path_tools import read_directory, read_directory_recursive, format_os_independent
from tools.util.format import get_leading_zeros_format_string, consecutive_indices_string
from tools.util.progress_factory import ProgressFactory
from tools.logger.logging import logger
from shutil import copy2
from tools.io.image import load_image, save_image
from tools.metric.torch.psnr import PSNR
import filecmp


@dataclass
class MoveUndistortedConfig(ArgparserMixin):

    config_path: str
    """Path to the config file."""

    shallow_identical_check: bool = field(default=True)
    """Check if the files are identical using shallow check."""

    source_dir: str = field(default=None)

    target_dir: str = field(default=None)

    source_subdir_pattern: Optional[str] = field(
        default=r"(?P<start_index>\d+)_(?P<end_index>\d+)/^UndistortedGT$")

    filename_pattern: Optional[str] = field(default=r"(?P<index>\d+)_\d+\.png")

    ground_truth_dir: Optional[str] = field(default=None)

    ground_truth_filename_pattern: Optional[str] = field(
        default=r"(?P<index>\d+)\.png")

    def prepare(self):
        self.source_dir = process_path(
            self.source_dir, need_exist=True, interpolate=True, interpolate_object=self, variable_name='source_dir')
        self.target_dir = process_path(self.target_dir, need_exist=False, make_exist=True,
                                       interpolate=True, interpolate_object=self, variable_name='target_dir')
        self.ground_truth_dir = process_path(self.ground_truth_dir, need_exist=False,
                                             interpolate=True, interpolate_object=self, variable_name='ground_truth_dir')


def current_filename() -> str:
    return os.path.basename(__file__).split('.')[0]


def config():
    import nsf
    load_dotenv()
    basic_config()
    set_exec_module(nsf)
    os.chdir(get_executed_module_root_path())


def get_config() -> MoveUndistortedConfig:
    parser = argparse.ArgumentParser(
        description='Move undistorted images and renames them.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config: MoveUndistortedConfig = MoveUndistortedConfig.parse_args(
        parser, add_config_path=False, sep="-")
    config.prepare()
    return config


def main(eval_cfg: MoveUndistortedConfig):
    # Creating index
    logger.info(f"Creating index for {eval_cfg.source_dir}")

    subdirs = read_directory_recursive(format_os_independent(str(
        eval_cfg.source_dir)) + "/" + eval_cfg.source_subdir_pattern, parser=dict(start_index=int, end_index=int))
    subdirs = sorted(subdirs, key=lambda x: x['start_index'])
    pf = ProgressFactory()

    # Load GT data
    df = read_directory(eval_cfg.ground_truth_dir,
                        pattern=eval_cfg.ground_truth_filename_pattern, parser=dict(index=int))
    index = pd.DataFrame(df).set_index('index', drop=True)
    index = index.rename(columns={'path': 'gt_path'})

    sub_dir_bar = pf.bar(total=len(subdirs), desc='Subdir')
    for sub_dir in subdirs:
        subdir_path = sub_dir["path"]
        sub_dir_bar.set_description(
            f"Subdir {sub_dir['start_index']} - {sub_dir['end_index']}")
        paths = read_directory(
            subdir_path, pattern=eval_cfg.filename_pattern, parser=dict(index=int))
        paths_df = pd.DataFrame(paths)

        # Add start_index to index
        paths_df["index"] = paths_df["index"].astype(
            int) + sub_dir["start_index"]
        paths_df = paths_df.set_index('index', drop=True)
        paths_df = paths_df.rename(columns={'path': 'source_path'})

        # Join with GT data
        if "source_path" not in index.columns:
            index = index.join(paths_df, how='left')
        else:
            # Set values
            index.loc[paths_df.index, "source_path"] = paths_df["source_path"]
        sub_dir_bar.update(1)

    # Check if any of source path is NaN
    if index["source_path"].isna().any():
        # Filter values and log
        missing_paths = index[index["source_path"].isna()]
        missing_index = missing_paths.index.tolist()
        logger.warning(f"Missing source paths for indices: {missing_index}")

        # Remove missing paths from index
        index = index[~index["source_path"].isna()]

    get_leading_zeros = get_leading_zeros_format_string(max(index.index))

    def compose_target(x):
        return os.path.join(eval_cfg.target_dir, get_leading_zeros.format(x.name) + ".png")

    index["target_path"] = index.apply(compose_target, axis=1)

    # Copy files
    logger.info(f"Copying files to {eval_cfg.target_dir}")

    copy_bar = pf.bar(total=len(index), desc='Copying files')
    skipped_identical_files = []
    for i, row in index.iterrows():
        source_path = row["source_path"]
        target_path = row["target_path"]
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if os.path.exists(target_path):
            if filecmp.cmp(source_path, target_path, shallow=eval_cfg.shallow_identical_check):
                copy_bar.update(1)
                skipped_identical_files.append(row.name)
                continue
            else:
                logger.warning(
                    f"File {target_path} already exists and is different. Overwriting.")
        copy2(source_path, target_path)
        copy_bar.update(1)

    if len(skipped_identical_files) > 0:
        numbers = consecutive_indices_string(skipped_identical_files)
        logger.info(
            f"Skipped {len(skipped_identical_files)} identical files: {numbers}")

    # Compute diff and PSNR
    from tools.transforms.to_tensor_image import ToTensorImage
    from tools.transforms.to_numpy_image import ToNumpyImage
    tensorify_image = ToTensorImage(output_dtype=torch.float32)
    numpyify_image = ToNumpyImage(output_dtype=np.uint8)
    logger.info(f"Computing diff and PSNR")

    psnr = PSNR(max_value=1.)

    def l1_loss(source, target):
        return torch.abs(source - target).mean()

    def l1_image(source, target):
        return torch.abs(source - target).mean(dim=(0, ))

    index["l1"] = np.nan
    index["psnr"] = np.nan

    bar = pf.bar(total=len(index), desc='Computing diff and PSNR')
    for i, row in index.iterrows():
        source_path = row["source_path"]
        gt_path = row["gt_path"]

        source_img = tensorify_image(load_image(source_path))
        gt_img = tensorify_image(load_image(gt_path))

        psnr_value = psnr(source_img, gt_img).item()
        l1_value = l1_loss(source_img, gt_img).item()

        index.at[i, "psnr"] = psnr_value
        index.at[i, "l1"] = l1_value

        diff_path = os.path.join(
            eval_cfg.target_dir, "diff", get_leading_zeros.format(i) + ".tiff")
        l1_img = numpyify_image(l1_image(source_img, gt_img))
        save_image(l1_img, diff_path)
        bar.update(1)

    # Save index to csv
    index.to_csv(os.path.join(eval_cfg.target_dir, "index.csv"), index=True)


if __name__ == '__main__':
    config()
    cfg = get_config()
    main(cfg)
