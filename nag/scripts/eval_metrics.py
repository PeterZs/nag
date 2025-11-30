#!/usr/bin/env python3
# Sample job file for an omni job. Can be used as a standalone file or within a job list.
from tools.agent.util.tracker import Tracker
from torch.utils.data import DataLoader, Dataset
from tools.transforms.to_tensor_image import ToTensorImage
import pandas as pd
from tools.io.image import load_image
from tools.torch.parse_device import parse_device
from tools.util.typing import DEFAULT, _DEFAULT
from tools.util.path_tools import process_path, read_directory
from tools.util.format import parse_type, format_dataframe_string, parse_enum
from tools.metric.metric import Metric
from dataclasses import dataclass, field
import argparse
import logging  # noqa
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib
import matplotlib.pyplot as plt
from tools.logger.logging import basic_config, logger, log_dataframe
import torch

from nag.config.nag_config import NAGConfig
from tools.config.experiment_output_config import ExperimentOutputConfig
from tools.util.package_tools import set_module_path
from tools.context.script_execution import ScriptExecution
from enum import Enum
from tools.transforms.to_numpy_image import ToNumpyImage
from tools.util.torch import torch_to_numpy_dtype, parse_dtype
from tools.util.format import get_leading_zeros_format_string

plt.ioff()
matplotlib.use('agg')


class ResizeLib(Enum):
    PIL = "pil"
    OPENCV = "opencv"


class ResizeMode(Enum):
    BILINEAR = "bilinear"


def resize_factory(
    lib: ResizeLib,
    mode: ResizeMode,
    height: int,
    width: int,
    in_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.float32,
    **kwargs
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Factory function to create a resize function based on the specified mode."""
    if lib.value == ResizeLib.PIL.value:
        from PIL import Image
        modes = {
            ResizeMode.BILINEAR.value: Image.BILINEAR
        }

        def resize(image: torch.Tensor) -> torch.Tensor:
            nonlocal height, width, in_dtype, out_dtype, kwargs, Image, mode, modes
            in_trans = ToNumpyImage(
                output_dtype=torch_to_numpy_dtype(in_dtype))
            image = in_trans(image)
            image = Image.fromarray(image)

            m = modes.get(mode.value, None)
            if m is None:
                raise ValueError(
                    f"Unsupported resize mode: {m} for pillow resize!")

            image = image.resize((width, height), m, **kwargs)
            out_trans = ToTensorImage(output_dtype=out_dtype)
            return out_trans(image)
        return resize
    elif lib.value == ResizeLib.OPENCV.value:
        import cv2
        modes = {
            ResizeMode.BILINEAR.value: cv2.INTER_LINEAR
        }

        def resize(image: torch.Tensor) -> torch.Tensor:
            nonlocal height, width, in_dtype, out_dtype, kwargs, cv2, modes, mode
            in_trans = ToNumpyImage(
                output_dtype=torch_to_numpy_dtype(in_dtype))
            image = in_trans(image)
            m = modes.get(mode.value, None)
            if m is None:
                raise ValueError(
                    f"Unsupported resize mode: {m} for cv2 resize!")
            image = cv2.resize(image, (width, height),
                               interpolation=m, **kwargs)
            out_trans = ToTensorImage(output_dtype=out_dtype)
            return out_trans(image)
        return resize
    else:
        raise ValueError(f"Unsupported resize library: {lib}.")


def _parse_metrics(metrics: List[Union[str, type, Metric, Tuple[Union[str, type, Metric], Dict[str, Any]]]]) -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    ret = dict()

    def _check_obj_vals(mt: Metric):
        if hasattr(met, "data_range"):
            met.data_range = 1.
        if hasattr(met, "max_value"):
            met.max_value = torch.tensor(1.)

    for i, metric_type in enumerate(metrics):
        if isinstance(metric_type, str) or isinstance(metric_type, type):
            metric_type = parse_type(metric_type, Metric)
            met = metric_type()
        elif isinstance(metric_type, tuple):
            mt = metric_type[0]
            args = metric_type[1] if len(metric_type) > 1 else dict()
            if not isinstance(args, dict):
                raise ValueError(
                    f"If metric is a tuple, the second element must be a dict, invalid for index {i}")
            metric_type = parse_type(mt, (Metric, torch.nn.Module))
            met = metric_type(**args)
        elif isinstance(metric_type, Metric):
            met = metric_type
        else:
            raise ValueError(
                f"Unknown metric type: {metric_type}")
        _check_obj_vals(met)
        ret[Tracker.get_metric_name(met)] = met
    return ret


def unpack_video(video_path: Union[str, Path]) -> str:
    from tools.video.reader import Reader
    from tools.io.image import save_image
    target_dir = os.path.dirname(video_path)
    name = os.path.basename(video_path).split(".")[0]
    output_image_dir = os.path.join(target_dir, name)
    os.makedirs(output_image_dir, exist_ok=True)
    reader = Reader(str(video_path)).generator()
    num_frames = len(reader)
    for i, frame in enumerate(reader):
        p = os.path.join(output_image_dir, get_leading_zeros_format_string(
            num_frames).format(i) + ".png")
        save_image(frame, p)
    return output_image_dir


@dataclass
class EvalMetricsConfig(ExperimentOutputConfig):

    target_folder: Union[str, Path] = field(default=None)
    """The target folder where the target images are stored."""

    target_filename_pattern: str = field(
        default=r"(?P<index>[0-9]+)(_[\w_\d]*)?\.((png)|(jpg)|(jpeg))")
    """The pattern of the target filenames."""

    source_folder: Union[str, Path] = field(default=None)
    """The source folder where the source images are stored."""

    source_filename_pattern: str = field(
        default=r"(?P<index>[0-9]+)(_[\w_\d]*)?\.((png)|(jpg)|(jpeg))")
    """The pattern of the source filenames."""

    scale_source: bool = field(default=True)
    """If the spatial resolution of the source and target images mismatch, whether to scale the source images to the target resolution. If False, the target images are scaled to the source resolution. Default is True."""

    allow_resize: bool = field(default=True)
    """If resizing of images is allowed when their sizes do not match. If False, an error is raised when sizes do not match."""

    resize_lib: Union[str, int, ResizeLib] = field(default="pil")
    """The library to use for resizing the images. Default is 'pil'."""

    resize_mode: Union[str, int, ResizeMode] = field(default="bilinear")
    """The mode to use for resizing the images. Default is 'bilinear'."""

    resize_in_dtype: Union[str, torch.dtype] = field(default=torch.uint8)
    """The dtype of the images before resizing. Default is torch.uint8."""

    resize_args: Dict[str, Any] = field(default_factory=dict)
    """Additional arguments to pass to the resize function."""

    name: str = field(default="EvalMetrics")
    """Name of the experiment."""

    dataloader_dtype: torch.dtype = field(default=torch.float32)
    """The dtype of the images after loading."""

    evaluation_dtype: torch.dtype = field(default=torch.float32)
    """The dtype of the images after loading."""

    device: str = field(default="cuda")
    """The device to use."""

    tracker: Optional[Union[Tracker, str]] = field(default=None)
    """Path of an existing tracker directory or a tracker object. If None, a new tracker is created."""

    batch_size: int = field(default=1)
    """The batch size to use."""

    metrics: List[Union[Type, str, Tuple[Union[Type, str],
                                         Dict[str, Any]]]] = field(default=None)
    """The metrics to use for the evaluation."""

    eval_mask_metrics: bool = field(default=False)
    """Whether to evaluate on masks. given in nag config."""

    nag_config: Optional[str] = field(default=None)
    """Path to a NAG config file to use for evaluation of masks."""

    eval_metrics: bool = field(default=True)
    """Whether to evaluate metrics."""

    compute_mean: bool = field(default=True)
    """Whether to compute the mean of the metrics."""

    is_target_video: bool = field(default=False)
    """Whether the target images are from a video."""

    is_source_video: bool = field(default=False)
    """Whether the source images are from a video."""

    truncate_on_shorter_masks: bool = field(default=True)
    """Whether to truncate the evaluation dataset if the NAG dataset masks are shorter than the evaluation dataset."""

    def prepare(self):
        super().prepare()
        self.target_folder = process_path(
            self.target_folder, need_exist=True, interpolate=True, interpolate_object=self)
        self.source_folder = process_path(
            self.source_folder, need_exist=True, interpolate=True, interpolate_object=self)
        self.output_path = process_path(
            self.output_path, need_exist=False, interpolate=True, interpolate_object=self)
        self.device = parse_device(self.device)
        self.resize_lib = parse_enum(ResizeLib, self.resize_lib)
        self.resize_mode = parse_enum(ResizeMode, self.resize_mode)
        self.resize_in_dtype = parse_dtype(self.resize_in_dtype)
        if self.tracker is not None:
            if isinstance(self.tracker, str):
                self.tracker = Tracker.from_directory(self.tracker)
        else:
            self.tracker = Tracker()

        if self.is_target_video:
            self.target_folder = unpack_video(str(self.target_folder))
            self.target_folder = process_path(
                self.target_folder, need_exist=True)
        if self.is_source_video:
            self.source_folder = unpack_video(str(self.source_folder))
            self.source_folder = process_path(
                self.source_folder, need_exist=True)

        metrics = []
        for metric in self.metrics:
            if isinstance(metric, (list, tuple)):
                metric_type, metric_kwargs = metric
            else:
                metric_type = metric
                metric_kwargs = dict()
            metric_type = parse_type(metric_type, (Metric, Callable))
            metrics.append((metric_type, metric_kwargs))
        self.metrics = metrics

        if self.nag_config is not None:
            self.nag_config = NAGConfig.load_from_file(self.nag_config)
            self.nag_config.prepare()

    @property
    def mask_metrics(self) -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        if not self.eval_mask_metrics:
            return {}
        if self.nag_config is None:
            raise ValueError(
                "NAG config must be provided for mask metrics evaluation.")
        return _parse_metrics(self.nag_config.final_mask_evaluation_metrics)

    def get_source_df(self) -> pd.DataFrame:
        paths = read_directory(
            self.source_folder, self.source_filename_pattern, parser=dict(index=int))
        paths = pd.DataFrame(paths).sort_values("index")
        paths = paths.set_index("index", drop=True)
        return paths

    def get_target_df(self) -> pd.DataFrame:
        paths = read_directory(
            self.target_folder, self.target_filename_pattern, parser=dict(index=int))
        paths = pd.DataFrame(paths).sort_values("index")
        paths = paths.set_index("index", drop=True)
        return paths

    def get_eval_data(self) -> pd.DataFrame:
        source_df = self.get_source_df()
        target_df = self.get_target_df()

        source_df.rename(dict(path="source_path"), axis=1, inplace=True)
        target_df.rename(dict(path="target_path"), axis=1, inplace=True)

        df = source_df.join(target_df, how="left",
                            lsuffix="_source", rsuffix="_target")
        # Check if all images are present
        missing_target = df["target_path"].isna()
        missing_source = df["source_path"].isna()

        if missing_target.any():
            logger.warning(
                f"Missing target images: {df[missing_target].index}")
        if missing_source.any():
            logger.warning(
                f"Missing source images: {df[missing_source].index}")
        # Opposide join to check if source is missing in target
        missing_source_in_target = target_df.join(
            source_df, how="left", lsuffix="_target", rsuffix="_source")["source_path"].isna()
        if missing_source_in_target.any():
            logger.warning(
                f"Missing source images in target: {target_df[missing_source_in_target].index.values}")

        if missing_target.any() or missing_source.any():
            missing_items = df[missing_target | missing_source]
            log_dataframe(missing_items, level=logging.ERROR)
            raise ValueError("Missing images. See log for details.")

        return df

    def get_metrics(self) -> List[Metric]:
        metrics = []
        for metric_type, metric_kwargs in self.metrics:
            metric = metric_type(**metric_kwargs)
            metrics.append(metric)
        return metrics


class EvalDataset(Dataset):

    def __init__(self,
                 index: pd.DataFrame,
                 config: EvalMetricsConfig,
                 ):
        super().__init__()
        self.index = index
        self.config = config
        self.tensorify = ToTensorImage(output_dtype=config.dataloader_dtype)
        self.nag_dataset = None
        self.masks = None
        self.mask_ids = None
        if self.config.eval_mask_metrics:
            from nag.dataset.nag_dataset import NAGDataset
            if self.config.nag_config is None:
                raise ValueError(
                    "NAG config must be provided for mask metrics evaluation.")
            self.config.nag_config.cache_images = False
            self.nag_dataset = NAGDataset(
                self.config.nag_config, None, allow_nan_on_load=True)
            self.masks = self.nag_dataset.load_mask_checked(
                None, init_size=False)
            self.mask_ids = self.nag_dataset.mask_ids.clone()
            if config.truncate_on_shorter_masks and len(self.masks) < len(self.index):
                # Truncate index
                logger.warning(
                    f"Truncating eval dataset index from {len(self.index)} to {len(self.masks)} due to NAG dataset masks length.")
                self.index = self.index.iloc[:len(self.masks)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        row = self.index.iloc[idx]
        source_path = row["source_path"]
        target_path = row["target_path"]

        source_image = load_image(source_path)
        target_image = load_image(target_path)

        HS, WS, CS = source_image.shape
        HT, WT, CT = target_image.shape

        if CS != CT:
            raise ValueError(
                f"Source and target images have different channels: {CS} != {CT}")

        if HS != HT or WS != WT:
            # Resize source image to target image size
            if self.config.scale_source:
                resize_fn = resize_factory(
                    lib=self.config.resize_lib,
                    mode=self.config.resize_mode,
                    height=HT,
                    width=WT,
                    in_dtype=self.config.resize_in_dtype,
                    out_dtype=self.config.dataloader_dtype,
                    **self.config.resize_args
                )
                source_image = resize_fn(source_image)
            else:
                resize_fn = resize_factory(
                    lib=self.config.resize_lib,
                    mode=self.config.resize_mode,
                    height=HS,
                    width=WS,
                    in_dtype=self.config.resize_in_dtype,
                    out_dtype=self.config.dataloader_dtype,
                    **self.config.resize_args
                )
                target_image = resize_fn(target_image)
        masks = torch.tensor([0])
        mask_ids = torch.tensor([0])
        if self.nag_dataset is not None:
            masks = self.masks[idx]
            mask_ids = self.mask_ids

        source_image = self.tensorify(source_image)
        target_image = self.tensorify(target_image)
        return source_image, target_image, masks, mask_ids


def config():
    from tools.mixin.argparser_mixin import set_warning_on_unsupported_type
    set_warning_on_unsupported_type(False)
    set_module_path(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    basic_config()
    from dotenv import load_dotenv
    load_dotenv()


def get_config() -> EvalMetricsConfig:
    parser = argparse.ArgumentParser(
        description='Performs a Metric evaluation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config: EvalMetricsConfig = EvalMetricsConfig.parse_args(parser)
    basic_config(filename=os.path.join(
        config.output_path, "eval_metrics.log"), reinit=True)
    return config


def calculate_mask_metrics(
    targets: torch.Tensor,
    outputs: torch.Tensor,
    masks: torch.Tensor,
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    also_only_mask: bool = True
) -> Dict[str, torch.Tensor]:
    T, HO, WO, C = outputs.shape

    results = {}
    B, C, H, W = outputs.shape
    if outputs.shape != targets.shape:
        raise ValueError(
            f"Outputs and targets must have the same shape, got {outputs.shape} and {targets.shape}.")

    _, O, _, _ = masks.shape

    for name, metric in metrics.items():
        if isinstance(metric, torch.nn.Module):
            metric = metric.to(outputs.device)
        res = torch.full((B, O), fill_value=torch.nan,
                         device=outputs.device, dtype=outputs.dtype)
        res_om = torch.full((B, O), fill_value=torch.nan, device=outputs.device,
                            dtype=outputs.dtype) if also_only_mask else None

        for b in range(B):
            for o in range(O):
                mask = masks[b, o].unsqueeze(0)  # [1, H, W]
                coords = torch.argwhere(mask)

                if len(coords) == 0:
                    # If there are no pixels in the mask, skip
                    continue

                y0, x0 = coords.amin(dim=0)[1:]
                y1, x1 = coords.amax(dim=0)[1:] + 1

                output = outputs[b, :, y0:y1, x0:x1]
                target = targets[b, :, y0:y1, x0:x1]

                res[b, o] = metric(output.unsqueeze(0), target.unsqueeze(0))

                if also_only_mask:
                    cropped_mask = mask[:, y0:y1, x0:x1]
                    output_om = output.clone()
                    output_om[~cropped_mask.expand(
                        3, -1, -1)] = 0.  # Set nonmask pixels to 0 to only calculate the metric on the mask
                    target_om = target.clone()
                    target_om[~cropped_mask.expand(3, -1, -1)] = 0.
                    res_om[b, o] = metric(
                        output_om.unsqueeze(0), target_om.unsqueeze(0))

        results[name] = res
        if also_only_mask:
            results[name + "_only_mask"] = res_om

    return results


def eval_mask_metrics(
        targets,
        outputs,
        masks,
        mask_ids,
        metrics,
        tracker,
        step: int,
):
    if len(metrics) > 0:
        merged_results = calculate_mask_metrics(
            targets=targets, outputs=outputs, masks=masks, metrics=metrics, also_only_mask=True)
        fmt_id = get_leading_zeros_format_string(len(mask_ids))
        for name, result in merged_results.items():
            B, O = result.shape
            for o in range(O):
                oname = "Mask_" + \
                    fmt_id.format(mask_ids[:, o].item()) + "_" + name
                tracker.step_metric(
                    oname, result[:, o].squeeze(), in_training=False, step=step)


@torch.no_grad()
def main(config: EvalMetricsConfig):
    logger.info("Starting evaluation.")
    yml = config.to_yaml(no_uuid=True)
    logger.info(f"Config:\n{yml}")
    config.save_to_file(os.path.join(config.output_path,
                        "eval_config.yaml"), no_uuid=True, override=True)
    dl = DataLoader(EvalDataset(config.get_eval_data(
    ), config),
        batch_size=config.batch_size,
        shuffle=False)
    tracker = config.tracker

    metrics = config.get_metrics()

    means = dict()
    num_samples = 0

    if tracker.get_epoch() != 0:
        tracker.epoch()

    current_epoch = tracker.get_epoch()
    init_step = tracker.global_steps

    if init_step != 0:
        tracker.step()
        init_step = tracker.global_steps

    bar = None
    if config.use_progress_bar:
        bar = config.progress_factory.bar(
            total=len(dl), desc="Evaluation", unit="batch")

    mask_metrics = None

    if config.eval_mask_metrics:
        mask_metrics = config.mask_metrics

    for i, (source, target, masks, mask_ids) in enumerate(dl):
        source = source.to(config.device).to(config.evaluation_dtype)
        target = target.to(config.device).to(config.evaluation_dtype)
        num_samples += source.shape[0]

        if config.eval_metrics:
            for metric in metrics:
                if isinstance(metric, torch.nn.Module):
                    metric = metric.to(config.device)

                value = metric(source, target)
                tracker.step_metric(Tracker.get_metric_name(
                    metric), value.detach().cpu(), False, False, step=init_step + i)

                if config.compute_mean:
                    if means.get(Tracker.get_metric_name(metric)) is None:
                        means[Tracker.get_metric_name(
                            metric)] = value.detach().cpu()
                    else:
                        means[Tracker.get_metric_name(
                            metric)] += value.detach().cpu()

        if config.eval_mask_metrics:
            eval_mask_metrics(
                targets=target,
                outputs=source,
                masks=masks.to(config.device),
                mask_ids=mask_ids,
                metrics=mask_metrics,
                tracker=tracker,
                step=init_step + i,
            )

        tracker.step()
        if bar is not None:
            bar.update(1)

    final_step = tracker.global_steps
    if config.compute_mean:
        for metric_name, mean in means.items():
            tracker.epoch_metric('Mean' + metric_name, mean /
                                 num_samples, False, False, step=current_epoch)

    tracker.save_to_directory(os.path.join(config.output_path, "tracker"))

    logger.info("Finished evaluation.")
    for metric in metrics:
        logger.info(f"Metric Results {Tracker.get_metric_name(metric)}:")
        df = tracker.get_metric(Tracker.get_metric_name(
            metric), scope="batch", mode="eval").values
        sl = slice(init_step, final_step + 1)
        df = df.loc[sl]
        formatted_df = format_dataframe_string(df)
        logger.info(formatted_df)
        logger.info("Stats:")
        stats = df.describe()
        stats_df = format_dataframe_string(stats)
        logger.info(stats_df)
        logger.info("")

    if config.compute_mean:
        logger.info("Mean Results:")
        for metric in metrics:
            logger.info(f"Mean {Tracker.get_metric_name(metric)}:")
            df = tracker.get_metric(
                'Mean' + Tracker.get_metric_name(metric), scope="epoch", mode="eval").values
            ep_df = df.loc[current_epoch].to_frame().T
            formatted_df = format_dataframe_string(ep_df)
            logger.info(formatted_df)
            logger.info("")


if __name__ == "__main__":
    config()
    cfg = get_config()
    with ScriptExecution(cfg):
        main(cfg)
