import argparse
import os
from typing import Optional

import numpy as np
from tools.util.path_tools import format_os_independent
from tools.logger.logging import basic_config
from tools.mixin.argparser_mixin import ArgparserMixin
import torch
from dataclasses import dataclass, field
import os
from tools.logger.logging import basic_config
from tools.util.package_tools import set_exec_module, get_executed_module_root_path
import numpy as np
from dotenv import load_dotenv

import torch
from nag.run.nag_runner import NAGRunner
from nag.config.nag_config import NAGConfig
from nag.model.timed_box_scene_node_3d import TimedBoxSceneNode3D
from tools.util.path_tools import process_path
from tools.labels.projected_timed_box_3d import ProjectedTimedBox3D
from tools.viz.matplotlib import plot_mask, figure_to_numpy
from matplotlib import pyplot as plt
from tools.video.utils import write_mp4_generator
from nag.model.timed_box_scene_node_3d import TimedBoxSceneNode3D
from tools.logger.logging import logger
import pandas as pd
import gc
from tools.util.sized_generator import SizedGenerator


@dataclass
class EvalBoxConverageConfig(ArgparserMixin):

    config_path: str
    """Path to the config file."""

    start_frame: Optional[int] = field(default=None)

    end_frame: Optional[int] = field(default=None)

    def prepare(self):
        self.config_path = os.path.normpath(self.config_path)
        self.config_path = format_os_independent(self.config_path)


def current_filename() -> str:
    return os.path.basename(__file__).split('.')[0]


def config():
    import nsf
    load_dotenv()
    basic_config()
    set_exec_module(nsf)
    os.chdir(get_executed_module_root_path())


def get_config() -> EvalBoxConverageConfig:
    parser = argparse.ArgumentParser(
        description='Creates box converage video.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config: EvalBoxConverageConfig = EvalBoxConverageConfig.parse_args(
        parser, add_config_path=False, sep="-")
    config.prepare()
    return config


def plot_masks_on_image(runner, index, images, masks, boxes, camera=None):

    gc.collect()
    torch.cuda.empty_cache()

    frame_time = runner.dataset.indices_to_times(torch.tensor(index))
    frame_idx = runner.dataset.indices_to_frame_indices(torch.tensor(index))

    mask = masks[index]
    image = images[index]

    labels = runner.dataset._loaded_mask_ids[:, 1].astype(int)
    labels

    mapped_box = dict()
    moving_box = dict()

    for k, box in boxes.items():
        if box.object_id in labels:
            mapped_box[k] = box
        else:
            if (np.linalg.norm(box.speed, axis=-1) > 0.2).any() and box.projected_label is not None:
                moving_box[k] = box

    mapped_boxes = []
    mapped_box_masks = []
    mapped_box_indices = []

    for k, v in mapped_box.items():
        v: ProjectedTimedBox3D
        if frame_time not in v.projected_label.frame_times:
            continue
        img = np.zeros_like(masks[0, ..., 0], dtype=bool)[..., None]
        inp = v.projected_label.inpaint(
            img, np.array(True), time_steps=frame_time)[..., 0]
        mapped_box_masks.append(inp)
        mapped_box_indices.append(k)
        mapped_boxes.append(v)

    mapped_box_masks = np.stack(
        mapped_box_masks, axis=-1) if len(mapped_box_masks) > 0 else np.array([])
    mapped_box_indices = np.array(mapped_box_indices)

    overlap_area = []

    m = [np.argwhere(labels == int(x) if str(x).isnumeric()
                     else None).squeeze() for x in mapped_box_indices]
    non_zero = [x.size > 0 for x in m]
    found_mask = np.stack(non_zero) if len(non_zero) > 0 else np.array([])
    # found_indices = np.stack([x for x in m if x.size > 0]) if len(m) > 0 else np.array([])

    found_mapped_box_indices = mapped_box_indices[found_mask].astype(
        int) if len(found_mask) > 0 else np.array([])
    found_mapped_box_masks = mapped_box_masks[..., found_mask] if len(
        found_mask) > 0 else np.array([])

    not_found_mapped_box_indices = mapped_box_indices[~found_mask] if len(
        found_mask) > 0 else np.array([])
    not_found_mapped_box_masks = mapped_box_masks[..., ~found_mask] if len(
        found_mask) > 0 else np.array([])

    cmap = plt.get_cmap('tab20')
    color_mapping = {k: cmap(i % cmap.N) for i, k in enumerate(labels)}

    non_mapped_colors = {l: cmap(i % cmap.N) for i, l in zip(range(len(labels), len(
        labels) + len(not_found_mapped_box_indices)), not_found_mapped_box_indices)}

    fig = plot_mask(image, mask,
                    labels=labels.round().astype(int),
                    color=[color_mapping[x] for x in labels],
                    inpaint_indices=True,
                    filled_contours=True,
                    lined_contours=False,
                    legend=False,
                    tight=True,
                    inpaint_title=False,
                    overlap_area=overlap_area)

    if camera is not None:
        for box in mapped_boxes:
            box: ProjectedTimedBox3D
            color = color_mapping.get(box.object_id, 'yellow')
            node = TimedBoxSceneNode3D.from_timed_box_3d(box)
            node.plot_2d_projection(ax=fig.gca(),
                                    camera=camera,
                                    box_color=color,
                                    t=frame_time,
                                    plot_forward_face_markers=False,
                                    image_resolution=image.shape[:2],
                                    linewidth=1)
    if len(found_mapped_box_indices) > 0:
        fig = plot_mask(None, found_mapped_box_masks,
                        labels=found_mapped_box_indices.astype(int),
                        color=[color_mapping[x]
                               for x in found_mapped_box_indices],
                        inpaint_indices=True,
                        ax=fig.gca(),
                        filled_contours=False, lined_contours=True,
                        legend=False, tight=True,
                        inpaint_title=False, overlap_area=overlap_area,
                        darkening_background=0.,
                        transparent_image=True,
                        save=True if len(
                            not_found_mapped_box_indices) == 0 else False,
                        path=runner.config.data_path /
                        "annotated_images" / f"{frame_idx}.png",
                        override=True
                        )
    else:
        fig.savefig(runner.config.data_path /
                    "annotated_images" / f"{frame_idx}.png")

    if len(not_found_mapped_box_indices) > 0:
        fig = plot_mask(None, not_found_mapped_box_masks,
                        labels=[x[:4] for x in not_found_mapped_box_indices],
                        color=[non_mapped_colors[x]
                               for x in not_found_mapped_box_indices],
                        inpaint_indices=True,
                        ax=fig.gca(),
                        filled_contours=False, lined_contours=True,
                        legend=False, tight=True,
                        inpaint_title=False, overlap_area=overlap_area,
                        darkening_background=0.,
                        transparent_image=True,
                        save=True,
                        path=runner.config.data_path /
                        "annotated_images" / f"{frame_idx}.png",
                        override=True
                        )
    arr = figure_to_numpy(fig)
    plt.close(fig)

    return arr


def main(eval_cfg: EvalBoxConverageConfig):
    config_path = process_path(eval_cfg.config_path)

    logger.info(f"Loading config from {config_path}")
    logger.info(f"Start frame: {eval_cfg.start_frame}")
    logger.info(f"End frame: {eval_cfg}")

    cfg = NAGConfig.load_from_file(config_path)

    start_frame = eval_cfg.start_frame
    end_frame = eval_cfg.end_frame

    sli = None
    if start_frame or end_frame:
        sli = slice(start_frame, end_frame)
        cfg.frame_indices_filter = sli

    cfg.experiment_logger = "tensorboard"
    cfg.skip_object_on_missing_box = True
    cfg.prepare(create_output_path=False)

    runner = NAGRunner(cfg)
    runner.build()

    masks = runner.dataset.load_mask_stack(init_size=False)
    images = runner.dataset.load_image_stack(init_size=False)
    boxes = runner.load_boxes(skip_unmapped_boxes=False)

    filtered_cols = ['id', 'object_id', 'filename']
    boxes_df = pd.DataFrame([box.to_dict() for box in boxes.values()])
    boxes_df['filename'] = boxes_df['save_path'].apply(
        lambda x: os.path.basename(x))

    start_idx = runner.dataset.times_to_frame_indices(torch.tensor([0.]))[
        0].item()
    end_idx = runner.dataset.times_to_frame_indices(torch.tensor([1.]))[
        0].item()

    boxes_df[filtered_cols].to_csv(
        runner.config.data_path / f"box_map_{start_idx}_{end_idx}.csv", index=True)

    start_offset = runner.dataset.indices_to_frame_indices(0)
    gen = SizedGenerator((plot_masks_on_image(runner, i, images, masks, boxes,
                         runner.model.camera) for i in range(0, images.shape[0])), len(images))

    slice_str = f"{start_idx}_{end_idx}"
    with plt.ioff():
        write_mp4_generator(gen, runner.config.data_path / f"annotated_video{('_' + slice_str) if len(slice_str) > 0 else ''}.mp4",
                            fps=5, frame_counter=True, frame_counter_offset=start_offset, progress_bar=True)


if __name__ == '__main__':
    config()
    cfg = get_config()
    main(cfg)
