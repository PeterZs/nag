#!/usr/bin/env python3
# Sample job file for an omni job. Can be used as a standalone file or within a job list.
from datetime import datetime
from nag.analytics.scene_nsf_result_model import NAGResultModel
from tools.util.path_tools import process_path
from tools.config.output_config import OutputConfig
from tools.mixin.argparser_mixin import ArgparserMixin
from dataclasses import dataclass, field
from tools.context.script_execution import ScriptExecution
from tools.util.package_tools import set_module_path
from nag.config.nag_config import NAGConfig
from nag.callbacks.nag_callback import NAGCallback
from tools.logger.logging import basic_config, logger
import matplotlib.pyplot as plt
import matplotlib
import argparse
import logging  # noqa
import os
from typing import Dict, List, Tuple, Union
from tools.util.torch import set_jit_enabled

set_jit_enabled(False)


plt.ioff()
matplotlib.use('agg')


@dataclass
class NAGEvalConfig(OutputConfig):

    run_path: str = field(default=None)
    """Path to the existing run which should be evaluated."""

    preserve_output_path: bool = field(default=True)
    """If the output path should be preserved. If False, all outputs will be places in a new generated output folder."""

    runner_index: int = field(default=-1)
    """Index of the runner to evaluate. In case multiple checkpoints exists will load the checkpoint with the given index. Default the last is used."""

    create_texture_editing_grid_background: bool = field(default=False)
    """If texture editing files should be created."""

    create_texture_editing_grid_foreground_oids: Union[str, int, List[int]] = field(
        default_factory=list)
    """List of object ids to be used where also a default texture editing grid shall be created."""

    create_individual_atlas_files: bool = field(default=True)
    """If individual atlas files should be created."""

    perform_texture_editing: bool = field(default=False)
    """If texture editing should be performed."""

    map_textures: Dict[int, List[Tuple[str, int]]
                       ] = field(default_factory=dict)
    """Mapping of textures to be used for texture editing."""

    def prepare(self):
        super().prepare()
        self.run_path = process_path(
            self.run_path, need_exist=True, interpolate=True, interpolate_object=self, variable_name="run_path")
        if self.preserve_output_path:
            self.output_folder = self.run_path

        if len(self.map_textures) > 0:
            for key, items in self.map_textures.items():
                for idx, (path, time) in enumerate(items):
                    parse_path = process_path(path, need_exist=True, interpolate=True, interpolate_object=self,
                                              variable_name="map_textures item " + str(key) + " index " + str(idx))
                    self.map_textures[key][idx] = (str(parse_path), time)
        if isinstance(self.create_texture_editing_grid_foreground_oids, str):
            # Split along comma and convert to int
            self.create_texture_editing_grid_foreground_oids = [
                int(x.strip()) for x in self.create_texture_editing_grid_foreground_oids.split(",")]


def config():
    from tools.mixin.argparser_mixin import set_warning_on_unsupported_type
    set_warning_on_unsupported_type(False)
    set_module_path(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    basic_config()


def get_config() -> NAGEvalConfig:
    parser = argparse.ArgumentParser(
        description='Performs NAG Evaluations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config: NAGEvalConfig = NAGEvalConfig.parse_args(parser)
    return config


def main(config: NAGEvalConfig):
    logger.info(f"Starting NAGation")
    logger.info(f"Using output directory: {str(config.output_folder)}")

    rm = NAGResultModel.from_path(config.run_path)
    logger.info(f"Loading NAG Result Model from {config.run_path} ...")
    runner = rm.get_runner(config.runner_index,
                           preserve_output_path=config.preserve_output_path)

    with plt.ioff():
        if config.create_texture_editing_grid_background:
            callback: NAGCallback = runner.callbacks[0]
            callback.save_texture_editing_background(model=runner.model)

        if config.create_texture_editing_grid_foreground_oids is not None:
            ids = None
            if isinstance(config.create_texture_editing_grid_foreground_oids, int):
                ids = [config.create_texture_editing_grid_foreground_oids]
            elif isinstance(config.create_texture_editing_grid_foreground_oids, list):
                ids = config.create_texture_editing_grid_foreground_oids
            else:
                raise ValueError(
                    "create_texture_editing_grid_foreground_oids must be either an int or a list of ints.")
            callback: NAGCallback = runner.callbacks[0]
            for oid in ids:
                try:
                    callback.save_texture_editing_oid(runner.model, oid)
                except Exception as e:
                    logger.error(
                        f"Error while saving texture editing for object {oid}: {e}")
                    continue

        if config.create_individual_atlas_files:
            runner.config.final_diff_plot = False
            runner.config.final_modalities_save = False
            runner.config.final_plot_without_view_dependency = False
            callback: NAGCallback = runner.callbacks[0]
            callback.save_final(model=runner.model, generate_per_object_images=True,
                                evaluate_metrics=False, evaluate_mask_metrics=False)
        if config.perform_texture_editing:
            callback: NAGCallback = runner.callbacks[0]
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = os.path.join(
                config.output_folder, "edits", "texture_editing_" + datetime_str)
            callback.perform_texture_editing(
                model=runner.model, textures=config.map_textures, output_path=output_path)


if __name__ == "__main__":
    config()
    cfg = get_config()
    with ScriptExecution(cfg):
        main(cfg)
