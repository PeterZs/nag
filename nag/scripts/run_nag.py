#!/usr/bin/env python3
import argparse
import logging  # noqa
import os

import matplotlib
import matplotlib.pyplot as plt
from tools.logger.logging import basic_config, logger

from nag.config.nag_config import NAGConfig

from tools.util.package_tools import set_module_path
from tools.context.script_execution import ScriptExecution
plt.ioff()
matplotlib.use('agg')


def config():
    from tools.mixin.argparser_mixin import set_warning_on_unsupported_type
    set_warning_on_unsupported_type(False)
    set_module_path(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    basic_config()


def get_config() -> NAGConfig:

    parser = argparse.ArgumentParser(
        description='Performs a SceneNSF training run.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config: NAGConfig = NAGConfig.parse_args(parser)

    return config


def main(config: NAGConfig):
    logging.info(f"Setup: {config.name}")
    logger.info(f"Using output directory: {str(config.output_path)}")
    # Setup
    from nag.run.nag_runner import NAGRunner
    runner = NAGRunner(config)
    runner.build()

    # Save config and log it
    cfg_file = runner.store_config(str(config.output_path))

    logger.info(f"Stored config in: {cfg_file}")
    # Training
    logger.info(f"Start training of: {config.name}")
    with plt.ioff():
        runner.train()


if __name__ == "__main__":
    config()
    cfg = get_config()
    with ScriptExecution(cfg):
        main(cfg)
