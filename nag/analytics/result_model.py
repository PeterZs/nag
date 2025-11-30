import copy
from dataclasses import dataclass, field
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from numpy import iterable
import torch
import pandas as pd
import re
from PIL import Image
from tools.serialization.json_convertible import JsonConvertible
from tools.config.config import Config
from tools.run.config_runner import ConfigRunner
from tools.util.path_tools import open_folder, read_directory, relpath, format_os_independent
import numpy as np
from nag.analytics.result_model_config import ResultModelConfig
from nag.config.nsf_config import NSFConfig
from tools.logger.logging import logger

from nag.run.nsf_runner import NSFRunner


def minmax(v: torch.Tensor, new_min: float = 0., new_max: float = 1.):
    v_min, v_max = v.min(), v.max()
    return (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min


def mean_std_norm(v: torch.Tensor, mu: Optional[float] = None, std: Optional[float] = None) -> torch.Tensor:
    if mu is None:
        mu = v.mean()
    if std is None:
        std = v.std()
    return (v - mu) / std


def inverse_mean_std_norm(v: torch.Tensor, mu: float, std: float) -> torch.Tensor:
    return (v * std) + mu


def get_circular_index(vals: Optional[iterable], index: int, max_val: Optional[int] = None) -> int:
    if vals is None and max_val is None:
        raise ValueError("Either vals or max_val must be set.")
    elif max_val is None:
        max_val = len(vals)
    if index < 0:
        index = (-1) * (abs(index) % max_val)
        if index < 0:
            index = max_val + index
    elif index >= max_val:
        index = index % max_val
    else:
        raise NotImplementedError("Never Reached")
    return index


@dataclass
class ResultModel():

    config: ResultModelConfig

    output_directory: str

    index: pd.DataFrame = field(
        default_factory=lambda: ResultModel.create_index_df())

    checkpoint_index: pd.DataFrame = field(
        default_factory=lambda: ResultModel.create_checkpoint_df())

    _run_config: Optional[Config] = field(default=None)
    """If experiment was done using a runner, there exists a run config in the folder which can be loaded."""

    _runners: Dict[int, ConfigRunner] = field(default_factory=dict)

    run_config_path: Optional[str] = field(default=None)
    """Path to the run config."""

    get_item_keys: List[str] = field(default_factory=lambda: [
                                     "alpha", "combined", "obstruction", "reference", "transmission"])

    get_item_return_format: Literal["list", "dict"] = "dict"

    @classmethod
    def get_default_config(cls, name: str) -> ResultModelConfig:
        return ResultModelConfig(name=name)

    @classmethod
    def create_checkpoint_df(cls, data: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        df = pd.DataFrame(data, columns=["epoch"])
        return df

    @classmethod
    def create_index_df(cls, data: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        return pd.DataFrame(data, columns=["epoch"])

    def get_runner(self, index: int = -1, config_kwargs: Optional[Dict[str, Any]] = None) -> NSFRunner:
        epoch = get_circular_index(None, index, self.run_config.max_epochs)
        return self.get_checkpoint_entry(epoch, config_kwargs=config_kwargs)

    @property
    def number(self) -> int:
        if self.config.number is None:
            pattern = re.compile(self.config.parse_number_pattern)
            match = pattern.match(self.config.name)
            if match is not None:
                self.config.number = int(match.group("number"))
        return self.config.number

    def get_index_keys(self) -> List[str]:
        return [x for x in self.index.columns if not x.endswith("_path") and x != "epoch"]

    def __repr__(self) -> str:
        self_dict = dict(vars(self))
        dont_show = ["index", "checkpoint_index"]
        for prop in dont_show:
            if prop in self_dict and self_dict.get(prop) is not None:
                self_dict[prop] = "[...]"
        return type(self).__name__ + f"({', '.join([k+'='+str(v) for k, v in self_dict.items()])})"

    def open_folder(self) -> None:
        """Opens the model output path in the operating system file explorer.

        Parameters
        ----------
        result_type : Optional[Union[ResultType, str]], optional
            The result type path to open. None opens base directory, dedicated result_types will open their own folder., by default None
        """
        path = os.path.normpath(self.output_directory)
        open_folder(path)

    def index_from_epoch(self, epoch: int):
        """Returns the index for an epoch to use within getitem."""
        if epoch not in self.config.epochs:
            raise ValueError(
                f"Invalid epoch: {epoch}. Avail: {', '.join([str(i) for i in self.config.epochs])}")
        return self.config.epochs.index(epoch)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Any:
        epoch = get_circular_index(None, index, self.run_config.max_epochs)

        if self.get_item_return_format == "dict":
            ret = dict()
        elif self.get_item_return_format == "list":
            ret = list()
        else:
            raise ValueError(
                f"Invalid return format: {self.get_item_return_format}")

        for key in self.get_item_keys:
            value = self.get_index_entry(epoch, key)
            if self.get_item_return_format == "dict":
                ret[key] = value
            elif self.get_item_return_format == "list":
                ret.append(value)
        return ret

    def get_index_entry(self, epoch: int, key: str) -> Any:
        if key not in self.index.columns:
            raise ValueError(f"Invalid key: {key}.")
        series = self.index[self.index["epoch"] == epoch].iloc[0]
        value = series[key]
        if (value is None or (isinstance(value, float) and np.isnan(value))) and key + "_path" in self.index.columns:
            path = os.path.join(self.output_directory, series[key + "_path"])
            value = self._load_entry(path)
            self.index.at[series.name, key] = value
        return value

    def get_checkpoint_entry(self, epoch: int, config_kwargs: Optional[Dict[str, Any]] = None, cache: bool = True) -> NSFRunner:
        series = self.checkpoint_index[self.checkpoint_index["epoch"]
                                       == epoch].iloc[0]
        runner = series["checkpoint"]
        if (runner is None or (isinstance(runner, float) and np.isnan(runner))) and "checkpoint_path" in self.checkpoint_index.columns:
            path = os.path.join(self.output_directory,
                                series["checkpoint_path"])
            cfg = self.run_config
            if config_kwargs is not None:
                for key, value in config_kwargs.items():
                    setattr(cfg, key, value)
            runner = NSFRunner(self.run_config)
            runner.load(path, logger=None)
            if cache:
                self.index.at[series.name, "checkpoint_path"] = runner
        return runner

    def _load_entry(self, path: str) -> Any:
        if path.endswith(".pth"):
            return torch.load(path)
        elif path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".png") or path.endswith(".jpg"):
            return np.array(Image.open(path))
        else:
            raise NotImplementedError(
                f"Loading of {path} not implemented yet.")

    @property
    def display_name(self) -> str:
        """Get a name for the result."""
        name = self.config.name or os.path.basename(self.output_directory)
        if self.numbering and self.config.number is not None:
            name = str(self.config.number) + ". " + name
        return name

    @display_name.setter
    def display_name(self, value: str):
        self.config.name = value

    @property
    def name(self) -> str:
        return os.path.basename(self.output_directory)

    def parse_name(self) -> Optional[str]:
        if self.config.parse_name_pattern is None:
            return None
        pattern = re.compile(self.config.parse_name_pattern)
        match = pattern.match(self.config.name)
        if match is None:
            return None
        return match.group("model_name")

    @property
    def parsed_name(self) -> str:
        if self.config.parsed_name is None:
            pname = self.parse_name()
            if pname is None:
                pname = self.config.name
            self.config.parsed_name = pname
        return self.config.parsed_name

    @property
    def run_config(self) -> Optional[Config]:
        """Gets the config which was used to run the experiment, when it was executed with a runner.

        Returns
        -------
        Optional[Config]
            Config of the experiment.
        """
        if self._run_config is None:
            if self.run_config_path is not None:
                try:
                    self._run_config = JsonConvertible.load_from_file(
                        self.run_config_path)
                except Exception as err:
                    logging.exception(
                        f"Could not load run config from {self.run_config_path}.")
        return self._run_config

    @run_config.setter
    def run_config(self, value: Config):
        self._run_config = value

    @classmethod
    def from_path(cls, path: str) -> 'ResultModel':
        if not os.path.exists(path):
            raise ValueError(
                f"Path: {path} does not exists. Can not create Resultmodel.")
        config = None
        config_path = os.path.join(path, "result_model_config.json")
        if os.path.exists(config_path):
            config = ResultModelConfig.load_from_file(config_path)
        else:
            config = cls.get_default_config(name=os.path.basename(path))
        model = cls(config=config, output_directory=path)
        model.create_index()
        model.save_config()
        return model

    def reload_config(self):
        """Reloads the config from the config.json file. Will override the existing one."""
        config_path = os.path.join(
            self.output_directory, "result_model_config.json")
        self.config = ResultModelConfig.load_from_file(config_path)

    def save_config(self):
        self.config.save_to_file(self.config_path, override=True)

    def scan_result_directory(self, path: str) -> Dict[str, Dict[str, Any]]:
        data = dict()
        for name, pattern in self.config.parse_result_patterns.items():
            results = read_directory(path, pattern=pattern)
            data[name] = results
        return data

    def scan_checkpoints(self) -> List[Dict[str, Any]]:
        data = dict()
        for name, pattern in self.config.parse_checkpoint_patterns.items():
            ckps = read_directory(self.output_directory, pattern)
            data[name] = ckps
        return data

    def _multi_pattern_extract(self, data: Dict[str, Any]) -> pd.DataFrame:
        # Collapse entries
        new_data = dict()  # Ordered by epoch

        for key, values in data.items():
            for v in values:
                if "path" not in v:
                    raise ValueError(f"Invalid entry: {v}")
                epoch = v.get("epoch", -1)
                if epoch is None:
                    epoch = -1
                res = new_data.get(epoch, dict())
                if epoch not in new_data:
                    new_data[epoch] = res
                path_key = key + "_path" if "path" not in key else key
                res[path_key] = format_os_independent(
                    relpath(self.output_directory, v["path"], is_from_file=False))

        di = pd.DataFrame.from_dict(new_data, orient="index").reset_index(
            drop=False, names="epoch")
        return di

    def create_index(self):
        data = dict()
        for sub_path in self.config.result_directories:
            path = os.path.join(self.output_directory, sub_path)
            out = self.scan_result_directory(path)
            for key, value in out.items():
                if key not in data:
                    data[key] = value
                else:
                    data[key].extend(value)

        columns = [k if "path" in k else k + "_path" for k in data.keys()]
        columns = set(columns + list(data.keys()))

        di = self._multi_pattern_extract(data)
        self.index = type(self).create_index_df(data)
        self.index = self.index.assign(**{k: np.nan for k in data.keys()})
        for k in data.keys():
            self.index[k] = self.index[k].astype(object)
        self.index = self.index.merge(di, how="right")

        # Checking for init config
        init_cfg = read_directory(
            self.output_directory, self.config.run_config_pattern)
        if len(init_cfg) > 0:
            self.run_config_path = init_cfg[0]['path']
        else:
            logger.warning(
                f"No run config found in {self.output_directory}. Create default.")
            self.run_config_path = None
            self.run_config = NSFConfig()

        ckp_data = self.scan_checkpoints()
        ckp_df = self._multi_pattern_extract(ckp_data)

        ckp_columns = [k if "path" in k else k +
                       "_path" for k in ckp_data.keys()]
        ckp_columns = set(ckp_columns + list(ckp_data.keys()))

        self.checkpoint_index = type(self).create_checkpoint_df(ckp_df)
        self.checkpoint_index = self.checkpoint_index.assign(
            **{k: np.nan for k in ckp_data.keys()})
        for k in ckp_data.keys():
            self.checkpoint_index[k] = self.checkpoint_index[k].astype(object)
        self.checkpoint_index = self.checkpoint_index.merge(
            ckp_df, how="right").sort_values("epoch")
        # Resolve epochs

        ep_map = {k: get_circular_index(
            None, k, self.run_config.max_epochs) for k in self.index["epoch"].unique()}
        self.index["epoch"] = self.index["epoch"].map(ep_map)
        ep_map_ckp = {k: get_circular_index(
            None, k, self.run_config.max_epochs) for k in self.checkpoint_index["epoch"].unique()}
        self.checkpoint_index["epoch"] = self.checkpoint_index["epoch"].map(
            ep_map_ckp)
        pass

    def get_checkpoint(self, index: int) -> Any:
        """Gets the agent checkpoint from the given index.

        Parameters
        ----------
        index : int
            The index of checkpoints.

        Returns
        -------
        Any
            Checkpoint
        """
        if len(self.checkpoint_index) == 0:
            raise IndexError("No checkpoints found!")
        index = get_circular_index(self.checkpoint_index.index, index)
        ckp_row = self.checkpoint_index.iloc[index]
        col_iloc = self.checkpoint_index.columns.get_loc("checkpoint")
        if pd.isna(ckp_row['checkpoint']):
            raise NotImplementedError(
                "Checkpoint loading not implemented yet.")
            # checkpoint = TorchAgentCheckpoint.load(ckp_row['path'])
            # self.checkpoint_index.iloc[index, col_iloc] = checkpoint
        return self.checkpoint_index.iloc[index, col_iloc]

    @property
    def config_path(self) -> str:
        path = os.path.join(self.output_directory, "result_model_config.json")
        return path
