import copy
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
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
from tools.util.path_tools import open_folder, read_directory, read_directory_recursive, relpath, format_os_independent
import numpy as np
from tools.logger.logging import logger
from tools.util.path_tools import read_directory_recursive
from nag.config.nag_config import NAGConfig
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_plane_scene_node_3d import TimedPlaneSceneNode3D
from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D
from nag.analytics.nag_result_model_config import NAGResultModelConfig
from nag.run.nag_runner import NAGRunner
from tools.transforms.geometric.mappings import unitquat_to_euler, rotmat_to_unitquat
from tools.viz.matplotlib import get_mpl_figure, saveable
from tools.agent.util.tracker import Tracker
from tools.io.image import load_image


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
class NAGResultModel():

    config: NAGResultModelConfig

    output_directory: str

    index: pd.DataFrame = field(
        default_factory=lambda: NAGResultModel.create_index_df())

    checkpoint_index: pd.DataFrame = field(
        default_factory=lambda: NAGResultModel.create_checkpoint_df())

    _run_config: Optional[Config] = field(default=None)
    """If experiment was done using a runner, there exists a run config in the folder which can be loaded."""

    _runners: Dict[int, ConfigRunner] = field(default_factory=dict)

    _final_tracker: Optional[Tracker] = field(default=None)

    run_config_path: Optional[str] = field(default=None)
    """Path to the run config."""

    getitem_epoch: int = -1
    """Epoch to use for getitem."""

    getitem_kind: str = "final"
    """Kind of data to retrieve in getitem."""

    getitem_object_composition_index: Optional[str] = "complete"
    """The object composition index to use for getitem."""

    @classmethod
    def get_default_config(cls, name: str) -> NAGResultModelConfig:
        return NAGResultModelConfig(name=name)

    @classmethod
    def create_checkpoint_df(cls, data: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        df = pd.DataFrame(data, columns=["epoch"])
        return df

    @classmethod
    def create_index_df(cls, data: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        return pd.DataFrame(data, columns=["epoch"])

    def get_runner(self,
                   index: int = -1,
                   config_kwargs: Optional[Dict[str, Any]] = None,
                   preserve_output_path: bool = False,
                   cache: bool = True) -> NAGRunner:
        epoch = get_circular_index(
            self.checkpoint_index["epoch"].unique(), index)
        return self.get_checkpoint_entry(epoch, config_kwargs=config_kwargs, preserve_output_path=preserve_output_path, cache=cache)

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

    @property
    def final_tracker(self) -> Optional[Tracker]:
        if self._final_tracker is None:
            final_tracker_path = os.path.join(
                self.output_directory, "tracker")
            if os.path.exists(final_tracker_path):
                try:
                    self._final_tracker = Tracker.from_directory(
                        final_tracker_path)
                except Exception as e:
                    logging.error(
                        f"Could not load final tracker from {final_tracker_path}: {str(e)}")
        return self._final_tracker

    def get_tracker(self, index: int) -> Tracker:
        """Gets the tracker of the given checkpoint index.

        Parameters
        ----------
        index : int
            the checkpoint index where the tracker should be loaded from.

        Returns
        -------
        Tracker
            The tracker containing state and metric info.
        """
        if index == -1:
            final_tracker = self.final_tracker
            if final_tracker is not None:
                return final_tracker
        return self.get_checkpoint(index).tracker

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

    def get_checkpoint_entry(self,
                             epoch_index: int,
                             config_kwargs: Optional[Dict[str, Any]] = None,
                             preserve_output_path: bool = False,
                             cache: bool = True
                             ) -> NAGRunner:
        series = self.checkpoint_index[self.checkpoint_index["epoch_index"]
                                       == epoch_index].iloc[0]
        runner = series.get("checkpoint", None)
        if (runner is None or (isinstance(runner, float) and np.isnan(runner))) and "path" in self.checkpoint_index.columns:
            path = os.path.join(self.output_directory,
                                series["path"])
            cfg = self.run_config
            if config_kwargs is not None:
                for key, value in config_kwargs.items():
                    setattr(cfg, key, value)

            runner = NAGRunner(self.run_config)
            runner.load(path, logger=None, tracker_directory=os.path.join(
                self.output_directory, "tracker"))
            if cache:
                if "checkpoint" in self.checkpoint_index.columns:
                    self.checkpoint_index.at[series.name,
                                             "checkpoint"] = runner
                else:
                    # Create new column
                    self.checkpoint_index["checkpoint"] = [[]
                                                           for _ in range(len(self.checkpoint_index))]
                    self.checkpoint_index.at[series.name,
                                             "checkpoint"] = runner
        if preserve_output_path:
            from tools.util.path_tools import process_path
            runner.config.output_path = process_path(
                self.output_directory, need_exist=True, variable_name="output_path")

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
                    self._run_config.prepare()
                except Exception as err:
                    logging.exception(
                        f"Could not load run config from {self.run_config_path}.")
        return self._run_config

    @run_config.setter
    def run_config(self, value: Config):
        self._run_config = value

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> 'NAGResultModel':
        if not os.path.exists(path):
            raise ValueError(
                f"Path: {path} does not exists. Can not create Resultmodel.")
        if isinstance(path, Path):
            path = str(path)
        # convert path into relative path
        path = os.path.normpath(path)
        try:
            path = os.path.relpath(path)
        except ValueError:
            pass
        config = None
        config_path = os.path.join(path, "result_model_config.json")
        if os.path.exists(config_path):
            config = NAGResultModelConfig.load_from_file(config_path)
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
        # self.config.save_to_file(self.config_path, override=True)
        pass

    def scan_result_directory(self, path: str, **parser_args) -> Dict[str, Dict[str, Any]]:
        data = dict()
        for name, pout in self.config.parse_result_patterns.items():
            if isinstance(pout, str):
                pattern = pout
                parser_args = dict()
            else:
                pattern = pout["pattern"]
                pargs = pout.pop("parser", dict())
                if "parser" in parser_args:
                    p = parser_args["parser"]
                    p.update(pargs)

            p = path + "/" + pattern
            results = read_directory_recursive(p, **parser_args)
            data[name] = results
        return data

    def multi_path_multi_pattern_scan(self, paths: List[Union[str, Dict[str, Any]]], patterns: Dict[str, Union[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Gets the results from multiple paths with multiple patterns.

        Parameters
        ----------
        paths : List[Union[str, Dict[str, Any]]]
            Multiple paths to scan for results. Can be a string (regex) or a dictionary with a pattern (regex) and a parser.

        patterns : Dict[str, Union[str, Dict[str, Any]]]
            Key: Name of the pattern.
            Value: Patterns to scan the directory folders for. Can be a string or a dictionary with a pattern and a parser.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Key: Pattern name.
            Value: Dictionary with the results.
        """
        data = dict()
        for rout in paths:
            parser_args = dict()
            if isinstance(rout, str):
                sub_path = rout
            else:
                r = dict(rout)
                sub_path = r.pop("pattern")
                parser_args.update(r)
            path = format_os_independent(
                self.output_directory) + "/" + sub_path
            out = self.scan_directories(path, patterns, **parser_args)
            for key, value in out.items():
                if key not in data:
                    data[key] = value
                else:
                    data[key].extend(value)
        return data

    def scan_directories(self, path: str, patterns: Dict[str, Union[str, Dict[str, Any]]], **parser_args) -> Dict[str, Dict[str, Any]]:
        """Scans a directory path for patterns.


        Parameters
        ----------
        path : str
            Path to the directory, which itself can be a pattern.

        patterns : Dict[str, Union[str, Dict[str, Any]]]
            Key: Name of the pattern.
            Value: Patterns to scan the directory folders for. Can be a string or a dictionary with a pattern and a parser.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Key: Pattern name.
            Value: Dictionary with the results.
        """
        data = dict()
        for name, pout in patterns.items():
            parse_args_cp = dict(parser_args)
            if isinstance(pout, str):
                pattern = pout
            else:
                pout = dict(pout)
                pattern = pout.pop("pattern")
                pargs = pout.pop("parser", dict())
                if "parser" in parse_args_cp:
                    p = parse_args_cp["parser"]
                    p.update(pargs)
                else:
                    parse_args_cp["parser"] = pargs

            p = path + "/" + pattern
            results = read_directory_recursive(p, **parse_args_cp)
            if name in data:
                data[name].extend(results)
            else:
                data[name] = results
        return data

    def scan_checkpoints(self, path: str, **parser_args) -> List[Dict[str, Any]]:
        data = dict()
        for name, args in self.config.parse_checkpoint_patterns.items():
            parser_args = dict()
            pattern = None
            if isinstance(args, str):
                pattern = args
            else:
                pattern = args.pop("pattern")
                parser_args.update(args)
            ckps = read_directory(self.output_directory,
                                  pattern, **parser_args)
            data[name] = ckps
        return data

    def localize_data(self, data: List[Dict[str, Any]], kind: Optional[str] = None) -> pd.DataFrame:
        # Collapse entries
        for item in data:
            item["path"] = format_os_independent(
                relpath(self.output_directory, item["path"], is_from_file=False, is_to_file=True))
            if kind is not None:
                item["kind"] = kind
            if "epoch" not in item:
                item["epoch"] = self.run_config.max_epochs

            if item["path"].startswith("final/complete/") or item["path"].startswith("/in_training/") and item.get("object_composition_index", None) is None:
                item["object_composition_index"] = "complete"

            if item["path"].startswith("final/complete_no_view_dependency/") and item.get("object_composition_index", None) is None:
                item["object_composition_index"] = "complete_no_view_dependency"

        di = pd.DataFrame(data)
        return di

    def create_index(self):

        # Checking for init config
        init_cfg = read_directory(
            self.output_directory, self.config.run_config_pattern)
        if len(init_cfg) > 0:
            self.run_config_path = init_cfg[0]['path']
        else:
            logger.warning(
                f"No run config found in {self.output_directory}. Create default.")
            self.run_config_path = None
            self.run_config = NAGConfig()

        data = self.multi_path_multi_pattern_scan(
            self.config.final_result_directories, self.config.parse_result_patterns)

        frames = []
        for key, value in data.items():
            if len(value) == 0:
                logger.debug(
                    f"No results found for {key}.")

            di = self.localize_data(value, kind=key)
            frames.append(di)

        data = self.multi_path_multi_pattern_scan(
            self.config.intermediate_result_directories, self.config.parse_intermediate_result_patterns)

        for key, value in data.items():
            if len(value) == 0:
                logger.debug(
                    f"No results found for {key}.")
                continue
            di = self.localize_data(value, kind=key)
            frames.append(di)
        di = pd.concat(frames)
        self.index = di

        frames = []
        ckp_data = self.multi_path_multi_pattern_scan(
            self.config.ckeckpoint_directories, self.config.parse_checkpoint_patterns)
        for key, value in ckp_data.items():
            if len(value) == 0:
                logger.debug(
                    f"No results found for {key}.")
                continue
            di = self.localize_data(value, kind=key)
            frames.append(di)
        self.checkpoint_index = pd.concat(frames)

        # Get unique epochs
        self.checkpoint_index["epoch"] = self.checkpoint_index["epoch"].astype(
            int)
        self.index["epoch"] = self.index["epoch"].astype(int)

        unique_tracked_images_epochs = set(self.index["epoch"].unique())
        unique_tracked_ckp_epochs = set(
            self.checkpoint_index["epoch"].unique())
        unique_tracked_epochs = list(
            unique_tracked_images_epochs.union(unique_tracked_ckp_epochs))

        ep_map = {k: i for i, k in enumerate(sorted(unique_tracked_epochs))}
        ep_map_img = {k: i for i, k in enumerate(
            sorted(unique_tracked_images_epochs))}
        ep_map_ckp = {k: i for i, k in enumerate(
            sorted(unique_tracked_ckp_epochs))}
        self.index["epoch_union_index"] = self.index["epoch"].map(ep_map)
        self.index["epoch_index"] = self.index["epoch"].map(ep_map_img)
        self.checkpoint_index["epoch_union_index"] = self.checkpoint_index["epoch"].map(
            ep_map)
        self.checkpoint_index["epoch_index"] = self.checkpoint_index["epoch"].map(
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

    def create_final_video(self, path: str = "{output_folder}/final_video.mp4", fps: int = 10) -> None:
        """Creates the final result video from the images.

        Parameters
        ----------
        path : str, optional
            Safe path of the output. Will be parsed , by default "{output_folder}/final_video.mp4"
        fps : int, optional
            Fps, by default 10
        """
        from tools.io.image import load_image_stack
        from tools.video.utils import write_mp4
        from tools.util.format import parse_format_string
        config = self.run_config
        path = parse_format_string(path, obj_list=[config])[0]
        image_rows = self.index[(self.index["object_composition_index"] == "complete") & (
            self.index["kind"] == "final")].sort_values("time")
        paths = image_rows["path"].tolist()
        paths = [os.path.join(config.output_folder, p) for p in paths]
        # Load images
        images = load_image_stack(sorted_image_paths=paths)
        write_mp4(images, path, fps=fps)

    def create_final_object_video(self,
                                  path: str = "{output_folder}/final_video_o_{object_index:02d}.mp4",
                                  object_index: int = 0,
                                  fps: int = 10,
                                  resolution: Optional[Tuple[int, int]] = None,
                                  transparent: bool = True) -> None:
        """Creates the final result video from the images.

        Parameters
        ----------
        path : str, optional
            Safe path of the output. Will be parsed , by default "{output_folder}/final_video.mp4"
        fps : int, optional
            Fps, by default 10
        """
        from tools.io.image import load_image_stack
        from tools.util.torch import tensorify
        from tools.video.utils import write_mp4
        from tools.util.format import parse_format_string
        config = self.run_config
        path = parse_format_string(path, obj_list=[config], additional_variables=dict(
            object_index=object_index))[0]
        image_rows = self.index[(self.index["object_composition_index"] == str(object_index)) & (
            self.index["kind"] == "final")].sort_values("time")
        paths = image_rows["path"].tolist()
        paths = [os.path.join(config.output_folder, p) for p in paths]
        # Load images
        images = load_image_stack(sorted_image_paths=paths, size=resolution)
        if not transparent:
            from tools.io.image import alpha_background_grid
            from nag.utils.utils import n_layers_alpha_compositing
            background = tensorify(alpha_background_grid(images.shape[1:3])[
                                   None, ...].repeat(len(images), axis=0)).float() / 255
            matted = n_layers_alpha_compositing(torch.stack([tensorify(images).float(
            ) / 255, background], axis=0), zbuffer=torch.tensor([0, 1])).numpy()
            images = (matted * 255).astype(np.uint8)
        write_mp4(images, path, fps=fps)

    def create_training_evolution_video(self,
                                        t: int,
                                        path: str = "{output_folder}/{t:02d}_in_training_video.mp4",
                                        fps: int = 10) -> None:
        from tools.io.image import load_image_stack
        from tools.video.utils import write_mp4
        from tools.util.format import parse_format_string
        config = self.run_config
        path = parse_format_string(
            path, obj_list=[config], additional_variables=dict(t=t))[0]
        image_rows = self.index[(self.index["kind"] == "in_training") & (
            self.index["time"] == t)].sort_values("epoch")
        if len(image_rows) == 0:
            # Check of any in_training images are available
            if len(self.index[(self.index["kind"] == "in_training")]) == 0:
                raise ValueError(
                    "No in_training images found. Was in_training plotting enabled?")
            raise ValueError(f"No images for time {t} found." +
                             "Existing times are: " + str(self.index[(self.index["kind"] == "in_training")]["time"].unique()))
        paths = image_rows["path"].tolist()
        paths = [os.path.join(config.output_folder, p) for p in paths]
        # Load images
        images = load_image_stack(sorted_image_paths=paths)
        write_mp4(images.astype(float) / 255, path, fps=fps)

    def get_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the objects positions as tracked within the tracker as global position matrix.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            1. np.ndarray: The positions of the objects. Shape: (N, E, T, 4, 4)
            Where N is the number of objects, E is the number of tracked epochs and T is the number of time steps.
            E.g. N is for each object which positions was tracked.
            2. np.ndarray: The labels of the objects as tag. Shape: (N,) string
            3. np.ndarray: The epochs of the tracked positions. Shape: (E,) int
        """
        import numpy as np
        import torch
        from tools.transforms.geometric.transforms3d import unitquat_to_rotmat
        runner = self.get_runner(-1)
        tracker = runner.tracker
        translation_labels = {
            x.replace("_trans", ""): x for x in tracker.metrics.keys() if "_trans" in x}
        orientation_labels = {
            x.replace("_quat", ""): x for x in tracker.metrics.keys() if "_quat" in x}

        label_pairs = [(k, (v, orientation_labels[k]))
                       for k, v in translation_labels.items() if k in orientation_labels]

        Ts = tracker.metrics[label_pairs[0][1][0]]
        E = len(Ts.values["value"])
        T = Ts.values["value"][0].shape[0]
        positions = torch.eye(4)[None, None, None, ...].repeat(
            (len(label_pairs), E, T, 1, 1)).numpy()

        for i, (k, (t_lab, o_lab)) in enumerate(label_pairs):
            Ts = tracker.metrics[t_lab]
            Os = tracker.metrics[o_lab]
            translations = np.stack(
                Ts.values["value"].values, axis=0)  # E x T x 3
            orientations = np.stack(
                Os.values["value"].values, axis=0)  # E x T x 4
            # Convert to position matricies
            positions[i, :, :, :3, 3] = translations
            positions[i, :, :, :3, :3] = unitquat_to_rotmat(
                torch.from_numpy(orientations))
        labels = np.array([x[0] for x in label_pairs])
        epochs = Ts.values.index.values
        return positions, labels, epochs

    @saveable(
        default_name="positions_surface_over_time",
        default_override=True,
        default_tight_layout=True,
    )
    def plot_positions_as_surface(self,
                                  positions: Optional[np.ndarray] = None,
                                  labels: Optional[np.ndarray] = None,
                                  epochs: Optional[np.ndarray] = None,
                                  ax: Optional[Axes] = None, **kwargs) -> Axes:
        """
        Plots the positions as surface plot.

        Parameters
        ----------
        positions : np.ndarray
            The positions of the objects. Shape: (N, E, T, 4, 4)
            Where N is the number of objects, E is the number of tracked epochs and T is the number of time steps.
            E.g. N is for each object which positions was tracked.
        labels : np.ndarray
            The labels of the objects as tag. Shape: (N,) string
        epochs : np.ndarray
            The epochs of the tracked positions. Shape: (E,) int
        ax : Optional[Axes], optional
            The axis to plot on, by default None

        Returns
        -------
        Axes
            The axis with the plot.
        """
        from tools.util.torch import tensorify
        if positions is None:
            positions, labels, epochs = self.get_positions()
            labels = [x.split("/")[-1].capitalize() for x in labels.tolist()]
            positions = torch.from_numpy(positions)
        if ax is None:
            fig, ax = get_mpl_figure(rows=positions.shape[0], cols=2, subplot_kw=dict(
                projection='3d'), ax_mode="2d")
        else:
            fig = ax.figure

        times = self.get_runner(-1).dataset.frame_timestamps
        coords_x = tensorify(times, dtype=torch.float32)
        coords_y = tensorify(epochs, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(
            coords_x, coords_y, indexing="xy"), dim=-1)

        trans_labels = ["x", "y", "z"]
        orientation_labels = ["a", "b", "c"]
        for row in range(ax.shape[0]):
            # Header
            if labels is not None:
                axw = fig.add_subplot(ax.shape[0], 1, 1 + row, frameon=False)
                axw.set_title(labels[row])
                axw.axis("off")

            translation = positions[row, :, :, :3, 3]
            rotmat = positions[row, :, :, :3, :3]
            euler = unitquat_to_euler(
                "xyz", rotmat_to_unitquat(rotmat), degrees=True)
            # Plot translation in x,y,z as surfaces for each time step and epoch
            x = grid[..., 0]
            y = grid[..., 1]
            if x.shape[-1] == 1:
                # Only one time step, repeat for surface plot
                x = x.repeat(1, 2)
                x[..., 1] = 1
                y = y.repeat(1, 2)

            for i in range(3):
                z_t = translation[..., i]
                z_e = euler[..., i]
                if x.shape[-1] == 1:
                    # Only one time step, repeat for surface plot
                    z_t = z_t.repeat(1, 2)
                    z_e = z_e.repeat(1, 2)

                ax[row, 0].plot_surface(x, y, z_t, label=f"{trans_labels[i]}")
                ax[row, 1].plot_surface(
                    x, y, z_e, label=f"{orientation_labels[i]}")

            ax[row, 0].set_xlabel("Time [t]")
            ax[row, 0].set_ylabel("Epochs")
            ax[row, 0].set_zlabel("Translation [U]")
            ax[row, 0].legend()

            ax[row, 1].set_xlabel("Time [t]")
            ax[row, 1].set_ylabel("Epochs")
            ax[row, 1].set_zlabel("Rotation [°]")
            ax[row, 1].legend()

        return fig

    def create_progress_scene(self, t_index: int = 0):
        positions, labels, epochs = self.get_positions()
        pos_mapping = {k.split("/")[-1]: positions[i]
                       for i, k in enumerate(labels)}
        world = TimedDiscreteSceneNode3D(
            name="world")
        runner = self.get_runner(-1)

        times = torch.linspace(0, 1, len(epochs))
        org_cam = runner.model.camera
        camera = TimedCameraSceneNode3D(
            image_resolution=org_cam._image_resolution.detach().clone().cpu(),
            lens_distortion=org_cam._lens_distortion.detach().clone().cpu(),
            intrinsics=org_cam._intrinsics.detach().clone().cpu(),
            times=times,
            position=torch.from_numpy(
                pos_mapping[org_cam.get_name()][:, t_index]),
            name=org_cam.get_name(),
            index=org_cam.get_index(),
        )
        world.add_scene_children(camera)

        # Add planes
        for plane in runner.model.objects:
            new_plane = TimedPlaneSceneNode3D(
                plane_scale=plane._plane_scale.detach().clone().cpu(),
                plane_scale_offset=plane._plane_scale_offset.detach().clone().cpu(),
                position=torch.from_numpy(
                    pos_mapping[plane.get_name()][:, t_index]),
                name=plane.get_name(),
                index=plane.get_index(),
                times=times,
            )
            world.add_scene_children(new_plane)
        return world

    def load_semantic_labels(self) -> Dict[int, str]:
        import json
        data_dir = self.run_config.data_path
        path = os.path.join(data_dir, "semantic_labels.json")
        if not os.path.exists(path):
            logger.warning(
                f"Semantic labels file {path} not found, no labels available.")
        try:
            with open(path, "r") as f:
                return {int(k): v for k, v in json.load(f).items()}
        except Exception as e:
            logger.warning(
                f"Error while loading semantic labels from {path}: {e}")
        return dict()

    def load_semantic_correspondences(self) -> Dict[int, int]:
        import json
        data_dir = self.run_config.data_path
        path = os.path.join(data_dir, "semantic_correspondence.json")
        if not os.path.exists(path):
            logger.warning(
                f"Semantic correspondence file {path} not found, no labels available.")
        try:
            with open(path, "r") as f:
                return {int(k): int(v) for k, v in json.load(f).items()}
        except Exception as e:
            logger.warning(
                f"Error while loading semantic labels from {path}: {e}")

    def get_semantic_mapping(self, used_ids: List[int]) -> Dict[int, Optional[str]]:
        labels = self.load_semantic_labels()
        correspondences = self.load_semantic_correspondences()
        mapping = dict()
        if correspondences is None:
            return dict()
        corresp = {k: correspondences.get(
            k, None) for k in used_ids} if used_ids is not None else correspondences

        for oid, class_ in corresp.items():
            if class_ in labels:
                mapping[oid] = labels[class_]
            else:
                mapping[oid] = None
        return mapping

    def __getitem__(self, index: int) -> np.ndarray:
        """Gets the result of the given index.

        Parameters
        ----------
        index : int
            The index of the result.

        Returns
        -------
        Any
            The result of the given index.
        """
        eps = sorted(self.index["epoch"].unique())
        epoch_filter = get_circular_index(eps, self.getitem_epoch)
        epoch_filter = eps[epoch_filter]
        kind_filter = self.getitem_kind
        object_filter = self.getitem_object_composition_index

        query = ((self.index["epoch"] == epoch_filter) &
                 (self.index["kind"] == kind_filter) &
                 (self.index["time"] == index))
        if object_filter is not None:
            query &= (self.index["object_composition_index"] == object_filter)
        else:
            query &= (pd.isna(self.index["object_composition_index"]))

        elems = self.index[query]
        if len(elems) > 1:
            raise ValueError(
                f"Multiple results found for epoch {epoch_filter}, kind {kind_filter}, object {object_filter}, time {index}.")

        if len(elems) == 0:
            raise ValueError(
                f"No result found for epoch {epoch_filter}, kind {kind_filter}, object {object_filter}, time {index}.")

        elem = elems.iloc[0]
        path = os.path.join(self.output_directory, elem["path"])
        return load_image(path)
