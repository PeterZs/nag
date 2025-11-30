
from matplotlib.image import AxesImage
from tools.logger.logging import logger
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from matplotlib.figure import Figure
from tools.error import ArgumentNoneError
from tqdm.auto import tqdm
from nag.analytics.result_model import ResultModel
import pandas as pd
from tools.util.path_tools import numerated_file_name, open_folder
from datetime import datetime
from tools.viz.matplotlib import saveable, plot_as_image, get_mpl_figure, render_text
from tools.context.temporary_property import TemporaryProperty
from tools.serialization.json_convertible import JsonConvertible
from tools.agent.util.tracker import Tracker
from tools.util.path_tools import read_directory, format_os_independent
from enum import Enum
from tools.metric.metric_summary import MetricSummary
import matplotlib.pyplot as plt

from pandas.io.formats.style import Styler


class MetricReference(Enum):
    BEST = "best"
    LAST = "last"
    ALL = "all"


class MetricMode(Enum):
    MIN = "min"
    MAX = "max"


def convert_metric_reference(value: Optional[Literal['best', 'last']]) -> MetricReference:
    if value is None:
        return MetricReference.BEST
    if isinstance(value, MetricReference):
        return value
    if isinstance(value, str):
        value = MetricReference(value)
    if isinstance(value, list):
        value = [convert_metric_reference(v) for v in value]
    return value


def convert_metric_mode(value: Optional[Literal['min', 'max']]) -> MetricMode:
    if value is None:
        return MetricMode.MIN
    if isinstance(value, str):
        value = MetricMode(value)
    if isinstance(value, list):
        value = [convert_metric_mode(v) for v in value]
    return value


class ResultComparison():

    models: List[ResultModel]

    numbering: bool
    """If results should be numbered."""

    def __init__(self, models: List[ResultModel], output_folder: str = None, numbering: bool = True):
        if models is None:
            raise ArgumentNoneError("models")
        self.models = models
        self.numbering = numbering
        if output_folder is None:
            output_folder = f"outputs/comparison/{datetime.now().strftime('%y_%m_%d_%H_%M_%S')}"
            os.makedirs(output_folder, exist_ok=True)
        self.output_folder = output_folder
        self.assign_numbers()

    def open_folder(self):
        """Opens the output folder of the comparison."""
        open_folder(os.path.normpath(self.output_folder))

    @classmethod
    def scan_path(cls, path: str, regex_filter: Optional[str] = None) -> List[str]:
        if regex_filter is not None:
            return [x.get("path") for x in read_directory(path, regex_filter)]
        ret = []
        for x in os.listdir(path):
            full_path = os.path.join(path, x)
            if os.path.isdir(full_path):
                ret.append(format_os_independent(full_path))
        return ret

    @classmethod
    def from_parent_path(cls, path: str, regex_filter: Optional[str] = None) -> "ResultComparison":
        paths = cls.scan_path(path, regex_filter)
        models = [ResultModel.from_path(x) for x in paths]
        return cls(models)

    @classmethod
    def from_paths(cls, paths: List[str]) -> "ResultComparison":
        models = [ResultModel.from_path(x) for x in paths]
        return cls(models)

    @property
    def numbering(self) -> bool:
        return self.models[0].numbering

    @numbering.setter
    def numbering(self, value: bool):
        max_num = max([x.number for x in self.models])
        num_format = f"{{:0{len(str(max_num))}d}}"
        return num_format

    def get_save_path(self, path_to_file: str, override: bool = False) -> str:
        """Returns a save path within the output directory.
        If override is false it will check for existing files an rename the path eventually.

        Parameters
        ----------
        path_to_file : str
            A subpath within the folder.
        override : bool, optional
            If a file should be overriden, by default False

        Returns
        -------
        str
            The existing path.
        """
        path = os.path.join(self.output_folder, path_to_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not override and not os.path.isdir(path):
            path = numerated_file_name(path)
        return path

    def assign_numbers(self, force: bool = False):
        if force:
            for i in range(0, len(self.models)):
                self.models[i].config.number = i + 1
                self.models[i].save_config()
        else:
            existing = [
                model.config.number for model in self.models if model.config.number is not None]
            free_numbers = [i for i in range(
                1, len(self.models) + 1) if i not in existing]
            for model in self.models:
                if model.config.number is None:
                    next_num = free_numbers[0]
                    model.config.number = next_num
                    model.save_config()
                    free_numbers.remove(next_num)
                elif model.config.number not in existing:
                    # Duplicate, reassign number
                    next_num = free_numbers[0]
                    model.config.number = next_num
                    model.save_config()
                    free_numbers.remove(next_num)
                else:
                    # Remove from existing
                    existing.remove(model.config.number)

    def get_number_format(self) -> Optional[str]:
        num_format = None
        if self.numbering:
            max_num = max([x.config.number for x in self.models])
            num_format = f"{{:0{len(str(max_num))}d}}"
        return num_format

    @saveable()
    def plot_visual_keys(self,
                         index: int,
                         keys: List[str],
                         how: Optional[Dict[str, Callable[[Any],
                                                          Union[Any, Dict[str, Any]]]]] = None,
                         render_info: bool = True,
                         size: float = 5,
                         tight: bool = False,
                         ) -> Figure:
        if how is None:
            how = dict()
        # Query all models for the keys
        data_dict = {i: dict() for i in range(len(self.models))}
        total_keys = set()

        ratio_img = None

        for i, model in enumerate(self.models):
            non_existing_keys = set(keys) - set(model.get_index_keys())
            existing_keys = set(keys) - non_existing_keys
            with TemporaryProperty(model, get_item_keys=existing_keys, get_item_return_format="dict"):
                value = model[index]
                res_value = dict()
                for k, v in value.items():
                    if k in how:
                        out = how[k](v)
                        if isinstance(out, dict):
                            for k2, v2 in out.items():
                                res_value[k2] = v2
                                total_keys.add(k2)
                        else:
                            res_value[k] = out
                            total_keys.add(k)
                    else:
                        res_value[k] = v
                        total_keys.add(k)
                if ratio_img is None:
                    ratio_img = next(iter(res_value.values()))
                data_dict[i] = res_value

        keys_index = {x: i for i, x in enumerate(keys)}
        total_keys = sorted(
            list(total_keys), key=lambda x: keys_index.get(x, len(total_keys)))

        rows = len(self.models)
        cols = len(total_keys)
        if render_info:
            cols += 1
        fig, axes = get_mpl_figure(
            rows=rows, cols=cols, ratio_or_img=ratio_img, ax_mode="2d", tight=tight, size=size)

        col_offset = 0
        if render_info:
            col_offset = 1

        for row in range(rows):
            # Render info if enabled
            if render_info:
                text = f"{self.models[row].config.number+1}. Model Name:\n{self.models[row].parsed_name}\n\n"
                obj = self.models[row].run_config.diff_config
                diff_config = JsonConvertible.convert_to_yaml_str(
                    obj, toplevel_wrapping=False)
                text += f"Diff Config:\n\n{diff_config}\n"
                wrap_width = ratio_img.shape[1]
                render_text(text, axes[row, 0], width=wrap_width)

            for col, key in enumerate(total_keys):
                col_idx = col + col_offset
                if key in data_dict[row]:
                    plot_as_image(
                        data_dict[row][key], axes=axes[row, col_idx], variable_name=key)
        fig.tight_layout()

        return fig

    def get_trackers(self) -> Dict[str, Tracker]:
        def get_tracker(model) -> Tracker:
            try:
                return model.get_tracker(-1)
            except IndexError as err:
                logger.warning(f"Model: {model.name} dont have a checkpoint!")
        trackers = {model.name: get_tracker(model) for model in tqdm(
            self.models, delay=2, desc="Loading Trackers")}

        return trackers

    def get_tracked_metrics(self) -> Dict[str, List[str]]:
        trackers = self.get_trackers()
        metrics = dict()
        for name, tracker in trackers.items():
            for metric, summary in tracker.metrics.items():
                has_metric = None
                if metric not in metrics:
                    has_metric = list()
                    metrics[metric] = has_metric
                else:
                    has_metric = metrics[metric]
                has_metric.append(name)
        return metrics

    def get_metrics(self, metric_name: str) -> Dict[str, MetricSummary]:
        trackers = self.get_trackers()
        metrics = dict()
        for name, tracker in trackers.items():
            if metric_name in tracker.metrics:
                metrics[name] = tracker.metrics[metric_name]
        return metrics

    @saveable()
    def plot_metric(self,
                    metric_name: str,
                    size: float = 5,
                    top_n: int = -1,
                    top_ref: Literal['min', 'max'] = 'min',
                    top_mode: Literal['best', 'last'] = 'best',
                    ylim: Tuple[float, float] = None,
                    xlim: Tuple[float, float] = None,
                    best_marker: bool = False,
                    best_marker_type: Literal['min', 'max'] = 'min',
                    marker_text_yformat: Optional[str] = None,
                    **kwargs) -> AxesImage:
        import matplotlib
        from tools.util.format import destinctive_number_float_format
        metrics = self.get_metrics(metric_name)
        models = {model.name: model for model in self.models}

        nrows = 1
        ncols = 1
        figsize = None
        if isinstance(size, tuple):
            figsize = size
        else:
            figsize = (ncols * size, nrows * size)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

        # Set default cmap
        cmap_name = "tab10"
        if len(metrics) > 10:
            cmap_name = "tab20"
        cmap = matplotlib.colormaps[cmap_name]

        if top_n > 0:
            def _top_key(name, summary: MetricSummary):
                if top_mode == "best":
                    if top_ref == "min":
                        return summary.values['value'].min()
                    elif top_ref == "max":
                        return summary.values['value'].max()
                    else:
                        raise NotImplementedError()
                elif top_mode == "last":
                    return summary.values.iloc[-1]['value']
            metrics = dict(sorted(metrics.items(), key=lambda x: _top_key(
                *x), reverse=(top_ref == "max"))[:top_n])
            metrics = dict(
                sorted(metrics.items(), key=lambda x: models[x[0]].display_name))

        if marker_text_yformat is None:
            def sel_val(arr: np.array):
                if best_marker_type == "min":
                    return arr.min()
                elif best_marker_type == "max":
                    return arr.max()
                else:
                    raise ValueError(
                        f"Unknown best marker type: {best_marker_type}")

            vals = pd.Series([sel_val(x.values["value"].values)
                             for x in metrics.values()])
            marker_text_yformat = destinctive_number_float_format(
                vals, max_decimals=10)

        for i, (model_name, summary) in enumerate(metrics.items()):
            model: ResultModel = models[model_name]
            color = cmap.colors[i % cmap.N]
            try:
                summary.plot(ax=ax, label=model.display_name, color=color, best_marker=best_marker,
                             best_marker_type=best_marker_type, marker_text_yformat=marker_text_yformat, **kwargs)
            except Exception as err:
                logger.exception(
                    f"Could not plot {metric_name} for tracker {model.name}")

        if ylim is not None:
            plt.ylim(*ylim)
        if xlim is not None:
            plt.xlim(*xlim)
        plt.legend()
        plt.grid(axis="y")
        return fig

    @saveable()
    def plot_metric_bar(self,
                        metric_name: str,
                        size: float = 5,
                        top_n: int = -1,
                        top_ref: Literal['min', 'max'] = 'min',
                        top_mode: Literal['best', 'last'] = 'best',
                        order: bool = True,
                        ylim: Tuple[float, float] = None,
                        xlim: Tuple[float, float] = None,
                        **kwargs) -> AxesImage:
        import matplotlib
        metrics = self.get_metrics(metric_name)
        models = {model.name: model for model in self.models}

        nrows = 1
        ncols = 1
        figsize = None
        if isinstance(size, tuple):
            figsize = size
        else:
            figsize = (ncols * size, nrows * size)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

        # Set default cmap
        cmap_name = "tab10"
        if len(metrics) > 10:
            cmap_name = "tab20"
        cmap = matplotlib.colormaps[cmap_name]

        def _top_key(name, summary: MetricSummary):
            if top_mode == "best":
                if top_ref == "min":
                    return summary.values['value'].min()
                elif top_ref == "max":
                    return summary.values['value'].max()
                else:
                    raise NotImplementedError()
            elif top_mode == "last":
                return summary.values.iloc[-1]['value']

        if top_n > 0:
            metrics = dict(sorted(metrics.items(), key=lambda x: _top_key(*x), reverse=(top_ref == "max"))
                           [:top_n])
            if not order:
                metrics = dict(
                    sorted(metrics.items(), key=lambda x: models[x[0]].display_name))
        elif order:
            metrics = dict(sorted(metrics.items(), key=lambda x: _top_key(
                *x), reverse=(top_ref == "max")))

        for i, (model_name, summary) in enumerate(metrics.items()):
            model: ResultModel = models[model_name]
            color = cmap.colors[i % cmap.N]
            try:
                summary.plot_bar(model.display_name, ax=ax,
                                 label=model.display_name, color=color, **kwargs)
            except Exception as err:
                logger.exception(
                    f"Could not plot {metric_name} for tracker {model.name}")

        # ax.set_xticklabels(ax.get_xticklabels(), rotation=315)
        for tick in ax.get_xticklabels():
            tick.set_rotation(315)
            tick.set_ha('left')
            tick.set_va('top')

        if ylim is not None:
            plt.ylim(*ylim)
        if xlim is not None:
            plt.xlim(*xlim)
        plt.legend()
        plt.grid(axis="y")
        return fig

    def relative_metric_table(self, reference_run_index: int, **kwargs) -> pd.DataFrame:
        pass

    def _metric_table(self,
                      metric_name: List[str],
                      ref: List[MetricReference],
                      mode: List[MetricMode],
                      add_time: bool = False,
                      ) -> pd.DataFrame:
        df = pd.DataFrame(index=[model.name for model in self.models])
        TIME = "time_step"
        for i, metric in enumerate(metric_name):
            metrics = self.get_metrics(metric)
            _ref = ref[i]
            _mode = mode[i]
            for model_name, summary in metrics.items():
                try:
                    if _ref == MetricReference.BEST:
                        x = None
                        if _mode == MetricMode.MIN:
                            x = summary.values['value'].argmin()
                        elif _mode == MetricMode.MAX:
                            x = summary.values['value'].argmax()
                        df.loc[model_name,
                               metric] = summary.values['value'].iloc[x]
                        if add_time:
                            df.loc[model_name, TIME + "_" + metric] = x
                        else:
                            pass
                            # raise NotImplementedError()
                    elif _ref == MetricReference.LAST:
                        df.loc[model_name,
                               metric] = summary.values.iloc[-1]['value']
                        if add_time:
                            df.loc[model_name, TIME + "_" +
                                   metric] = summary.values.iloc[-1].name
                    elif _ref == MetricReference.ALL:
                        _metric = [metric + "_" + str(x)
                                   for x in summary.values.index]
                        df.loc[model_name,
                               _metric] = summary.values['value'].values
                    else:
                        raise NotImplementedError()
                except Exception as err:
                    logger.warning(
                        f"Could not compare metric {metric} for model {model_name} due to {err}")
        return df

    def _metric_table_formatting(self,
                                 df: pd.DataFrame,
                                 metric_name: List[str],
                                 ref: List[MetricReference],
                                 mode: List[MetricMode],
                                 add_time: bool = False,
                                 max_formatting_decimals: int = 10,
                                 best_format: Literal['bold',
                                                      'underline'] = 'underline',
                                 column_alias: Dict[str, str] = None,
                                 custom_formatting: Optional[Callable[[
                                     pd.DataFrame, Dict[str, Any]], pd.DataFrame]] = None,
                                 arrow_metric_indicator: bool = True,
                                 value_format: Optional[Union[str, List[str]]] = None) -> Styler:
        from tools.util.format import destinctive_number_float_format
        if value_format is None:
            value_format = []
            # Auto determine
            for column in df.columns:
                col = df[column]
                value_format.append(destinctive_number_float_format(
                    col, max_decimals=max_formatting_decimals))
        else:
            if not isinstance(value_format, list):
                value_format = [value_format for _ in metric_name]
        # Mapping names and aliases

        def _split_number_text(x):
            return x.split(".", maxsplit=1)[1].strip() if "." in x else x

        def _split_number_number(x):
            return int(x.split(".", maxsplit=1)[0])

        names = {x.name: _split_number_text(
            x.display_name) for x in self.models}
        numbers = {x.name: str(_split_number_number(
            x.display_name)) + "." for x in self.models}
        ser = pd.Series(numbers)
        ser.name = "number"
        df = pd.concat([df, ser], axis=1)

        if arrow_metric_indicator:
            def _mode_to_arrow(mode: MetricMode):
                if mode == MetricMode.MIN:
                    return '↓'
                elif mode == MetricMode.MAX:
                    return '↑'
                else:
                    return ''
            arrows = {metric_name[i]: _mode_to_arrow(
                mode[i]) for i in range(len(metric_name))}
            for k, v in arrows.items():
                column_alias[k] = f"{column_alias.get(k, k)} {v}".strip()
        df.index.name = "model"
        df.rename(names, inplace=True, axis=0)
        df.reset_index(inplace=True)
        df.rename(column_alias, inplace=True, axis=1)
        cols = [column_alias.get('number', 'number'), column_alias.get(
            'model', 'model')] + list(df.columns[1: -1])
        df = df[cols]

        best_val = dict()
        for metric, alias in [(k, v) for k, v in column_alias.items() if k in metric_name]:
            val = df[alias].min() if mode[metric_name.index(
                metric)] == MetricMode.MIN else df[alias].max()
            entries = df[alias] == val
            best_val[alias] = entries

        # Column best
        def _best_mark(x):
            try:
                if x.name == "index":
                    return pd.Series(np.array([''] * len(x)))
                ret = []
                for k, v in x.items():
                    if k in best_val:
                        _v = best_val[k]
                        if _v[x.name]:
                            if best_format == 'underline':
                                ret.append('text-decoration: underline;')
                            elif best_format == 'bold':
                                ret.append('font-weight: bold;')
                            else:
                                raise NotImplementedError()
                        else:
                            ret.append('')
                    else:
                        ret.append('')
                return ret
            except Exception as err:
                print(err)
            return None

        if custom_formatting is not None:
            df = custom_formatting(df)

        formats = {column_alias[metric_name[i]]: value_format[i]
                   for i in range(len(metric_name))}
        style = df.style.apply(_best_mark, axis=1)
        st = style.format(formats, na_rep="-")
        return st

    def metric_table(self,
                     metric_name: Union[str, List],
                     ref: Union[MetricReference,
                                List[MetricReference]] = MetricReference.BEST,
                     mode: Union[Literal['min', 'max'], List] = 'min',
                     add_time: bool = False,
                     formatting: bool = False,
                     max_formatting_decimals: int = 10,
                     best_format: Literal['bold', 'underline'] = 'underline',
                     column_alias: Optional[Dict[str, str]] = None,
                     custom_formatting: Optional[Callable[[
                         pd.DataFrame, Dict[str, Any]], pd.DataFrame]] = None,
                     arrow_metric_indicator: bool = True,
                     value_format: Optional[Union[str, List[str]]] = None,
                     ) -> pd.DataFrame:
        if column_alias is None:
            column_alias = dict()
        if isinstance(metric_name, str):
            metric_name = [metric_name]
        ref = convert_metric_reference(ref)
        mode = convert_metric_mode(mode)
        if not isinstance(ref, list):
            ref = [ref for _ in metric_name]
        if not isinstance(mode, list):
            mode = [mode for _ in metric_name]

        df = self._metric_table(metric_name=metric_name,
                                ref=ref, mode=mode, add_time=add_time)
        if formatting:
            return self._metric_table_formatting(df=df,
                                                 metric_name=metric_name,
                                                 ref=ref,
                                                 mode=mode,
                                                 add_time=add_time,
                                                 max_formatting_decimals=max_formatting_decimals,
                                                 best_format=best_format,
                                                 column_alias=column_alias,
                                                 custom_formatting=custom_formatting,
                                                 arrow_metric_indicator=arrow_metric_indicator,
                                                 value_format=value_format)
        return df

    def apply(self, fnc: Callable[[ResultModel], Dict[str, Any]]) -> pd.DataFrame:
        """
        Applies a function to each model and returns a dataframe with the results.

        Parameters
        ----------
        fnc : Callable[[ResultModel], Dict[str, Any]]
            Function to apply to each model.
            Gets a model as input and returns a dictionary with the results.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the results.
            Columns are the keys of the dictionary and the rows are the models.
            The key 'model' contains the name of the model. When numbering is enabled the key 'index' contains the number of the model.
        """
        df = pd.DataFrame()
        for model in self.models:
            data = fnc(model)
            data['model'] = model.name
            if self.numbering:
                data['index'] = model.config.number

            row = pd.Series(data)
            df = pd.concat([df, row.to_frame().T], axis=0)
        if self.numbering:
            df = df.set_index('index')
        return df

    def get_metric_df(self, metric_name: str) -> pd.DataFrame:
        """Gets a wide-form dataframe of the metric values,
        whereby each models metrics goes into a column defined by its name.


        Parameters
        ----------
        metric_name : str
            Name of metric to get a df for.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the metric values.
        """
        import pandas as pd
        metrics = self.get_metrics(metric_name)
        models = {model.name: model for model in self.models}
        df = pd.DataFrame(columns=['global_step'] + list(metrics.keys()))

        for model_name, summary in metrics.items():
            frame = summary.values[['value', 'global_step']]
            existing = frame['global_step'].isin(df['global_step'])
            df.loc[df['global_step'].isin(
                frame['global_step']), model_name] = frame[existing]['value']
            non_existing = frame[~existing]
            non_existing = non_existing.rename(dict(value=model_name), axis=1)
            df = pd.concat([df, non_existing])
        df = df.sort_values("global_step")
        return df
