from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from tools.serialization.json_convertible import JsonConvertible

from nag.homography.homography_finder import HomographyFinder, HomographyFinderConfig
from nag.homography.sequence_homography_finder import SequenceHomographyFinder, SequenceHomographyFinderConfig
from nag.homography.loftr_match_finder import LoftrMatchFinder, LoftrFinderConfig
from tools.util.format import parse_type
import cv2
import numpy as np
import torch
from tools.transforms.to_tensor_image import ToTensorImage
from tools.util.typing import VEC_TYPE
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims, tensorify
from tools.viz.matplotlib import get_mpl_figure, plot_as_image, saveable, plot_mask
import pandas as pd
from tools.util.path_tools import read_directory, numerated_file_name, process_path
from tools.util.numpy import numpyify, index_of_first as np_index_of_first
from tools.util.torch import index_of_first
from os import PathLike
from tools.util.progress_factory import ProgressFactory
from tools.logger.logging import logger
from tools.util.container import Container
from tools.util.torch import cummatmul
from tools.util.reflection import class_name
from tools.util.diff import dict_diff


@dataclass
class CachedVideoHomographyFinderConfig():

    cache_directory: PathLike
    """Directory where the homography cache entries are stored."""

    homography_finder_type: Union[str, Type[SequenceHomographyFinder]] = field(
        default=SequenceHomographyFinder)
    """Type of the homography finder used to calculate the homographies."""

    homography_finder_config: SequenceHomographyFinderConfig = field(
        default_factory=SequenceHomographyFinderConfig)
    """Configuration of the homography finder used to calculate the homographies."""

    def __post_init__(self):
        self.cache_directory = process_path(
            self.cache_directory, need_exist=False, make_exist=True, allow_none=False)
        self.homography_finder_type: Type[SequenceHomographyFinder] = parse_type(
            self.homography_finder_type, SequenceHomographyFinder, default_value=SequenceHomographyFinder)
        homography_config: Type[SequenceHomographyFinder] = self.homography_finder_type.config_type(
        )
        if isinstance(self.homography_finder_config, dict):
            self.homography_finder_config = homography_config.from_object_dict(
                self.homography_finder_config, force_cls=True)
        elif isinstance(self.homography_finder_config, SequenceHomographyFinderConfig):
            pass
        elif not isinstance(self.homography_finder_config, homography_config):
            raise ValueError(
                f"homography_finder_config should be of type {homography_config}, not {type(self.homography_finder_config)}")
        else:
            raise ValueError(
                f"homography_finder_config should be of type {homography_config}, not {type(self.homography_finder_config)} or dict")


@dataclass()
class HomographyCacheEntry(JsonConvertible):

    num_objects: int
    """Number of objects in the homography matrix."""

    num_homographies: int
    """Number of homographies in the cache entry, corresponding to the number of frames -1."""

    frame_indices: List[int]
    """Indices of the frames for which the homographies are calculated.
    These are the source frames for the homographies.
    """

    target_frame_indices: List[int]
    """Indices of the frames for which the homographies are calculated.
    These are the target frames for the homographies."""

    object_indices: List[int]
    """Indices of the objects for which the homographies are calculated."""

    homography_finder_type: Type[HomographyFinder]
    """Type of the homography finder used to calculate the homographies."""

    homography_finder_config: HomographyFinderConfig
    """Configuration of the homography finder used to calculate the homographies."""

    homography_path: str
    """Path to the file where the homographies are stored."""


class CachedVideoHomographyFinder(SequenceHomographyFinder):

    def __init__(self,
                 config: CachedVideoHomographyFinderConfig,
                 progress_bar: bool = True,
                 progress_factory: Optional[ProgressFactory] = None,
                 **kwargs) -> None:
        self.config = config
        self.index = pd.DataFrame(columns=[])
        self.tensorify_mask = ToTensorImage(output_dtype=torch.bool)
        self.tensorify_image = ToTensorImage(output_dtype=torch.float32)
        self.load_cache()
        self.progress_bar = progress_bar
        self.progress_factory = progress_factory if (
            progress_factory is not None or not progress_bar) else ProgressFactory()
        self.finder = self.config.homography_finder_type(
            self.config.homography_finder_config, progress_bar=progress_bar, progress_factory=self.progress_factory)

    def load_cache(self):
        pattern = r"homography_cache_.+\.yaml"
        files = read_directory(self.config.cache_directory, pattern=pattern)
        df = pd.DataFrame(files, columns=['path'])
        df['config'] = df['path'].apply(
            lambda x: HomographyCacheEntry.load_from_file(x))
        df['frame_indices'] = df['config'].apply(
            lambda x: np.array(x.frame_indices))
        df['target_frame_indices'] = df['config'].apply(
            lambda x: np.array(x.target_frame_indices))
        df['object_indices'] = df['config'].apply(
            lambda x: np.array(x.object_indices))
        df['num_homographies'] = df['frame_indices'].apply(lambda x: len(x))
        df['num_objects'] = df['object_indices'].apply(lambda x: len(x))
        df['homography_finder_type'] = df['config'].apply(
            lambda x: x.homography_finder_type)
        df['frame_indices_mapping'] = df.apply(lambda x: self._compute_mapping(
            x['frame_indices'], x['target_frame_indices']), axis=1)

        self.index = df

    def _compute_mapping(self, frame_indices: np.ndarray, target_frame_indices: np.ndarray) -> np.ndarray:
        arr = np.concatenate([frame_indices[..., None].astype(
            str), target_frame_indices[..., None].astype(str)], axis=-1)
        mapping = np.array(['-'.join(v) for v in arr])
        return mapping

    def save_cache(self,
                   finder: HomographyFinder,
                   finder_config: HomographyFinderConfig,
                   frame_indices: VEC_TYPE,
                   target_frame_indices: VEC_TYPE,
                   object_indices: VEC_TYPE,
                   homographies: np.ndarray
                   ):
        B, C, _, _ = homographies.shape
        frame_indices = numpyify(frame_indices)
        target_frame_indices = numpyify(target_frame_indices)
        object_indices = numpyify(object_indices)
        desc = f"nhom_{B}_o_{C}_{type(finder).__name__}_{type(finder_config).__name__}"
        filename_config = f"homography_cache_{desc}.yaml"
        filename_arr = f"homography_cache_{desc}.npy"
        filename_arr_path = numerated_file_name(
            os.path.join(self.config.cache_directory, filename_arr))
        filename_config_path = numerated_file_name(
            os.path.join(self.config.cache_directory, filename_config))
        cache_entry = HomographyCacheEntry(num_homographies=B,
                                           num_objects=C,
                                           frame_indices=frame_indices.tolist(),
                                           target_frame_indices=target_frame_indices.tolist(),
                                           object_indices=object_indices.tolist(),
                                           homography_finder_type=type(finder),
                                           homography_finder_config=finder_config, homography_path=filename_arr_path)
        cache_entry.save_to_file(filename_config_path, override=True)
        np.save(filename_arr_path, homographies)

        row = pd.Series(
            dict(
                path=filename_config_path,
                frame_indices=frame_indices.tolist(),
                object_indices=object_indices.tolist(),
                config=cache_entry,
                num_homographies=B,
                num_objects=C,
                homography_finder_type=type(finder),
            )
        )

        index = pd.concat([self.index, row.to_frame().T], ignore_index=True)
        self.index = index

    def check_cache(self,
                    frame_indices: VEC_TYPE,
                    target_frame_indices: VEC_TYPE,
                    object_indices: VEC_TYPE,
                    ) -> Optional[np.ndarray]:
        frame_indices = numpyify(frame_indices)
        target_frame_indices = numpyify(target_frame_indices)
        object_indices = numpyify(object_indices)
        df = self.index
        # Filter first by number of images and objects
        df = df[((df['num_homographies'] >= len(frame_indices))
                 & (df['num_objects'] >= len(object_indices))
                 )]
        # Return None if no cache entry is found
        if df.empty:
            return None
        # Filter by matching config and type
        current_type = self.config.homography_finder_type
        current_config = self.config.homography_finder_config

        df = df[(df['homography_finder_type'].apply(lambda x: class_name(x)) == class_name(current_type))
                & (df['config'].apply(lambda x: x.homography_finder_config == current_config))
                ]
        if df.empty:
            return None

        needed_mappings = self._compute_mapping(
            frame_indices, target_frame_indices)

        # Filter by matching frame and object indices
        df = df[(df['frame_indices_mapping'].apply(lambda x: np.isin(needed_mappings, x).all()))
                & (df['object_indices'].apply(lambda x: np.isin(object_indices, x).all()))
                ]
        if df.empty:
            return None

        # Return the first entry
        entry = df.iloc[0]
        existing_frames_mapping = entry['frame_indices_mapping']
        existing_objects = entry['object_indices']

        # Get mask for the the needed frames and objects
        needed_frames_mask = np.isin(existing_frames_mapping, needed_mappings)
        needed_objects_mask = np.isin(existing_objects, object_indices)

        # Load the homographies
        homographies = np.load(entry['config'].homography_path)

        # Get the indices of the needed frames and objects
        frame_index_in_homography = np_index_of_first(
            needed_mappings, existing_frames_mapping)
        object_index_in_homography = index_of_first(torch.tensor(
            object_indices), torch.tensor(existing_objects)).numpy()

        if any(frame_index_in_homography == -1) or any(object_index_in_homography == -1):
            # Should not happen
            raise ValueError("Some indices were not found in the cache")

        out_homographies = np.zeros(
            (len(frame_indices), len(object_indices), 3, 3))

        nfidx = frame_index_in_homography
        noidx = object_index_in_homography

        repeated_nfidx = nfidx[None, ...].repeat(len(noidx), 1).reshape(
            nfidx.shape[0] * len(noidx))  # Repeat interleave
        repeated_noidx = noidx[None, ...].repeat(len(nfidx), 0).reshape(
            len(nfidx) * len(noidx))  # Repeat interleave

        out_homographies[repeated_nfidx, repeated_noidx] = homographies[needed_frames_mask][:,
                                                                                            needed_objects_mask].reshape(len(nfidx) * len(noidx), 3, 3)

        logger.info(f"Loaded homographies from cache: {entry['path']}")
        return out_homographies

    def find_homography(self,
                        image1: VEC_TYPE,
                        image2: VEC_TYPE,
                        mask1: VEC_TYPE,
                        mask2: VEC_TYPE,
                        frame_indices: VEC_TYPE,
                        target_frame_indices: VEC_TYPE,
                        object_indices: VEC_TYPE,
                        used_cache: Optional[Container[bool]] = None,
                        **kwargs
                        ):
        return self.get_or_create_homography(
            image_source=image1,
            image_target=image2,
            mask_source=mask1,
            mask_target=mask2,
            frame_indices=frame_indices,
            target_frame_indices=target_frame_indices,
            object_indices=object_indices,
            used_cache=used_cache,
            **kwargs
        )

    def find_cumulative_homographies(self,
                                     images: VEC_TYPE,
                                     masks: VEC_TYPE,
                                     frame_indices: VEC_TYPE,
                                     object_indices: VEC_TYPE,
                                     ) -> np.ndarray:

        img_source = images[:-1]
        img_target = images[1:]
        mask_source = masks[:-1]
        mask_target = masks[1:]
        fi = frame_indices[:-1]
        target_frame_indices = frame_indices[1:]

        used_cache = Container(False)

        homo = self.find_homography(image1=img_source,
                                    image2=img_target,
                                    mask1=mask_source,
                                    mask2=mask_target,
                                    frame_indices=fi,
                                    target_frame_indices=target_frame_indices,
                                    object_indices=object_indices,
                                    used_cache=used_cache
                                    )

        homo = tensorify(homo)
        target = torch.zeros_like(homo)
        for i in range(homo.shape[1]):
            target[:, i] = cummatmul(homo[:, i])

        if not used_cache.value and self.config.homography_finder_config.plot_visualizations and self.config.homography_finder_config.plot_visualization_path is not None:
            import matplotlib.pyplot as plt
            with plt.ioff():
                self.plot_sequence_homography_warps(images, masks=masks, homographies=homo,
                                                    frame_indices=frame_indices,
                                                    object_indices=object_indices, save_directory=self.config.homography_finder_config.plot_visualization_path)
        return target

    def get_or_create_homography(
            self,
            image_source: VEC_TYPE,
            image_target: VEC_TYPE,
            mask_source: VEC_TYPE,
            mask_target: VEC_TYPE,
            frame_indices: VEC_TYPE,
            target_frame_indices: VEC_TYPE,
            object_indices: VEC_TYPE,
            used_cache: Optional[Container[bool]] = None,
            **kwargs
    ):
        frame_indices = flatten_batch_dims(tensorify(frame_indices), -1)[0]
        target_frame_indices = flatten_batch_dims(
            tensorify(target_frame_indices), -1)[0]

        if len(frame_indices) != len(target_frame_indices):
            raise ValueError(
                "frame_indices and target_frame_indices should have the same length")

        homographies = self.check_cache(frame_indices,
                                        target_frame_indices,
                                        object_indices)
        if homographies is not None:
            if used_cache is not None:
                used_cache.value = True
            return homographies
        else:
            if used_cache is not None:
                used_cache.value = False

        # Query finder for every continuous sequence of frames
        # and every object
        image1 = image_source
        image2 = image_target
        mask1 = mask_source
        mask2 = mask_target

        mapping = [str(s.item()) + " -> " + str(t.item())
                   for s, t in zip(frame_indices, target_frame_indices)]
        logger.debug(
            f"Calculating homographies for objects {object_indices} and frames {mapping}")

        homographies = self.finder(image1, image2, mask1, mask2)
        self.save_cache(self.finder,
                        self.config.homography_finder_config,
                        frame_indices=frame_indices,
                        target_frame_indices=target_frame_indices,
                        object_indices=object_indices,
                        homographies=homographies)
        return homographies

    def __call__(self,
                 image1: VEC_TYPE,
                 image2: VEC_TYPE,
                 mask1: VEC_TYPE,
                 mask2: VEC_TYPE,
                 frame_indices: VEC_TYPE,
                 target_frame_indices: VEC_TYPE,
                 object_indices: VEC_TYPE,
                 **kwargs
                 ):
        return self.get_or_create_homography(image_source=image1, image_target=image2, mask_source=mask1,
                                             mask_target=mask2,
                                             frame_indices=frame_indices,
                                             target_frame_indices=target_frame_indices,
                                             object_indices=object_indices,
                                             **kwargs)
