from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Type
import numpy as np
from tools.serialization.json_convertible import JsonConvertible
from tools.util.typing import VEC_TYPE
from abc import ABC, abstractmethod
from tools.viz.matplotlib import plot_as_image, get_mpl_figure, saveable
import matplotlib.pyplot as plt
import torch


class MatchFindingError(Exception):
    """Exception raised for errors in the match finding process."""
    pass

@dataclass
class MatchFinderConfig(JsonConvertible):
    """Configuration for match finder."""
    pass


class MatchFinder:
    """Base class for match finder.
    Extracts keypoints from two images and finds matching keypoints between them."""

    def __init__(self, config: MatchFinderConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config    

    @classmethod
    @abstractmethod
    def config_type(cls) -> Type[MatchFinderConfig]:
        """Get the configuration class for this match finder."""
        raise NotImplementedError

    @abstractmethod
    def find_matches(self, image1: VEC_TYPE, image2: VEC_TYPE) -> Tuple[VEC_TYPE, VEC_TYPE, Dict[str, Any]]:
        """Find matching keypoints between two images.

        Parameters
        ----------
        image1 : VEC_TYPE
            Image 1, also referred as the source image.
            If numpy array, the shape should be ([B,] H, W, C) where H is the height, W is the width and C is the number of channels.
            If tensor, the shape should be ([B,] C, H, W) where C is the number of channels, H is the height and W is the width.

        image2 : VEC_TYPE
            Image 2, also referred as the target image.
            If numpy array, the shape should be ([B,] H, W, C) where H is the height, W is the width and C is the number of channels.
            If tensor, the shape should be ([B,] C, H, W) where C is the number of channels, H is the height and W is the width.

        Returns
        -------
        Tuple[VEC_TYPE, VEC_TYPE, Dict[str, Any]]
            1. keypoints from image1 Shape: (N, 3) (B_idx, x, y)
            2. keypoints from image2 Shape: (N, 3) (B_idx, x, y)
        """
        pass

    def __call__(self, image1: VEC_TYPE, image2: VEC_TYPE, **kwargs) -> Tuple[VEC_TYPE, VEC_TYPE]:
        return self.find_matches(image1, image2, **kwargs)
    
    def plot_keypoints(
            self,
            keypoints1: VEC_TYPE,
            keypoints2: VEC_TYPE,
            axes: np.ndarray,
    ):
        from matplotlib.patches import ConnectionPatch
        cmap = plt.get_cmap('gist_rainbow') 
        irange = torch.arange(keypoints1.shape[0]) * (cmap.N // keypoints1.shape[0])
        colors = irange % cmap.N
        colors = [cmap(c) for c in colors]
        axes[0].scatter(keypoints1[:, 0], keypoints1[:, 1], c=colors)
        axes[1].scatter(keypoints2[:, 0], keypoints2[:, 1], c=colors)

        for i in range(keypoints1.shape[0]):
            con = ConnectionPatch(xyA=keypoints1[i], xyB=keypoints2[i], coordsA="data", coordsB="data",
                        axesA=axes[0], axesB=axes[1], color=colors[i])
            axes[1].add_artist(con)

    @saveable()
    def plot_matches(self, 
                     image1: VEC_TYPE, 
                     image2: VEC_TYPE, 
                     keypoints1: VEC_TYPE,
                     keypoints2: VEC_TYPE,
                     tight: bool = False,
                     **kwargs) -> None:
        """Plot the matches between two images.

        Parameters
        ----------
        image1 : VEC_TYPE
            Image 1, also referred as the source image.
            If numpy array, the shape should be (H, W, C) where H is the height, W is the width and C is the number of channels.
            If tensor, the shape should be (C, H, W) where C is the number of channels, H is the height and W is the width.

        image2 : VEC_TYPE
            Image 2, also referred as the target image.
            If numpy array, the shape should be (H, W, C) where H is the height, W is the width and C is the number of channels.
            If tensor, the shape should be (C, H, W) where C is the number of channels, H is the height and W is the width.

        keypoints1 : VEC_TYPE
            Keypoints from image1. Shape: (N, 2) (x, y)

        keypoints2 : VEC_TYPE
            Keypoints from image2. Shape: (N, 2) (x, y)

        """
        from matplotlib.patches import ConnectionPatch
        fig, axes = get_mpl_figure(1, 2, 
                                   ratio_or_img=image1,
                                    tight=tight,
                                   **kwargs)
        
        plot_as_image(image1, axes=axes[0], variable_name="Image 1")
        plot_as_image(image2, axes=axes[1], variable_name="Image 2")

        self.plot_keypoints(keypoints1, keypoints2, axes)

        return fig

