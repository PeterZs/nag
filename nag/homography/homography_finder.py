from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from tools.serialization.json_convertible import JsonConvertible

from nag.homography.match_finder import MatchFinder, MatchFinderConfig, MatchFindingError
from nag.homography.loftr_match_finder import LoftrMatchFinder, LoftrFinderConfig
from nag.homography.sift_match_finder import SiftFinderConfig, SiftMatchFinder
from tools.util.format import parse_type
import cv2
import numpy as np
import torch
from tools.transforms.to_tensor_image import ToTensorImage
from tools.util.typing import VEC_TYPE
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims, tensorify
from tools.viz.matplotlib import get_mpl_figure, plot_as_image, saveable, plot_mask
from tools.util.progress_factory import ProgressFactory
from tools.util.format import parse_enum


def is_coord_in_mask(coord: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Check if a coordinate is in a mask.

    Parameters
    ----------
    coord : torch.Tensor
        Coordinates to check.
        Shape: (N, C) C is the number of dimensions within the mask in the same order.

    mask : torch.Tensor
        Mask to check against.
        Shape: (C1, C2, ..., CN)

    Returns
    -------
    torch.Tensor
        Boolean tensor indicating if the coordinate is in the mask.
        Shape (N,)
    """
    ret = torch.ones(coord.shape[0], dtype=torch.bool, device=coord.device)
    if coord.numel() == 0:
        # If no coordinates are present, return empty tensor.
        return ret
    mask_shape = mask.shape
    # Check if bound overflows
    for i in range(len(mask_shape)):
        ret = ret & ~(coord[:, i] < 0)
        ret = ret & ~(coord[:, i] >= mask_shape[i])

    chk = list()
    for i in range(len(mask_shape)):
        chk.append(coord[ret, i])

    in_bounds = ret.clone()
    inmask_res = ret[in_bounds] & mask[tuple(chk)]
    ret[in_bounds] = inmask_res
    return ret


class NoMaskFoundError(Exception):
    pass


class HomographyEstimationFunctions(Enum):

    OPEN_CV_FIND_HOMOGRAPHY = "open_cv_find_homography"

    OPEN_CV_FUNDAMENTAL_MATRIX = "open_cv_fundamental_matrix"


@dataclass
class HomographyFinderConfig(JsonConvertible):

    match_finder_type: Union[str, Type[MatchFinder]
                             ] = field(default=SiftMatchFinder)
    """Match finder type to use for finding matches / keypoints between two frames."""

    match_finder_config: Union[Dict[str, Any], MatchFinderConfig] = field(
        default_factory=SiftFinderConfig)
    """Keyword arguments to pass to the match finder constructor."""

    homography_estimation_function: Union[str, HomographyEstimationFunctions] = field(
        default=HomographyEstimationFunctions.OPEN_CV_FIND_HOMOGRAPHY)

    ransac_threshold: float = field(default=None)
    """RANSAC threshold for homography estimation."""

    def __eq__(self, value):
        if not isinstance(value, HomographyFinderConfig):
            return False
        if self.match_finder_type != value.match_finder_type:
            return False
        if self.match_finder_config != value.match_finder_config:
            return False
        if self.ransac_threshold != value.ransac_threshold:
            return False
        return True

    def __post_init__(self):
        self.match_finder_type: Type[MatchFinder] = parse_type(
            self.match_finder_type, MatchFinder, default_value=LoftrMatchFinder)
        matcher_config: Type[MatchFinderConfig] = self.match_finder_type.config_type(
        )
        self.homography_estimation_function = parse_enum(
            HomographyEstimationFunctions, self.homography_estimation_function)
        if isinstance(self.match_finder_config, dict):
            self.match_finder_config = matcher_config.from_object_dict(
                self.match_finder_config, force_cls=True)
        elif isinstance(self.match_finder_config, MatchFinderConfig):
            pass
        elif not isinstance(self.match_finder_config, matcher_config):
            raise ValueError(
                f"match_finder_config should be of type {matcher_config}, not {type(self.match_finder_config)}")
        else:
            raise ValueError(
                f"match_finder_config should be of type {matcher_config}, not {type(self.match_finder_config)} or dict")

    def after_decoding(self, **kwargs):
        super().after_decoding(**kwargs)
        if isinstance(self.match_finder_config, dict):
            matcher_config: Type[MatchFinderConfig] = self.match_finder_type.config_type(
            )
            self.match_finder_config = matcher_config.from_object_dict(
                self.match_finder_config, force_cls=True)


class HomographyFinder():
    """Finds homographies between two images, by identifying matching keypoints between them."""

    _compute_homography: callable
    """Internal function to compute homography between two sets of points."""

    def __init__(self,
                 config: HomographyFinderConfig,
                 progress_bar: bool = False,
                 progress_factory: Optional[ProgressFactory] = None,
                 ) -> None:
        self.config = config
        self.matcher = config.match_finder_type(config.match_finder_config)
        self.tensorify_mask = ToTensorImage(output_dtype=torch.bool)
        self.tensorify_image = ToTensorImage(output_dtype=torch.float32)
        self.progress_bar = progress_bar
        self.progress_factory = progress_factory if (
            progress_factory is not None or not progress_bar) else ProgressFactory()
        self.setup_homography_function()

    def setup_homography_function(self):
        if self.config.homography_estimation_function == HomographyEstimationFunctions.OPEN_CV_FIND_HOMOGRAPHY:
            homog_args = dict()
            if self.config.ransac_threshold is not None:
                homog_args["ransacReprojThreshold"] = self.config.ransac_threshold if self.config.ransac_threshold > 0 else 3.0

            def _find_homography(pts1, pts2, output_mask: Optional[np.ndarray] = None, **kwargs):
                args = dict(homog_args)
                M, mask = cv2.findHomography(
                    pts1, pts2, method=cv2.RANSAC, **args)
                if output_mask is not None:
                    if output_mask.shape != mask.shape[:-1]:
                        raise ValueError(
                            f"Provided output_mask shape {output_mask.shape} does not match the intended shape of {mask.shape[:-1]}.")
                    output_mask[...] = (mask > 0).squeeze(-1)
                return M
            self._compute_homography = _find_homography
        # elif self.config.homography_estimation_function == HomographyEstimationFunctions.OPEN_CV_FUNDAMENTAL_MATRIX:
        #     homog_args = dict()
        #     if self.config.ransac_threshold is not None:
        #         homog_args["ransacReprojThreshold"] = self.config.ransac_threshold
        #     def _find_homography(pts1, pts2, **kwargs):
        #         M, msk = cv2.findFundamentalMat(pts1, pts2, method=cv2.RANSAC, **homog_args)
        #         return M

        #     self._compute_homography = _find_homography
        else:
            raise ValueError(
                f"Unknown homography estimation function: {self.config.homography_estimation_function}")

    @classmethod
    def config_type(cls) -> Type[HomographyFinderConfig]:
        return HomographyFinderConfig

    def find_homography(self,
                        image1: VEC_TYPE,
                        image2: VEC_TYPE,
                        mask1: VEC_TYPE,
                        mask2: VEC_TYPE,
                        **kwargs
                        ) -> np.ndarray:
        """Find homographies between two images, by identifying matching keypoints between them.
        Returns a homography matrix per image in a stack and per object in the image indentified by the mask.

        Parameters
        ----------
        image1 : VEC_TYPE
            Image 1, also referred as the source image.
            If numpy array, the shape should be ([B,] H, W, C) where H is the height, W is the width and C is the number of channels.
            If tensor, the shape should be ([B,] C, H, W) where C is the number of channels, H is the height and W is the width.
            Can be grayscale or color image. If color image, the number of channels should be 3 and the color format should be RGB.

        image2 : VEC_TYPE
            Image 2, also referred as the target image.
            If numpy array, the shape should be ([B,] H, W, C) where H is the height, W is the width and C is the number of channels.
            If tensor, the shape should be ([B,] C, H, W) where C is the number of channels, H is the height and W is the width.
            Can be grayscale or color image. If color image, the number of channels should be 3 and the color format should be RGB.

        mask1 : VEC_TYPE
            The masks for the keypoints in image1. Shape: (B, H, W, C)
            If tensor, the shape should be ([B,] C, H, W) where C is the number of channels, H is the height and W is the width.
            Channels is equivalent to the number of objects in the image, where each channel is a binary mask for the object.

        mask2 : VEC_TYPE
            The masks for the keypoints in image2. Shape: (B, H, W, C)
            If tensor, the shape should be ([B,] C, H, W) where C is the number of channels, H is the height and W is the width.
            Channels is equivalent to the number of objects in the image, where each channel is a binary mask for the object

        Returns
        -------
        np.ndarray
            Homographies between the two images. Shape: (B, C, 3, 3)
            Can be used to convert coordinates from image1 to image2.
            May contain NaN values if the homography could not be estimated.
        """

        # Bp1, Bp2 = self.matcher(image1, image2)

        # mask1 = flatten_batch_dims(self.tensorify_mask(mask1), -4)[0]
        # mask2 = flatten_batch_dims(self.tensorify_mask(mask2), -4)[0]

        # Bp1 = tensorify(Bp1, device="cpu").round().int()
        # Bp2 = tensorify(Bp2, device="cpu").round().int()

        # if Bp1.shape[-1] == 2:
        #     Bp1 = torch.cat([torch.zeros_like(Bp1[:, 0])[:, None], Bp1], dim=-1)
        #     Bp2 = torch.cat([torch.zeros_like(Bp2[:, 0])[:, None], Bp2], dim=-1)

        # Bids = torch.unique(Bp1[:, 0])

        # homog_args = dict()
        # if self.config.ransac_threshold is not None:
        #     homog_args["ransacReprojThreshold"] = self.config.ransac_threshold

        # B, C, H, W = mask1.shape
        # if B != len(Bids):
        #     raise ValueError(f"Number of batches in mask1 should be equal to the number of batches in the keypoints. Got {B} and {len(Bids)}")

        # out_trans = np.eye(3)[None, None, :,:].repeat(B, axis=0).repeat(C, axis=1) # [B, C, 3, 3]
        # for i in range(B):
        #     img_mask = Bp1[:, 0] == i
        #     img_p1 = Bp1[img_mask, 1:]
        #     img_p2 = Bp2[img_mask, 1:]

        #     for obj in range(C):
        #         omask1 = mask1[i, obj]
        #         omask2 = mask2[i, obj]
        #         obj_p1_filter = is_coord_in_mask(img_p1.flip(-1), omask1) # Flip to (y, x) format
        #         obj_p2_filter = is_coord_in_mask(img_p2.flip(-1), omask2) # Flip to (y, x) format
        #         combined_filter = obj_p1_filter & obj_p2_filter
        #         obj_p1 = img_p1[combined_filter].numpy().copy()
        #         obj_p2 = img_p2[combined_filter].numpy().copy()

        #         if len(obj_p1) < 4 or len(obj_p2) < 4:
        #             out_trans[i, obj] = np.nan
        #         else:
        #             M, _ = cv2.findHomography(obj_p1, obj_p2, method=cv2.RANSAC, **homog_args)
        #             out_trans[i, obj] = M
        # return out_trans
        return self.find_homography_mask_wise(image1, image2, mask1=mask1, mask2=mask2)

    def find_homography_mask_wise(self,
                                  image1: VEC_TYPE,
                                  image2: VEC_TYPE,
                                  mask1: VEC_TYPE,
                                  mask2: VEC_TYPE,
                                  return_keypoints: bool = False,
                                  return_loss: bool = False,
                                  return_used_points: bool = False,
                                  **kwargs
                                  ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Find homographies between two images, by identifying matching keypoints between them.
        Returns a homography matrix per image in a stack and per object in the image indentified by the mask.

        Operates mask wise, so each mask in the image is treated as a separate object.
        Masks are used to pre-crop the images to the bounding box of the masks, and then the keypoints are matched.


        Parameters
        ----------
        image1 : VEC_TYPE
            Image 1, also referred as the source image.
            If numpy array, the shape should be ([B,] H, W, C) where H is the height, W is the width and C is the number of channels.
            If tensor, the shape should be ([B,] C, H, W) where C is the number of channels, H is the height and W is the width.
            Can be grayscale or color image. If color image, the number of channels should be 3 and the color format should be RGB.

        image2 : VEC_TYPE
            Image 2, also referred as the target image.
            If numpy array, the shape should be ([B,] H, W, C) where H is the height, W is the width and C is the number of channels.
            If tensor, the shape should be ([B,] C, H, W) where C is the number of channels, H is the height and W is the width.
            Can be grayscale or color image. If color image, the number of channels should be 3 and the color format should be RGB.

        mask1 : VEC_TYPE
            The masks for the keypoints in image1. Shape: (B, H, W, C)
            If tensor, the shape should be ([B,] C, H, W) where C is the number of channels, H is the height and W is the width.
            Channels is equivalent to the number of objects in the image, where each channel is a binary mask for the object.

        mask2 : VEC_TYPE
            The masks for the keypoints in image2. Shape: (B, H, W, C)
            If tensor, the shape should be ([B,] C, H, W) where C is the number of channels, H is the height and W is the width.
            Channels is equivalent to the number of objects in the image, where each channel is a binary mask for the object

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], np.ndarray]]
            1.Homographies between the two images. Shape: (B, C, 3, 3)
              Can be used to convert coordinates from image1 to image2.
              May contain NaN values if the homography could not be estimated.

            2. Dictionary containing the following keys:
                "keypoints": A nested list of tuples containing the keypoints in image1 and image2 for each object. If return_keypoints is True.
                The list is of shape (B, O) where B is the batch size and O is the number of objects in the image, and each element is a tuple of the form (obj_p1, obj_p2). Contains the keypoints in image1 and image2 respectively. as (N, 2) array.
                If no keypoints are found, the tuple is (None, None).
                "loss": A tensor of shape (B, O) containing the L1 loss between the warped image1 and image2. If return_loss is True.
                "used_points": A (B, O) list, of which values are boolean numPy arrays of shape (N,) indicating which points were used in the homography estimation. If return_used_points is True.
        """
        mask1 = flatten_batch_dims(self.tensorify_mask(mask1), -4)[0]
        mask2 = flatten_batch_dims(self.tensorify_mask(mask2), -4)[0]

        image1, shp1 = flatten_batch_dims(self.tensorify_image(image1), -4)
        image2, _ = flatten_batch_dims(self.tensorify_image(image2), -4)

        B, O, H, W = mask1.shape
        _, C, _, _ = image1.shape
        found_keypoints = []
        found_used_points = []

        loss = None
        if return_loss:
            loss = np.zeros((B, O))

        batch_iter = None
        object_iter = None

        if self.progress_bar:
            batch_iter = self.progress_factory.bar(
                total=B, desc="Creating Homographies along batches", is_reusable=True, tag="HomographyFinder_Batch")
        out_trans = np.eye(3)[None, None, :, :].repeat(
            B, axis=0).repeat(O, axis=1)  # [B, C, 3, 3]
        for i in range(B):
            bimg1 = image1[i]
            bimg2 = image2[i]

            if self.progress_bar:
                object_iter = self.progress_factory.bar(
                    total=O, desc="Creating Homographies along objects", is_reusable=True, tag="HomographyFinder_Object")

            ids = list(range(O))

            object_keypoints = []
            object_used_points = []

            for o in ids:
                omask1 = mask1[i, o]
                omask2 = mask2[i, o]

                try:
                    cimg1, cimg2, cmask1, cmask2, crop1min, crop2min = self.prepare_for_matching(
                        bimg1, bimg2, omask1[None, ...], omask2[None, ...])

                    Bp1, Bp2 = self.matcher(cimg1, cimg2, is_batch_mode=True if (
                        (i * O + o) < (B * O - 1)) else False)  # Batch mode only if not the last object
                    Bp1 = tensorify(Bp1, device="cpu")
                    Bp2 = tensorify(Bp2, device="cpu")

                    if Bp1.numel() == 0 or Bp2.numel() == 0:
                        raise MatchFindingError("No keypoints found")

                    if Bp1.shape[-1] == 3:
                        Bp1 = Bp1[:, 1:]
                        Bp2 = Bp2[:, 1:]

                    obj_p1_filter = is_coord_in_mask(
                        Bp1.flip(-1).round().int(), cmask1[0])  # Flip to (y, x) format
                    obj_p2_filter = is_coord_in_mask(
                        Bp2.flip(-1).round().int(), cmask2[0])  # Flip to (y, x) format
                    combined_filter = obj_p1_filter & obj_p2_filter

                    obj_p1 = Bp1[combined_filter]
                    obj_p2 = Bp2[combined_filter]

                    # self.matcher.plot_matches(cimg1, cimg2, obj_p1, obj_p2) # Debugging

                    # Need to shift the coordinates back to the original image, as the keypoints are in the cropped image
                    # Flip to (x, y) format
                    obj_p1 = obj_p1 + crop1min[None, :].flip(-1)
                    obj_p2 = obj_p2 + crop2min[None, :].flip(-1)

                    obj_p1 = obj_p1.numpy()
                    obj_p2 = obj_p2.numpy()

                    if len(obj_p1) < 4 or len(obj_p2) < 4:
                        out_trans[i, o] = np.nan
                        if return_keypoints:
                            object_keypoints.append((None, None))
                        if return_used_points:
                            object_used_points.append(None)
                    else:
                        if return_keypoints:
                            object_keypoints.append((obj_p1, obj_p2))
                        used_points = np.zeros(obj_p1.shape[0], dtype=bool)
                        M = self._compute_homography(
                            obj_p1, obj_p2, output_mask=used_points)
                        out_trans[i, o] = M
                        if return_used_points:
                            object_used_points.append(used_points)
                    if self.progress_bar:
                        object_iter.update(1)

                except (NoMaskFoundError, MatchFindingError) as err:
                    # If no mask is found, set the homography to NaN
                    out_trans[i, o] = np.nan
                    if return_keypoints:
                        object_keypoints.append((None, None))
                    if return_used_points:
                        object_used_points.append(None)

                if return_keypoints:
                    found_keypoints.append(object_keypoints)
                if return_used_points:
                    found_used_points.append(object_used_points)

            if self.progress_bar:
                batch_iter.update(1)

        if return_loss:
            # Compute L1 Distance of reprojected images
            hom = tensorify(out_trans, dtype=torch.float32)
            hom = torch.inverse(hom)
            img1_for_homography = image1.unsqueeze(
                1).expand(-1, O, -1, -1, -1)  # [B, O, C, H, W]
            mask1_for_homography = mask1.unsqueeze(2)  # [B, O, 1, H, W]
            img1_homog = _apply_homography(img1_for_homography.reshape(
                B * O, C, H, W), hom.reshape(B * O, 3, 3))
            mask1_homog = _apply_homography(mask1_for_homography.reshape(
                B * O, 1, H, W).float(), hom.reshape(B * O, 3, 3)).round().to(torch.bool)

            img2_cmp = image2.unsqueeze(
                1).expand(-1, O, -1, -1, -1).reshape(B * O, C, H, W)  # [B * O, C, H, W]
            mask2_cmp = mask2.unsqueeze(2).reshape(
                B * O, 1, H, W)  # [B * O, 1, H, W]
            loss[...] = masked_l1(img1_homog, img2_cmp,
                                  mask1_homog, mask2_cmp).reshape(B, O)

        ret = dict()
        if return_keypoints:
            ret["keypoints"] = found_keypoints
        if return_used_points:
            ret["used_points"] = found_used_points
        if return_loss:
            ret["loss"] = loss
        if len(ret) > 0:
            return out_trans, ret
        return out_trans

    def prepare_for_matching(self,
                             image1: torch.Tensor,
                             image2: torch.Tensor,
                             mask1: torch.Tensor,
                             mask2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        crop_image_1, crop_image_2, crop_mask_1, crop_mask_2, c1, c2 = crop_mask_image(
            image1, image2, mask1, mask2)
        # Replace colors with black if the mask is not present
        crop_image_1 = crop_image_1 * crop_mask_1
        crop_image_2 = crop_image_2 * crop_mask_2
        return crop_image_1, crop_image_2, crop_mask_1, crop_mask_2, c1, c2

    def __call__(self,
                 image1: VEC_TYPE,
                 image2: VEC_TYPE,
                 mask1: VEC_TYPE,
                 mask2: VEC_TYPE,
                 **kwargs
                 ) -> np.ndarray:
        return self.find_homography_mask_wise(image1, image2, mask1=mask1, mask2=mask2, **kwargs)

    def _plot_diff(self, img1_warped, mask1_warped, img2, mask2, axes):
        if axes is None:
            fig, axes = get_mpl_figure(
                1, 3, ratio_or_img=img1_warped, tight=False, ax_mode="1d")
        else:
            fig = axes[0].figure

        img1_warped_repl = img1_warped.clone()
        img1_warped_repl[:, ~mask1_warped[0]] = 0
        img2_copy = img2.clone()
        img2_copy[:, ~mask2] = 0

        plot_mask(img2, mask2, ax=axes[0], darkening_background=0.4,
                  title="Image 2 (Original)", frame_on=False)
        plot_mask(img1_warped, mask1_warped, ax=axes[1], darkening_background=0.4,
                  title="Image 1 (Warped) - Mask {}".format(i), frame_on=False)

        l1img = (img2_copy - img1_warped_repl).abs()
        mask_diff = mask1_warped[0] | mask2
        loss = l1img[:, mask_diff].mean()
        plot_as_image(
            l1img, axes=axes[2], variable_name=f"L1 Difference {loss:.3f}", frame_on=False)
        return fig

    @saveable()
    def plot_homography_warp(
            self,
            img1: VEC_TYPE,
            img2: VEC_TYPE,
            mask1: VEC_TYPE,
            mask2: VEC_TYPE,
            homography: VEC_TYPE,
            img1_index: Optional[VEC_TYPE] = None,
            img2_index: Optional[VEC_TYPE] = None,
            object_index: Optional[VEC_TYPE] = None,
            keypoints: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Any:
        """Plot the warped image after applying the homography.


        Parameters
        ----------
        img1 : VEC_TYPE
            The source image to warp. Shape: ([..., B,] C, H, W) if tensor, else ([..., B,] H, W, C)
        img2 : VEC_TYPE
            The target image to warp to. Shape: ([..., B,] C, H, W) if tensor, else ([..., B,] H, W, C)
        mask1 : VEC_TYPE
            The mask for the source image. Shape: ([..., B,] O, H, W) if tensor, else ([..., B,] H, W, O)
        mask2 : VEC_TYPE
            The mask for the target image. Shape: ([..., B,] O, H, W) if tensor, else ([..., B,] H, W, O)
        homography : VEC_TYPE
            The homography to apply to the source image. Shape: ([..., B,] O, H, W) if tensor, else ([..., B,] H, W, O)

        img1_index : VEC_TYPE, optional
            The indices of the source images. Shape: ([..., B])

        img2_index : VEC_TYPE, optional
            The indices of the target images. Shape: ([..., B])

        object_index : VEC_TYPE, optional
            The indices / name of the object in the mask. Shape: (O)

        Returns
        -------
        Matplotlib Figure
            Figure with the following subplots:
            1. Image 2 (Original)
            2. Image 1 (Warped)
            3. L1 Difference between the warped image and the original image
        """
        from tools.viz.matplotlib import get_mpl_figure, plot_as_image
        from tools.util.numpy import numpyify, flatten_batch_dims as np_flatten_batch_dims
        mask1, _ = flatten_batch_dims(self.tensorify_mask(mask1).float(), -4)
        mask2, _ = flatten_batch_dims(self.tensorify_mask(mask2), -4)

        homography = flatten_batch_dims(
            tensorify(homography, dtype=torch.float32), -4)[0]

        img1: torch.Tensor = flatten_batch_dims(
            self.tensorify_image(img1), -4)[0]
        img2: torch.Tensor = flatten_batch_dims(
            self.tensorify_image(img2), -4)[0]

        img1_index = np_flatten_batch_dims(numpyify(
            img1_index), -1)[0] if img1_index is not None else np.array([str(i) + "-1" for i in range(img1.shape[0])])
        img2_index = np_flatten_batch_dims(numpyify(
            img2_index), -1)[0] if img2_index is not None else np.array([str(i) + "-2" for i in range(img2.shape[0])])

        object_index = np_flatten_batch_dims(numpyify(object_index), -1)[
            0] if object_index is not None else np.array([str(i) for i in range(mask1.shape[-3])])

        # if len(homography.shape) != 2 or homography.shape[0] != 3 or homography.shape[1] != 3:
        #     raise ValueError("Homography should be of shape (3, 3). Got: " + str(homography.shape))

        B, O, H, W = mask1.shape
        _, C, _, _ = img1.shape

        if len(object_index) != O:
            raise ValueError("Number of object indices should be equal to the number of objects in the mask. Got {} and {}".format(
                len(object_index), O))

        # img_warped = torch.zeros(O, C, H, W, device=img1.device, dtype=img1.dtype)

        BO = B * O
        fig, maxes = get_mpl_figure(
            BO, 3, ratio_or_img=img1, tight=False, ax_mode="2d")

        for b in range(B):
            for i in range(O):
                axes = maxes[b * O + i]
                for a in axes:
                    a.axis("off")
                img1_warped = apply_homography(
                    img1[b], torch.inverse(homography[b, i]))
                mask1_warped = apply_homography(
                    mask1[b, i][None, ...], torch.inverse(homography[b, i])).round().bool()

                img1_warped_repl = img1_warped.clone()
                img1_warped_repl[:, ~mask1_warped[0]] = 0
                img2_copy = img2[b].clone()
                img2_copy[:, ~mask2[b, i]] = 0

                plot_mask(img2[b], mask2[b, i], ax=axes[0], darkening_background=0.4,
                          title=f"Image {img2_index[b]} (Original)", frame_on=False)
                plot_mask(img1_warped, mask1_warped, ax=axes[1], darkening_background=0.4,
                          title=f"Image {img1_index[b]} (Warped) - Mask {object_index[i]}", frame_on=False)

                l1img = (img2_copy - img1_warped_repl).abs()
                mask_diff = mask1_warped[0] | mask2[b, i]
                loss = l1img[:, mask_diff].mean()
                plot_as_image(
                    l1img, axes=axes[2], variable_name=f"L1 Difference {loss:.3f}", frame_on=False)
        return fig


@torch.jit.script
def masked_l1(
        image1: torch.Tensor,
        image2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor) -> torch.Tensor:
    """
    Compute the L1 difference between two images, as a mean on the intersection of the masks.

    Parameters
    ----------
    image1 : torch.Tensor
        The first image. Shape: (B, C, H, W) where C is the number of channels, H is the height and W is the width.
    image2 : torch.Tensor
        The second image. Shape: (B, C, H, W)
    mask1 : torch.Tensor
        The mask for the first image. Shape: (B, 1, H, W)
    mask2 : torch.Tensor
        The mask for the second image. Shape: (B, 1, H, W)

    Returns
    -------
    torch.Tensor
        The L1 difference between the two images. Shape: (B,)
    """
    image1, shp = flatten_batch_dims(image1, -4)
    image2, _ = flatten_batch_dims(image2, -4)
    mask1 = flatten_batch_dims(mask1, -4)[0].to(torch.bool)
    mask2 = flatten_batch_dims(mask2, -4)[0].to(torch.bool)

    B, C, H, W = image1.shape

    diff = torch.zeros(B, device=image1.device, dtype=image1.dtype)
    mask_diff = mask1 | mask2
    for i in range(B):
        l1diff = (image2[i, :, mask_diff[i, 0]] -
                  image1[i, :, mask_diff[i, 0]]).abs()
        diff[i] = l1diff.mean(dim=(-1, -2))
    return unflatten_batch_dims(diff, shp)


@torch.jit.script
def crop_mask_image(image1: torch.Tensor, image2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crops the images to the maximum bounding box of the masks, and returns the cropped images and masks.
    As well as the shift in the coordinates of the cropped images.

    Parameters
    ----------
    image1 : torch.Tensor
        The first image.
        Shape: (C, H, W) where C is the number of channels, H is the height and W is the width.
    image2 : torch.Tensor
        The second image.
        Shape: (C, H, W)
    mask1 : torch.Tensor
        The mask for the first image.
        Shape: (1, H, W)

    mask2 : torch.Tensor
        The mask for the second image.
        Shape: (1, H, W)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        1. Cropped image 1. Shape: (C, CH, CW)
        2. Cropped image 2. Shape: (C, CH, CW)
        3. Cropped mask 1. Shape: (1, CH, CW)
        4. Cropped mask 2. Shape: (1, CH, CW)
        5. Shift of the crop in image1. Shape: (2,) The shift in the coordinates of the cropped image1. (y, x)
        6. Shift of the crop in image2. Shape: (2,) The shift in the coordinates of the cropped image2. (y, x)

    Raises
    ------
    NoMaskFoundError
        If no mask is found in either of the images.
    """

    # Compute the minimum bounding box for the masks, and determine the max box size

    C, H, W = mask1.shape
    if C != 1:
        raise ValueError("Mask should be single channel. Got: " + str(C))
    img_shape = torch.tensor([H, W], device=mask1.device)
    coords1 = torch.argwhere(mask1[0])
    coords2 = torch.argwhere(mask2[0])

    if len(coords1) == 0 or len(coords2) == 0:
        raise NoMaskFoundError("No mask found in the image")

    min1 = coords1.amin(dim=0)
    max1 = coords1.amax(dim=0)
    min2 = coords2.amin(dim=0)
    max2 = coords2.amax(dim=0)

    size1 = max1 - min1
    size2 = max2 - min2

    max_size = torch.max(size1, size2)

    box1_center = ((min1 + max1) / 2)
    box2_center = ((min2 + max2) / 2)

    # Crop the images to the max_size box
    crop1_min = (box1_center.round() - (max_size / 2)).floor().int()
    crop1_max = (box1_center.round() + (max_size / 2)).ceil().int()

    crop2_min = (box2_center.round() - (max_size / 2)).floor().int()
    crop2_max = (box2_center.round() + (max_size / 2)).ceil().int()

    # Ensure the crop is within the image bounds
    crop1_min_delta = torch.where(crop1_min < 0, -crop1_min, 0)
    crop2_min_delta = torch.where(crop2_min < 0, -crop2_min, 0)

    crop1_min += crop1_min_delta
    crop1_max += crop1_min_delta

    crop2_min += crop2_min_delta
    crop2_max += crop2_min_delta

    crop1_max_delta = torch.where(
        crop1_max > img_shape, crop1_max - img_shape, 0)
    crop2_max_delta = torch.where(
        crop2_max > img_shape, crop2_max - img_shape, 0)

    crop1_min -= crop1_max_delta
    crop1_max -= crop1_max_delta

    crop2_min -= crop2_max_delta
    crop2_max -= crop2_max_delta

    if torch.any(crop1_min < 0):
        raise ValueError("Crop1 min is less than 0")
    if torch.any(crop2_min < 0):
        raise ValueError("Crop2 min is less than 0")

    image_crop1 = image1[..., crop1_min[0]                         :crop1_max[0], crop1_min[1]:crop1_max[1]]
    image_crop2 = image2[..., crop2_min[0]                         :crop2_max[0], crop2_min[1]:crop2_max[1]]

    mask_crop1 = mask1[..., crop1_min[0]                       :crop1_max[0], crop1_min[1]:crop1_max[1]]
    mask_crop2 = mask2[..., crop2_min[0]                       :crop2_max[0], crop2_min[1]:crop2_max[1]]

    return image_crop1, image_crop2, mask_crop1, mask_crop2, crop1_min, crop2_min


def apply_homography(image: VEC_TYPE, transform: VEC_TYPE) -> torch.Tensor:
    """
    Apply homography to an image.
    Uses grid_sample and the transform on the image coordinates to deform the image.

    Parameters
    ----------
    image : VEC_TYPE
        Image to apply homography to.
        Shape: ([B,] H, W, C) if numpy array, else ([B,] C, H, W) if tensor,
        where H is the height, W is the width and C is the number of channels.
        Can be grayscale or color image. If color image, the number of channels should be 3 and the color format should be RGB.

    transform : VEC_TYPE
        Homography to apply to the image.
        Shape: ([B,] 3, 3)

    Returns
    -------
    torch.Tensor
        Image after applying the homography.
        Shape: ([B,] H, W, C)
    """
    from tools.transforms.to_tensor_image import ToTensorImage
    tensor_image = ToTensorImage(output_dtype=torch.float32)(image)
    return _apply_homography(tensor_image, tensorify(transform, dtype=torch.float32))


@torch.jit.script
def _apply_homography(image: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Apply homography to an image.
    Uses grid_sample and the transform on the image coordinates to deform the image.

    Parameters
    ----------
    image : torch.Tensor
        Image to apply homography to.
        Shape: (B, C, H, W) where C is the number of channels, H is the height and W is the width.

    transform : torch.Tensor
        Homography to apply to the image.
        Shape: (B, 3, 3)

    Returns
    -------
    torch.Tensor
        Image after applying the homography.
        Shape: (B, C, H, W)
    """
    transform, tfshp = flatten_batch_dims(transform, -3)
    tensor_image, shp = flatten_batch_dims(image, -4)

    B, C, H, W = tensor_image.shape

    x = torch.arange(W, device=tensor_image.device)
    y = torch.arange(H, device=tensor_image.device)

    xx, yy = torch.meshgrid(x, y, indexing="xy")
    grid = torch.stack([xx, yy, torch.ones((H, W), dtype=x.dtype,
                       device=tensor_image.device)], dim=-1).float().unsqueeze(0).repeat(B, 1, 1, 1)

    mats = transform.unsqueeze(1).unsqueeze(1).repeat(
        1, H, W, 1, 1).reshape(B * H * W, 3, 3)
    new_coords = torch.bmm(
        mats, grid.unsqueeze(-1).reshape(B * H * W, 3, 1)).reshape(B, H, W, 3, 1)[..., 0]
    new_coords = (new_coords / new_coords[..., 2:3])[..., :2]  # Normalize by z

    # Normalize to [-1, 1]
    new_coords = new_coords / \
        torch.tensor([W, H], device=tensor_image.device).float()
    new_coords = new_coords * 2 - 1
    out_grid = torch.nn.functional.grid_sample(
        tensor_image, new_coords, align_corners=True)

    return unflatten_batch_dims(out_grid, shp)
