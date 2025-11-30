from dataclasses import dataclass, field
from typing import Literal, Type, Union, Any, Dict, Tuple
from tools.serialization.json_convertible import JsonConvertible
from nag.homography.match_finder import MatchFinderConfig, MatchFinder, MatchFindingError
from tools.util.torch import tensorify, flatten_batch_dims
from tools.torch.parse_device import parse_device
from tools.util.typing import VEC_TYPE
import torch
from tools.transforms.to_tensor_image import ToTensorImage
from tools.transforms.to_numpy_image import ToNumpyImage
from tools.context.temporary_device import TemporaryDevice
import cv2
import kornia as K
import numpy as np
import matplotlib.pyplot as plt
from tools.logger.logging import logger


@dataclass
class SiftFinderConfig(MatchFinderConfig):

    flann_trees: int = field(default=5)

    flann_checks: int = field(default=50)


class SiftMatchFinder(MatchFinder):

    def __init__(self, config: SiftFinderConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.tensorify_image = ToTensorImage(
            output_dtype=torch.float32, output_device="cpu")
        self.numpify_image = ToNumpyImage(output_dtype=np.uint8)

        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(dict(
            algorithm=1, trees=self.config.flann_trees), dict(checks=self.config.flann_checks))

    @classmethod
    def config_type(cls) -> Type[MatchFinderConfig]:
        return SiftFinderConfig

    def find_matches(self,
                     image1: VEC_TYPE,
                     image2: VEC_TYPE,
                     is_batch_mode: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray]:
        image1, shp = flatten_batch_dims(self.tensorify_image(image1), -4)
        image2, _ = flatten_batch_dims(self.tensorify_image(image2), -4)
        if image1.shape != image2.shape:
            raise ValueError(
                f"Image shapes must match. Got {image1.shape} and {image2.shape}")
        if image1.shape[1] != 1:
            image1 = K.color.rgb_to_grayscale(image1)
            image2 = K.color.rgb_to_grayscale(image2)
        image1 = self.numpify_image(image1.cpu())
        image2 = self.numpify_image(image2.cpu())

        rets_source = []
        rets_target = []

        for i in range(image1.shape[0]):
            try:
                img1 = image1[i]
                img2 = image2[i]
                kp1, des1 = self.sift.detectAndCompute(img1, None)
                kp2, des2 = self.sift.detectAndCompute(img2, None)

                if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
                    # No keypoints found
                    logger.debug(
                        "No Sift Matches found at all for for idx: " + str(i))
                    continue

                matches = self.matcher.knnMatch(des1, des2, k=2)

                good = []
                # Need to draw only good matches, so create a mask
                matchesMask = [[0, 0] for i in range(len(matches))]
                # ratio test as per Lowe's paper
                for j, (m, n) in enumerate(matches):
                    if m.distance < 0.7*n.distance:
                        matchesMask[j] = [1, 0]
                        good.append(m)

                if len(good) == 0:
                    # No good matches found
                    logger.debug(
                        "No good Sift Matches found for idx: " + str(i))
                    continue

                source_pt = np.array([kp1[m.queryIdx].pt for m in good])
                target_pt = np.array([kp2[m.trainIdx].pt for m in good])

                # Add batch index
                source_pt = np.hstack(
                    [np.full((source_pt.shape[0], 1), i), source_pt])
                target_pt = np.hstack(
                    [np.full((target_pt.shape[0], 1), i), target_pt])

                rets_source.append(source_pt)
                rets_target.append(target_pt)
            except Exception as e:
                logger.debug(
                    f"Error while finding matches: {e}", exc_info=True)
            # draw_params = dict(matchColor = (0,255,0),
            #         singlePointColor = (255,0,0),
            #         matchesMask = matchesMask,
            #         flags = cv2.DrawMatchesFlags_DEFAULT)
            # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
            # plt.imshow(img3,),plt.show()
        if len(rets_source) == 0 and len(rets_target) == 0:
            return np.array([]), np.array([])
        return np.concatenate(rets_source), np.concatenate(rets_target)
