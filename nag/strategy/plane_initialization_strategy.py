

from typing import Any, Dict, Optional, Tuple
from nag.config.nag_config import NAGConfig
from nag.dataset.nag_dataset import NAGDataset
from nag.model.learned_image_plane_scene_node_3d import LearnedImagePlaneSceneNode3D
from nag.model.nag_model import NAGModel
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.strategy.strategy import Strategy
import torch
from abc import abstractmethod

from typing import Any, Dict
from nag.config.nag_config import NAGConfig
from nag.dataset.nag_dataset import NAGDataset
from nag.model.nag_model import NAGModel
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
import torch
from tools.util.typing import DEFAULT
from tools.util.format import parse_type


def get_plane_config_properties(object_index: int,
                                times: torch.Tensor,
                                config: NAGConfig,
                                name: str = None
                                ) -> Dict[str, Any]:
    """Gets the basic plane config properties for the plane

    Parameters
    ----------
    object_index : int
        The object index

    times : torch.Tensor
        The times of used frames for the object.
        Shape: (T,) in range [0, 1].
        Should be equivalent to the number of images.

    config : NAGConfig
        The NAG config

    Returns
    -------
    Dict[str, Any]
        The plane config properties.
    """
    with torch.no_grad():
        num_flow_control_points = config.plane_flow_control_points if config.plane_flow_control_points is not None else int(round(len(
            times) * config.plane_flow_control_points_ratio))
        object_rigid_control_points = int(round(len(
            times) * config.object_rigid_control_points_ratio)) if config.object_rigid_control_points == DEFAULT or config.object_rigid_control_points == None else config.object_rigid_control_points

        if len(times) == 1:
            object_rigid_control_points = 1
            num_flow_control_points = 1

        set_name = ""
        if config.plane_names is not None and object_index < len(config.plane_names):
            set_name = config.plane_names[object_index]

        if len(set_name) > 0:
            set_name = ": " + set_name

        return dict(
            learnable_translation=config.is_plane_translation_learnable,
            learnable_rotation=config.is_plane_rotation_learnable,
            translation_offset_weight=config.plane_translation_offset_weight,
            rotation_offset_weight=config.plane_rotation_offset_weight,
            num_flow_control_points=num_flow_control_points,
            num_rigid_control_points=object_rigid_control_points,
            dtype=config.dtype,
            encoding_image_config=config.plane_encoding_image_config,
            encoding_alpha_config=config.plane_encoding_alpha_config,
            encoding_flow_config=config.plane_encoding_flow_config,
            network_image_config=config.plane_network_image_config,
            network_alpha_config=config.plane_network_alpha_config,
            network_flow_config=config.plane_network_flow_config,
            flow_weight=config.plane_flow_weight,
            rgb_weight=config.plane_color_weight,
            coarse_to_fine_color=config.plane_coarse_to_fine_color,
            coarse_to_fine_alpha=config.plane_coarse_to_fine_alpha,
            alpha_weight=config.plane_alpha_weight,
            times=times,
            name=(
                f"{object_index}" if (name is None or len(name) == 0) else name) + set_name,
            index=object_index,
            rgb_rescaling=config.plane_rgb_rescaling,
            alpha_rescaling=config.plane_alpha_rescaling,
            flow_rescaling=config.plane_flow_rescaling,
            network_dtype=config.tinycudann_network_dtype,
            **(config.plane_kwargs if config.plane_kwargs is not None else dict()),
            deprecated_flow=config.deprecated_flow,
            align_corners=config.plane_align_corners,
        )


class PlaneInitializationStrategy(Strategy):
    """A plane initialization strategy. Determines how to initialize a plane.
    Returns a dictionary of init arguments for the plane.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def execute(self,
                object_index: int,
                mask_index: int,
                images: torch.Tensor,
                masks: torch.Tensor,
                depths: torch.Tensor,
                times: torch.Tensor,
                camera: TimedCameraSceneNode3D,
                nag_model: "NAGModel",
                dataset: NAGDataset,
                config: "NAGConfig",
                name: str = None,
                proxy_init: bool = False,
                runner: Optional[Any] = None,
                **kwargs) -> Dict[str, Any]:
        """Execute the plane initialization strategy.

        Parameters
        ----------
        object_index : int
            The index of the object. Starts from 0 to the number of objects.

        mask_index : int
            The index of the mask. Used to select the correct mask in the masks.

        images : torch.Tensor
            The images. Shape (T, C, H, W). (C = 3)

        masks : torch.Tensor
            The masks. Shape (T, O, H, W). (O = number of objects)

        depth : torch.Tensor
            The estimated depths in the scene. Shape (T, H, W).

        camera : TimedCameraSceneNode3D
            The camera.

        nag_model : NAGModel
            The NAG model.

        config : NAGConfig
            The NAG config.

        Returns
        -------
        Dict[str, Any]
            A dictionary of valid init arguments for the plane.
        """
        plane_properties = get_plane_config_properties(
            object_index, times, config, name)

        if proxy_init:
            from nag.model.view_dependent_spline_scene_node_3d import ViewDependentSplineSceneNode3D
            plane_type = plane_type = parse_type(
                config.plane_type, LearnedImagePlaneSceneNode3D)
            if issubclass(plane_type, ViewDependentSplineSceneNode3D):
                plane_properties["view_dependent_data_range"] = torch.stack(
                    [torch.tensor([-torch.pi, -torch.pi]), torch.tensor([torch.pi, torch.pi])], dim=0)
                num_control_points = int(
                    images.shape[0] * config.plane_view_dependent_control_point_ratio)
                plane_properties["num_view_dependent_control_points"] = num_control_points
            if runner is not None:
                if not hasattr(runner, "boxes"):
                    runner.boxes = runner.load_boxes()
                # Try getting box
                box = runner.boxes.get(
                    runner.dataset.mask_ids[object_index].item(), None)
                if box is not None:
                    from nag.model.timed_box_scene_node_3d import TimedBoxSceneNode3D
                    box_node = TimedBoxSceneNode3D.from_timed_box_3d(box)
                    plane_properties["box"] = box_node

        return plane_properties

    def get_base_image_alpha(
        self,
        padded_bottom_left: torch.Tensor,
        padded_top_right: torch.Tensor,
        mask: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the base image and base alpha by padding the image and mask.

        The image and alpha is retrived between the padded bottom left and top right corners.
        And will have the shape padded_top_right - padded_bottom_left. (PH, PW)

        Parameters
        ----------
        padded_bottom_left : torch.Tensor
            The bottom left corner within the mask or image + eventual padding. Shape: (2, ) (y, x)

        padded_top_right : torch.Tensor
            The top right corner within the mask or image + eventual padding. Shape: (2, ) (y, x)

        mask : torch.Tensor
            The mask of the object. Shape: (1, H, W)

        image : torch.Tensor
            The image. Shape: (C, H, W)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            1. torch.Tensor
                The base image. Shape: (C, PH, PW)
            2. torch.Tensor
                The base alpha. Shape: (PH, PW)
        """
        C, H, W = mask.shape
        resolution = torch.tensor([H, W], dtype=padded_bottom_left.dtype)
        # As bounds are maybe outside the image, we need to pad it if necessary
        pad_d = torch.tensor([0, 0, 0, 0], dtype=padded_bottom_left.dtype)
        if (padded_bottom_left < 0).any():
            pad_d[:2][padded_bottom_left <
                      0] = padded_bottom_left[padded_bottom_left < 0] * -1
        if (padded_top_right >= resolution).any():
            pad_d[2:][padded_top_right >= resolution] = (padded_top_right[padded_top_right >= torch.tensor(
                [H, W])] - torch.tensor([H, W], dtype=padded_top_right.dtype)[padded_top_right >= resolution])

        # Add from bounds_bl the padded values
        padded_top_right = padded_top_right + pad_d[:2]
        padded_bottom_left = padded_bottom_left + pad_d[:2]

        # bounds_tr = bounds_tr + pad_d[2:]
        img = image
        alpha = mask[0]

        if pad_d.sum() > 0:
            # pad_d is in formax bl_y, bl_x, tr_y, tr_x => padding_bottom, padding_left, padding_top, padding_right
            # For fpad need to (padding_left,padding_right, padding_top,padding_bottom)
            pd = (pad_d[1], pad_d[3], pad_d[0], pad_d[2])
            img = torch.nn.functional.pad(img, pd, mode="replicate")
            alpha = torch.nn.functional.pad(
                alpha, pd, mode="constant", value=0.)
        base_image = img[:,
                         padded_bottom_left[0]:padded_top_right[0],
                         padded_bottom_left[1]:padded_top_right[1]]

        base_alpha = alpha[
            padded_bottom_left[0]:padded_top_right[0],
            padded_bottom_left[1]:padded_top_right[1]]
        return base_image, base_alpha
