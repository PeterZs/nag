import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger as PLLogger
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from tools.agent.util.tracker import Tracker
from tools.labels.projected_timed_box_3d import ProjectedTimedBox3D
from tools.logger.experiment_logger import ExperimentLogger
from tools.logger.logging import logger
from tools.run.trainable_runner import TrainableRunner
from tools.transforms.min_max import MinMax
from tools.util.format import parse_type, raise_on_none
from tools.util.path_tools import read_directory
from tools.util.seed import seed_all
from tools.util.torch import tensorify, tensorify_image
from tools.util.typing import _DEFAULT, DEFAULT
from tools.viz.matplotlib import set_default_output_dir
from torch.utils.data import DataLoader

from nag.callbacks.nag_callback import NAGCallback
from nag.config.nag_config import NAGConfig
from nag.dataset.nag_dataset import NAGDataset
from nag.error.box_not_found_error import BoxNotFoundError
from nag.error.model_build_exception import ModelBuildException
from nag.model.background_image_plane_scene_node_3d import \
    BackgroundImagePlaneSceneNode3D
from nag.model.background_plane_scene_node_3d import BackgroundPlaneSceneNode3D
from nag.model.epoch_state_mixin import EpochStateMixin
from nag.model.learned_camera_scene_node_3d import LearnedCameraSceneNode3D
from nag.model.learned_image_plane_scene_node_3d import \
    LearnedImagePlaneSceneNode3D
from nag.model.mixed_precision_custom import MixedPrecisionCustom
from nag.model.nag_functional_model import NAGFunctionalModel
from nag.model.nag_model import NAGModel
from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D
from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D
from nag.sampling.dataset_grid_sampler import DatasetGridSampler
from nag.sampling.random_timed_uv_grid_sampler import RandomTimedUVGridSampler
from nag.strategy.border_condition_plane_position_strategy import \
    BorderConditionPlanePositionStrategy
from nag.strategy.plane_initialization_strategy import \
    PlaneInitializationStrategy
from nag.strategy.plane_position_strategy import PlanePositionStrategy
from nag.strategy.tilted_plane_initialization_strategy import \
    TiltedPlaneInitializationStrategy
from nag.utils import utils


class NAGRunner(TrainableRunner, EpochStateMixin):
    """Runner to run the nag experiments."""

    config: NAGConfig

    def __init__(self, config: NAGConfig, **kwargs) -> None:
        raise_on_none(config)
        super().__init__(config=config, **kwargs)
        det = self.config.deterministic
        self.experiment_logger: ExperimentLogger = None
        self.loggers = None
        if isinstance(det, str):
            det = True
        warn_only = False
        if isinstance(self.config.deterministic, str):
            if self.config.deterministic.lower() == "warn":
                warn_only = True
        seed_all(self.config.seed, deterministic=det, warn_only=warn_only)
        pl.seed_everything(self.config.seed, workers=True)

    def build(self,
              tracker: Optional[Tracker] = None,
              dataset: Optional[NAGDataset] = None,
              model: Optional[NAGModel] = None,
              context: Optional[Dict[str, Any]] = None,
              **kwargs):
        # Check if checkpoint path is set
        if self.config.checkpoint_path is not None:
            logger.info(
                f"Checkpoint path is set. Loading model from {self.config.checkpoint_path}.")
            self.load(self.config.checkpoint_path,
                      tracker_directory=self.config.checkpoint_tracker_path)
        else:
            logger.info("Building runner...")
            # Load bundle
            bundle = utils.load_bundle(
                self.config.bundle_path) if self.config.bundle_path is not None else None
            if bundle is None:
                if self.config.pinhole_camera_config is None and self.config.synthetic_camera_config is None:
                    logger.warning(
                        "Did not found any camera config. Using default values and not relying on any inertial data.")
                else:
                    if self.config.pinhole_camera_config is not None:
                        logger.info(
                            "Initializing camera poses from pinhole camera config.")
                    elif self.config.synthetic_camera_config is not None:
                        logger.info(
                            "Initialize camera poses from synthetic camera config.")
            self.tracker = Tracker() if tracker is None else tracker
            self.dataset = self.create_bundle_dataset(
                bundle) if dataset is None else dataset
            self.model = self.create_model(
                bundle, context=context) if model is None else model
            self.grid_sampler = self.create_grid_dataset(
                self.dataset, self.model.camera)
            self.train_loader = self.create_dataloader(self.grid_sampler)
            self.trainer = self.create_trainer()

    def load(self,
             checkpoint_path: str,
             callbacks: Optional[
                 Union[List[pl.Callback],
                       _DEFAULT]] = DEFAULT,
             logger: Optional[
                 Union[Iterable[PLLogger], PLLogger,
                       _DEFAULT]] = DEFAULT,
             tracker_directory: Optional[str] = None) -> None:
        from tools.logger.logging import logger as logging_logger

        # Load tracker if exists
        bundle = utils.load_bundle(
            self.config.bundle_path) if self.config.bundle_path is not None else None
        if bundle is None:
            pass
        if tracker_directory is not None:
            self.tracker = Tracker.from_directory(tracker_directory)
        else:
            # Warn if tracker is not loaded
            logging_logger.warning("Tracker not loaded. Creating new tracker.")
            self.tracker = Tracker()
        self.dataset = self.create_bundle_dataset(bundle)
        dataset = self.dataset

        mt = parse_type(self.config.model_type, NAGModel,
                        variable_name="model_type")

        # images, masks, depth, oids = self.load_data(dataset)
        # Create proxy data
        H, W = self.dataset.initial_image_shape
        T, _, _, C = self.dataset.get_image_shape()

        # Create a dummy world as proxy data, since model gets created from checkpoint which will adjust all dimensions.
        templ = torch.zeros((1, 1, 1, 1), dtype=self.config.dtype)
        images = templ.expand(T, C, H, W)
        oids = self.get_oids()
        if self.config.has_background_plane:
            # Remove the background plane as masks and create world function requires just ids for the masks not the bg
            oids = oids[:-1]

        if self.config.has_camera_aberration_plane:
            oids = oids[:-1]

        masks = torch.zeros((1, 1, 1, 1), dtype=torch.bool).expand(
            T, len(oids), H, W)
        _ = self.dataset.load_mask_stack(init_size=True)
        depth = templ.expand(T, -1, H, W)

        dummy_model: NAGModel = self.init_model(mt, dataset)
        world = self.create_world(dataset, bundle, dummy_model, images=images,
                                  masks=masks, depth=depth, oids=oids, init_strategy=PlaneInitializationStrategy(), proxy_init=True)  # Use the default strategy as values will be overriden by state dict.
        dummy_model.set_world(world)

        self.model = mt.load_from_checkpoint(checkpoint_path, **self._get_model_args(
            mt, dataset), world=world, allow_loading_unmatching_parameter_sizes=True)
        self.model.world._after_checkpoint_loaded()

        if isinstance(self.model, NAGFunctionalModel):
            # Sync the _translations etc.
            # Cc
            if (self.model._node_indices == -1).all():
                # Older model, neeed to set the indices
                indices = [x.get_index() for x in world._scene_children]
                if self.config._old_scene_children_order:
                    # Order shall be camera, background, objects
                    # Move cam + back to last
                    if self.config.has_background_plane:
                        indices = indices[2:] + indices[:2]
                    else:
                        indices = indices[1:] + indices[:1]
                else:
                    # Order should be Camera (last-idx), Foreground planes (0, 1, 2, ...), Background plane (last-idx-1)
                    if self.config.has_background_plane:
                        # Objects, Camera, Background
                        indices = indices[1:-1] + indices[0:1] + indices[-1:]
                    else:
                        indices = indices[1:] + indices[:1]
                
                self.model._node_indices = torch.tensor(
                    indices, dtype=torch.int32, device=self.model.device)
            self.model.push_to_objects()

        self.trainer = self.create_trainer(callbacks=callbacks, logger=logger)
        self.model.eval()

    def load_data(self,
                  dataset: NAGDataset,
                  init_size: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        image_tensor = tensorify_image(
            dataset.load_image_stack(init_size=init_size))
        masks = tensorify_image(dataset.load_mask_checked(init_size=init_size))
        depth = tensorify_image(dataset.load_depth_stack(init_size=init_size))

        # Create a summary text including shapes of masks and images as well as mask object ids
        # and the number of objects

        text = f"Init-Image shape: {list(image_tensor.shape)}\n\t"
        text += f"Init-Mask shape: {list(masks.shape)}\n\t"
        text += f"Init-Depth shape: {list(depth.shape)}\n\t"
        text += f"Mask object ids: {self.dataset.mask_ids.cpu().tolist()}\n\t"
        text += f"Number of objects: {len(self.dataset.mask_ids)}"
        logger.info("Loaded initialization data: \n\t" + text)

        if self.config.use_distance_quantiles and self.config.pinhole_camera_config is not None and self.config.pinhole_camera_config.distance_quantiles is not None:
            max_dist = tensorify(
                self.config.pinhole_camera_config.distance_quantiles).amax()
            depth *= max_dist
            logger.info(
                f"Using distance quantile to scale depth. Max distance: {max_dist}."
            )

        oids = list(range(masks.shape[-3]))
        bar = None

        if self.config.mask_indices_filter is not None:
            oids = [x for x in oids if x in set(
                self.config.mask_indices_filter)]

        # Set number of objects to len of oids
        N_obj = (len(oids) + 1) if self.config.has_background_plane else len(oids)
        if self.config.has_camera_aberration_plane:
            N_obj += 1
        self.config.num_objects = N_obj
        return image_tensor, masks, depth, oids

    def load_boxes(self,
                   convert: bool = True,
                   skip_unmapped_boxes: bool = True,
                   normalize_box_positions: bool = True,
                   waymo_boxes: bool = True
                   ) -> Dict[int, ProjectedTimedBox3D]:
        from tools.scene.coordinate_system_3d import CoordinateSystem3D
        from tools.transforms.geometric.mappings import (rotmat_to_rotvec,
                                                         rotvec_to_rotmat)
        from tools.util.numpy import flatten_batch_dims as nflatten_batch_dims
        from tools.util.torch import (flatten_batch_dims, tensorify,
                                      unflatten_batch_dims)

        boxes = dict()
        if self.config.boxes_path is None or not os.path.exists(self.config.boxes_path):
            return boxes
        boxes_paths = read_directory(
            self.config.boxes_path, self.config.boxes_filename_pattern)

        ignored_boxes = []
        duplicated_boxes = []
        no_id_boxes = []

        if waymo_boxes:
            camera_movement_path = tensorify(
                self.config.pinhole_camera_config.position, dtype=self.config.dtype)
            camera_movement_times = tensorify(
                self.config.pinhole_camera_config.times, dtype=self.config.dtype)
            max_t = len(camera_movement_times)
            camera_movement_times = (
                camera_movement_times * (max_t - 1)).round().int()

            if self.config.frame_indices_filter is not None:
                camera_movement_times = camera_movement_times[self.config.frame_indices_filter]
                camera_movement_path = camera_movement_path[self.config.frame_indices_filter]

            camera_movement_times = self.dataset.frame_indices_to_times(
                camera_movement_times)

        for p in boxes_paths:
            path = p["path"]
            box = ProjectedTimedBox3D.load_from_file(path)
            box.save_path = path
            if skip_unmapped_boxes and box.object_id is None:
                no_id_boxes.append((path, box))
                continue

            mask_idx = self.dataset.mask_ids

            if skip_unmapped_boxes and box.object_id in boxes and convert:
                duplicated_boxes.append((path, box))
                continue

            if skip_unmapped_boxes and box.object_id not in mask_idx and convert:
                ignored_boxes.append((path, box))
                continue
            if skip_unmapped_boxes:
                boxes[int(box.object_id)] = box
            else:
                if box.object_id is None or (int(box.object_id) in boxes):
                    boxes[box.id] = box
                else:
                    boxes[int(box.object_id)] = box

            our_coord = CoordinateSystem3D.from_string("rdf")

            # Filter box data for the current frame indices
            if convert:
                from tools.util.torch import index_of_first
                needed_indices = self.dataset._index["index"].values
                pos = index_of_first(torch.tensor(
                    box.frame_times), torch.tensor(needed_indices))
                filtered_pos = pos[pos > -1]
                if len(filtered_pos) == 0:
                    logger.info(
                        f"Box {box.object_id if box.object_id is not None else box.id} has no valid frame indices it will be skipped. Its indices are {box.frame_times}."
                    )
                    if box.object_id is not None:
                        boxes.pop(box.object_id)
                    else:
                        boxes.pop(box.id)
                    continue

                sel_pos_mask = flatten_batch_dims(
                    (pos > -1).argwhere().squeeze(), -1)[0]
                norm_box_stamps = self.dataset.frame_timestamps[sel_pos_mask]

                box.center = nflatten_batch_dims(
                    box.center[filtered_pos], -2)[0]
                box.depth = nflatten_batch_dims(box.depth[filtered_pos], -1)[0]
                box.width = nflatten_batch_dims(box.width[filtered_pos], -1)[0]
                box.height = nflatten_batch_dims(
                    box.height[filtered_pos], -1)[0]
                box.heading = nflatten_batch_dims(
                    box.heading[filtered_pos], -2)[0]
                box.corners = nflatten_batch_dims(
                    box.corners[filtered_pos], -3)[0]
                box.frame_times = flatten_batch_dims(norm_box_stamps, -1)[0]

                lab = box.projected_label
                if lab is not None:
                    projected_pos = index_of_first(torch.tensor(
                        lab.frame_times), torch.tensor(needed_indices))
                    sel_mask = flatten_batch_dims(
                        (projected_pos > -1).argwhere().squeeze(), -1)[0]
                    norm_label_stamps = self.dataset.frame_timestamps[sel_mask]

                    filtered_proj_pos = projected_pos[projected_pos > -1]

                    if len(filtered_proj_pos) == 0:
                        logger.warning(
                            f"Box {box.object_id if box.object_id is not None else box.id} has no valid projection to camera. Its projection indices are {lab.frame_times}."
                        )
                        if box.object_id is not None:
                            boxes.pop(box.object_id)
                        else:
                            boxes.pop(box.id)
                        continue

                    lab.center = nflatten_batch_dims(
                        lab.center[filtered_proj_pos], -2)[0]
                    lab.width = nflatten_batch_dims(
                        lab.width[filtered_proj_pos], -1)[0]
                    lab.height = nflatten_batch_dims(
                        lab.height[filtered_proj_pos], -1)[0]
                    lab.frame_times = flatten_batch_dims(
                        norm_label_stamps, -1)[0]

            # If coordinate system is not rfu, convert the boxes
            if str(box.coordinate_system) != str(our_coord) and convert:
                box.center = box.coordinate_system.convert_vector(
                    our_coord, box.center)
                box.corners = box.coordinate_system.convert_vector(
                    our_coord, box.corners)
                box.heading = box.coordinate_system.convert_vector(
                    our_coord, box.heading)

                # For waymo, the heading is defined as angle of the front of the car, so we need to rotate it by 90 degrees
                box.heading = box.heading - np.array([0, np.pi/2, 0])

                sizes = np.stack([box.width, box.height, box.depth], axis=-1)
                sizes_cvt = np.abs(
                    box.coordinate_system.convert_vector(our_coord, sizes))
                box.width = sizes_cvt[..., 0]
                box.height = sizes_cvt[..., 1]
                box.depth = sizes_cvt[..., 2]

            if self.config.pinhole_camera_config is not None and convert:
                if waymo_boxes:
                    # Waymo boxes are defined in vehicle coordinate system, so we need to convert them to world coordinate system by including the vehicle / camera motion
                    select_times = index_of_first(
                        camera_movement_times, box.frame_times)
                    if (select_times == -1).any():
                        raise ValueError(
                            "Could not find all times in the camera movement times.")
                    cam_to_world = camera_movement_path[select_times]
                    vehicle_to_cam = tensorify(
                        self.config.pinhole_camera_config.normalization_position, dtype=self.config.dtype)[None, ...]

                    vehicle_to_cam = flatten_batch_dims(
                        vehicle_to_cam, -3)[0].expand(len(cam_to_world), -1, -1)
                    vehicle_to_world = torch.bmm(cam_to_world, vehicle_to_cam)

                    T = cam_to_world.shape[0]
                    box_center = torch.tensor(
                        box.center, dtype=self.config.dtype)

                    pos_cat = torch.cat(
                        [box_center, torch.ones_like(box_center[..., :1])], dim=-1)
                    pos_in_vehicle = torch.eye(4, dtype=self.config.dtype)[
                        None, ...].repeat(T, 1, 1)
                    pos_in_vehicle[:, :3, 3] = pos_cat[:, :3]
                    pos_in_world = torch.bmm(vehicle_to_world, pos_in_vehicle)

                    box_center = pos_in_world[:, :3, 3].numpy()
                    box.center = box_center

                    box_corners = torch.tensor(
                        box.corners, dtype=self.config.dtype)
                    corners_cat = torch.cat(
                        [box_corners, torch.ones_like(box_corners[..., :1])], dim=-1)
                    corners_in_vehicle = torch.eye(4, dtype=self.config.dtype)[
                        None, None, ...].repeat(T, 8, 1, 1)
                    corners_in_vehicle[..., :3, 3] = corners_cat[..., :3]

                    corners_in_world = torch.bmm(vehicle_to_world[:, None].repeat(1, 8, 1, 1).reshape(
                        T * 8, 4, 4), corners_in_vehicle.reshape(T * 8, 4, 4)).reshape(T, 8, 4, 4)
                    box.corners = corners_in_world[:, :, :3, 3].numpy()

                    # Recompute the heading
                    heading = torch.tensor(
                        box.heading, dtype=self.config.dtype)
                    R = torch.eye(4, dtype=self.config.dtype)[
                        None, ...].repeat(T, 1, 1)
                    R[:, :3, :3] = rotvec_to_rotmat(heading)
                    new_heading = torch.bmm(vehicle_to_world, R)[:, :3, :3]
                    box.heading = rotmat_to_rotvec(new_heading).numpy()

                elif normalize_box_positions:

                    center = tensorify(box.center, dtype=self.config.dtype)
                    flattened_corners, fc = flatten_batch_dims(
                        tensorify(box.corners, dtype=self.config.dtype), -2)
                    center = torch.cat(
                        [center, torch.ones_like(center[..., :1])], dim=-1)
                    flattened_corners = torch.cat(
                        [flattened_corners, torch.ones_like(flattened_corners[..., :1])], dim=-1)

                    A = tensorify(
                        self.config.pinhole_camera_config.normalization_position, dtype=self.config.dtype)[None, ...]
                    if str(box.coordinate_system) != str(our_coord):
                        A = torch.tensor(
                            box.coordinate_system.convert(our_coord, A))

                    B, _ = center.shape
                    BC, _ = flattened_corners.shape

                    new_center = torch.bmm(
                        A.expand(B, -1, -1), center[..., None])[..., 0]
                    new_corners = torch.bmm(
                        A.expand(BC, -1, -1), flattened_corners[..., None])[..., 0]

                    box.center = new_center[:, :3].numpy()
                    box.corners = unflatten_batch_dims(new_corners, fc)[
                        :, :, :3].numpy()

        if len(ignored_boxes) > 0:
            paths = [os.path.basename(x[0]) for x in ignored_boxes]
            bids = [str(x[1].object_id) for x in ignored_boxes]
            logger.info(
                f"Ignored boxes {', '.join(paths)} with ids {', '.join(bids)} as they are not in the mask indices.")

        if len(duplicated_boxes) > 0:
            paths = [os.path.basename(x[0]) for x in duplicated_boxes]
            bids = [str(x[1].object_id) for x in duplicated_boxes]
            logger.warning(
                f"Ignored boxes {', '.join(paths)} with ids {', '.join(bids)} as they are duplicated.")

        if len(no_id_boxes) > 0:
            paths = [os.path.basename(x[0]) for x in no_id_boxes]
            logger.info(
                f"Ignored boxes {', '.join(paths)} as they have no object id.")

        box_mappings = {k: v.id for k, v in boxes.items()}
        logger.info(f"Box-ID mapping: {box_mappings}")
        missing_ids = set(self.dataset.mask_ids.cpu().tolist()
                          ) - set(box_mappings.keys())
        if len(missing_ids) > 0:
            logger.warning(
                f"Missing boxes for mask ids {missing_ids}. Please check the box files.")
        else:
            logger.info(
                f"All boxes are mapped to mask ids. {len(boxes)} boxes found.")
        return boxes

    def get_oids(self) -> List[int]:
        return list(range(self.config.num_objects))

    def get_plane_position_exec_args(self,
                                     position_strategy: Type[PlanePositionStrategy],
                                     object_index: int,
                                     mask_index: int,
                                     images: torch.Tensor,
                                     masks: torch.Tensor,
                                     depths: torch.Tensor,
                                     skip_object_on_missing_box: bool = False
                                     ) -> Dict[str, Any]:
        from nag.strategy.box_plane_position_strategy import \
            BoxPlanePositionStrategy
        args = dict()
        if issubclass(position_strategy, (BoxPlanePositionStrategy, BorderConditionPlanePositionStrategy)):
            if not hasattr(self, "boxes"):
                self.boxes = self.load_boxes()
            mask_id = self.dataset.mask_ids[mask_index].item()
            box = self.boxes.get(mask_id, None)
            if box is None:
                if skip_object_on_missing_box:
                    raise BoxNotFoundError(f"Box with id {mask_id} not found.")
                if issubclass(position_strategy, BoxPlanePositionStrategy):
                    raise ValueError(f"Box with id {mask_id} not found.")
                else:
                    pass
            else:
                args['box'] = box
        return args

    def get_background_camera_distance(self,
                                       masks: torch.Tensor,
                                       camera: TimedCameraSceneNode3D,
                                       position_strategy: Optional[Type[PlanePositionStrategy]] = None,
                                       world: TimedDiscreteSceneNode3D = None,
                                       ) -> float:
        from tools.util.torch import index_of_first

        from nag.strategy.box_plane_position_strategy import \
            BoxPlanePositionStrategy
        if not hasattr(self, "boxes"):
            self.boxes = self.load_boxes()
        will_use_boxes = position_strategy is not None and issubclass(
            position_strategy, (BoxPlanePositionStrategy, BorderConditionPlanePositionStrategy))
        if len(self.boxes) > 0 and will_use_boxes:
            # Check distance of the boxes to the camera if they are visible
            max_vis_dist = 0
            for obj_id, box in self.boxes.items():
                mask_index = np.argwhere(self.dataset.mask_ids == obj_id)
                if len(mask_index) == 0:
                    continue
                if len(mask_index) > 1:
                    raise ValueError(
                        f"Multiple masks with the same id {obj_id} found.")
                missing_in_frame = masks[:, mask_index].sum(
                    dim=(-2, -1)).squeeze() == 0
                indices = index_of_first(
                    box.frame_times, self.dataset.frame_timestamps)
                missing_in_frame = missing_in_frame | (indices == -1)

                existing_times = self.dataset.frame_timestamps[~missing_in_frame]
                existing_indices = indices[~missing_in_frame]

                # Use the center as a proxy, shall be enough
                camera_positions = camera.get_global_position(t=existing_times)[
                    :, :3, 3].detach()
                box_positions = torch.tensor(box.center[existing_indices])

                dists = torch.norm(camera_positions - box_positions, dim=-1)
                try:
                    if dists.max() > max_vis_dist:
                        max_vis_dist = dists.max()
                except Exception as e:
                    pass

            if self.config.background_camera_distance > max_vis_dist * 2:
                return self.config.background_camera_distance

            background_dist = max_vis_dist * 2
            logger.warning(
                f"Background camera distance is smaller than the maximum distance of one box {max_vis_dist:.2f} to the camera. Setting background camera distance to: {background_dist:.2f} (twice the maximum distance).")
            return background_dist
        else:
            # Check for already created objects their distance to the camera
            max_vis_dist = 0
            for obj in world.get_scene_children():
                if not isinstance(obj, LearnedImagePlaneSceneNode3D):
                    continue
                idx = obj.get_index()
                # If obj is visible
                missing_in_frame = masks[:, idx].sum(
                    dim=(-2, -1)).squeeze() == 0
                existing_times = self.dataset.frame_timestamps[~missing_in_frame]
                camera_positions = camera.get_global_position(t=existing_times)[
                    :, :3, 3].detach()
                plane_positions = obj.get_global_position(t=existing_times)[
                    :, :3, 3].detach()
                dists = torch.norm(camera_positions - plane_positions, dim=-1)
                try:
                    if dists.max() > max_vis_dist:
                        max_vis_dist = dists.max()
                except Exception as e:
                    pass

            if self.config.background_camera_distance > max_vis_dist * 2:
                return self.config.background_camera_distance

            background_dist = max_vis_dist * 2
            logger.warning(
                f"Background camera distance is smaller than the maximum distance of a plane {max_vis_dist:.2f} to the camera. Setting background camera distance to: {background_dist:.2f} (twice the maximum distance).")
            return background_dist
        # else:
        #     return self.config.background_camera_distance

    def create_world(self,
                     dataset: NAGDataset,
                     bundle: Optional[dict],
                     model: NAGModel,
                     images: torch.Tensor,
                     masks: torch.Tensor,
                     depth: torch.Tensor,
                     oids: List[int],
                     init_strategy: Optional[PlaneInitializationStrategy] = DEFAULT,
                     proxy_init: bool = False,
                     context: Optional[Dict[str, Any]] = None,
                     ) -> TimedDiscreteSceneNode3D:
        """
        Create the scene world with the given dataset and model.

        Parameters
        ----------
        dataset : NAGDataset
            The dataset.

        bundle : Optional[dict]
            The bundle. Can be None if not available.

        model : NAGModel
            The model.

        images : torch.Tensor
            The images. Maybe not the real images if proxy_init is True.
            Shape: (T, C, H, W)

        masks : torch.Tensor
            The masks. Maybe not the real masks if proxy_init is True.
            Shape: (T, O, H, W)

        depth : torch.Tensor
            The depth. Maybe not the real depth if proxy_init is True.
            Shape: (T, 1, H, W)

        oids : List[int]
            The object indices per object.
            The length of the list is the number of actual objects / masks.
            The background plane is NOT included in this list.
            Shape: (O,)

        init_strategy : Optional[PlaneInitializationStrategy], optional
            The plane initialization strategy, by default DEFAULT.

        proxy_init : bool, optional
            Whether to use proxy initialization, by default False.
            Proxy init will be used on loading a model from a checkpoint - so proxy objects are created and filled with checkpoint values afterwards.

        Returns
        -------
        TimedDiscreteSceneNode3D
            The world scene node.
        """
        if init_strategy is None or init_strategy == DEFAULT:
            init_strategy_type: Type[PlaneInitializationStrategy] = parse_type(
                self.config.plane_init_strategy, PlaneInitializationStrategy)
            self.config.plane_init_strategy = init_strategy_type
            args = self.config.plane_init_strategy_kwargs
            if args is None:
                args = dict()
            if issubclass(init_strategy_type, TiltedPlaneInitializationStrategy):
                args['position_spline_fitting'] = self.config.object_position_spline_approximation
            init_strategy: PlaneInitializationStrategy = init_strategy_type(
                **args)
        with torch.no_grad():
            object_rigid_control_points = int(round(len(
                dataset) * self.config.object_rigid_control_points_ratio)) if self.config.object_rigid_control_points == DEFAULT or self.config.object_rigid_control_points == None else self.config.object_rigid_control_points
            if images.shape[0] == 1:
                object_rigid_control_points = 1
            camera_idx = len(
                oids) + 1 if self.config.has_background_plane else len(oids)
            background_idx = len(oids)
            logger.info(f"Creating camera...")
            if bundle is not None:
                camera = LearnedCameraSceneNode3D.from_bundle(
                    bundle, name="camera",
                    num_rigid_control_points=object_rigid_control_points,
                    learnable_translation=self.config.is_camera_translation_learnable,
                    learnable_rotation=self.config.is_camera_rotation_learnable,
                    translation_offset_weight=self.config.camera_translation_offset_weight,
                    rotation_offset_weight=self.config.camera_rotation_offset_weight,
                    dtype=self.config.dtype,
                    frame_indices_filter=self.config.frame_indices_filter,
                    model=model,
                    index=camera_idx,
                    position_spline_approximation=self.config.object_position_spline_approximation,
                    position_spline_control_points=object_rigid_control_points
                )
            else:
                camera = LearnedCameraSceneNode3D.from_images(
                    images, name="camera",
                    num_rigid_control_points=object_rigid_control_points,
                    learnable_translation=self.config.is_camera_translation_learnable,
                    learnable_rotation=self.config.is_camera_rotation_learnable,
                    translation_offset_weight=self.config.camera_translation_offset_weight,
                    rotation_offset_weight=self.config.camera_rotation_offset_weight,
                    dtype=self.config.dtype,
                    frame_indices_filter=self.config.frame_indices_filter,
                    model=model,
                    camera_config=self.config.pinhole_camera_config if self.config.pinhole_camera_config is not None else self.config.synthetic_camera_config,
                    index=camera_idx,
                    resolution=self.dataset.image_shape[1:3],
                    position_spline_approximation=self.config.object_position_spline_approximation,
                    position_spline_control_points=object_rigid_control_points,
                    need_image_filter=False,
                    normalize_camera=self.config.normalize_camera
                )
            world = TimedDiscreteSceneNode3D(
                name="world", dtype=self.config.dtype)
            world.add_scene_children(camera)

            if self.config.use_progress_bar:
                bar = self.config.progress_factory.tqdm(
                    total=len(oids), desc="Creating scene nodes", tag="scene_nodes", is_reusable=False, delay=2)

            for i, x in enumerate(oids):
                mask_idx = self.dataset.mask_ids[i]

                additional_args = dict()
                if hasattr(init_strategy, "plane_position_strategy"):
                    try:
                        additional_args = self.get_plane_position_exec_args(
                            position_strategy=init_strategy.plane_position_strategy,
                            object_index=i,
                            mask_index=x,
                            images=images,
                            masks=masks,
                            depths=depth,
                            skip_object_on_missing_box=self.config.skip_object_on_missing_box
                        )
                    except BoxNotFoundError as e:
                        logger.warning(str(e))
                        continue

                obj_name = f"{i}: M-id: {mask_idx}"
                if self.config.plane_names is not None and i < len(self.config.plane_names):
                    obj_name += ' ' + self.config.plane_names[i]

                logger.info(
                    f"Creating object No. {i}, mask id {mask_idx} using name: \n\t{obj_name}")

                args = init_strategy(
                    object_index=i,
                    mask_index=x,
                    images=images,
                    masks=masks,
                    depths=depth,
                    times=dataset.frame_timestamps,
                    camera=camera,
                    nag_model=model,
                    dataset=dataset,
                    config=self.config,
                    name=obj_name,
                    **additional_args,
                    proxy_init=proxy_init,
                    runner=self,
                )
                args['proxy_init'] = proxy_init
                if isinstance(model, NAGFunctionalModel):
                    args = model.patch_plane_args(object_idx=i, args=args)

                plane_type = parse_type(
                    self.config.plane_type, LearnedImagePlaneSceneNode3D, variable_name="plane_type")
                self.config.plane_type = plane_type

                plane = plane_type(**args)
                world.add_scene_children(plane)
                if bar is not None:
                    bar.update(1)

             # Add background plane
            if self.config.has_background_plane:
                logger.info(f"Creating background...")
                used_masks = masks[:, oids, :, :]
                background_plane_type = self._get_background_plane_type(
                    used_masks, proxy_init)
                args = self._get_background_init_kwargs(background_plane_type)

                background_camera_distance = self.get_background_camera_distance(
                    masks=masks,
                    camera=camera,
                    position_strategy=init_strategy.plane_position_strategy if hasattr(
                        init_strategy, "plane_position_strategy") else None,
                    world=world,
                )

                # Background for camera init should attach itself to the camera and for functional model the patching will happen before the init within the method aswell
                background_plane = background_plane_type.for_camera(
                    camera, images=images,
                    masks=used_masks,
                    depths=depth,
                    times=dataset.frame_timestamps,
                    scene_cutoff_distance=background_camera_distance,
                    dtype=self.config.dtype,
                    relative_scale_margin=self.config.background_relative_scale_margin,
                    name=f"Background Plane {len(oids)}",
                    index=background_idx,
                    nag_model=model,
                    world=world,
                    config=self.config,
                    dataset=dataset,
                    proxy_init=proxy_init,
                    align_corners=self.config.plane_align_corners,
                    **args
                )

                # Multiply depth which is in range [0, 1] with the scene cutoff distance
                depth = depth * self.config.background_camera_distance

            if self.config.has_camera_aberration_plane:
                from nag.model.learned_aberration_plane_scene_node_3d import \
                    LearnedAberrationPlaneSceneNode3D

                aberration_idx = camera_idx + 1

                aberration_plane = LearnedAberrationPlaneSceneNode3D.for_camera(
                    camera, images=images,
                    masks=masks,
                    depths=depth,
                    times=dataset.frame_timestamps,
                    dataset=dataset,
                    camera_distance=1e-3,
                    nag_model=model,
                    dtype=self.config.dtype,
                    world=world,
                    name=f"Aberration Plane",
                    index=aberration_idx,
                    config=self.config,
                )

            order = list(world.get_scene_children())
            order = sorted(order, key=lambda x: x.get_index())
            from tools.util.torch import sort_module_list
            sort_module_list(world._scene_children, order)

            # Move Camera to first position
            cam = world._scene_children.pop(len(world._scene_children) - 1)
            world._scene_children.insert(0, cam)
            # Order should be now Camera (last-idx), Foreground planes (0, 1, 2, ...), Background plane (last-idx-1)

            if self.config._old_scene_children_order:
                # Move background plane to 2nd position
                v = world._scene_children.pop(len(world._scene_children) - 1)
                world._scene_children.insert(1, v)


            return world

    def _get_background_plane_type(self, masks: Optional[torch.Tensor], proxy_init: bool = False) -> Type[BackgroundPlaneSceneNode3D]:
        if self.config.background_plane_type != DEFAULT:
            return parse_type(self.config.background_plane_type, BackgroundPlaneSceneNode3D, variable_name="background_plane_type")
        if proxy_init:
            return BackgroundPlaneSceneNode3D  # Backwards compatibility
        covered_pixels = (masks.sum(dim=1) > 0).sum(
            dim=(-2, -1))  # Shape: (T, )
        ppim = masks.shape[-2] * masks.shape[-1]
        ratio = covered_pixels / ppim
        background_type = BackgroundPlaneSceneNode3D
        if (ratio < self.config.mask_background_coverage_threshold).any():
            background_type = BackgroundImagePlaneSceneNode3D
        self.config.background_plane_type = background_type
        return background_type

    def _get_background_init_kwargs(self, plane_type: Type[BackgroundPlaneSceneNode3D]) -> Dict[str, Any]:
        if issubclass(plane_type, BackgroundImagePlaneSceneNode3D):
            args = dict(
                **self.config.background_plane_kwargs if self.config.background_plane_kwargs is not None else dict(),
            )
            args["rgb_weight"] = self.config.plane_color_weight
            args["flow_weight"] = self.config.plane_flow_weight
            args["deprecated_flow"] = self.config.deprecated_flow
            return args
        else:
            return dict(
                background_color_fadeout=False,
                is_background_learnable=self.config.is_background_color_learnable,
                **self.config.background_plane_kwargs if self.config.background_plane_kwargs is not None else dict(),
            )

    def _get_model_args(self,
                        model_type: Type[NAGModel],
                        dataset: NAGDataset) -> Dict[str, Any]:
        resolution = (self.dataset.image_shape[2], self.dataset.image_shape[1])

        if model_type == NAGModel:
            return dict(config=self.config, resolution=resolution)
        elif model_type == NAGFunctionalModel:
            times = dataset.frame_timestamps
            if self.config.object_position_spline_approximation:
                K = self.config.object_rigid_control_points
                if K is None:
                    K = int(
                        round(len(times) * self.config.object_rigid_control_points_ratio))
                    self.config.object_rigid_control_points = K
                times = torch.linspace(0, 1, K, dtype=self.config.dtype)
            return dict(num_objects=self.config.num_objects, times=times, config=self.config, resolution=resolution)
        else:
            raise ValueError(f"Model type {model_type} not supported")

    def init_model(self,
                   model_type: Type[NAGModel],
                   dataset: NAGDataset) -> NAGModel:
        return model_type(**self._get_model_args(model_type, dataset))

    def create_model(self, bundle: Optional[dict], context: Optional[Dict[str, Any]] = None) -> NAGModel:
        from tools.util.torch import TensorUtil
        dataset: NAGDataset = self.dataset
        torch.set_float32_matmul_precision('high')
        logger.info("Loading initial data...")

        data = dict()
        if context is not None and "images" in context and "masks" in context and "depth" in context and "oids" in context:
            logger.info("Using context data instead of loading...")
            images = context["images"]
            masks = context["masks"]
            depth = context["depth"]
            oids = context["oids"]
        else:
            images, masks, depth, oids = self.load_data(dataset)
            data["images"] = images
            data["masks"] = masks
            data["depth"] = depth
            data["oids"] = oids

            if context is not None:
                context.update(data)

        try:
            logger.info("Initialize model...")
            mt = parse_type(self.config.model_type, NAGModel,
                            variable_name="model_type")
            self.config.model_type = mt
            model: NAGModel = self.init_model(mt, dataset)
            data["model"] = model

            logger.info("Creating world...")
            world = self.create_world(dataset, bundle, model, images=images,
                                      masks=masks, depth=depth, oids=oids, context=context)
            model.set_world(world)

            model.to(model.device)
        except Exception as e:
            raise ModelBuildException(
                f"Could not build model due to: {str(e)}", data=data).with_traceback(e.__traceback__)

        # model.enable_flow(False)
        # if self.config.pretraining_epochs > 0:
        #     model.enable_position_learning(False)

        # def print_fn(x: torch.Tensor, path: str):
        #     print("Device: ", x.device, "Path: ", path, "Shape: ", x.shape)
        #     return x
        # TensorUtil.apply_deep(model, print_fn)
        return model

    def create_bundle_dataset(self, bundle: Optional[dict]) -> NAGDataset:
        # Create world
        # Load images, masks and depth
        if bundle is None:
            return NAGDataset(self.config, None)
        frame_timestamps = utils.get_frame_timestamps(bundle)

        # Check if a filter is set
        if self.config.frame_indices_filter is not None:
            frame_timestamps = frame_timestamps[self.config.frame_indices_filter]

        # Normalize the time stamps
        mm = MinMax(0, 1, dim=-1)
        normalized_frame_timestamps = mm.fit_transform(
            frame_timestamps).to(self.config.dtype)

        return NAGDataset(self.config, normalized_frame_timestamps)

    def create_grid_dataset(self, dataset: NAGDataset, camera: LearnedCameraSceneNode3D):
        gs = DatasetGridSampler.for_camera(camera, dataset)
        self.notify_on_epoch_change(gs)
        return gs

    def create_dataloader(self, grid_sampler: DatasetGridSampler):
        return DataLoader(grid_sampler,
                          batch_size=1,
                          num_workers=self.config.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          prefetch_factor=(
                              1 if self.config.num_workers > 0 else None),
                          persistent_workers=True if self.config.num_workers > 0 else False,
                          )

    def create_logger(self) -> Any:
        # Parent directory of the lightning log directory
        base_dir = os.path.normpath(
            os.path.join(self.config.output_path, ".."))
        basename = os.path.basename(self.config.output_path)

        if self.config.experiment_logger == "wandb":
            try:
                import wandb
            except (ModuleNotFoundError, ImportError) as e:
                logger.error(
                    f"Could not import wandb. Please install wandb to use it as a logger. Falling back to tensorboard.")
                self.config.experiment_logger = "tensorboard"

        if self.config.experiment_logger == "tensorboard":
            from tools.logger.tensorboard import Tensorboard
            tb = TensorBoardLogger(
                save_dir=base_dir, version=".", name=basename)
            loggers = [tb]
            self.experiment_logger = Tensorboard(
                basename, base_dir,
                summary_writer=tb.experiment)
        elif self.config.experiment_logger == "wandb":
            from pytorch_lightning.loggers import WandbLogger as PLWandbLogger
            from tools.logger.wandb_logger import WandbLogger
            self.experiment_logger: WandbLogger = WandbLogger.for_experiment_config(
                self.config)
            loggers = [PLWandbLogger(
                name=self.experiment_logger.name,
                save_dir=self.experiment_logger.output_path,
                experiment=self.experiment_logger.run,
            )]

        else:
            raise ValueError(
                f"Experiment logger {self.config.experiment_logger} not supported.")
        return loggers

    def create_callbacks(self) -> List[pl.Callback]:
        # training
        lr_callback = pl.callbacks.LearningRateMonitor()
        validation_callback = NAGCallback(self)
        callbacks = [validation_callback, lr_callback]

        if not self.config.use_progress_bar:
            from nag.callbacks.logging_callback import LoggingCallback
            callbacks.append(LoggingCallback())
        return callbacks

    def create_trainer(self,
                       callbacks: Optional[
                           Union[List[pl.Callback],
                                 _DEFAULT]] = DEFAULT,
                       logger: Optional[
                           Union[Iterable[PLLogger], PLLogger,
                                 _DEFAULT]] = DEFAULT) -> pl.Trainer:
        if callbacks == DEFAULT:
            callbacks = self.create_callbacks()
        if logger == DEFAULT:
            logger = self.create_logger()

        if not isinstance(logger, Iterable):
            logger = [logger]

        self.callbacks = callbacks
        self.loggers = logger

        acc_devices = "auto"
        accelerator = "cpu"
        progress_bar = True
        if self.config.accelerator != None:
            accelerator = self.config.accelerator
            if self.config.accelerator == "gpu":
                acc_devices = self.config.accelerator_devices

        trainer = pl.Trainer(accelerator=accelerator, devices=acc_devices, num_nodes=1,
                             strategy=SingleDeviceStrategy(device="cuda:0"),
                             max_epochs=self.config.max_epochs,
                             deterministic=self.config.deterministic,
                             log_every_n_steps=self.config.log_every_n_steps,
                             enable_progress_bar=progress_bar,
                             detect_anomaly=self.config.detect_anomaly,
                             logger=logger, callbacks=callbacks, enable_checkpointing=False, fast_dev_run=self.config.debug)
        return trainer

    def train(self) -> None:
        try:
            set_default_output_dir(self.config.output_path / "viz")
            # Model manually to device
            self.model.train()
            if isinstance(self.grid_sampler._sampler, RandomTimedUVGridSampler):
                if self.config.deprecated_flow:
                    self.grid_sampler._sampler.fixed_t = self.model.get_flow_reference_times()

            # Clean Cuda cache
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            self.trainer.fit(self.model, self.train_loader,
                             ckpt_path=self.config.checkpoint_path)

        finally:
            set_default_output_dir(None)
            if self.loggers is not None:
                for l in self.loggers:
                    if isinstance(l, ExperimentLogger):
                        try:
                            l.finish()
                        except Exception as e:
                            logger.error(
                                f"Error during finishing logger : {e}")
