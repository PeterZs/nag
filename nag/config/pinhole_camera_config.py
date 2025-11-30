from dataclasses import dataclass, field
from matplotlib.figure import Figure
import numpy as np
from typing import List, Optional, Tuple, Union
from nag.config.intrinsic_camera_config import IntrinsicCameraConfig
import torch
from tools.util.typing import VEC_TYPE
from tools.transforms.to_numpy import numpyify
from tools.util.path import PATH_TYPE
from tools.util.path_tools import process_path
from matplotlib.animation import FuncAnimation
from tools.logger.logging import logger
from tools.scene.coordinate_system_3d import CoordinateSystem3D


@dataclass
class PinholeCameraConfig(IntrinsicCameraConfig):
    """Configuration parameters for a pinhole camera model."""

    resolution: Union[Tuple[int, int], VEC_TYPE] = field(default=None)
    """Resolution of the camera image (width, height)."""

    times: Union[List[float], VEC_TYPE] = field(default=None)
    """Times of the camera poses. If None, the camera is static. Should be of shape (N,) or (N, 1). With values in range [0, 1]."""

    position: Union[List[List[float]], VEC_TYPE] = field(default=None)
    """Positions of the camera poses as affine matrix. Should be of shape ([T,] [3 | 4], 4)."""

    distance_quantiles: Optional[Union[List[List[float]], VEC_TYPE]] = field(
        default=None)
    """Distance quantiles for the camera poses. Should be of shape ([T,] 2). (Closest, farthest)
    Used in NeRF for the depth of the camera poses.
    """

    normalization_position: Optional[VEC_TYPE] = field(default=None)
    """If the position of the camera was normalized P = A @ P_WL, this shall contain the transformation matrix A (4, 4) which was used to normalize the position."""

    def get_intrinsics(self,
                       resolution: Tuple[int, int],
                       dtype: torch.dtype = torch.float32,
                       device: torch.device = None) -> torch.Tensor:
        """Get the intrinsic matrix K for the camera.

        Parameters
        ----------

        resolution : Tuple[int, int]
            Resolution of the image (width, height).

        dtype : torch.dtype, optional
            Data type of the returned tensor.

        device : torch.device, optional
            Device of the returned tensor.

        Returns
        -------
        torch.Tensor
            3x3 intrinsic matrix K.
        """
        W, H = resolution
        intrinsics = torch.eye(3, dtype=dtype, device=device)
        # Set optical axis / principal point to the center of the image
        if self.principal_point is None:
            intrinsics[:2, 2] = (torch.tensor(
                [W, H], dtype=dtype, device=device) / 2)
        else:
            if W != self.resolution[0] or H != self.resolution[1]:
                pwr = self.principal_point[0] / self.resolution[0]
                phr = self.principal_point[1] / self.resolution[1]
                intrinsics[:2, 2] = torch.tensor(
                    [W * pwr, H * phr], dtype=dtype, device=device)
            else:
                intrinsics[:2, 2] = torch.tensor(
                    self.principal_point, dtype=dtype, device=device)

        intrinsics[0, 0] = self.focal_length * W
        intrinsics[1, 1] = self.focal_length * W
        intrinsics[0, 1] = self.skew
        return intrinsics

    def fields_to_native(self):
        """Convert fields to lists."""
        if self.resolution is not None:
            self.resolution = tuple(self.resolution)
        if self.times is not None and isinstance(self.times, (np.ndarray, torch.Tensor)):
            self.times = self.times.tolist()
        if self.position is not None and isinstance(self.position, (np.ndarray, torch.Tensor)):
            self.position = self.position.tolist()
        if self.distance_quantiles is not None and isinstance(self.distance_quantiles, (np.ndarray, torch.Tensor)):
            self.distance_quantiles = self.distance_quantiles.tolist()
        if self.normalization_position is not None and isinstance(self.normalization_position, (np.ndarray, torch.Tensor)):
            self.normalization_position = self.normalization_position.tolist()

    @classmethod
    def from_nerf_pose(cls, pose: VEC_TYPE) -> 'PinholeCameraConfig':
        """Create a PinholeCameraConfig from a pose matrix in the format used in NERFs.


        E.g. as in:

        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" - Mildenhall et al. (2020)
        https://github.com/bmild/nerf/blob/master/run_nerf.py#L406

        Robust Dynamic Radiance Fields - Liu et al. (2023)
        https://github.com/facebookresearch/robust-dynrf/blob/main/renderer.py#L849

        Parameters
        ----------
        pose : torch.Tensor
            Flattened combined pose matrix.
            Shape: ([N,] 17)

        Returns
        -------
        PinholeCameraConfig
            PinholeCameraConfig instance.
        """
        from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D, compose_translation_orientation
        poses = numpyify(pose)
        prev = poses[:, :15]
        near_far = poses[:, 15:17]
        ext_hwf = prev.reshape(-1, 3, 5)
        extrinsic = ext_hwf[:, :, :4]
        vec = np.zeros_like(extrinsic[:, :1, :])
        extrinsic = np.concatenate([extrinsic, vec], axis=1)
        extrinsic[:, 3, 3] = 1.

        hwf = ext_hwf[:, :, 4]
        H = hwf[0, 0]
        W = hwf[0, 1]
        f_raw = hwf[0, 2]
        f = f_raw / W
        # if W < H:
        #     f = f_raw / W  # Normalize focal length
        # else:
        #     f = f_raw / H
        principal_point = (W / 2, H / 2)

        T = extrinsic.shape[0]
        times = np.linspace(0, 1, T)

        # Nerf uses OpenGL coordinate system, so we need to rotate the camera by 180 degrees around the x-axis
        # From Camera view, Nerf uses X right, Y Up, Z backward, while we use X right, Y down, Z forward

        from tools.transforms.geometric.transforms3d import component_rotation_matrix
        mat = component_rotation_matrix(angle_x=180, mode="deg")

        rot = torch.tensor(extrinsic).float()
        extrinsic = torch.bmm(mat.unsqueeze(0).repeat(rot.shape[0], 1, 1), rot)

        return cls(
            resolution=(W, H),
            times=times,
            focal_length=f,
            principal_point=principal_point,
            position=extrinsic,
            distance_quantiles=near_far
        )

    @classmethod
    def from_nerf_pose_file(cls, path: Union[str, PATH_TYPE]) -> 'PinholeCameraConfig':
        """Create a PinholeCameraConfig from a pose matrix in the format used in NERFs.

        E.g. as in:

        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" - Mildenhall et al. (2020)
        https://github.com/bmild/nerf/blob/master/run_nerf.py#L406

        Robust Dynamic Radiance Fields - Liu et al. (2023)
        https://github.com/facebookresearch/robust-dynrf/blob/main/renderer.py#L849

        Parameters
        ----------
        path : Union[str, PATH_TYPE]
            Path to the pose matrix file. Should be a numpy file which contains the pose matrix
            in the format (N, 17).

        Returns
        -------
        PinholeCameraConfig
            PinholeCameraConfig instance.
        """
        path = process_path(path, need_exist=True, interpolate=True,
                            interpolate_object=dict(), variable_name="path")
        pose = np.load(str(path), allow_pickle=False)
        return cls.from_nerf_pose(pose)

    @classmethod
    def from_waymo_poses(cls,
                         extrinsics: VEC_TYPE,
                         intrinsics: VEC_TYPE,
                         width: int, height: int,
                         cam_to_vehicle: Optional[VEC_TYPE] = None
                         ) -> 'PinholeCameraConfig':
        """
        Create a PinholeCameraConfig from a pose matrix in the format used in Waymo Open Dataset.

        Parameters
        ----------
        extrinsics : torch.Tensor
            Flattened combined pose matrix.
            Shape: (N, 4, 4)
            Each pose matrix is of shape (4, 4) and represents the camera pose.

        intrinsics : torch.Tensor
            Intrinsic parameters
            Shape: (N, 9)
            These are: [f_u, f_v, c_u, c_v, k{1, 2, 3}, p{1, 2}]
            f_u, f_v: focal length in pixels
            c_u, c_v: principal point in pixels
            k{1, 2, 3}: radial distortion coefficients
            p{1, 2}: tangential distortion coefficients

        width : int
            Width of the image in pixels.

        height : int
            Height of the image in pixels.

        Returns
        -------
        PinholeCameraConfig
            PinholeCameraConfig instance.
        """
        from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D, compose_translation_orientation
        from nag.model.timed_discrete_scene_node_3d import plot_position
        from tools.viz.matplotlib import plot_vectors

        extrinsics = numpyify(extrinsics)
        intrinsics = numpyify(intrinsics)
        if intrinsics.shape[1] != 9:
            raise ValueError(
                f"Invalid shape {intrinsics.shape} for intrinsics. Should be (N, 9).")
        if extrinsics.shape[1:] != (4, 4):
            raise ValueError(
                f"Invalid shape {extrinsics.shape} for extrinsics. Should be (N, 4, 4).")
        if len(extrinsics.shape) != 3:
            raise ValueError(
                f"Invalid shape {extrinsics.shape} for extrinsics. Should be (N, 4, 4).")
        if len(intrinsics.shape) != 2:
            raise ValueError(
                f"Invalid shape {intrinsics.shape} for intrinsics. Should be (N, 9).")
        if intrinsics.shape[0] != extrinsics.shape[0]:
            raise ValueError(
                f"Number of intrinsics {intrinsics.shape[0]} and extrinsics {extrinsics.shape[0]} should be the same.")

        times = torch.linspace(0, 1, extrinsics.shape[0])

        # Waymo uses a right-handed world coordinate system "East-North-Up", (https://waymo.com/open/data/perception/)
        # Waymos camera system is z up, x forward, y left
        # We use X right, Y down, Z forward Therefore we need to rotate the camera by 90 degrees around the x-axis in counter-clockwise direction.
        from tools.transforms.geometric.transforms3d import component_rotation_matrix, compose_transformation_matrix

        # TODO Add cam to vehicle transformation in normalization position
        # As Objects are defined in vehicle coordinate system, we add the cam_to_vehicle transformation, as
        # inverse in the normalization position
        vehicle_to_cam = None
        if cam_to_vehicle is not None:
            cam_to_vehicle = torch.tensor(cam_to_vehicle).float()
            if cam_to_vehicle.shape[-2:] != (4, 4):
                raise ValueError(
                    f"Invalid shape {cam_to_vehicle.shape} for cam_to_vehicle. Should be (4, 4).")
            # Check difference along time
            diff = torch.diff(cam_to_vehicle, dim=0)
            if torch.any(torch.abs(diff) > 1e-3):
                logger.warning(f"cam_to_vehicle changes along time: {diff}")
            vehicle_to_cam = torch.linalg.inv(cam_to_vehicle[0]).numpy()

        vehicle_to_world = torch.tensor(extrinsics).double()

        # position = rot[:, :3, 3]
        # rotmat = rot[:, :3, :3]

        # # Rotate by 90 degrees around the x-axis and 90 degrees around the y-axis
        # # mat = component_rotation_matrix(angle_x=90, angle_y=90, mode="deg")
        # # new_rot = torch.bmm(mat[:3, :3].unsqueeze(0).expand_as(rotmat), rotmat)
        # # rot_extrinsics = compose_transformation_matrix(position, new_rot)
        # # While rotations would be fine, due to normalization of the rotations to be 0 at the first camera position,
        # # those would be undone, so we need swap the axes instead

        # # OR Swap x and z
        # # Swap new x, y
        # # Negate new x, new y
        # # Old x -> new z
        # # Old z -> new -y
        # # Old y -> new -x
        # swap = position[:, [1, 2, 0]]
        # swap[:, 0] = -swap[:, 0]
        # swap[:, 1] = -swap[:, 1]
        # rot_extrinsics = compose_transformation_matrix(swap, rotmat)

        vehicle_to_world = torch.bmm(vehicle_to_world[0:1].inverse().expand_as(
            vehicle_to_world), vehicle_to_world)  # Reset origin to the first camera position

        # fig = plot_position(vehicle_to_world , times, title="Waymo Vehicle Coordinate System")
        # fig.show()

        waymo_glob_coord = CoordinateSystem3D.from_string(
            "flu")  # Right (East), Forward (North), Up(Gravity)
        waymo_cam_coord = CoordinateSystem3D.from_string(
            "flu")  # X forway (barrel), Y left, Z up
        our_coord = CoordinateSystem3D.from_string("rdf")

        v2w = torch.tensor(waymo_glob_coord.convert(
            our_coord, vehicle_to_world), dtype=torch.float64)
        vehicle_to_cam = torch.tensor(waymo_glob_coord.convert(
            our_coord, vehicle_to_cam), dtype=torch.float64)
        c2v = torch.tensor(waymo_cam_coord.convert(
            our_coord, cam_to_vehicle[0]), dtype=torch.float64)

        cam_to_world = v2w @ c2v

        # Reset origin to the first camera position

        org_cam_to_world_0 = cam_to_world[0:1]
        rot0 = torch.inverse(org_cam_to_world_0)
        cam_to_world = torch.bmm(rot0.expand_as(cam_to_world), cam_to_world)

        # fig = plot_position(cam_to_world, times, title="Own Coordinate System")
        # fig.show()

        f_u, f_v, c_u, c_v, k1, k2, k3, p1, p2 = intrinsics.T
        if (np.abs(np.diff(f_u)) > 1e-3).any() or (np.abs(np.diff(f_v)) > 1e-3).any():
            raise ValueError(
                "Focal lengths should be the same for all camera positions.")
        if (np.abs(np.diff(c_u)) > 1e-3).any() or (np.abs(np.diff(c_v)) > 1e-3).any():
            raise ValueError(
                "Principal points should be the same for all camera positions.")
        if (np.abs(np.diff(k1)) > 1e-3).any() or (np.abs(np.diff(k2)) > 1e-3).any() or (np.abs(np.diff(k3)) > 1e-3).any():
            raise ValueError(
                "Radial distortion coefficients should be the same for all camera positions.")
        if (np.abs(np.diff(p1)) > 1e-3).any() or (np.abs(np.diff(p2)) > 1e-3).any():
            raise ValueError(
                "Tangential distortion coefficients should be the same for all camera positions.")

        return cls(
            resolution=(width, height),
            times=times,
            focal_length=f_u[0] / width,
            principal_point=[c_u[0], c_v[0]],
            position=cam_to_world,
            lens_distortion=[k1[0], k2[0], k3[0], p1[0], p2[0]],
            normalization_position=vehicle_to_cam,
        )

    @classmethod
    def from_waymo_poses_file(cls,
                              extrinsics: Union[str, PATH_TYPE],
                              intrinsics: Union[str, PATH_TYPE],
                              width: int, height: int,
                              cam_to_vehicle: Optional[Union[str,
                                                             PATH_TYPE]] = None,
                              ) -> 'PinholeCameraConfig':
        """
        Create a PinholeCameraConfig from a pose matrix in the format used in Waymo Open Dataset.

        Parameters
        ----------
        extrinsics : Union[str, PATH_TYPE]
            Path to the extrinsics file. Should be a numpy file which contains the pose matrix
            in the format (N, 4, 4).

        intrinsics : Union[str, PATH_TYPE]
            Path to the intrinsics file. Should be a numpy file which contains the intrinsics matrix
            in the format (N, 9).

        width : int
            Width of the image in pixels.

        height : int
            Height of the image in pixels.

        cam_to_vehicle : Optional[Union[str, PATH_TYPE]], optional
            Path to the cam to vehicle file. Should be a numpy file which contains the pose matrix
            in the format (N, 4, 4). If None, the camera is assumed to be in the vehicle coordinate system.


        Returns
        -------
        PinholeCameraConfig
            PinholeCameraConfig instance.
        """
        extrinsics = process_path(extrinsics, need_exist=True, interpolate=True,
                                  interpolate_object=dict(), variable_name="extrinsics")
        intrinsics = process_path(intrinsics, need_exist=True, interpolate=True,
                                  interpolate_object=dict(), variable_name="intrinsics")
        if cam_to_vehicle is not None:
            cam_to_vehicle = process_path(cam_to_vehicle, need_exist=True, interpolate=True,
                                          interpolate_object=dict(), variable_name="cam_to_vehicle")
        extrinsics = np.load(str(extrinsics), allow_pickle=False)
        intrinsics = np.load(str(intrinsics), allow_pickle=False)
        if cam_to_vehicle is not None:
            cam_to_vehicle = np.load(str(cam_to_vehicle), allow_pickle=False)
        return cls.from_waymo_poses(extrinsics, intrinsics, width, height, cam_to_vehicle=cam_to_vehicle)

    def plot_camera_motion(self, **kwargs) -> Tuple[Figure, FuncAnimation]:
        """Plot the camera motion in 3D.

        Returns
        -------
        Tuple[Figure, FuncAnimation]
            Tuple with the figure and the animation for timed camera motion.
        """

        from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D
        from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D

        world = TimedDiscreteSceneNode3D(name="world")
        camera = TimedCameraSceneNode3D.from_pinhole_camera_config(
            self, normalize=True, name="camera")

        world.add_scene_children(camera)
        args = dict(kwargs)
        if "plot_camera_traces" not in args:
            args["plot_camera_traces"] = True
        if "elevation" not in args:
            args["elevation"] = -45
        if "azimuth" not in args:
            args["azimuth"] = -90
        return world.plot_scene_animation(**args)

    def plot_camera(self, normalize: bool = True, **kwargs) -> Tuple[Figure, FuncAnimation]:
        """Plot the camera motion in 3D.

        Returns
        -------
        Tuple[Figure, FuncAnimation]
            Tuple with the figure and the animation for timed camera motion.
        """

        from nag.model.timed_discrete_scene_node_3d import TimedDiscreteSceneNode3D
        from nag.model.timed_camera_scene_node_3d import TimedCameraSceneNode3D

        world = TimedDiscreteSceneNode3D(name="world")
        camera = TimedCameraSceneNode3D.from_pinhole_camera_config(
            self, normalize=normalize, name="camera")

        world.add_scene_children(camera)
        args = dict(kwargs)
        if "plot_camera_traces" not in args:
            args["plot_camera_traces"] = True
        if "elevation" not in args:
            args["elevation"] = -45
        if "azimuth" not in args:
            args["azimuth"] = 90
        return world.plot_scene(**args)
