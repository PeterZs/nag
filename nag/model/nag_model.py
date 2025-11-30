from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from nag.config.nag_config import NAGConfig
from nag.model.background_image_plane_scene_node_3d import BackgroundImagePlaneSceneNode3D
from nag.model.learned_aberration_plane_scene_node_3d import LearnedAberrationPlaneSceneNode3D
from nag.model.background_plane_scene_node_3d import BackgroundPlaneSceneNode3D
from nag.model.discrete_plane_scene_node_3d import DiscretePlaneSceneNode3D
from nag.model.learned_camera_scene_node_3d import LearnedCameraSceneNode3D
from nag.model.learned_image_plane_scene_node_3d import LearnedImagePlaneSceneNode3D
from nag.model.phase import Phase
from nag.model.timed_plane_scene_node_3d import TimedPlaneSceneNode3D
from tools.util.typing import VEC_TYPE, NUMERICAL_TYPE
from tools.util.torch import tensorify
import torch
import pytorch_lightning as pl
from tools.model.module_scene_node_3d import ModuleSceneNode3D
from tools.util.progress_factory import ProgressFactory
from tools.util.typing import DEFAULT
from tools.logger.logging import logger
from tools.viz.matplotlib import saveable

from tools.context.temporary_device import TemporaryDevice
from tools.context.temporary_training import TemporaryTraining
from tools.util.torch import index_of_first

from nag.sampling.regular_uv_grid_sampler import RegularUVGridSampler


@torch.jit.script
def get_object_intersection_points(
    global_plane_positions: torch.Tensor,
    local_plane_scale: torch.Tensor,
    local_plane_scale_offset: torch.Tensor,
    global_ray_origins: torch.Tensor,
    global_ray_directions: torch.Tensor,
    eps: float = 1e-6,
    return_plane_positions: bool = False,
    bound_eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Calculate the intersection points of rays with objects / planes in the scene.

    If a ray intersects not with a plane, the intersection point will be set to -100 and is_inside will be False.
    If a ray intersects with a plane, but not within the planed bounds defined by the scale, the intersection point is stated but is_inside will be False.


    Parameters
    ----------
    plane_positions : torch.Tensor
        Position matricies of the planes in the scene. Shape (N, T, 4, 4)

    local_plane_scale : torch.Tensor
        Local plane scale. Shape (N, 2)

    local_plane_scale_offset : torch.Tensor
        Local plane scale offset. Shape (N, 2)

    global_ray_origins : torch.Tensor
        Global ray origins. Shape (B, T, 3)

    global_ray_directions : torch.Tensor
        Global ray directions. Shape (B, T, 3)

    eps : float
        Epsilon for numerical stability.

    return_plane_positions : bool
        If the plane positions should be returned. Default False.
        If True, the plane positions will be returned as well.

    bound_eps : float
        Epsilon for the bounds of the planes. Default 1e-5.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        1. Intersection points of the rays with the infinite planes in the scene. Shape (N, B, T, 3)
        2. Boolean tensor indicating if the rays intersect with the plane within its bounds. Shape (N, B, T)
    Or if return_plane_positions is True:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        1. & 2. as above
        3. Plane positions per ray as used to compute the intersection. Shape (N, B, T, 3)
        4. Plane normals per ray as used to compute the intersection. Shape (N, B, T, 3)
    """
    # plane_positions = torch.stack([obj.get_global_position(t=t) for obj in self.objects], dim=0) # (N, T, 4, 4)
    B, T = global_ray_origins.shape[:2]
    N = global_plane_positions.shape[0]

    normal = torch.eye(4, device=global_ray_origins.device,
                       dtype=global_ray_origins.dtype).unsqueeze(0).unsqueeze(0)
    normal[..., 2, 3] = 1  # z = 1
    normal = normal.repeat(N, T, 1, 1)  # (N, T, 4, 4)
    plane_normal_target = torch.bmm(global_plane_positions.reshape(
        N * T, 4, 4), normal.reshape(N * T, 4, 4)).reshape(N, T, 4, 4)[..., :3, 3]  # (N, T, 3)
    plane_normals = plane_normal_target - global_plane_positions[..., :3, 3]

    # Plane normals to ray_dims
    plane_n = plane_normals.unsqueeze(1).repeat(
        1, B, 1, 1)  # (N, 1, T, 3) => (N, B, T, 3)
    plane_p = global_plane_positions[..., :3, 3].unsqueeze(
        1).repeat(1, B, 1, 1)  # (N, 1, T, 3) => (N, B, T, 3)

    d = (plane_n * plane_p).sum(-1)  # dot product

    # Add N to rays
    ray_p = global_ray_origins.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, B, T, 3)
    ray_v = global_ray_directions.unsqueeze(
        0).repeat(N, 1, 1, 1)  # (N, B, T, 3)

    intersection_points = torch.empty(
        (N, B, T, 3), device=ray_v.device, dtype=ray_v.dtype)
    # Fill with -100 to indicate no intersection
    intersection_points.fill_(-100)

    # Calculate intersection points
    denom = (plane_n * ray_v).sum(-1)  # dot product
    # This should be True for parallel rays Shape (N, B, T)
    is_not_intersecting = torch.abs(denom) < eps

    dot_n_p = (plane_n * ray_p).sum(-1)  # dot product

    t = ((-(dot_n_p.unsqueeze(-1).repeat(1, 1, 1, 3) - d.unsqueeze(-1).repeat(1, 1, 1, 3))
          )[~is_not_intersecting] / denom[~is_not_intersecting].unsqueeze(-1).repeat(1, 1, 1, 3)).squeeze()

    intersection_points[~is_not_intersecting] = ray_p[~is_not_intersecting] + \
        t * ray_v[~is_not_intersecting]  # Intersection points for unlimited planes

    is_not_intersecting[~is_not_intersecting] = (
        t <= (0 - eps)).any(dim=-1)  # Check if plane is behind the camera
    intersection_points[is_not_intersecting] = -100
    # Get Plane bl
    PC = 4
    bl = torch.tensor([0, 0], device=ray_v.device,
                      dtype=ray_v.dtype)
    br = torch.tensor([0, 1], device=ray_v.device,
                      dtype=ray_v.dtype)
    tr = torch.tensor([1, 1], device=ray_v.device,
                      dtype=ray_v.dtype)
    tl = torch.tensor([1, 0], device=ray_v.device,
                      dtype=ray_v.dtype)

    # Gets the plane corners
    plane_corners = torch.stack([bl, br, tr, tl], dim=0)
    # Convert plane corners to local coordinates

    local_plane_corners = (plane_corners.unsqueeze(1).repeat(
        1, N, 1) + local_plane_scale_offset.unsqueeze(0)) * local_plane_scale.unsqueeze(0)  # Shape (2, N, 2)
    # Add z=0, and w=1
    local_plane_corners = torch.cat(
        [local_plane_corners, torch.zeros_like(local_plane_corners[..., :1]), torch.ones_like(local_plane_corners[..., :1])], dim=-1)  # Shape (2, N, 4)

    # Repeat to include T, and to vec
    # Shape (PC, N, T, 1, 4)
    local_plane_corners = local_plane_corners.unsqueeze(
        -2).repeat(1, 1, T, 1).unsqueeze(-2)

    usqp = global_plane_positions.unsqueeze(0).repeat(
        PC, 1, 1, 1, 1)  # Shape (PC, N, T, 4, 4)

    # Flatten 2, N, T
    global_bl_tr = torch.bmm(usqp.reshape(PC * N * T, 4, 4), local_plane_corners.reshape(
        PC * N * T, 4, 1)).reshape(PC, N, T, 4)[..., :3]  # Shape (2, N, T, 3)
    gbl = global_bl_tr.amin(0).unsqueeze(1).repeat(
        1, B, 1, 1)  # Shape (N, T, 3) to (N, B, T, 3)
    gtr = global_bl_tr.amax(0).unsqueeze(1).repeat(
        1, B, 1, 1)  # Shape (N, T, 3) to (N, B, T, 3)

    # Each point must be larger than the bottom left corner, and smaller than the top right corner, this is not differentiable - lets hope that the bounds are not too tight

    # Having an epsilon to avoid numerical issues
    larger_than_lower_bound = ((intersection_points - gbl) + bound_eps) > 0
    # Having an epsilon to avoid numerical issues
    smaller_than_upper_bound = ((gtr - intersection_points) + bound_eps) > 0

    is_inside = (larger_than_lower_bound & smaller_than_upper_bound).all(
        dim=-1)  # Shape (N, B, T)

    # Set intersection points to inf if not inside
    is_not_intersecting = (is_not_intersecting) | (
        ~is_inside)  # Shape (N, B, T, 1)

    if not return_plane_positions:
        return intersection_points, ~is_not_intersecting, None, None
    return intersection_points, ~is_not_intersecting, plane_p, plane_n


def get_object_information(
        objects: List[TimedPlaneSceneNode3D],
        t: torch.Tensor,
        global_ray_origins: torch.Tensor,
        global_ray_directions: torch.Tensor,
        sin_epoch: torch.Tensor
):
    N = len(objects)
    T = len(t)
    B = global_ray_origins.shape[0]

    # Check where these rays collide with the objects / planes
    global_plane_positions = torch.zeros(
        (N, T, 4, 4), device=global_ray_origins.device, dtype=global_ray_origins.dtype)
    local_plane_scale = torch.zeros(
        (N, 2), device=global_ray_origins.device, dtype=global_ray_origins.dtype)
    local_plane_scale_offset = torch.zeros(
        (N, 2), device=global_ray_origins.device, dtype=global_ray_origins.dtype)

    for i, obj in enumerate(objects):
        obj: LearnedImagePlaneSceneNode3D
        global_plane_positions[i] = obj.get_global_position(t=t)
        local_plane_scale[i] = obj.get_plane_scale()
        local_plane_scale_offset[i] = obj.get_plane_scale_offset()

    # Get the intersection points
    intersection_points, is_inside = get_object_intersection_points(
        global_plane_positions, local_plane_scale, local_plane_scale_offset, global_ray_origins, global_ray_directions)

    # Intersection Points shape is (N, B, T, 3)
    # is_inside shape is (N, B, T)

    colors = torch.zeros(
        (N, B, T, 3), device=global_ray_origins.device, dtype=global_ray_origins.dtype)
    alphas = torch.zeros(
        (N, B, T, 1), device=global_ray_origins.device, dtype=global_ray_origins.dtype)
    for i, obj in enumerate(objects):
        # Trace each ray through the object if it is inside
        obj: LearnedImagePlaneSceneNode3D
        colors[i], alphas[i] = obj(intersection_points[i],
                                   ray_origins=global_ray_origins,
                                   ray_directions=global_ray_directions,
                                   t=t,
                                   sin_epoch=sin_epoch,
                                   global_position=global_plane_positions[i],
                                   plane_scale=local_plane_scale[i],
                                   plane_scale_offset=local_plane_scale_offset[i])

    return intersection_points, is_inside, colors, alphas, global_plane_positions


@torch.jit.script
def compute_object_rgba(colors: torch.Tensor, alphas: torch.Tensor, t: torch.Tensor,
                        global_ray_origins: torch.Tensor,
                        global_ray_directions: torch.Tensor,
                        is_inside: torch.Tensor, intersection_points: torch.Tensor,
                        get_object_alpha_chain: bool = False
                        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute the RGBA values of the objects in the scene, by multi level alpha matting.

    Parameters
    ----------
    colors : torch.Tensor
        Colors of the objects. Shape (N, B, T, 3)
    alphas : torch.Tensor
        Alphas of the objects. Shape (N, B, T, 1)
    t : torch.Tensor
        Time steps to evaluate the objects at. Shape (T)
    global_ray_origins : torch.Tensor
        The global ray origins. Shape (B, T, 3)
    global_ray_directions : torch.Tensor
        The global ray directions. Shape (B, T, 3)
    is_inside : torch.Tensor
        If the rays are inside the objects. Shape (N, B, T)
    intersection_points : torch.Tensor
        The intersection points of the rays with the objects. Shape (N, B, T, 3)
    get_object_alpha_chain : bool
        If the object alpha chain should be returned. Default False.
        False refers to return 1, True to Return 2 .

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    The order of N is for 1. and 2. the order of the objects beeing hit. This is NOT the order of colors and alphas as they are sorted by distance.
    3. will give the order of the objects by distance. This can also be used to "unsort" the RGB and alpha values again to the order given in colors and alphas.

        1. The RGB values of the objects (sorted_colors). Shape (N, B, T, 3)
        2. The alpha values of the objects (sorted_alphas). Shape (N, B, T, 1)
    Doing a (sorted_alphas * sorted_colors).sum(dim=0) will give the final color of each ray.

    If get_object_alpha_chain is True, the following will be returned:

    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        1. S.a
        2. S.a
        3. The order of the objects. Shape (N, B, T)
        4. The alpha chain of the objects. Shape (N, B, T, 1)
           Already integrated in alphas, but can be used to debug the alpha matting.
        5. The alphas of the objects. Shape (N, B, T, 1)
           The alphas adjusted for non hits. Also already integrated in sorted_alphas, but can be used to debug the alpha matting.

    """
    N = colors.shape[0]
    B = global_ray_origins.shape[0]
    T = t.shape[0]

    # Determine the order of plane hits by calculating the distance to the camera / ray origin
    direction_vec = intersection_points - \
        global_ray_origins.unsqueeze(0)  # (N, B, T, 3)
    distance = torch.norm(direction_vec, dim=-1)  # (N, B, T)

    # Set distance to inf if not inside
    # distance[~is_inside] = 0

    # Set alphas to 0 if not inside
    alphas = alphas * \
        is_inside.unsqueeze(-1).to(dtype=alphas.dtype)  # (N, B, T, 1)

    # Set alpha to 0 if not inside
    alphas[~is_inside] = 0.

    # Find the order of planes by sorting the distance
    order = torch.argsort(distance, dim=0).to(dtype=torch.int32)  # (N, B, T)

    inv_alphas = (1 - alphas)  # (N, B, T, 1)

    sorted_inv_alphas = torch.zeros_like(inv_alphas).unsqueeze(
        0).repeat(N, 1, 1, 1, 1)  # (N, N, B, T, 1)

    # Apply N object alpha matting. This is done by multiplying the alpha of the object with the inverse of the alphas of the objects before it.
    # We do it by calculating 1-alpha for each object, and
    bidx = torch.arange(B, device=inv_alphas.device).unsqueeze(
        0).unsqueeze(-1).repeat(N, 1, T)  # (N, T, B)
    tidx = torch.arange(T, device=inv_alphas.device).unsqueeze(
        0).unsqueeze(0).repeat(N, B, 1)

    # Sort them in the first dimension
    sorted_inv_alphas = inv_alphas[order, bidx, tidx]  # (N, B, T, 1)
    sorted_alphas = alphas[order, bidx, tidx]  # (N, B, T, 1)
    sorted_colors = colors[order, bidx, tidx]  # (N, B, T, 3)

    rolled_inv_alpha = torch.roll(sorted_inv_alphas, 1, dims=0)
    rolled_inv_alpha[0] = 1.  # The first object has no previous object

    # 1 - alpha terms for each object if in front of the current object
    alpha_chain = torch.cumprod(rolled_inv_alpha, dim=0)
    sorted_per_layer_alphas = alpha_chain * sorted_alphas
    # ray_color = (sorted_per_layer_alphas * sorted_colors).sum(dim=0)  # (B, T, 3)

    # Check if the alphas sum to 1
    # torch.allclose((torch.cumprod(rolled_inv_alpha, dim=0) * sorted_alphas).sum(dim=0), torch.tensor(1.))
    if not get_object_alpha_chain:
        return sorted_colors, sorted_per_layer_alphas
    return sorted_colors, sorted_per_layer_alphas, order, alpha_chain, sorted_alphas


@torch.jit.script
def undo_intersection_ordering(
    order: torch.Tensor,
    sorted_color: Optional[torch.Tensor] = None,
    sorted_assembled_alpha: Optional[torch.Tensor] = None,
    sorted_alpha_chain: Optional[torch.Tensor] = None,
    sorted_alpha: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Undo the ordering of colors etc. based on the order tensor.
    Assumes all inputs are of shape (N, B, T, ...) where N is the number of objects, B is the batch size, and T is the number of time steps.
    Will invert the ordering of when the objects were hit to the original order of planes within the model.

    Parameters
    ----------
    order : torch.Tensor
        Order of the objects. Shape (N, B, T)
    sorted_color : Optional[torch.Tensor], optional
        Sorted colors per object Shape (N, B, T, 3), by default None
    sorted_assembled_alpha : Optional[torch.Tensor], optional
        Sorted assembled alphas (alpha chain * object alphas) per object Shape (N, B, T, 1), by default None
    sorted_alpha_chain : Optional[torch.Tensor], optional
        Sorted alpha chain per object Shape (N, B, T, 1), by default None
    sorted_alpha : Optional[torch.Tensor], optional
        Sorted object alphas per object Shape (N, B, T, 1), by default None
        Only the alpha as predicted by the planes networks.
    Returns
    -------
    Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
        Returns the unsorted tensors in the same shape as the input tensors.
        1. The unsorted color per object. Shape (N, B, T, 3)
        2. The unsorted assembled alpha per object. Shape (N, B, T, 1)
        3. The unsorted alpha chain per object. Shape (N, B, T, 1)
        4. The unsorted alpha per object. Shape (N, B, T, 1)
        Values are None if the input was None.
    """
    arg_order = torch.argsort(order, dim=0)
    res_sorted_object_colors: Optional[torch.Tensor] = None
    if sorted_color is not None:
        res_sorted_object_colors = sorted_color.gather(
            0, arg_order.unsqueeze(-1).repeat(1, 1, 1, 3))
    res_sorted_object_assembled_alphas: Optional[torch.Tensor] = None
    if sorted_assembled_alpha is not None:
        res_sorted_object_assembled_alphas = sorted_assembled_alpha.gather(
            0, arg_order.unsqueeze(-1))
    res_sorted_alpha_chain: Optional[torch.Tensor] = None
    if sorted_alpha_chain is not None:
        res_sorted_alpha_chain = sorted_alpha_chain.gather(
            0, arg_order.unsqueeze(-1))
    res_sorted_object_alphas: Optional[torch.Tensor] = None
    if sorted_alpha is not None:
        res_sorted_object_alphas = sorted_alpha.gather(
            0, arg_order.unsqueeze(-1))
    return res_sorted_object_colors, res_sorted_object_assembled_alphas, res_sorted_alpha_chain, res_sorted_object_alphas


class NAGModel(pl.LightningModule):
    """Pytorch Module class optimizing a NAG world with objects and a camera."""

    _camera: List[LearnedCameraSceneNode3D]
    """Camera of the scene. This is just a reference to the camera in the world for easier access. Wrapped in a list to avoid torch doubled indexing."""

    _focal_length: torch.Tensor
    """Focal length of the camera. This is just a reference to the focal length of the camera in the world for easier access."""

    _world: ModuleSceneNode3D
    """The world scene node containing all objects and the camera."""

    _objects: List[ModuleSceneNode3D]
    """List of objects in the scene. These are just references to the objects in the world for easier access."""

    _config: NAGConfig
    """Configuration for the NAG model"""

    sin_epoch: Optional[torch.Tensor]
    """Sine of the epoch for time dependent functions. sin(epoch/max epoch)"""

    next_sin_epoch: Optional[torch.Tensor]
    """Sine of the next epoch for time dependent functions. sin((epoch+1)/max epoch)"""

    num_batches: Optional[int]
    """Number of batches within the current epoch."""

    batch_idx: Optional[int]
    """Current batch index within the current epoch."""

    allow_loading_unmatching_parameter_sizes: bool
    """If the model should allow loading of unmatching parameter sizes. If False, it will raise an error if the sizes do not match. Default False"""

    load_missing_parameters_as_defaults: bool
    """If the model should load missing parameters as defaults. Default False"""

    active_training_phase: int
    """The active training phase index of the model."""

    training_phases: List[Phase]
    """The training phase of the model."""

    _flow_reference_times: Optional[torch.Tensor]
    """Reference times for any object that has a flow. These times must be included within each query to ensure correct flow computation."""

    # region Init- and Setup-Methods

    def __init__(self,
                 config: NAGConfig,
                 world: Optional[ModuleSceneNode3D] = None,
                 allow_loading_unmatching_parameter_sizes: bool = False,
                 load_missing_parameters_as_defaults: bool = True,
                 resolution: Optional[Tuple[int, int]] = None,
                 ):
        super().__init__()
        # Register world as buffer, reference parameters are normal properties
        self.allow_loading_unmatching_parameter_sizes = allow_loading_unmatching_parameter_sizes
        self.load_missing_parameters_as_defaults = load_missing_parameters_as_defaults
        self._world = None
        self._camera = [None]
        self._focal_length = None
        self._objects = None
        self._config = config
        self._lr_scheduler_enabled = config.lr_scheduler
        self.register_buffer("sin_epoch", torch.tensor(1., dtype=config.dtype))
        self.register_buffer("_flow_reference_times",
                             torch.zeros(0), persistent=False)
        self._flow_reference_loaded = False
        if world is not None:
            self._world = world
            self.setup_scene(world, config)
        self.loss = self.setup_loss(resolution)
        self.setup_training_phases(config)
        self.setup_progress(config)

    def setup_progress(self, config: NAGConfig):
        self.register_buffer("next_sin_epoch", torch.tensor(
            1., dtype=config.dtype), persistent=False)
        self.register_buffer("sin_epoch", torch.tensor(1., dtype=config.dtype))
        self.register_buffer(
            "num_batches", torch.tensor(-1, dtype=torch.int32), persistent=False)
        self.register_buffer("batch_idx", torch.tensor(-1,
                             dtype=torch.int32), persistent=False)

    def setup_loss(self, resolution: Optional[Tuple[int, int]]) -> None:
        from tools.metric.torch.reducible import Metric
        from tools.metric.torch.module_mixin import ModuleMixin
        from tools.util.format import parse_type

        """Setup the loss function for the NAG model."""
        loss_type = self.config.loss_type
        loss_args = self.config.loss_kwargs if self.config.loss_kwargs is not None else {}
        if issubclass(loss_type, ModuleMixin):
            loss_args["module"] = self
        loss = loss_type(**loss_args)

        return loss

    def setup_training_phases(self, config: NAGConfig):
        """Setup the training phases for the model."""
        self.active_training_phase = -1
        phases = Phase.parse(config.training_phases,
                             max_time=config.max_epochs)
        self.training_phases = phases

    def setup_scene(self, world: ModuleSceneNode3D, config: NAGConfig):
        """Setup the scene with the world and camera."""
        # Look for camera in world
        cam = next(world.find(lambda x: isinstance(
            x, LearnedCameraSceneNode3D), include_self=False, find_first=True), None)
        # if no camera found, raise error
        if cam is None:
            raise ValueError("No camera found in world.")
        objects = world.find(lambda x: isinstance(
            x, DiscretePlaneSceneNode3D), include_self=False)
        objects = sorted(objects, key=lambda x: x.get_index())
        self.objects = objects
        self.set_camera(cam)

        # Get flow reference times
        self._flow_reference_times = self.get_flow_reference_times()

    # endregion

    def get_index_object_mapping(self) -> Mapping[int, Any]:
        """Get the mapping of object indices to the objects.

        Returns
        -------
        Mapping[int, Any]
            The mapping of object indices to the objects.
        """
        objs = list(self.objects) + [self.camera]
        return {obj.get_index(): obj for obj in objs}

    def get_flow_reference_times(self) -> Optional[torch.Tensor]:
        """Get the reference times for the flow objects."""
        return torch.tensor(sorted(list(set([x.flow_reference_time.item() for x in self.objects if isinstance(x, (LearnedImagePlaneSceneNode3D, BackgroundImagePlaneSceneNode3D))]))), dtype=self.config.dtype, device=self.device)

    @property
    def camera(self) -> LearnedCameraSceneNode3D:
        """Get the camera of the scene."""
        return self._camera[0]

    @property
    def focal_length(self) -> torch.Tensor:
        """Get the focal length of the camera."""
        if self._focal_length is None and self.camera is not None:
            self._focal_length = self.camera.focal_length
        return self._focal_length

    @camera.setter
    def camera(self, value: LearnedCameraSceneNode3D):
        """Set the camerase of the scene."""
        if value is not None and not isinstance(value, LearnedCameraSceneNode3D):
            raise ValueError("Camera must be a LearnedCameraSceneNode3D.")
        self._camera = [value]

    @property
    def background(self) -> Optional[BackgroundPlaneSceneNode3D]:
        """Get the background plane of the scene."""
        return next((x for x in self.objects if isinstance(x, BackgroundPlaneSceneNode3D)), None)

    def set_camera(self, value: LearnedCameraSceneNode3D):
        """Set the camera of the scene."""
        if value is not None and not isinstance(value, LearnedCameraSceneNode3D):
            raise ValueError("Camera must be a LearnedCameraSceneNode3D.")
        self._camera = [value]

    def get_background_plane(self) -> Optional[BackgroundPlaneSceneNode3D]:
        """Get the background plane of the scene."""
        return next((x for x in self.objects if isinstance(x, BackgroundPlaneSceneNode3D)), None)

    def get_background_color(self) -> Optional[torch.Tensor]:
        """Get the background color of the scene."""
        bg = self.get_background_plane()
        if bg is not None:
            return bg._background_color
        return torch.zeros(3, dtype=self.config.dtype)

    @property
    def world(self) -> ModuleSceneNode3D:
        """Get the world scene node."""
        return self._world

    @world.setter
    def world(self, value: ModuleSceneNode3D):
        """Set the world scene node."""
        if value == self._world:
            return
        self._world = value
        if value is not None:
            self.setup_scene(value, self.config)

    def set_world(self, value: ModuleSceneNode3D):
        """Set the world scene node."""
        if value == self._world:
            return
        self._world = value
        if value is not None:
            self.setup_scene(value, self.config)

    @property
    def objects(self) -> List[ModuleSceneNode3D]:
        """Get the objects in the scene. Returns a copy of the list."""
        return list(self._objects)

    @objects.setter
    def objects(self, objects: List[ModuleSceneNode3D]):
        """Set the objects in the scene."""
        self._objects = list(objects)

    @property
    def config(self) -> NAGConfig:
        """Get the configuration of the NAG model."""
        return self._config

    @property
    def flow_reference_times(self) -> torch.Tensor:
        """Get the reference times for the objects with flow.

        Returns
        -------
        torch.Tensor
            The unique reference times for the objects with flow. Shape (TF, )
        """
        if self._flow_reference_loaded:
            return self._flow_reference_times
        else:
            self._flow_reference_times = self.get_flow_reference_times()
            self._flow_reference_loaded = True
            return self._flow_reference_times

    def forward(self, uv: torch.Tensor, t: torch.Tensor, **kwargs):
        """Forward pass of the NAG model.

        Parameters
        ----------
        uv : torch.Tensor
            UV coordinates to evaluate the scene at. Shape (B, 2)
            UV coordinates should be in camera space (x, y). [0, width) and [0, height)

        t : torch.Tensor
            Time steps to evaluate the scene at. Shape (T,)

        Returns
        -------
        torch.Tensor
            The color of the scene at the given UV coordinates and time steps. Shape (B, T, 3)
        """
        global_ray_origins, global_ray_directions = self.camera.get_global_rays(
            uv, t, uv_includes_time=False)  # UV has shape (B, 2)

        # global_ray_origins shape is (B, T, 3) and global_ray_directions shape is (B, T, 3)
        intersection_points, is_inside, colors, alphas, global_plane_positions = get_object_information(objects=self.objects, t=t,
                                                                                                        global_ray_origins=global_ray_origins,
                                                                                                        global_ray_directions=global_ray_directions,
                                                                                                        sin_epoch=self.sin_epoch)
        object_colors, object_alphas = compute_object_rgba(colors=colors, alphas=alphas, t=t,
                                                           global_ray_origins=global_ray_origins, global_ray_directions=global_ray_directions,
                                                           is_inside=is_inside, intersection_points=intersection_points)

        return (object_alphas * object_colors).sum(dim=0)

    def object_specific_forward(self,
                                uv: torch.Tensor,
                                t: torch.Tensor,
                                enabled_objects: torch.Tensor,
                                enabled_background: bool = True,
                                enabled_aberration: bool = True,
                                color_composing: bool = True,
                                context: Optional[Dict[str, Any]] = None,
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        objects = [self.objects[i] for i in torch.argwhere(enabled_objects)]
        if not enabled_background:
            objects = [o for o in objects if not isinstance(
                o, BackgroundPlaneSceneNode3D)]
        if not enabled_aberration:
            objects = [o for o in objects if not isinstance(
                o, LearnedAberrationPlaneSceneNode3D)]
        global_ray_origins, global_ray_directions = self.camera.get_global_rays(
            uv, t, uv_includes_time=False)  # UV has shape (B, 2)

        # global_ray_origins shape is (B, T, 3) and global_ray_directions shape is (B, T, 3)
        intersection_points, is_inside, colors, alphas, global_plane_positions = get_object_information(objects=objects, t=t,
                                                                                                        global_ray_origins=global_ray_origins,
                                                                                                        global_ray_directions=global_ray_directions,
                                                                                                        sin_epoch=self.sin_epoch)

        if color_composing:
            object_colors, object_composed_alphas = compute_object_rgba(colors=colors, alphas=alphas, t=t,
                                                                        global_ray_origins=global_ray_origins, global_ray_directions=global_ray_directions,
                                                                        is_inside=is_inside, intersection_points=intersection_points)
            ray_color = (object_composed_alphas * object_colors).sum(dim=0)
            return ray_color, object_composed_alphas.sum(dim=0)
        else:
            # TODO This will not support actual composing of multiple objects right now. As alpha chain is not used.
            object_colors, object_composed_alphas, order, alpha_chain, sorted_per_plane_alphas = compute_object_rgba(colors=colors, alphas=alphas, t=t,
                                                                                                                     global_ray_origins=global_ray_origins, global_ray_directions=global_ray_directions,
                                                                                                                     is_inside=is_inside, intersection_points=intersection_points, get_object_alpha_chain=True)
            # Undo the order
            arg_order = torch.argsort(order, dim=0)
            original_sorted_colors = object_colors.gather(
                0, arg_order.unsqueeze(-1).repeat(1, 1, 1, 3))
            original_sorted_alphas = object_composed_alphas.gather(
                0, arg_order.unsqueeze(-1))
            original_sorted_object_alphas = sorted_per_plane_alphas.gather(
                0, arg_order.unsqueeze(-1))
            return original_sorted_colors, original_sorted_object_alphas, None

    def generate_outputs(self,
                         config: NAGConfig,
                         t: Optional[Union[VEC_TYPE, NUMERICAL_TYPE]] = None,
                         resolution: Optional[VEC_TYPE] = None,
                         objects: Optional[Union[List[TimedPlaneSceneNode3D],
                                                 List[List[TimedPlaneSceneNode3D]]]] = None,
                         progress_bar: bool = True,
                         progress_factory: Optional[ProgressFactory] = DEFAULT,
                         device: Optional[torch.device] = None,
                         context: Optional[Dict[str, Any]] = None,
                         **kwargs
                         ) -> torch.Tensor:
        """
        Generates images of the encoded scene at respective timestamps.

        config: NAGConfig
            Configuration for the NAG model.
            Can be used to override certain settings during generation.

        t: Optional[Union[VEC_TYPE, NUMERICAL_TYPE]]
            Time steps to evaluate the scene at. Shape (T, ).
            If None, uses the camera's times.

        resolution: Optional[VEC_TYPE]
            Resolution of the generated images. Shape (2, ).

        objects: Optional[Union[List[TimedPlaneSceneNode3D], List[List[TimedPlaneSceneNode3D]]]]
            List of objects to generate outputs for.
            If a list of lists is provided, multiple compositions will be generated.
            If None, generates outputs for all objects in the scene.
            E.g. [[obj1, obj2], [obj3]] will generate two outputs, one with obj1 and obj2, and one with obj3.
            Yet, different combinations of the same object [[obj1, obj2], [obj2, obj1]] are not fully supported and produce incorrect results.
            To yield correct results, call this function multiple times with different object lists.

            Yet disjoint sets of objects are supported in one call, e.g. [[obj1], [obj2], [obj3]]].

            Output shape will be 
        progress_bar: bool
            If a progress bar should be shown during generation.

        progress_factory: Optional[ProgressFactory]
            Factory to create progress bars. If None, uses the default factory.
        device: Optional[torch.device]
            Device to generate the images on. If None, uses the current device of the model.

        Returns
        -------
        torch.Tensor
            Generated images. Shape [N, T, C, H, W]
        """
        # Create a UV Grid sampler
        simple_composition = True
        object_compositions = []
        if objects is not None:
            if len(objects) > 0 and isinstance(objects[0], list):
                simple_composition = False
                # Find a set of objects to generate outputs for
                obj = set()
                for obj_list in objects:
                    for o in obj_list:
                        obj.add(o)
                object_compositions = objects
                objects = list(obj)
            else:
                object_compositions = [objects]

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        if progress_bar:
            if progress_factory is None:
                progress_factory = ProgressFactory()
            if progress_factory == DEFAULT:
                progress_factory = config.progress_factory

        resolution = tensorify(resolution).detach().cpu(
        ) if resolution is not None else None
        if resolution is None:
            resolution = self.camera._image_resolution.flip(-1).detach().cpu()

        t = tensorify(t, dtype=self.camera._translation.dtype,
                      device=device) if t is not None else self.camera._times
        if len(t.shape) == 0:
            t = t.unsqueeze(0)

        object_mask = None
        object_mask_output_indices = None
        enabled_background = False
        enabled_aberration = False
        background_idx = -1
        aberration_idx = -1
        if objects is not None:
            N = len(self.objects)
            if self.config.has_background_plane:
                N -= 1
            if self.config.has_camera_aberration_plane:
                N -= 1

            object_mask = torch.zeros(N, dtype=torch.bool, device=device)
            all_object_indices = self._node_indices.to(device)
            all_mask = torch.ones(len(all_object_indices),
                                  dtype=torch.bool, device=device)
            all_mask[self._camera_index_in_node_indices] = False
            if self.config.has_background_plane:
                all_mask[self._background_index_in_node_indices] = False
            filtered_indices = all_object_indices[all_mask]

            for i, obj in enumerate(objects):
                if isinstance(obj, BackgroundPlaneSceneNode3D):
                    enabled_background = True
                    background_idx = torch.tensor(next((i for i, x in enumerate(
                        self.objects) if isinstance(x, BackgroundPlaneSceneNode3D))), device=device)
                    continue
                if isinstance(obj, LearnedAberrationPlaneSceneNode3D):
                    enabled_aberration = True
                    aberration_idx = torch.tensor(next((i for i, x in enumerate(
                        self.objects) if isinstance(x, LearnedAberrationPlaneSceneNode3D))), device=device)
                    continue
                idx = obj.get_index()
                object_mask[filtered_indices == idx] = True
            if not object_mask.any() and not enabled_background and not enabled_aberration:
                raise ValueError(
                    "No objects specified to generate outputs for.")
            if object_mask.all() and simple_composition and enabled_background:
                object_mask = None
        # Initialize image tensor

        if object_mask is not None:
            # Add alpha channel
            object_mask_output_indices = torch.argwhere(
                object_mask).squeeze(-1)
            if enabled_background:
                object_mask_output_indices = torch.cat(
                    [object_mask_output_indices, background_idx.unsqueeze(0)])
            if enabled_aberration:
                object_mask_output_indices = torch.cat(
                    [object_mask_output_indices, aberration_idx.unsqueeze(0)])

        return_composing_indices = []

        if not simple_composition:
            # Define the composing indices to reassemble the images from a combined query
            selected_objects = torch.argwhere(object_mask).squeeze(-1)
            for i, obj_list in enumerate(object_compositions):
                # Subselect the relevant objects
                idx = torch.tensor([o.get_index(
                ) for o in obj_list], device=selected_objects.device, dtype=selected_objects.dtype)
                contains = torch.isin(selected_objects, idx)
                hasbg = any(isinstance(o, BackgroundPlaneSceneNode3D)
                            for o in obj_list)
                hasabr = any(isinstance(o, LearnedAberrationPlaneSceneNode3D)
                             for o in obj_list)
                selobj = selected_objects[contains]
                if hasbg:
                    selobj = torch.cat([selobj, background_idx.unsqueeze(0)])
                if hasabr:
                    selobj = torch.cat([selobj, aberration_idx.unsqueeze(0)])
                return_composing_indices.append(selobj)

        with TemporaryDevice(self, device) as dm:
            with TemporaryTraining(self, False), torch.no_grad():
                # Iterate over the grid sampler
                return self._generate_outputs(t=t.to(device=dm.device),
                                              device=dm.device,
                                              config=config,
                                              resolution=resolution,
                                              object_mask=object_mask,
                                              simple_composition=simple_composition,
                                              enabled_background=enabled_background,
                                              enabled_aberration=enabled_aberration,
                                              return_composing_indices=return_composing_indices,
                                              object_mask_output_indices=object_mask_output_indices,
                                              progress_bar=progress_bar, progress_factory=progress_factory,
                                              context=context,
                                              )

    def _get_t_with_flow_reference_times(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the times with the flow reference times added if they are missing.

        Parameters
        ----------
        t : torch.Tensor
            The times to evaluate the scene at. Shape (T, )

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            1. The times with the flow reference times added if they are missing. Shape (TF, )
            2. The boolean mask indicating which times were added. Shape (TF, )
        """
        flow_added_times = torch.zeros_like(
            t, dtype=torch.bool, device=t.device)
        if self.config.plane_flow_weight > 0 and self.config.deprecated_flow:
            flow_reference_times = self.flow_reference_times
            # Add the flow reference times to the times if they are missing
            contains = index_of_first(t, flow_reference_times)
            missing = contains == -1
            if missing.any():
                missing_times = flow_reference_times[missing]
                t = torch.cat([t, missing_times], dim=0).sort().values
                flow_added_times = torch.zeros_like(
                    t, dtype=torch.bool, device=t.device)
                flow_added_times[index_of_first(t, missing_times)] = True
        return t, flow_added_times

    def _generate_outputs(self,
                          t: torch.Tensor,
                          device: torch.device,
                          config: NAGConfig,
                          resolution: torch.Tensor,
                          object_mask: Optional[torch.Tensor] = None,
                          simple_composition: bool = True,
                          enabled_background: bool = True,
                          enabled_aberration: bool = True,
                          return_composing_indices: Optional[List[torch.Tensor]] = None,
                          object_mask_output_indices: Optional[torch.Tensor] = None,
                          progress_bar: bool = True,
                          progress_factory: Optional[ProgressFactory] = None,
                          context: Optional[Dict[str, Any]] = None,
                          ):
        with torch.no_grad(), TemporaryTraining(self, False):
            import gc
            t, flow_added_times = self._get_t_with_flow_reference_times(t)

            uv_max = self.camera._image_resolution.flip(-1).detach().cpu()
            gs = RegularUVGridSampler(resolution=tuple(resolution.int().tolist()),
                                      uv_max=uv_max,
                                      inter_pixel_noise_fnc=None,
                                      t=t, config=config,
                                      max_total_batch_size=config.max_total_batch_size_inference
                                      )

            image, subsample, subsample_offsets = gs.get_proto_image_tensor(
                t=t)

            # Image Shape T, C, H, W

            if object_mask is not None:
                # If we subselect objects, we may get transparency so we need to add an alpha channel
                image = torch.cat(
                    [image, torch.ones_like(image[:, :1])], dim=1)

            T, C, H, W = image.shape
            N = None
            if not simple_composition:
                O = object_mask.sum()  # Number of total query objects

            if not simple_composition:
                # Repeat the image tensor for each object composition
                image = image.unsqueeze(0).repeat(
                    len(return_composing_indices), 1, 1, 1, 1)  # N, T, C, H, W
                N = image.shape[0]

            bar = None
            if progress_bar:
                bar = progress_factory.bar(total=len(
                    gs), desc="Generating Image", tag="nag_model_generate_outputs", is_reusable=True)

            inner_context = None
            if context is not None:
                # Prepare context to add intersection points and is_inside checks.
                if not simple_composition:
                    O = object_mask.sum()
                    # Intersection points in plane local coordinates
                    intersection_points = torch.zeros(
                        O, T, H, W, 2, dtype=torch.float32)
                    # Is inside the object
                    is_inside = torch.zeros(O, T, H, W, dtype=torch.bool)
                    inner_context = dict()
            for i in range(len(gs)):
                uv, t = gs[i]
                uv = uv.to(device=device)
                t = t.to(device=device)
                # Get the predicted color
                alpha = None
                alpha_chain = None
                if object_mask is None:
                    rgb = self(uv, t)
                else:
                    if simple_composition:
                        rgb, alpha = self.object_specific_forward(
                            uv, t, object_mask,
                            enabled_background=enabled_background,
                            enabled_aberration=enabled_aberration,
                            color_composing=simple_composition,
                            context=inner_context
                        )
                    else:
                        rgb, alpha, alpha_chain = self.object_specific_forward(
                            uv, t, object_mask, enabled_background=enabled_background,
                            enabled_aberration=enabled_aberration,
                            color_composing=simple_composition,
                            context=inner_context
                        )

                if simple_composition:
                    rgb = gs.batch_tensor_to_image_tensor(
                        rgb, i).detach().cpu()
                    # Fill the image tensor
                    image[:, :3,
                          subsample_offsets[i, 1]::subsample[1],
                          subsample_offsets[i, 0]::subsample[0]] = rgb
                    if alpha is not None:
                        alpha = gs.batch_tensor_to_image_tensor(
                            alpha, i).detach().cpu()
                        image[:, 3:4,
                              subsample_offsets[i, 1]::subsample[1],
                              subsample_offsets[i, 0]::subsample[0]] = alpha
                else:
                    object_rgb, obj_alpha = rgb, alpha
                    for img_idx, indices in enumerate(return_composing_indices):
                        # Subselect the relevant objects
                        indices_mask = index_of_first(
                            indices, object_mask_output_indices) >= 0
                        object_ray_rgb = object_rgb[indices_mask]
                        object_ray_alpha = obj_alpha[indices_mask]

                        if len(indices) > 1:
                            # TODO This will be wrong if the indices is different than the one queried with as the cum_prod to form the alpha chain is not ignoring the missing objects
                            # Could be fixed, but its not worth the effort right now, as this only brings a benefit if the function shall be used to query multiply compositions in one pass, currently this is just handled in multiple passes
                            # Slightly more inefficient, but not a big deal.
                            if not indices_mask.all():
                                logger.warning(
                                    "The indices queried for composing the objects are not the same as the ones used for the forward pass. This will lead to incorrect results for opacity.")
                            object_ray_alpha = object_ray_alpha * \
                                alpha_chain[indices_mask]

                        ray_color = (object_ray_alpha *
                                     object_ray_rgb).sum(dim=0)
                        ray_alpha = object_ray_alpha.sum(dim=0)
                        rgb = gs.batch_tensor_to_image_tensor(
                            ray_color, i).detach().cpu()
                        alpha = gs.batch_tensor_to_image_tensor(
                            ray_alpha, i).detach().cpu()
                        image[img_idx, :, :3,
                              subsample_offsets[i, 1]::subsample[1],
                              subsample_offsets[i, 0]::subsample[0]] = rgb
                        image[img_idx, :, 3:4,
                              subsample_offsets[i, 1]::subsample[1],
                              subsample_offsets[i, 0]::subsample[0]] = alpha
                        del ray_color, ray_alpha, rgb, alpha

                    if inner_context is not None:
                        # Pop the items to save memory
                        i_H, i_W = gs.get_batch_shape(i)
                        inter = inner_context.pop("intersection_points")[
                            0]  # Shape (O, B, T, 2)
                        intersection_points[:, :,
                                            subsample_offsets[i, 1]::subsample[1],
                                            subsample_offsets[i, 0]::subsample[0]] = inter.reshape(O, i_H, i_W, T, 2).permute(0, 3, 1, 2, 4).detach().cpu()
                        inside = inner_context.pop("is_inside")[
                            0]  # Shape (O, B, T, 2)
                        is_inside[:, :,
                                  subsample_offsets[i, 1]::subsample[1],
                                  subsample_offsets[i, 0]::subsample[0]] = inside.reshape(O, i_H, i_W, T).permute(0, 3, 1, 2).detach().cpu()
                        del inter, inside
                if progress_bar:
                    bar.update()
                torch.cuda.empty_cache()
                gc.collect()

            if context is not None:
                # Add the intersection points and is_inside checks to the context
                context["intersection_points"] = intersection_points
                context["is_inside"] = is_inside

            if flow_added_times.any():
                # Remove the unwanted times from image
                if simple_composition:
                    image = image[~flow_added_times.cpu()]
                else:
                    image = image[:, ~flow_added_times.cpu()]

                if context is not None:
                    # Remove the unwanted times from context
                    context["intersection_points"] = context["intersection_points"][
                        :, ~flow_added_times.cpu()]
                    context["is_inside"] = context["is_inside"][
                        :, ~flow_added_times.cpu()]
            return image

    def training_step(self, train_batch: Any, batch_idx: torch.Tensor) -> torch.Tensor:
        # torch.autograd.set_detect_anomaly(True)
        uv, true_rgb, t, weight_t, data = train_batch
        # Squeeze the Dataloader batch dim
        uv = uv.squeeze(0)
        true_rgb = true_rgb.squeeze(0)
        t = t.squeeze(0)
        weight_t = weight_t.squeeze(0)

        pred_rgb = self(uv, t, batch_idx, context=data)

        loss = self.loss(pred_rgb, true_rgb, time_weight=weight_t)

        self.log('loss/sum', loss.detach().cpu(),
                 prog_bar=True, on_step=True, on_epoch=True)
        self.log('sin_epoch', self.sin_epoch.detach().cpu(), prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):

        eps = 1e-7 if self.config.use_amp else 1e-8
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(self.config.lr),
            eps=eps,
        )
        # constant lr
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #    optimizer, step_size=1, gamma=1.0)

        args = {"optimizer": optimizer}

        if self.config.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=4,
                threshold=0.01, threshold_mode='rel', cooldown=self.config.lr_scheduler_cooldown, min_lr=0, eps=eps)
            args["lr_scheduler"] = scheduler
            args["monitor"] = "loss/sum_epoch"
        return args

    def _set_lr_scheduler_enabled(self, enable: bool):
        if not self.config.lr_scheduler:
            return
        lrs = self.lr_schedulers()

        if enable:
            lrs.cooldown_counter = 0
            lrs._reset()
        else:
            # Set cooldown to near inf and enable the cooldown
            lrs.cooldown_counter = 1000000000

    @property
    def lr_scheduler_enabled(self):
        return self._lr_scheduler_enabled

    @lr_scheduler_enabled.setter
    def lr_scheduler_enabled(self, value: bool):
        if value == self._lr_scheduler_enabled:
            return
        self._set_lr_scheduler_enabled(value)
        self._lr_scheduler_enabled = value

    def get_node_indices(self) -> torch.Tensor:
        """Get the indices of the nodes in the model.

        Returns
        -------
        torch.Tensor
            The indices of the nodes in the model.
        """
        OBJS = len(self.objects) + 1  # Objects + Camera
        idx = [x.get_index() for x in self.objects]
        cam = self.camera.get_index()

        order = torch.zeros(OBJS, dtype=torch.int32)
        order.fill_(-1)
        order[:len(idx)] = idx
        order[-1] = cam
        return order

    def get_global_positions(self, t: torch.Tensor) -> torch.Tensor:
        """Get the global positions of the objects in the scene.

        Parameters
        ----------
        t : torch.Tensor
            Time steps to evaluate the objects at. Shape (T)

        Returns
        -------
        torch.Tensor
            Global positions of the objects in the scene. Shape (N, T, 4, 4)
        """
        objs = [obj.get_global_position(t=t) for obj in self.objects]
        cam = [self.camera.get_global_position(t=t)]
        return torch.stack(objs + cam, dim=0)

    def enable_flow(self, value: bool):
        """Enable or disable the flow model in the objects."""
        for o in self.objects:
            if isinstance(o, LearnedImagePlaneSceneNode3D):
                o.encoding_flow.requires_grad_(value)
                o.network_flow.requires_grad_(value)
                o.network_flow.train(value)
                o.encoding_flow.train(value)
                if o.independent_rgba_flow:
                    o.encoding_flow_alpha.requires_grad_(value)
                    o.network_flow_alpha.requires_grad_(value)
                    o.encoding_flow_alpha.train(value)
                    o.network_flow_alpha.train(value)
            elif isinstance(o, BackgroundImagePlaneSceneNode3D):
                o.network_flow.requires_grad_(value)
                o.network_flow.train(value)
                o.encoding_flow.requires_grad_(value)
                o.encoding_flow.train(value)

    def enable_position_learning(self, value: bool):
        """Enable or disable the position learning in the objects."""
        for o in self.objects:
            if isinstance(o, LearnedImagePlaneSceneNode3D):
                if self.config.is_plane_translation_learnable:
                    o._offset_translation.requires_grad_(value)
                if self.config.is_plane_orientation_learnable:
                    o._offset_rotation_vector.requires_grad_(value)
        if self.config.is_camera_translation_learnable:
            self.camera._offset_translation.requires_grad_(value)
        if self.config.is_camera_orientation_learnable:
            self.camera._offset_rotation_vector.requires_grad_(value)

    def enable_color_alpha(self, value: bool):
        for o in self.objects:
            if isinstance(o, LearnedImagePlaneSceneNode3D):
                o.network_alpha.requires_grad_(value)
                o.network_image.requires_grad_(value)
                o.network_alpha.train(value)
                o.network_image.train(value)
                o.encoding_alpha.requires_grad_(value)
                o.encoding_image.requires_grad_(value)
                o.encoding_alpha.train(value)
                o.encoding_image.train(value)
            elif isinstance(o, LearnedAberrationPlaneSceneNode3D):
                o.network_alpha.requires_grad_(value)
                o.network_alpha.train(value)
                o.encoding_alpha.requires_grad_(value)
                o.encoding_alpha.train(value)
                o.color.requires_grad_(value)
            elif isinstance(o, BackgroundPlaneSceneNode3D):
                if isinstance(o, BackgroundImagePlaneSceneNode3D):
                    o.network_image.requires_grad_(value)
                    o.network_image.train(value)
                    o.encoding_image.requires_grad_(value)
                    o.encoding_image.train(value)
                else:
                    if o._is_background_learnable:
                        o._background_color.requires_grad_(value)

    def enable_view_dependence(self, value: bool):
        from nag.model.view_dependent_image_plane_scene_node_3d import ViewDependentImagePlaneSceneNode3D
        from nag.model.view_dependent_background_image_plane_scene_node_3d import ViewDependentBackgroundImagePlaneSceneNode3D
        for o in self.objects:
            if isinstance(o, (ViewDependentImagePlaneSceneNode3D, ViewDependentBackgroundImagePlaneSceneNode3D)):
                o.encoding_view_dependence.requires_grad_(value)
                o.network_view_dependence.requires_grad_(value)
                o.network_view_dependence.train(value)
                o.encoding_view_dependence.train(value)

    def enable_time_dependence(self, value: bool):
        from nag.model.time_dependent_background_image_plane_scene_node_3d import TimeDependentBackgroundImagePlaneSceneNode3D
        for o in self.objects:
            if isinstance(o, (TimeDependentBackgroundImagePlaneSceneNode3D,)):
                o.encoding_time_dependence.requires_grad_(value)
                o.network_time_dependence.requires_grad_(value)
                o.network_time_dependence.train(value)
                o.encoding_time_dependence.train(value)

    def on_after_backward(self) -> None:
        pass

    # region Loading and Saving

    def on_load_checkpoint(self, checkpoint: torch.Dict[str, Any]) -> None:
        # After loading the checkpoint, the flow reference times must be reloaded
        self._flow_reference_loaded = False
        if self.allow_loading_unmatching_parameter_sizes or self.load_missing_parameters_as_defaults:
            current_state_dict = self.state_dict(keep_vars=True)
            loaded_state_dict = checkpoint['state_dict']
            loaded_keys = set(loaded_state_dict.keys())
            current_keys = set(current_state_dict.keys())
            missing_keys = loaded_keys - current_keys

            shape_changes = []
            missing_parameters = []
            for k in current_state_dict:
                if k in loaded_state_dict:
                    current_shape = current_state_dict[k].shape
                    loaded_shape = loaded_state_dict[k].shape
                    if self.allow_loading_unmatching_parameter_sizes:
                        if current_shape != loaded_shape:
                            # Set the current parameter with an empty tensor like the loaded one
                            p = current_state_dict[k]
                            p.data = torch.empty_like(
                                loaded_state_dict[k], device=p.device, dtype=p.dtype)
                            shape_changes.append(
                                "{}: {} -> {}".format(k, list(current_shape), list(loaded_shape)))
                else:
                    if self.load_missing_parameters_as_defaults:
                        loaded_state_dict[k] = current_state_dict[k].detach(
                        ).clone()
                        missing_parameters.append("{}: {}".format(
                            k, list(current_state_dict[k].shape)))
            if self.allow_loading_unmatching_parameter_sizes:
                if len(shape_changes) > 0:
                    logger.info("Loaded model has different parameter sizes than current model. The following parameters have been changed: \n{}".format(
                        ",\n".join(shape_changes)))
            if self.load_missing_parameters_as_defaults:
                if len(missing_parameters) > 0:
                    logger.info("Loaded model is missing parameters. The following parameters have been added: \n{}".format(
                        ",\n".join(missing_parameters)))

            if len(missing_keys) > 0:
                logger.warning(
                    "Loaded model is missing keys: {}".format(missing_keys))
                box_pattern = r"_world\._scene_children\.(?P<index>\d+)\.box\."
                import re
                can_be_ignored = set()
                missing_boxes_index = set()
                for k in missing_keys:
                    m = re.match(box_pattern, k)
                    if m is not None:
                        can_be_ignored.add(k)
                        missing_boxes_index.add(int(m.group("index")))
                        # Remove key from loaded state dict
                        del loaded_state_dict[k]

                if len(missing_boxes_index) > 0:
                    oll_nodes = list(self._world._scene_children)
                    missing_box_nodes = [oll_nodes[i]
                                         for i in missing_boxes_index]
                    logger.warning("The following nodes where originally initialized with a box attribute which is now missing. Box will be unavailable. \n{}".format(
                        '\n'.join([repr(x) for x in missing_box_nodes])))

    # endregion

    # region Plotting

    def plot_scene(
        self,
        plot_coordinate_systems: bool = False,
        plot_coordinate_annotations: bool = True,
        plot_pixel_grid: bool = False,
        plot_ray_origins: bool = True,
        pixel_grid_subsample: int = 800,
        plot_ray_cast: bool = True,
        ray_distance: float = 1.5,
        supersample: int = 2,
        t: float = 0.,
        plot_plane_alpha: bool = False,
        plot_plane_image: bool = False,
        plot_plane_image_subsample: int = 5,
        elevation: Optional[float] = -45,
        azimuth: Optional[float] = -90,
        **kwargs
    ) -> Figure:
        t = tensorify(t, dtype=self.camera._translation.dtype,
                      device=self.device)
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        return self.world.plot_scene(
            plot_coordinate_systems=plot_coordinate_systems,
            plot_coordinate_annotations=plot_coordinate_annotations,
            plot_pixel_grid=plot_pixel_grid,
            plot_ray_origins=plot_ray_origins,
            pixel_grid_subsample=pixel_grid_subsample,
            plot_ray_cast=plot_ray_cast,
            ray_distance=ray_distance,
            supersample=supersample,
            t=t,
            plot_plane_alpha=plot_plane_alpha,
            plot_plane_image=plot_plane_image,
            plot_plane_image_subsample=plot_plane_image_subsample,
            elevation=elevation,
            azimuth=azimuth,
            **kwargs
        )

    def plot_scene_animation(
            self,
            plot_coordinate_systems: bool = False,
            plot_coordinate_annotations: bool = False,
            plot_pixel_grid: bool = True,
            plot_ray_origins: bool = True,
            pixel_grid_subsample: int = 800,
            plot_ray_cast: bool = True,
            ray_distance=1.5,
            plot_camera_traces: bool = True,
            plot_plane_image: bool = False,
            plot_plane_alpha: bool = False,
            plot_plane_image_subsample: int = 50,
            elevation: float = 15,
            azimuth: float = 0,
            roll: float = 0,
            **kwargs
    ) -> Tuple[Figure, FuncAnimation]:
        return self.world.plot_scene_animation(
            plot_coordinate_systems=plot_coordinate_systems,
            plot_coordinate_annotations=plot_coordinate_annotations,
            plot_pixel_grid=plot_pixel_grid,
            plot_ray_origins=plot_ray_origins,
            pixel_grid_subsample=pixel_grid_subsample,
            plot_ray_cast=plot_ray_cast,
            ray_distance=ray_distance,
            plot_camera_traces=plot_camera_traces,
            plot_plane_image=plot_plane_image,
            plot_plane_alpha=plot_plane_alpha,
            plot_plane_image_subsample=plot_plane_image_subsample,
            elevation=elevation,
            azimuth=azimuth,
            roll=roll,
            **kwargs
        )

    @saveable(
        default_name="object_positions",
        default_dpi=150
    )
    def plot_object_positions(self,
                              object_idx: Optional[List[int]] = None, t: Optional[torch.Tensor] = None, **kwargs):
        import matplotlib.pyplot as plt
        from tools.viz.matplotlib import get_mpl_figure
        if object_idx is None:
            objects = list(self.objects)
            objects += [self.camera]
        else:
            objects = [self.objects[i] for i in object_idx]

        rows = len(objects)
        size = 5
        cols = 2

        fig = plt.figure(figsize=(size * cols, size * rows))
        for i, obj in enumerate(objects):
            ax1 = fig.add_subplot(rows, 2, 2*i+1)
            ax2 = fig.add_subplot(rows, 2, 2*i+2)
            obj.plot_position(ax=[ax1, ax2], t=t, **kwargs)

            axw = fig.add_subplot(rows, 1, 1 + i, frameon=False)
            axw.set_title(obj.get_name())
            axw.axis("off")
        return fig

    @saveable(
        default_name="object_positions",
        is_figure_collection=True,
        default_dpi=150
    )
    def plot_objects_positions(self, object_idx: List[List[int]], t: Optional[torch.Tensor] = None, **kwargs):
        figs = []
        for obj_idx in object_idx:
            fig = self.plot_object_positions(object_idx=obj_idx, t=t, **kwargs)
            figs.append(fig)
        return figs

    # endregion
