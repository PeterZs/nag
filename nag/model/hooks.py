from typing import Tuple
import torch
from nag.model.timed_discrete_scene_node_3d import compose_translation_orientation


@torch.jit.script
def get_total_hit_scales(
    is_inside: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device
):
    """Get the total hit scales.

    This should equalize the impact of planes of different sizes.

    Parameters
    ----------
    is_inside : torch.Tensor
        Tensor indicating whether the ray hits the object.
        Shape: (N, B, T).

    Returns
    -------
    torch.Tensor
        Total hit scales.
        Shape (N, T)
    """
    N, B, T = is_inside.shape
    is_inside_wo_background = is_inside[: -1]

    sis = is_inside_wo_background.sum(dim=(-2))

    non_zero = sis > 0
    scale = torch.zeros(N-1, T, dtype=dtype, device=device)
    scale[non_zero] = (B / sis[non_zero])
    devider = scale[non_zero].mean() if non_zero.any(
    ) else torch.tensor(1., dtype=dtype, device=device)
    scale[non_zero] = scale[non_zero] / devider
    return torch.cat([scale, torch.ones((1, T), dtype=dtype, device=device)], dim=0)


@torch.jit.script
def alpha_chain_scaling_hook(
    grad: torch.Tensor,
    alpha_chain: torch.Tensor,
    alpha: torch.Tensor,
    is_inside: torch.Tensor,
    atol: float = 1e-6,
    alpha_chain_scaling: bool = True,
    total_hit_scaling: bool = True,
    patch_nans: bool = True
) -> torch.Tensor:
    """Scale the gradient by the alpha chain.

    Parameters
    ----------
    grad : torch.Tensor
        Gradient tensor. Shape: (N, B, T, C).

    alpha_chain : torch.Tensor
        Alpha chain tensor.
        Shape: (N, B, T, 1).

    Returns
    -------
    torch.Tensor
        Updated gradient tensor.
    """
    N, B, T, C = grad.shape
    if patch_nans:
        # Due to numerical instability / gradient overflow in tinycuda, gradients might get NaN, we might get NaNs in the gradients
        grad = torch.where(~torch.isfinite(grad), torch.zeros_like(grad), grad)

    if total_hit_scaling:
        scales = get_total_hit_scales(is_inside, grad.dtype, grad.device)
        per_ray_scale = scales.unsqueeze(1).unsqueeze(-1).repeat(1, B, 1, C)
        new_grad = grad * per_ray_scale
        # assert new_grad.isfinite().all(), f"New grad is not finite, Values: {new_grad[~new_grad.isfinite()]}, Coords: {torch.argwhere(~new_grad.isfinite())}, Scale: {per_ray_scale[~new_grad.isfinite()]}"
        grad = new_grad

    if alpha_chain_scaling:
        c_wo_zero = torch.where(torch.isclose(alpha_chain,
                                              torch.tensor(0., dtype=grad.dtype, device=grad.device), atol=atol),
                                torch.tensor(1., dtype=grad.dtype,
                                             device=grad.device),
                                alpha_chain.detach()).detach()
        new_grad = grad * (1 / c_wo_zero)
        # assert new_grad.isfinite().all(), f"New grad is not finite, Values: {new_grad[~new_grad.isfinite()]}, Coords: {torch.argwhere(~new_grad.isfinite())}, Scale: {1 / c_wo_zero[~new_grad.isfinite()]}"
        grad = new_grad

    return grad


@torch.jit.script
def sample_positions_hook(
    translations: torch.Tensor,
    orientations: torch.Tensor,
    camera_idx: int,
    sin_epoch: torch.Tensor,
    sigma: float = 0.02
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Offset object positions based on a gaussian distribution closer to or further away from the camera.

    Parameters
    ----------
    translations : torch.Tensor
        Positions of objects and camera. Shape: (N, T, 3).

    orientations : torch.Tensor
        Orientations of objects and camera as unit quaternion. Shape: (N, T, 4).

    camera_idx : int
        Index of the camera within the translations tensor.

    sin_epoch : torch.Tensor
        Sine of the epoch.
        Progresively increases from 0 to 1.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Updated positions and orientations tensors.
    """
    if sin_epoch == 1:
        return translations, orientations

    N, T, _ = translations.shape
    O = N - 1

    object_positions = torch.ones_like(translations[:, 0, 0], dtype=torch.bool)
    object_positions[camera_idx] = False

    sample_sigma = sigma * (1 - sin_epoch)

    sigma_tensor = torch.full((O, T), sample_sigma,
                              dtype=translations.dtype, device=translations.device)
    sampled_z_dist = torch.normal(mean=0., std=sigma_tensor)
    pos = torch.zeros((O, T, 4), dtype=translations.dtype,
                      device=translations.device)
    pos[:, :, 2] = sampled_z_dist

    global_camera_position = compose_translation_orientation(translations[camera_idx].unsqueeze(0).detach(),
                                                             orientations[camera_idx].unsqueeze(0).detach())
    target_pos = torch.matmul(global_camera_position.repeat(O, 1, 1, 1).reshape(O * T, 4, 4),
                              pos.unsqueeze(-1).reshape(O * T, 4, 1)).reshape(O, T, 4, 1)[..., :3, 0]
    delta_pos = target_pos - global_camera_position[..., :3, 3]

    translations[object_positions] = translations[object_positions] + delta_pos
    return translations, orientations
