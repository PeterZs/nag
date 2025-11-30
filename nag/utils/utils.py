from typing import Dict, Any, Optional, Union, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import torch.nn.functional as F
from enum import Enum
import tools
from tools.util.format import parse_enum
from tqdm import tqdm
from tools.io.image import load_image
from tools.util.path_tools import open_folder, read_directory
from tools.io.image import load_image_stack
from nag.transforms.transforms_timed_3d import interpolate_orientation


class ColorFilterArangement(Enum):
    """Demosaicing pattern of the camera sensor."""
    RGGB = 0
    """Red, Green, Green, Blue"""
    GRBG = 1
    """Green, Red, Blue, Green"""
    GBRG = 2
    """Green, Blue, Red, Green"""
    BGGR = 3
    """Blue, Green, Green, Red"""


get_image_stack = load_image_stack


def load_bundle(bundle_path: str) -> Dict[str, Any]:
    bundle = dict(np.load(bundle_path, allow_pickle=True))
    de_item(bundle)
    return bundle


def parse_color_correction_gains(data_string: str) -> np.ndarray:
    red_pattern = r"R:\s?[0-9]+(((,|\.))[0-9]+)?"
    green_even_pattern = r"G_even:\s?[0-9]+(((,|\.))[0-9]+)?"
    green_odd_pattern = r"G_odd:\s?[0-9]+(((,|\.))[0-9]+)?"
    blue_pattern = r"B:\s?[0-9]+(((,|\.))[0-9]+)?"

    R_gain = float(re.search(red_pattern, data_string).group().split(
        ':')[-1].strip().replace(',', '.'))
    G_even_gain = float(re.search(green_even_pattern, data_string).group().split(
        ':')[-1].strip().replace(',', '.'))
    G_odd_gain = float(re.search(green_odd_pattern, data_string).group().split(
        ':')[-1].strip().replace(',', '.'))
    B_gain = float(re.search(blue_pattern, data_string).group().split(
        ':')[-1].strip().replace(',', '.'))
    color_correction_gains = np.array(
        [R_gain, G_even_gain, G_odd_gain, B_gain], dtype=np.float32)
    return color_correction_gains


def get_color_correction_gains(bundle: dict, image_index: int = 0) -> torch.Tensor:
    """Get color correction gains from bundle.
    For the image at index `image_index` in the bundle, extract the color correction gains.

    Parameters
    ----------
    bundle : dict
        Bundle containing the raw frames.
    image_index : int, optional
        Image index, by default 0

    Returns
    -------
    torch.Tensor
        Tensor containing the color correction gains.
        Color correction gains are in the order: [R_gain, G_odd_gain, G_even_gain, B_gain]
        => [R, G1, G2, B]
    """
    gains = bundle[f'raw_{image_index}']['android']['colorCorrection.gains']
    return torch.tensor(parse_color_correction_gains(gains))


def get_raw_frames(bundle: dict) -> torch.Tensor:
    """Get raw frames from bundle.

    Parameters
    ----------
    bundle : dict
        Dictionary containing the raw frames.

    Returns
    -------
    torch.Tensor
        Raw frames tensor of shape (T, C, H, W).
    """
    raw_frames = torch.tensor(np.array([bundle[f'raw_{i}']['raw'] for i in range(
        bundle['num_raw_frames'])]).astype(np.int32), dtype=torch.float32)[None]  # C,T,H,W
    return raw_frames.permute(1, 0, 2, 3)  # T,C,H,W


def get_raw_frame(bundle: dict, image_index: int) -> torch.Tensor:
    """Get raw frame from bundle.

    Parameters
    ----------
    bundle : dict
        Dictionary containing the raw frames.

    image_index : int
        Image index.

    Returns
    -------
    torch.Tensor
        Raw frame tensor of shape (1, C, H, W).
    """
    raw_frames = torch.tensor(np.array([bundle[f'raw_{image_index}']['raw']]).astype(
        np.int32), dtype=torch.float32)[None]  # C,T,H,W
    return raw_frames.permute(1, 0, 2, 3)  # T,C,H,W


def get_frame_timestamps(bundle: dict) -> torch.Tensor:
    """Get frame timestamps from bundle.

    Parameters
    ----------
    bundle : dict
        Dictionary containing the raw frames.

    Returns
    -------
    torch.Tensor
        Tensor containing the frame timestamps. Shape is (T,).
    """
    times = [bundle[f'raw_{i}']['timestamp'] for i in range(
        bundle['num_raw_frames'])]
    return torch.tensor(times, dtype=torch.float64)


def get_color_filter_arrangement(bundle: dict) -> ColorFilterArangement:
    """Get color filter arrangement from bundle.

    Parameters
    ----------
    bundle : dict
        Dictionary containing the raw frames.

    Returns
    -------
    ColorFilterArangement
        Color filter arrangement.
    """
    return ColorFilterArangement(bundle['characteristics']['color_filter_arrangement'])


def get_blacklevel(bundle: dict) -> torch.Tensor:
    """Get blacklevel from bundle.

    Parameters
    ----------
    bundle : dict
        Dictionary containing the raw frames.

    Returns
    -------
    torch.Tensor
        Tensor containing the blacklevel.
    """    """"""
    return torch.tensor(np.array([bundle[f'raw_{i}']['blacklevel'] for i in range(bundle['num_raw_frames'])]))[:, :, None, None]


def get_whitelevel(bundle: dict) -> torch.Tensor:
    """Get whitelevel from bundle.

    Parameters
    ----------
    bundle : dict
        Dictionary containing the raw frames.

    Returns
    -------
    torch.Tensor
        Tensor containing the whitelevel.
    """
    return torch.tensor(np.array([bundle[f'raw_{i}']['whitelevel'] for i in range(bundle['num_raw_frames'])]))[:, None, None, None]


def get_shade_maps(bundle: dict, image_shape: Tuple[int, int]) -> torch.Tensor:
    """Get shade maps from bundle.

    Parameters
    ----------
    bundle : dict
        Dictionary containing the raw frames.

    image_shape : Tuple[int, int]
        Image shape (H, W) to interpolate the shade maps.

    Returns
    -------
    torch.Tensor
        Tensor containing the shade maps.
    """
    shade_maps = torch.tensor(np.array([bundle[f'raw_{i}']['shade_map'] for i in range(
        bundle['num_raw_frames'])])).permute(0, 3, 1, 2)  # T,C,H,W
    return F.interpolate(shade_maps, size=image_shape, mode='bilinear', align_corners=False)


def get_ccm(bundle: dict, image_index: int = 0) -> torch.Tensor:
    """Get color correction matrix (CCM) from bundle.
    Should be the same for all images.
    Shape of the CCM is (3, 3).

    Parameters
    ----------
    bundle : dict
        Dictionary containing the raw frames.

    Returns
    -------
    torch.Tensor
        Tensor containing the color correction matrix.
    """
    return torch.tensor(np.array(bundle[f'raw_{image_index}']['ccm']))


def get_tonemap_curve(bundle: dict, image_index: int = 0) -> torch.Tensor:
    """Get tonemap curve from bundle.
    Should be the same for all images.

    Parameters
    ----------
    bundle : dict
        Dictionary containing the raw frames.

    image_index : int, optional
        Image index, by default 0

    Returns
    -------
    torch.Tensor
        Tensor containing the tonemap curve.
    """
    return torch.tensor(np.array(bundle[f'raw_{image_index}']['tonemap_curve']))


def get_intrinsics(bundle: dict) -> torch.Tensor:
    """Get the intrinsics from the bundle.

    Parameters
    ----------
    bundle : dict
        Dictionary containing the raw frames.

    Returns
    -------
    torch.Tensor
        Intrinsics tensor of shape (T, 3, 3).
    """
    intrinsics = torch.tensor(np.array(
        [bundle[f'raw_{i}']['intrinsics'] for i in range(bundle['num_raw_frames'])])).float()  # T,3,3
    # swap cx,cy -> landscape to portrait
    cx, cy = intrinsics[:, 2, 1].clone(), intrinsics[:, 2, 0].clone()
    intrinsics[:, 2, 0], intrinsics[:, 2, 1] = cx, cy
    # transpose to put cx,cy in right column
    intrinsics = intrinsics.transpose(1, 2)
    return intrinsics


def get_lens_distortion(bundle: dict) -> torch.Tensor:
    """Get the lens distortion from the bundle.

    Parameters
    ----------
    bundle : dict
        Dictionary containing the raw frames.

    Returns
    -------
    torch.Tensor
        Lens distortion tensor. Shape is (5, ).
    """
    return torch.tensor(bundle['raw_0']['lens_distortion'])


@torch.no_grad()
def raw_to_rgb(bundle):
    """ Convert RAW mosaic into three-channel RGB volume
        by only in-filling empty pixels.
        Returns volume of shape: (T, C, H, W)
    """

    raw_frames = torch.tensor(np.array([bundle[f'raw_{i}']['raw'] for i in range(
        bundle['num_raw_frames'])]).astype(np.int32), dtype=torch.float32)[None]  # C,T,H,W
    raw_frames = raw_frames.permute(1, 0, 2, 3)  # T,C,H,W

    color_correction_gains = get_color_correction_gains(bundle)
    color_filter_arrangement = bundle['characteristics']['color_filter_arrangement']
    blacklevel = torch.tensor(np.array([bundle[f'raw_{i}']['blacklevel'] for i in range(
        bundle['num_raw_frames'])]))[:, :, None, None]
    whitelevel = torch.tensor(np.array([bundle[f'raw_{i}']['whitelevel'] for i in range(
        bundle['num_raw_frames'])]))[:, None, None, None]
    shade_maps = torch.tensor(np.array([bundle[f'raw_{i}']['shade_map'] for i in range(
        bundle['num_raw_frames'])])).permute(0, 3, 1, 2)  # T,C,H,W
    # interpolate to size of image
    shade_maps = F.interpolate(shade_maps, size=(
        raw_frames.shape[-2]//2, raw_frames.shape[-1]//2), mode='bilinear', align_corners=False)

    top_left = raw_frames[:, :, 0::2, 0::2]
    top_right = raw_frames[:, :, 0::2, 1::2]
    bottom_left = raw_frames[:, :, 1::2, 0::2]
    bottom_right = raw_frames[:, :, 1::2, 1::2]

    # figure out color channels
    if color_filter_arrangement == ColorFilterArangement.RGGB.value:  # RGGB
        R, G1, G2, B = top_left, top_right, bottom_left, bottom_right
    elif color_filter_arrangement == ColorFilterArangement.GRBG.value:  # GRBG
        G1, R, B, G2 = top_left, top_right, bottom_left, bottom_right
    elif color_filter_arrangement == ColorFilterArangement.GBRG.value:  # GBRG
        G1, B, R, G2 = top_left, top_right, bottom_left, bottom_right
    elif color_filter_arrangement == ColorFilterArangement.BGGR.value:  # BGGR
        B, G1, G2, R = top_left, top_right, bottom_left, bottom_right

    color_filter_arrangement = parse_enum(
        ColorFilterArangement, color_filter_arrangement)

    # apply color correction gains, flip to portrait
    if color_filter_arrangement == ColorFilterArangement.RGGB:  # RGGB
        R, G1, G2, B = top_left, top_right, bottom_left, bottom_right
        red_slice = slice(0, 1)
        green_slice_odd = slice(1, 2)
        green_slice_even = slice(2, 3)
        blue_slice = slice(3, 4)

    elif color_filter_arrangement == ColorFilterArangement.GRBG:  # GRBG
        G1, R, B, G2 = top_left, top_right, bottom_left, bottom_right
        green_slice_odd = slice(0, 1)
        red_slice = slice(1, 2)
        blue_slice = slice(2, 3)
        green_slice_even = slice(3, 4)

    elif color_filter_arrangement == ColorFilterArangement.GBRG:  # GBRG
        G1, B, R, G2 = top_left, top_right, bottom_left, bottom_right
        green_slice_odd = slice(0, 1)
        blue_slice = slice(1, 2)
        red_slice = slice(2, 3)
        green_slice_even = slice(3, 4)

    elif color_filter_arrangement == ColorFilterArangement.BGGR:  # BGGR
        B, G1, G2, R = top_left, top_right, bottom_left, bottom_right
        blue_slice = slice(0, 1)
        green_slice_odd = slice(1, 2)
        green_slice_even = slice(2, 3)
        red_slice = slice(3, 4)
    else:
        raise ValueError(
            f"Invalid color filter arrangement: {color_filter_arrangement}.")

    # apply color correction gains, flip to portrait
    R = ((R - blacklevel[:, red_slice]) / (whitelevel -
         blacklevel[:, red_slice]) * color_correction_gains[0])
    R *= shade_maps[:, red_slice]
    G1 = ((G1 - blacklevel[:, green_slice_odd]) / (whitelevel -
          blacklevel[:, green_slice_odd]) * color_correction_gains[1])
    G1 *= shade_maps[:, green_slice_odd]
    G2 = ((G2 - blacklevel[:, green_slice_even]) / (whitelevel -
          blacklevel[:, green_slice_even]) * color_correction_gains[2])
    G2 *= shade_maps[:, green_slice_even]
    B = ((B - blacklevel[:, blue_slice]) / (whitelevel -
         blacklevel[:, blue_slice]) * color_correction_gains[3])
    B *= shade_maps[:, blue_slice]

    rgb = torch.zeros(
        raw_frames.shape[0], 3, raw_frames.shape[-2], raw_frames.shape[-1], dtype=torch.float32)

    # Fill gaps in blue channel
    rgb[:, 2, 0::2, 0::2] = B.squeeze(1)
    rgb[:, 2, 0::2, 1::2] = (B + torch.roll(B, -1, dims=3)).squeeze(1) / 2
    rgb[:, 2, 1::2, 0::2] = (B + torch.roll(B, -1, dims=2)).squeeze(1) / 2
    rgb[:, 2, 1::2, 1::2] = (B + torch.roll(B, -1, dims=2) + torch.roll(
        B, -1, dims=3) + torch.roll(B, [-1, -1], dims=[2, 3])).squeeze(1) / 4

    # Fill gaps in green channel
    rgb[:, 1, 0::2, 0::2] = G1.squeeze(1)
    rgb[:, 1, 0::2, 1::2] = (
        G1 + torch.roll(G1, -1, dims=3) + G2 + torch.roll(G2, 1, dims=2)).squeeze(1) / 4
    rgb[:, 1, 1::2, 0::2] = (
        G1 + torch.roll(G1, -1, dims=2) + G2 + torch.roll(G2, 1, dims=3)).squeeze(1) / 4
    rgb[:, 1, 1::2, 1::2] = G2.squeeze(1)

    # Fill gaps in red channel
    rgb[:, 0, 0::2, 0::2] = R.squeeze(1)
    rgb[:, 0, 0::2, 1::2] = (R + torch.roll(R, -1, dims=3)).squeeze(1) / 2
    rgb[:, 0, 1::2, 0::2] = (R + torch.roll(R, -1, dims=2)).squeeze(1) / 2
    rgb[:, 0, 1::2, 1::2] = (R + torch.roll(R, -1, dims=2) + torch.roll(
        R, -1, dims=3) + torch.roll(R, [-1, -1], dims=[2, 3])).squeeze(1) / 4

    # rotate 90 degrees clockwise to portrait mode
    rgb = torch.flip(rgb.transpose(-1, -2), [-1])

    return rgb


def raw_to_rgb_single(raw_frame: torch.Tensor,
                      color_correction_gains: torch.Tensor,
                      color_filter_arrangement: Union[ColorFilterArangement, int],
                      blacklevel: torch.Tensor,
                      whitelevel: torch.Tensor,
                      shade_maps: torch.Tensor,
                      ccm: Optional[torch.Tensor],
                      tonemap_curve: Optional[torch.Tensor],
                      downsampling_factor: int = 1) -> torch.Tensor:

    color_filter_arrangement = parse_enum(
        ColorFilterArangement, color_filter_arrangement)

    top_left = raw_frame[..., 0::2 *
                         downsampling_factor, 0::2 * downsampling_factor]
    top_right = raw_frame[..., 0::2 *
                          downsampling_factor, 1::2 * downsampling_factor]
    bottom_left = raw_frame[..., 1::2 *
                            downsampling_factor, 0::2 * downsampling_factor]
    bottom_right = raw_frame[..., 1::2 *
                             downsampling_factor, 1::2 * downsampling_factor]

    if color_filter_arrangement == ColorFilterArangement.RGGB:  # RGGB
        R, G1, G2, B = top_left, top_right, bottom_left, bottom_right
        red_slice = slice(0, 1)
        green_slice_odd = slice(1, 2)
        green_slice_even = slice(2, 3)
        blue_slice = slice(3, 4)

    elif color_filter_arrangement == ColorFilterArangement.GRBG:  # GRBG
        G1, R, B, G2 = top_left, top_right, bottom_left, bottom_right
        green_slice_odd = slice(0, 1)
        red_slice = slice(1, 2)
        blue_slice = slice(2, 3)
        green_slice_even = slice(3, 4)

    elif color_filter_arrangement == ColorFilterArangement.GBRG:  # GBRG
        G1, B, R, G2 = top_left, top_right, bottom_left, bottom_right
        green_slice_odd = slice(0, 1)
        blue_slice = slice(1, 2)
        red_slice = slice(2, 3)
        green_slice_even = slice(3, 4)

    elif color_filter_arrangement == ColorFilterArangement.BGGR:  # BGGR
        B, G1, G2, R = top_left, top_right, bottom_left, bottom_right
        blue_slice = slice(0, 1)
        green_slice_odd = slice(1, 2)
        green_slice_even = slice(2, 3)
        red_slice = slice(3, 4)
    else:
        raise ValueError(
            f"Invalid color filter arrangement: {color_filter_arrangement}.")

    unsqueezed = False

    if len(raw_frame.shape) not in [3, 4]:
        raise ValueError(f"Invalid raw_frames shape: {raw_frame.shape}.")
    if len(raw_frame.shape) == 3:
        raw_frame = raw_frame.unsqueeze(0)
        unsqueezed = True

    if len(blacklevel.shape) not in [3, 4]:
        raise ValueError(f"Invalid blacklevel shape: {blacklevel.shape}.")
    if len(blacklevel.shape) == 3:
        blacklevel = blacklevel.unsqueeze(0)

    if len(whitelevel.shape) not in [3, 4]:
        raise ValueError(f"Invalid whitelevel shape: {whitelevel.shape}.")
    if len(whitelevel.shape) == 3:
        whitelevel = whitelevel.unsqueeze(0)

    if len(shade_maps.shape) not in [3, 4]:
        raise ValueError(f"Invalid shade_maps shape: {shade_maps.shape}.")
    if len(shade_maps.shape) == 3:
        shade_maps = shade_maps.unsqueeze(0)

    # If downsampling factor is not 1, subsample also the shade maps
    if downsampling_factor > 1:
        shade_maps = shade_maps[...,
                                ::downsampling_factor, ::downsampling_factor]

    # apply color correction gains, flip to portrait
    R = ((R - blacklevel[:, red_slice]) / (whitelevel -
         blacklevel[:, red_slice]) * color_correction_gains[0])
    R *= shade_maps[:, red_slice]
    G1 = ((G1 - blacklevel[:, green_slice_odd]) / (whitelevel -
          blacklevel[:, green_slice_odd]) * color_correction_gains[1])
    G1 *= shade_maps[:, green_slice_odd]
    G2 = ((G2 - blacklevel[:, green_slice_even]) / (whitelevel -
          blacklevel[:, green_slice_even]) * color_correction_gains[2])
    G2 *= shade_maps[:, green_slice_even]
    B = ((B - blacklevel[:, blue_slice]) / (whitelevel -
         blacklevel[:, blue_slice]) * color_correction_gains[3])
    B *= shade_maps[:, blue_slice]

    frame_shape = (raw_frame.shape[0], 3) + tuple([raw_frame.shape[-2] //
                                                   downsampling_factor, raw_frame.shape[-1] // downsampling_factor])
    rgb = torch.zeros(frame_shape, dtype=torch.float32)

    # Fill gaps in blue channel
    rgb[..., 2, 0::2, 0::2] = B.squeeze(1)
    rgb[..., 2, 0::2, 1::2] = (B + torch.roll(B, -1, dims=3)).squeeze(1) / 2
    rgb[..., 2, 1::2, 0::2] = (B + torch.roll(B, -1, dims=2)).squeeze(1) / 2
    rgb[..., 2, 1::2, 1::2] = (B + torch.roll(B, -1, dims=2) + torch.roll(
        B, -1, dims=3) + torch.roll(B, [-1, -1], dims=[2, 3])).squeeze(1) / 4

    # Fill gaps in green channel
    rgb[..., 1, 0::2, 0::2] = G1.squeeze(1)
    rgb[..., 1, 0::2, 1::2] = (
        G1 + torch.roll(G1, -1, dims=3) + G2 + torch.roll(G2, 1, dims=2)).squeeze(1) / 4
    rgb[..., 1, 1::2, 0::2] = (
        G1 + torch.roll(G1, -1, dims=2) + G2 + torch.roll(G2, 1, dims=3)).squeeze(1) / 4
    rgb[..., 1, 1::2, 1::2] = G2.squeeze(1)

    # Fill gaps in red channel
    rgb[..., 0, 0::2, 0::2] = R.squeeze(1)
    rgb[..., 0, 0::2, 1::2] = (R + torch.roll(R, -1, dims=3)).squeeze(1) / 2
    rgb[..., 0, 1::2, 0::2] = (R + torch.roll(R, -1, dims=2)).squeeze(1) / 2
    rgb[..., 0, 1::2, 1::2] = (R + torch.roll(R, -1, dims=2) + torch.roll(
        R, -1, dims=3) + torch.roll(R, [-1, -1], dims=[2, 3])).squeeze(1) / 4

    # rotate 90 degrees clockwise to portrait mode
    rgb = torch.flip(rgb.transpose(-1, -2), [-1])

    # Apply CCMs and tonemap curves

    if ccm is not None:
        pix = ccm.to(dtype=rgb.dtype) @ rgb.reshape(3, -1)
        rgb = pix.reshape(3, rgb.shape[-2], rgb.shape[-1])

    if tonemap_curve is not None:
        for i in range(3):
            x_vals, y_vals = tonemap_curve[i][:, 0], tonemap_curve[i][:, 1]
            rgb[i, ...] = torch.tensor(
                np.interp(rgb[i].numpy(), x_vals, y_vals))

    if unsqueezed:
        rgb = rgb.squeeze(0)

    return rgb


def de_item(bundle):
    """ Call .item() on all dictionary items
        removes unnecessary extra dimension
    """

    bundle['motion'] = bundle['motion'].item()
    bundle['characteristics'] = bundle['characteristics'].item()

    for i in range(bundle['num_raw_frames']):
        bundle[f'raw_{i}'] = bundle[f'raw_{i}'].item()


@torch.jit.script
def mask(encoding: torch.Tensor, mask_coef: torch.Tensor, b: float = 0.4) -> torch.Tensor:
    """Granualrity function to use more of the encoding based on mask_coef.

    Parameters
    ----------
    encoding : torch.Tensor
        Output of MultiLevel HashGrid Encoding.

    mask_coef : torch.Tensor
        Coefficient to mask the encoding. Range [0, 1]. Shape (1,)
    b : float, optional
        Start / offset of the encoding., by default 0.4

    Returns
    -------
    torch.Tensor
        Positional encoding which is partially masked.
    """
    mask_coef = b + (1 - b) * mask_coef
    # interpolate to size of encoding
    mask = torch.zeros_like(encoding[0:1])
    mask_ceil = torch.ceil(mask_coef * encoding.shape[1]).int()
    mask[:, :mask_ceil] = 1.0
    return encoding * mask


def interpolate(signal: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    if signal.shape[-1] == 1:
        return signal.squeeze(-1)
    elif signal.shape[-1] == 2:
        return interpolate_linear(signal, times)
    else:
        return interpolate_cubic_hermite(signal, times)


def motion_to_rotmat(motion: Dict[str, Any], frame_timestamps: torch.Tensor, method: str = "cubic") -> torch.Tensor:
    """Converts the motions dict from a bundle to rotation matrices.

    Parameters
    ----------
    motion : Dict[str, Any]
        Dictionary with timestamps and quaternions of the rotation motion.

    frame_timestamps : torch.Tensor
        Frame timestamps. Should be in same scale as motion timestamps. (unnormalized) Shape (T,).

    Returns
    -------
    torch.Tensor
        Batched rotation matrices of shape (T, 3, 3).
    """
    motion_timestamps = torch.tensor(motion['timestamp'], dtype=torch.float64)

    # T',4, has different timestamps from frames
    quaternions = torch.tensor(motion['quaternion'], dtype=torch.float64)
    # our scene is +z towards scene convention, but phone is +z towards face convention
    # so we need to rotate 180 degrees around y axis, or equivalently flip over z,y
    quaternions[:, 2] = -quaternions[:, 2]  # invert y
    quaternions[:, 3] = -quaternions[:, 3]  # invert z
    quaternions = interpolate_orientation(quaternions, motion_timestamps, frame_timestamps.to(
        torch.float64), method=method).to(frame_timestamps.dtype)

    rotations = convert_quaternions_to_rot(quaternions)
    return rotations


def motion_to_unitquat(motion: Dict[str, Any], frame_timestamps: torch.Tensor, method: str = "cubic") -> torch.Tensor:
    motion_timestamps = torch.tensor(motion['timestamp'], dtype=torch.float64)
    # T',4, has different timestamps from frames
    quaternions = torch.tensor(motion['quaternion'], dtype=torch.float64)

    # our scene is +z towards scene convention, but phone is +z towards face convention
    # so we need to rotate 180 degrees around y axis, or equivalently flip over z,y
    quaternions[:, 2] = -quaternions[:, 2]  # invert y
    quaternions[:, 3] = -quaternions[:, 3]  # invert z

    quat = interpolate_orientation(quaternions, motion_timestamps, frame_timestamps.to(
        torch.float64), method=method).to(frame_timestamps.dtype)
    return quat


@torch.jit.script
def interpolate_cubic_hermite(signal, times):
    # Interpolate a signal using cubic Hermite splines
    # signal: (B, C, T) or (B, T)
    # times: (B, T)

    if len(signal.shape) == 3:  # B,C,T
        times = times.unsqueeze(1)
        times = times.repeat(1, signal.shape[1], 1)

    N = signal.shape[-1]

    times_scaled = times * (N - 1)
    indices = torch.floor(times_scaled).long()

    # Clamping to avoid out-of-bounds indices
    indices = torch.clamp(indices, 0, N - 2)
    left_indices = torch.clamp(indices - 1, 0, N - 1)
    right_indices = torch.clamp(indices + 1, 0, N - 1)
    right_right_indices = torch.clamp(indices + 2, 0, N - 1)

    t = (times_scaled - indices.float())

    p0 = torch.gather(signal, -1, left_indices)
    p1 = torch.gather(signal, -1, indices)
    p2 = torch.gather(signal, -1, right_indices)
    p3 = torch.gather(signal, -1, right_right_indices)

    # One-sided derivatives at the boundaries
    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices,
                     (p2 - p1), (p3 - p1) / 2)

    # Hermite basis functions
    h00 = (1 + 2*t) * (1 - t)**2
    h10 = t * (1 - t)**2
    h01 = t**2 * (3 - 2*t)
    h11 = t**2 * (t - 1)

    interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1

    if len(signal.shape) == 3:  # remove extra singleton dimension
        interpolation = interpolation.squeeze(-1)

    return interpolation


@torch.jit.script
def interpolate_linear(signal, times):
    # Interpolate a signal using linear interpolation
    # signal: (B, C, T) or (B, T)
    # times: (B, T)

    if len(signal.shape) == 3:  # B,C,T
        times = times.unsqueeze(1)
        times = times.repeat(1, signal.shape[1], 1)

    # Scale times to be between 0 and N - 1
    times_scaled = times * (signal.shape[-1] - 1)

    indices = torch.floor(times_scaled).long()
    right_indices = (indices + 1).clamp(max=signal.shape[-1] - 1)

    t = (times_scaled - indices.float())

    p0 = torch.gather(signal, -1, indices)
    p1 = torch.gather(signal, -1, right_indices)

    # Linear basis functions
    h00 = (1 - t)
    h01 = t

    interpolation = h00 * p0 + h01 * p1

    if len(signal.shape) == 3:  # remove extra singleton dimension
        interpolation = interpolation.squeeze(-1)

    return interpolation


@torch.jit.script
def convert_quaternions_to_rot(quaternions):
    """ Convert quaternions (wxyz) to 3x3 rotation matrices.
        Adapted from: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix
    """

    qw, qx, qy, qz = quaternions[:, 0], quaternions[:,
                                                    1], quaternions[:, 2], quaternions[:, 3]

    R00 = 2 * ((qw * qw) + (qx * qx)) - 1
    R01 = 2 * ((qx * qy) - (qw * qz))
    R02 = 2 * ((qx * qz) + (qw * qy))

    R10 = 2 * ((qx * qy) + (qw * qz))
    R11 = 2 * ((qw * qw) + (qy * qy)) - 1
    R12 = 2 * ((qy * qz) - (qw * qx))

    R20 = 2 * ((qx * qz) - (qw * qy))
    R21 = 2 * ((qy * qz) + (qw * qx))
    R22 = 2 * ((qw * qw) + (qz * qz)) - 1

    R = torch.stack([R00, R01, R02, R10, R11, R12, R20, R21, R22], dim=-1)
    R = R.reshape(-1, 3, 3)

    return R


def multi_interp(x, xp, fp):
    """ Simple extension of np.interp for independent
        linear interpolation of all axes of fp
        sample signal fp with timestamps xp at new timestamps x
    """
    if torch.is_tensor(fp):
        out = [torch.tensor(np.interp(x, xp, fp[:, i]), dtype=fp.dtype)
               for i in range(fp.shape[-1])]
        return torch.stack(out, dim=-1)
    else:
        out = [np.interp(x, xp, fp[:, i]) for i in range(fp.shape[-1])]
        return np.stack(out, axis=-1)


def parse_ccm(data_string: str):
    ccm = np.array([float(x.group("nominator")) / float(x.group("denominator")) for x in re.finditer(
        r"(?P<nominator>[-+]?\d+)/(?P<denominator>\d+|[-+]?\d+\.\d+|[-+]?\d+)", data_string)])
    ccm = ccm.reshape(3, 3)
    return ccm


def parse_tonemap_curve(data_string: str) -> np.ndarray:
    channels = re.findall(r'(R|G|B):\[(.*?)\]', data_string)
    result_array = np.zeros((3, len(channels[0][1].split('),')), 2))

    for i, (_, channel_data) in enumerate(channels):
        pairs = channel_data.split('),')
        for j, pair in enumerate(pairs):
            x, y = map(float, re.findall(r'([\d\.]+)', pair))
            result_array[i, j] = (x, y)
    return result_array


def apply_tonemap_curve(image, tonemap):
    # apply tonemap curve to each color channel
    image_toned = image.clone().cpu().numpy()

    for i in range(3):
        x_vals, y_vals = tonemap[i][:, 0], tonemap[i][:, 1]
        image_toned[i] = np.interp(image_toned[i], x_vals, y_vals)

    # Convert back to PyTorch tensor
    image_toned = torch.tensor(image_toned, dtype=torch.float32)

    return image_toned


def debatch(batch):
    """ Collapse batch and channel dimension together
    """
    debatched = []

    for x in batch:
        if len(x.shape) <= 1:
            raise Exception("This tensor is to small to debatch.")
        elif len(x.shape) == 2:
            debatched.append(x.reshape(x.shape[0] * x.shape[1]))
        else:
            debatched.append(x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]))

    return debatched


def colorize_tensor(value, vmin=None, vmax=None, cmap=None, colorbar=False, height=9.6, width=7.2):
    """ Convert tensor to 3 channel RGB array according to colors from cmap
        similar usage as plt.imshow
    """
    assert len(value.shape) == 2  # H x W

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(width, height)
    a = ax.imshow(value.detach().cpu(), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_axis_off()
    if colorbar:
        cbar = plt.colorbar(a, fraction=0.05)
        cbar.ax.tick_params(labelsize=30)
    plt.tight_layout()
    plt.close()

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0

    return torch.tensor(img).permute(2, 0, 1).float()


def write_mp4(frames: np.ndarray,
              path: str = 'test.mp4',
              fps: float = 24.0,
              progress_bar: bool = False):
    """Writes the frames to a video file.

    Parameters
    ----------
    frames : np.ndarray
        Frames to write to video in shape BxHxWxC or BxHxW. C is either 1 or 3.
    path : str, optional
        Path to the video file, by default 'test.mp4'
    fps : float, optional
        Fps in the video, by default 24.0
    progress_bar : bool, optional
        Show progress bar, by default False

    Raises
    ------
    ValueError
        If wrong number of channels in frames.
    """
    if len(frames.shape) not in [3, 4]:
        raise ValueError(f"Unsupported frame shape: {frames.shape}.")

    if len(frames.shape) == 4:
        num_frames, height, width, channels = frames.shape
    elif len(frames.shape) == 3:
        num_frames, height, width = frames.shape
        channels = 1
    else:
        raise ValueError(f"Unsupported frame shape: {frames.shape}.")

    frames = ((frames - frames.min()) / frames.max() * 255).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))

    bar = None
    if progress_bar:
        bar = tqdm(total=num_frames, desc='Writing video frames')

    if channels in [3, 4]:  # RGB(A) -> BGR
        for frame in frames:
            video.write(frame[:, :, [2, 1, 0]])
            if progress_bar:
                bar.update(1)
    elif channels == 1:  # grayscale
        for frame in frames:
            video.write(frame[:, :, None].repeat(3, 2))
            if progress_bar:
                bar.update(1)
    else:
        raise ValueError(f"Unsupported channel size: {channels}.")

    cv2.destroyAllWindows()
    video.release()


@torch.jit.script
def n_layers_alpha_compositing(
        images: torch.Tensor, zbuffer: torch.Tensor) -> torch.Tensor:
    """
    Applies N layers alpha compositing to the input images,
    bases on the z-buffer values.

    Parameters
    ----------
    images: torch.Tensor
        The input images with shape (N, [..., B], C).
        C must be 4 (RGBA).
        Image values are expected to be in the range [0, 1].

    zbuffer: torch.Tensor
        The z-buffer values for the images with shape (N).

    Returns
    -------
    torch.Tensor
        The alpha composited image with shape ([..., B], C).
    """
    N = images.shape[0]
    flattened_shape = images.shape[1:-1]
    C = images.shape[-1]
    if C != 4:
        raise ValueError(
            "The last dimension of the input images must be 4 (RGBA).")

    B = torch.prod(torch.tensor(flattened_shape)).item()
    images = images.reshape(N, B, C)  # (N, B, C)

    order = torch.argsort(zbuffer).unsqueeze(1)
    colors = images[:, :, :3]  # (N, B, 3)
    alphas = images[:, :, 3]
    inv_alphas = (1 - alphas)

    # Apply N object alpha matting. This is done by multiplying the alpha of the object with the inverse of the alphas of the objects before it.
    # We do it by calculating 1-alpha for each object, and
    bidx = torch.arange(B, device=inv_alphas.device).unsqueeze(
        0).repeat(N, 1)  # (N, B)
    sorted_inv_alphas = inv_alphas[order, bidx]
    sorted_alphas = alphas[order, bidx]  # (N, B, T, 1)
    sorted_colors = colors[order, bidx]

    rolled_inv_alpha = torch.roll(sorted_inv_alphas, 1, dims=0)
    rolled_inv_alpha[0] = 1.

    alpha_chain = torch.cumprod(rolled_inv_alpha, dim=0)
    sorted_per_layer_alphas = alpha_chain * sorted_alphas
    fused_color = (sorted_per_layer_alphas.unsqueeze(-1).repeat(1,
                   1, 3) * sorted_colors).sum(dim=0)  # (B, 3)
    out_image = torch.zeros(B, 4)
    out_image[:, :3] = fused_color  # (3, H, W)
    out_image[:, 3] = sorted_per_layer_alphas.sum(dim=0)
    return out_image.reshape(flattened_shape + (4,))
