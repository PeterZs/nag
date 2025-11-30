
import math
from typing import Any, Dict, Optional
from matplotlib.figure import Figure
import numpy as np
from tools.viz.matplotlib import saveable, get_mpl_figure, plot_as_image, figure_to_numpy
import matplotlib.pyplot as plt


def make_colorwheel() -> np.ndarray:
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
    -------
    colorwheel : np.ndarray
        RGB Color wheel. Output shape is [55, 3] in the range [0, 255] np.uint8.

    Credits
    -------
    Author: Tom Runia
    Date Created: 2018-08-03

    https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis/flow_vis.py

    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0, YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0, GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0, BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u: np.ndarray, v: np.ndarray, convert_to_bgr: bool = False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Parameters
    ----------
    u : np.ndarray
        Horizontal flow component of shape [H,W]
    v : np.ndarray
        Vertical flow component of shape [H,W]
    convert_to_bgr : bool, optional
        Convert output image to BGR, by default False

    Returns
    -------
    np.ndarray
        Flow visualization image of shape [H,W,3]

    Credits
    -------

    Author: Tom Runia
    Date Created: 2018-08-03

    https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis/flow_vis.py
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))

    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def get_wheel_image(image_size: int = 256):
    center_x, center_y = image_size // 2, image_size // 2
    # Create a grid of x, y coordinates
    x = np.linspace(-center_x, center_x, image_size)
    y = np.linspace(-center_y, center_y, image_size)

    # Create a meshgrid from x, y
    xx, yy = np.meshgrid(x, y)

    # Create a circle of radius half the image size
    circle = (xx ** 2 + yy ** 2) < (center_x ** 2)
    wheel = make_colorwheel()

    xx_circ = xx[circle]
    yy_circ = yy[circle]

    dist = np.sqrt(xx_circ ** 2 + yy_circ ** 2)
    # Compute the angle for each pixel
    angles = (np.arctan2(- yy_circ, - xx_circ) +
              np.pi) % (2 * np.pi)  # Normalize to [0, 2 * pi]

    # Interpolate the wheel colors
    base_color = np.ones((1, 3))  # White

    # Supersample the wheel
    xp = np.linspace(0., 2 * np.pi, len(wheel))
    wheel_exact_r = np.interp(angles, xp, wheel[:, 0], period=2 * np.pi)
    wheel_exact_g = np.interp(angles, xp, wheel[:, 1], period=2 * np.pi)
    wheel_exact_b = np.interp(angles, xp, wheel[:, 2], period=2 * np.pi)
    wheel_exact = np.stack(
        [wheel_exact_r, wheel_exact_g, wheel_exact_b], axis=1) / 255.

    # Interpolate the wheel colors with the distance from the center / base color
    angles_colors = wheel_exact
    color_frac = dist / (image_size // 2)
    base_frac = 1 - color_frac

    pixel_colors = base_frac[:, None] * base_color + \
        color_frac[:, None] * angles_colors

    wheel_img = np.zeros((image_size, image_size, 4))
    wheel_img[circle, :3] = pixel_colors
    wheel_img[circle, 3] = 1.
    return wheel_img


@saveable()
def get_wheel_figure(
    size: float = 3,
    marker_size: int = 12,
    axis: bool = True,
    labels: bool = True,
    circle: bool = True,
) -> Figure:
    from tools.viz.matplotlib import align_marker
    from mpl_toolkits.axisartist.axislines import SubplotZero, AxesZero
    from matplotlib import patches

    fig = plt.figure(figsize=(size, size))
    fig.set_size_inches(
        size,
        size,
        forward=False)
    ax = AxesZero(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    # fig.add_subplot(ax)

    pix = round(math.ceil(size * fig.dpi))

    wheel_img = get_wheel_image(pix)

    ax.imshow(wheel_img, extent=(-1., 1., -1., 1.))

    for direction in ["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        # ax.axis[direction].set_axisline_style("-|>")
        # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(axis)
        ax.axis[direction].lim = (-1, 1)

    for direction in ["left", "right", "bottom", "top"]:
        # hides borders
        ax.axis[direction].set_visible(False)

    axis_eps = 0
    ax.set_xlim(-1 - axis_eps, 1. + axis_eps)
    ax.set_ylim(-1 - axis_eps, 1. + axis_eps)

    ax.set_xticks([])
    ax.set_yticks([])

    if labels:
        ax.text(s="$u$", x=0.5, y=-0.1, va="top", ha="center")
        ax.text(s="$v$", y=0.5, x=-0.1, ha="right", va="center")

    if axis:
        ax.plot((1), (0), ls="", marker=align_marker('>', ha='right'), ms=marker_size, color="k",
                transform=ax.get_yaxis_transform(), clip_on=True)
        ax.plot((0), (1), ls="", marker=align_marker('^', ha='center', va="top"), ms=marker_size, color="k",
                transform=ax.get_xaxis_transform(), clip_on=True)

    if circle:
        circle1 = patches.Circle((0, 0), 1, color='k', fill=False)
        ax.add_patch(circle1)

    return fig


def flow_to_color(flow_uv: np.ndarray, clip_flow: Optional[float] = None, convert_to_bgr: bool = False) -> np.ndarray:
    """
    Expects a two dimensional flow image of shape.

    Parameters
    ----------
    flow_uv : np.ndarray
        Flow UV image of shape [H,W,2]
    clip_flow : float, optional
        Clip maximum of flow values, by default None
    convert_to_bgr : bool, optional
        Convert output image to BGR, by default False

    Returns
    -------
    np.ndarray
        Flow visualization image of shape [H,W,3]

    Credits
    -------

    Author: Tom Runia
    Date Created: 2018-08-03

    https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis/flow_vis.py
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def flow_to_color_legend(
        flow_uv: np.ndarray,
        clip_flow: Optional[float] = None,
        legend_location: str = "top right",
        legend_kwargs: Optional[Dict[str, Any]] = None) -> np.ndarray:

    if legend_kwargs is None:
        legend_kwargs = {}

    legend_ypos, legend_xpos = legend_location.split(" ")
    if legend_xpos not in ["left", "right"]:
        raise ValueError("legend_xpos must be 'left' or 'right'")
    if legend_ypos not in ["top", "bottom"]:
        raise ValueError("legend_ypos must be 'top' or 'bottom'")

    dpi = 300
    min_shape = min(flow_uv.shape[:2])

    # Default size 1 / 5 of the smallest dimension
    legend_size = legend_kwargs.pop(
        "size", min_shape // dpi / 5) if legend_kwargs is not None else 10
    color_flow = flow_to_color(flow_uv, clip_flow=clip_flow)

    with plt.ioff():
        wf = get_wheel_figure(size=legend_size, **legend_kwargs)
        npwf = figure_to_numpy(wf, dpi=dpi, transparent=True)

    legend_ypos, legend_xpos = legend_location.split(" ")
    if legend_xpos not in ["left", "right"]:
        raise ValueError("legend_xpos must be 'left' or 'right'")
    if legend_ypos not in ["top", "bottom"]:
        raise ValueError("legend_ypos must be 'top' or 'bottom'")

    fig_size_inch = min_shape // dpi
    legend_frac_size = legend_size / fig_size_inch

    # Y is swapped as in numpy top left corner is the origin
    if legend_xpos == "left":
        xl_start = 0
    elif legend_xpos == "right":
        xl_start = 1 - legend_frac_size
    if legend_ypos == "bottom":
        yl_start = 1 - legend_frac_size
    elif legend_ypos == "top":
        yl_start = 0

    min_x_idx = np.round(xl_start * color_flow.shape[-2]).astype(int)
    raw_x_max = min_x_idx + npwf.shape[-2]
    if raw_x_max > color_flow.shape[-2]:
        min_x_idx = color_flow.shape[-2] - npwf.shape[-2]

    min_y_idx = np.round(yl_start * color_flow.shape[-3]).astype(int)
    raw_y_max = min_y_idx + npwf.shape[-3]
    if raw_y_max > color_flow.shape[-3]:
        min_y_idx = color_flow.shape[-3] - npwf.shape[-3]

    color_flow_rgba = np.zeros(
        (color_flow.shape[0], color_flow.shape[1], 4), dtype=np.uint8)
    color_flow_rgba[..., :3] = color_flow
    color_flow_rgba[..., 3] = 255

    transparencies = npwf[..., 3:4] / 255.
    leg_px = (transparencies * (npwf[..., :3]))
    norm_px = ((1 - transparencies) *
               color_flow[min_y_idx:raw_y_max, min_x_idx:raw_x_max, :3])

    color_flow[min_y_idx:raw_y_max, min_x_idx:raw_x_max,
               :3] = np.round(leg_px + norm_px).astype(np.uint8)
    return color_flow
