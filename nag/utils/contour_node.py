from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import cv2 as cv
from tools.viz.matplotlib import plot_as_image
from matplotlib import pyplot as plt


def draw_contours(points, ref_mask):
    mask = np.zeros_like(ref_mask).astype(np.uint8)
    mask = cv.drawContours(mask, [points], -1, 255.)
    return mask


@dataclass
class ContourNode:

    index: int
    """Index of the contour in the hierarchy"""

    points: np.ndarray
    """Contour points in format (n, 2) (y, x)"""

    hierarchy: np.ndarray
    """Hierarchy of the contour in format (1, 4) [Next, Previous, First_Child, Parent]"""

    color: np.ndarray
    """Color of the contour True for white, False for black"""

    children: list = field(default_factory=list)
    """List of children nodes"""

    parent: "ContourNode" = None
    """Parent node"""

    points_length: int = field(default=0)
    """Number of points in the contour"""

    def __post_init__(self):
        self.points_length = len(self.points)

    def get_recursive_points(self, pts: Optional[np.ndarray] = None):
        if pts is None:
            pts = self.points.copy()
        else:
            pts = np.concatenate([pts, self.points])
        for child in self.children:
            pts = child.get_recursive_points(pts)
        return pts

    def set_recursive_points(self, pts: np.ndarray) -> np.ndarray:
        if self.points_length > len(pts):
            raise ValueError("Not enough points to set")
        if self.points_length == 0:
            raise ValueError("No points to set")
        mp = pts[:self.points_length].copy()
        self.points = mp
        new_pts = pts[self.points_length:]
        for c, child in enumerate(self.children):
            new_pts = child.set_recursive_points(new_pts)
        return new_pts

    @classmethod
    def from_contour_hierarchy(cls,
                               index: int,
                               contours: np.ndarray,
                               hierarchies: np.ndarray,
                               mask: np.ndarray,
                               parent: Optional["ContourNode"] = None):
        contour = contours[index][:, 0]
        color = mask[contour[0, 1], contour[0, 0]]
        if parent is not None and parent.color:
            color = False
        node = cls(index=index, points=contour, color=color,
                   parent=parent, hierarchy=hierarchies[index])
        children = [(i) for i, x in enumerate(hierarchies) if x[3] == index]
        for (i) in children:
            child = cls.from_contour_hierarchy(
                index=i, contours=contours, hierarchies=hierarchies, mask=mask, parent=node)
            node.children.append(child)
        return node

    def mask_apply(self, mask: np.ndarray, positive_lines: Optional[np.ndarray] = None):
        # Use fill convex poly
        if positive_lines is None:
            positive_lines = np.zeros_like(mask).astype(np.uint8)
        points = self.points
        color = (np.array(self.color).astype(float) * 255)
        # Contour lines seem to be always on the contour line on boolean mask
        positive_lines = cv.drawContours(
            positive_lines, [points], -1, 255).astype(np.uint8)
        if color == 255:
            mask = cv.fillPoly(mask.astype(float), [
                               points], color).astype(np.uint8)
        else:
            proto = np.zeros_like(mask).astype(np.uint8)
            proto = cv.fillPoly(proto.astype(float), [
                                points], 255).astype(np.uint8)
            # If color is black, dont mask color all contour lines white
            proto[positive_lines == 255] = 0
            mask[proto == 255] = 0
        # return mask
        for child in self.children:
            mask = child.mask_apply(mask, positive_lines=positive_lines)
        return mask

    def image_apply(self, image: np.ndarray, positive_lines: Optional[np.ndarray] = None):
        # Use fill convex poly
        if positive_lines is None:
            positive_lines = np.zeros_like(image).astype(np.uint8)
        points = self.points
        color = np.array([self.color])
        if len(color.shape) > 1:
            color = color[0]
        #if color == 255:
        image = cv.fillPoly(image, [points], color.item())
        # else:
        #     proto = np.zeros_like(mask).astype(np.uint8)
        #     proto = cv.fillPoly(proto.astype(float), [
        #                         points], 255).astype(np.uint8)
        #     # If color is black, dont mask color all contour lines white
        #     proto[positive_lines == 255] = 0
        #     mask[proto == 255] = 0
        # # return mask
        for child in self.children:
            image = child.image_apply(image, positive_lines=positive_lines)
        return image

    def print(self, indent: int = 4, base_indent: int = 0):
        cindent = " " * base_indent
        aindent = " " * base_indent + " " * (indent // 2)
        text = ""
        text += cindent + f"ContourNode: {self.index}" + "\n"
        text += aindent + f"Color: {self.color}" + "\n"
        for child in self.children:
            text += child.print(indent=indent,
                                base_indent=base_indent + indent)
        return text

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> List["ContourNode"]:
        contours, hierarchy = cv.findContours(mask.astype(
            np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        parents = [i for i, x in enumerate(
            hierarchy[0]) if x is not None and x[3] == -1]
        parent_nodes = []
        for pindex in parents:
            node = cls.from_contour_hierarchy(
                index=pindex, contours=contours, hierarchies=hierarchy[0], mask=mask)
            parent_nodes.append(node)
        return parent_nodes
    
    @classmethod
    def from_image(cls, image: np.ndarray) -> List["ContourNode"]:
        contours, hierarchy = cv.findContours(image.astype(
            np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        parents = [i for i, x in enumerate(
            hierarchy[0]) if x is not None and x[3] == -1]
        parent_nodes = []
        for pindex in parents:
            node = cls.from_contour_hierarchy(
                index=pindex, contours=contours, hierarchies=hierarchy[0], mask=image)
            parent_nodes.append(node)
        return parent_nodes

    @staticmethod
    def to_mask(nodes: List["ContourNode"], mask_shape: tuple, base_color: np.ndarray) -> np.ndarray:
        mask = np.zeros(mask_shape).astype(np.uint8)
        mask[...] = base_color
        lines = np.zeros(mask_shape).astype(np.uint8)
        for node in nodes[::-1]:
            mask = node.mask_apply(mask, positive_lines=lines)
        return mask

    @staticmethod
    def to_image(nodes: List["ContourNode"], image_shape: tuple, base_color: np.ndarray) -> np.ndarray:
        image = np.zeros(image_shape).astype(np.uint8)
        image[...] = base_color
        lines = np.zeros(image_shape).astype(np.uint8)
        for node in nodes[::-1]:
            image = node.image_apply(image, positive_lines=lines)
        return image

    @staticmethod
    def print_nodes(nodes: List["ContourNode"]):
        text = ""
        for node in nodes:
            text += node.print()
        return text

    @staticmethod
    def plot_contours_list(nodes: List["ContourNode"], mask_shape: tuple):
        mask = np.zeros(mask_shape).astype(np.uint8)
        for i, node in enumerate(nodes):
            mask = node.plot_contours(mask)
        return mask

    @staticmethod
    def get_recursive_points_list(nodes: List["ContourNode"]):
        pts = None
        for node in nodes:
            pts = node.get_recursive_points(pts)
        return pts

    @staticmethod
    def set_recursive_points_list(nodes: List["ContourNode"], pts: np.ndarray):
        new_pts = pts
        for node in nodes:
            new_pts = node.set_recursive_points(new_pts)
        return new_pts
