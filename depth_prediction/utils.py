"""Utils for monoDepth.
"""
import numpy as np
import cv2
import torch
from numba import njit


def resize_image(img):
    """Resize image and make it fit for network.

    Args:
        img (array): image

    Returns:
        tensor: data ready for network
    """
    height_orig = img.shape[0]
    width_orig = img.shape[1]

    if width_orig > height_orig:
        scale = width_orig / 384
    else:
        scale = height_orig / 384

    height = (np.ceil(height_orig / scale / 32) * 32).astype(int)
    width = (np.ceil(width_orig / scale / 32) * 32).astype(int)

    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img_resized = (
        torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).contiguous().float()
    )
    img_resized = img_resized.unsqueeze(0)

    return img_resized


def resize_depth(depth, width, height):
    """Resize depth map and bring to CPU (numpy).

    Args:
        depth (tensor): depth
        width (int): image width
        height (int): image height

    Returns:
        array: processed depth
    """
    depth = torch.squeeze(depth[0, :, :, :]).to("cpu")

    depth_resized = cv2.resize(
        depth.numpy(), (width, height), interpolation=cv2.INTER_CUBIC
    )

    return depth_resized


@njit
def write_depth(depth, bits=1, eps=1e-2):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=np.float32)

    return out.astype(np.uint8)
