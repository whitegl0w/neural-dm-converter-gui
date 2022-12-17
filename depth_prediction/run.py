"""Compute depth maps for images in the input folder.
"""
import os
import glob

import numpy as np
import torch
from . import utils
import cv2

from torchvision.transforms import Compose
from .models.midas_net import MidasNet
from .models.transforms import Resize, NormalizeImage, PrepareForNet


model = None
device = None

transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )


def prepare(model_path):
    global model, device
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    model = MidasNet(model_path, non_negative=True)

    model.to(device)
    model.eval()


def run(image):
    global model, device
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    img_input = transform({"image": img})["image"]

    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return utils.write_depth(prediction, bits=1)
