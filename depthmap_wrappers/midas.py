from typing import Any

import numpy as np
import torch
from depthmap.midas_sources.midas.model_loader import load_model
from depthmap.midas_sources.run import process as midas_process
from depthmap_wrappers.base import BaseDmWrapper
from depthmap_wrappers.models import Model


class MidasDmWrapper(BaseDmWrapper):
    _model: Any
    _transform: Any
    _net_size: Any
    _device: Any
    _model_type: Any

    def prepare_model(self, model: Model, *args):
        self._model_type = model.type
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self._transform, *self._net_size \
            = load_model(self._device, model.path, model.type, False, None, False)

    def process(self, image, *args):
        transformed_image = self._transform({"image": image})["image"]
        with torch.no_grad():
            prediction = midas_process(self._device, self._model, self._model_type, transformed_image, self._net_size,
                                       image.shape[1::-1], False, False)
        return self._convert_to_uint8(prediction)

    @staticmethod
    def _convert_to_uint8(prediction):
        if not np.isfinite(prediction).all():
            prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)

        depth_min = prediction.min()
        depth_max = prediction.max()

        max_val = (2 ** 8) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (prediction - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(prediction.shape, dtype=prediction.dtype)

        return out.astype("uint8")
