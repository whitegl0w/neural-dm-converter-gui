import glob
import os

import cv2
from cv2 import VideoCapture
from numpy import typing as npt
from typing import Optional, Generator
from .converter import DmMediaReader, DmMediaParams


class DmVideoReader(DmMediaReader):
    def __init__(self, file_path: Optional[str] = None, cam_number: Optional[int] = None):
        self._source = file_path if file_path is not None else cam_number
        self._cap: Optional[VideoCapture] = None

    def prepare(self) -> DmMediaParams:
        self._cap = cv2.VideoCapture(self._source)

        return DmMediaParams(
            fps=int(self._cap.get(cv2.CAP_PROP_FPS)),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def data(self) -> Generator[npt.NDArray, any, None]:
        while self._cap is not None and self._cap.isOpened():
            ret, img = self._cap.read()

            if not ret or img is None:
                break

            yield img

    def close(self):
        self._cap.release() if self._cap is not None else None


class DmImagesReader(DmMediaReader):
    def __init__(self, directory: str):
        self._directory = directory
        self._files: Optional[list[str]] = None

    def prepare(self) -> DmMediaParams:
        self._files = glob.glob(os.path.join(self._directory, "*"))
        return DmMediaParams(
            frame_count=len(self._files)
        )

    def data(self) -> Generator[npt.NDArray, any, None]:
        return (cv2.imread(file) for file in self._files)
