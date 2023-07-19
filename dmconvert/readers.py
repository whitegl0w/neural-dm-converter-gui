import glob
import os
import threading
import cv2

from functools import cached_property
from cv2 import VideoCapture
from numpy import typing as npt
from typing import Optional, Generator
from .converter import DmMediaReader, DmMediaParams, DmMediaSeekableReader


class DmVideoReader(DmMediaSeekableReader):
    def __init__(self, file_path: Optional[str] = None, cam_number: Optional[int] = None):
        self._source = file_path if file_path is not None else cam_number
        self._cap: Optional[VideoCapture] = None
        self._media_param: Optional[DmMediaParams] = None
        self.lock = threading.Lock()

    def prepare(self) -> DmMediaParams:
        self._cap = cv2.VideoCapture(self._source)

        self._media_param = DmMediaParams(
            fps=int(self._cap.get(cv2.CAP_PROP_FPS)),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

        return self._media_param

    def data(self) -> Generator[npt.NDArray, any, None]:
        while self._cap is not None and self._cap.isOpened():
            self.lock.acquire()
            ret, img = self._cap.read()
            self.lock.release()

            if not ret or img is None:
                break

            yield img

    def close(self):
        self._cap and self._cap.release()

    def seek(self, position_ms: int):
        self.lock.acquire()
        self._cap.set(cv2.CAP_PROP_POS_MSEC, position_ms)
        self.lock.release()

    @cached_property
    def duration(self) -> (int, int):
        return int(self._media_param.frame_count / self._media_param.fps)


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
