import glob
import os
import threading
import cv2

from functools import cached_property
from cv2 import VideoCapture
from numpy import typing as npt
from typing import Optional, Generator
from .converter import DmMediaReader, DmMediaParams, DmMediaSeekableReader, ReaderError


def _create_media_params(cap: VideoCapture) -> DmMediaParams:
    return DmMediaParams(
        fps=int(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def raise_if_path_not_existed(path: str):
    if not os.path.exists(path):
        raise ReaderError(f"Путь {path} не существует")


class DmVideoReader(DmMediaSeekableReader):
    @property
    def progress(self) -> int:
        params = self.prepare_and_get_params()
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES) / params.fps)

    @staticmethod
    def display_name() -> str:
        return "Чтение видео из файла"

    def __init__(self, file_path: str):
        raise_if_path_not_existed(file_path)
        self._source = file_path
        self._cap: Optional[VideoCapture] = None
        self._media_param: Optional[DmMediaParams] = None
        self.lock = threading.Lock()

    def prepare_and_get_params(self) -> DmMediaParams:
        if not self.is_ready():
            self._cap = cv2.VideoCapture(self._source)
            self._media_param = _create_media_params(self._cap)
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

    def seek(self, position_sec: int):
        self.lock.acquire()
        self._cap.set(cv2.CAP_PROP_POS_MSEC, position_sec * 1000)
        self.lock.release()

    @cached_property
    def duration(self) -> (int, int):
        params = self.prepare_and_get_params()
        return int(params.frame_count / params.fps)

    def is_ready(self) -> bool:
        return self._cap and self._cap.isOpened()


class DmCameraReader(DmMediaReader):
    @staticmethod
    def display_name() -> str:
        return "Чтение видеопотока с камеры"

    def __init__(self, cam_number: str):
        self._source = int(cam_number)
        self._cap: Optional[VideoCapture] = None
        self._media_param: Optional[DmMediaParams] = None

    def prepare_and_get_params(self) -> DmMediaParams:
        if not self.is_ready():
            self._cap = cv2.VideoCapture(self._source)
            self._media_param = _create_media_params(self._cap)
        return self._media_param

    def data(self) -> Generator[npt.NDArray, any, None]:
        while self._cap is not None and self._cap.isOpened():
            ret, img = self._cap.read()

            if not ret or img is None:
                break

            yield img

    def close(self):
        self._cap and self._cap.release()

    def is_ready(self) -> bool:
        return self._cap and self._cap.isOpened()


class DmImagesReader(DmMediaReader):
    @staticmethod
    def display_name() -> str:
        return "Чтение изображений из директории"

    def __init__(self, directory: str):
        raise_if_path_not_existed(directory)
        self._directory = directory
        self._files: Optional[list[str]] = None

    def prepare_and_get_params(self) -> DmMediaParams:
        self._files = glob.glob(os.path.join(self._directory, "*"))
        return DmMediaParams(
            frame_count=len(self._files)
        )

    def data(self) -> Generator[npt.NDArray, any, None]:
        return (cv2.imread(file) for file in self._files)

    def is_ready(self) -> bool:
        return self._files and len(self._files) != 0
