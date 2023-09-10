import os
from dataclasses import dataclass
from typing import Optional, Callable

import cv2
import numpy as np
from numpy import typing as npt
from .converter import DmMediaWriter, DmMediaParams


class DmVideoWriter(DmMediaWriter):
    @staticmethod
    def display_name() -> str:
        return "Запись видеофайла"

    def __init__(self, file_name: str, codec: str = 'mp4v'):
        self._file_name = file_name
        self._codec = codec
        self._cap: Optional[cv2.VideoWriter] = None

    def prepare(self, media_params: DmMediaParams):
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        self._cap = cv2.VideoWriter(self._file_name, fourcc, media_params.fps,
                                    (media_params.width, media_params.height))

    def write(self, img: npt.NDArray, dm: npt.NDArray):
        if self._cap is None:
            return

        self._cap.write(img)

    def close(self):
        self._cap.release() if self._cap is not None else None


@dataclass
class DmImageWriter(DmMediaWriter):
    @staticmethod
    def display_name() -> str:
        return "Запись в виде набора изображений"

    def __init__(self, directory: str, name_rule: Callable[[int], str] = None, write_dm: bool = False,
                 write_img: bool = False, write_concat: bool = False):
        self._directory = directory
        self._name_rule = name_rule
        self._img_num = 0
        self._write_dm = write_dm
        self._write_img = write_img
        self._write_concat = write_concat

    def prepare(self, media_params: DmMediaParams):
        os.makedirs(self._directory, exist_ok=True)

    def write(self, img: npt.NDArray, dm: npt.NDArray):
        self._img_num += 1

        name = self._name_rule(self._img_num) if self._name_rule else str(self._img_num)
        file = os.path.join(self._directory, name)

        if self._write_dm or self._write_concat:
            cv2.imwrite(f"{file}_dm.png", dm)

        if self._write_img or self._write_concat:
            cv2.imwrite(f"{file}_img.png", img)

        if self._write_concat:
            f_dm = cv2.imread(f"{file}_dm.png")
            f_img = cv2.imread(f"{file}_img.png")
            concat = np.concatenate((f_dm, f_img))
            cv2.imwrite(f"{file}_concat.png", concat)
            if not self._write_dm:
                os.remove(f"{file}_dm.png")
            if not self._write_img:
                os.remove(f"{file}_img.png")


class DmScreenWriter(DmMediaWriter):
    @staticmethod
    def display_name() -> str:
        return "Вывод на экран в помощью opencv"

    def write(self, img: npt.NDArray, dm: npt.NDArray):
        cv2.imshow('img', img)
        cv2.imshow('dm', dm)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


class DmCallbackWriter(DmMediaWriter):
    @staticmethod
    def display_name() -> str:
        return "Передача через callback"

    def __init__(self, callback):
        self._callback = callback

    def write(self, img: npt.NDArray, dm: npt.NDArray):
        self._callback(img, dm)
