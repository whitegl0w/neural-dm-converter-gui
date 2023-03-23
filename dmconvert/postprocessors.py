import cv2
from numba import njit
from .converter import RED, GREEN, BLUE
import math
import numpy as np
import numpy.typing as npt


def create_anaglyph_processor(max_offset: int, direction: int = 1):
    """
    Создает конвертор 2D кадра в анаглифный формат по карте глубины
    :param max_offset: Максимальное значение параллакса (пикселей)
    :param direction: направление сдвига
    """

    @njit
    def convert(img: npt.NDArray, dm: npt.NDArray):
        anaglyph: npt.NDArray = np.zeros(img.shape).astype(np.uint8)

        anaglyph[:, :, GREEN] = img[:, :, GREEN]
        anaglyph[:, :, BLUE] = img[:, :, BLUE]

        for x in range(dm.shape[0]):
            for y in range(dm.shape[1]):
                offset = math.floor((dm[x][y] / 255) * max_offset * direction)
                for k in range(min(offset + 1, dm.shape[1] - y)):
                    if not anaglyph[x, y + k, RED]:
                        anaglyph[x, y + k, RED] = img[x, y, RED]

        return anaglyph, dm

    return convert


def create_dm_correcter(windows_size: int, return_num: int, move_factor: int):
    """
    Создает постпроцессор для устранения колебания карты глубины в соседних кадрах
    :param windows_size: размер окна усреднения
    :param return_num: номер кадра для возврата (позволяет выбирать: усреднять кадр с предыдущими или следующими)
    :param move_factor: порог для определения движения в кадре
    :return:
    """

    dm_frame_holder: list[np.ndarray] = []
    img_frame_holder: list[np.ndarray] = []

    def convert(img: npt.NDArray, dm: npt.NDArray):

        def is_static(x: npt.NDArray, y: npt.NDArray):
            return (np.sum(cv2.absdiff(x, y)) / x.size * 100) < move_factor

        frames_for_avg = [frame for frame in dm_frame_holder if is_static(frame, dm)]
        new_dm = dm.copy() / (len(frames_for_avg) + 1)
        for frame in frames_for_avg:
            new_dm = new_dm + (frame / (len(frames_for_avg) + 1))

        new_dm = new_dm.astype(np.uint8)

        dm_frame_holder.append(dm.copy())
        img_frame_holder.append(img.copy())
        while len(dm_frame_holder) > windows_size:
            dm_frame_holder.remove(dm_frame_holder[0])
            img_frame_holder.remove(img_frame_holder[0])

        result_num = min(len(dm_frame_holder) - 1, return_num)
        return img_frame_holder[result_num], new_dm

    return convert
