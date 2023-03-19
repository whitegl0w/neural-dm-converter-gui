from numba import njit
from .converter import RED, GREEN, BLUE
import math
import numpy as np
import numpy.typing as npt


def create_anaglyph_processor(max_offset: int, direction: int):
    """
    Выполняет конвертацию 2D кадра в анаглифный формат по карте глубины
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
