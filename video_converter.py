import math
import cv2
import numpy as np
import numpy.typing as npt
from numba import njit
from depth_prediction.run import run, prepare

# model_path = "depth_prediction/model.pt"
model_path = "depth_prediction/dpt_large-midas-2f21e586.pt"

RED = 2
GREEN = 1
BLUE = 0

DIRECTION = 1
MAX_PIXEL_OFFSET = 23

@njit
def convert2anaglyph(frame: npt.NDArray, depth_map: npt.NDArray):
    """
    Выполняет конвертацию 2D кадра в анаглифный формат по карте глубины.
    :param offset: максимальное значение параллакса (пикселей)
    :param direction: направление сдвига.
    :param frame: Кадр для конвертации.
    :param depth_map: Карта глубины.
    :return: кадр в анаглифном формате.
    """
    anaglyph: npt.NDArray = np.zeros(frame.shape).astype(np.uint8)

    anaglyph[:, :, GREEN] = frame[:, :, GREEN]
    anaglyph[:, :, BLUE] = frame[:, :, BLUE]

    for x in range(depth_map.shape[0]):
        for y in range(depth_map.shape[1]):
            offset = math.floor((depth_map[x][y] / 255) * MAX_PIXEL_OFFSET * DIRECTION)
            for k in range(min(offset + 1, depth_map.shape[1] - y)):
                if not anaglyph[x, y + k, RED]:
                    anaglyph[x, y + k, RED] = frame[x, y, RED]

    return anaglyph


def prepare_frame(frame: npt.NDArray):
    """
    Конвертирует кадр видео в анаглифный формат.
    :param offset: максимальное значение параллакса (пикселей)
    :param direction: направление сдвига.
    :param frame: кадр видео в виде двумерного массива.
    :return: конвертированный кадр в анаглифный формат.
    """
    depth_map = run(frame)
    depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)

    return convert2anaglyph(frame, depth_map)


def convert(source: str, output: str, anaglyph: bool = True):
    """
    Конвертирует видео файл в анаглифный формат.
    :param anaglyph: true - выводит в анаглифном формате, false - выводит карту глубины
    :param source: путь к входному 2D видео.
    :param output: путь для записи конвертированного видео.
    """
    cap = cv2.VideoCapture(source)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'X264')

    video_out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    prepare(model_path)

    n = 0
    try:
        while cap.isOpened():
            n += 1
            print(f"{n} - {frame_count}")

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            convert_func = prepare_frame if anaglyph else run
            output_frame = convert_func(frame)
            video_out.write(output_frame)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        video_out.release()


def capture_camera(source: int, anaglyph: bool = False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print('Ошибка открытия камеры')
        exit(1)

    prepare(model_path)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Ошибка получения кадра')
                break

            convert_func = prepare_frame if anaglyph else run
            output = convert_func(frame)

            cv2.imshow('frame', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
