import glob
import math
import os

import cv2
import numpy as np
import numpy.typing as npt
from numba import njit
from depth_prediction.run import run, prepare

# model_path = "depth_prediction/model.pt"
# model_path =

models = {
    'dpt_large': "depth_prediction/dpt_large-midas-2f21e586.pt",
    'midas_v21': "depth_prediction/model.pt"
}


active_model = 'midas_v21'


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
    prepare(active_model, models[active_model])

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

    prepare(active_model, models[active_model])

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


def load_image(in_path, out_path, anaglyph: bool = False):
    # get input
    img_names = glob.glob(os.path.join(in_path, "*"))
    num_images = len(img_names)
    print(num_images)

    # create output folder
    os.makedirs(out_path, exist_ok=True)

    prepare(active_model, models[active_model])

    print("start processing")

    for ind, img_name in enumerate(img_names):
        print(f"{ind}: {img_name}")

        img = cv2.imread(img_name)

        filename = os.path.join(
            out_path, os.path.splitext(os.path.basename(img_name))[0]
        )

        # convert_func = prepare_frame if anaglyph else run
        # rez_img = convert_func(img)

        # cv2.imwrite(filename + ".png", rez_img)

        depth_map = run(img)
        cv2.imwrite(filename + "_dm.png", depth_map)

        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        d3d = convert2anaglyph(img, depth_map)
        cv2.imwrite(filename + "_3d.png", d3d)

        ldm = cv2.imread(filename + "_dm.png")
        l3d = cv2.imread(filename + "_3d.png")
        common = np.concatenate((l3d, ldm, img))
        cv2.imwrite(filename + "_concat.png", common)

        os.remove(filename + "_dm.png")
        os.remove(filename + "_3d.png")


        # depth_map = cv2.imread(filename + ".png")
        # out = np.concatenate((depth_map, img))
        # cv2.imwrite(filename + "_concat.png", out)
