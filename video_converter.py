import math

import cv2
import numpy as np
import numpy.typing as npt
from depth_prediction.run import run, prepare
from numba import njit

DIRECTION = 1
MAX_PIXEL_OFFSET = 14
model_path = "depth_prediction/model.pt"

RED = 2
GREEN = 1
BLUE = 0


@njit
def convert2anaglyph(frame, depth_map):
    anaglyph: npt.NDArray = np.zeros(frame.shape).astype(np.uint8)

    anaglyph[:, :, GREEN] = frame[:, :, GREEN]
    anaglyph[:, :, BLUE] = frame[:, :, BLUE]
    # anaglyph[:, :, :] = frame[:, :, :]

    for x in range(depth_map.shape[0]):
        for y in range(depth_map.shape[1]):
            offset = math.floor((depth_map[x][y] / 255) * MAX_PIXEL_OFFSET * DIRECTION)
            if offset > 20 or offset < 0:
                print(offset)
            for k in range(min(offset + 1, depth_map.shape[1] - y)):
                if not anaglyph[x, y + k, RED]:
                    anaglyph[x, y + k, RED] = frame[x, y, RED]

    return anaglyph


def prepare_frame(frame: npt.NDArray):
    depth_map = run(frame)
    depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
    # depth_map = np.ones(frame.shape[:-1]) * 125

    return convert2anaglyph(frame, depth_map)


def convert(source: str, output: str):
    cap = cv2.VideoCapture(source)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    video_out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    prepare(model_path)

    n = 0
    while cap.isOpened():
        ret, frame = cap.read()
        n += 1
        print(f"{n} - {frame_count}")
        if frame is None:
            break
        anaglyph = prepare_frame(frame)
        # cv2.imshow('Data', anaglyph)
        video_out.write(anaglyph)

    cap.release()
    video_out.release()



