import sys
from video_converter import convert
import os


def main():
    if len(sys.argv) > 1:
        convert(sys.argv[1], "out.avi")


if __name__ == '__main__':
    path, _ = os.path.split(os.path.abspath(__file__))
    os.chdir(path)
    main()
