import sys
from video_converter import convert


def main():
    if len(sys.argv) > 1:
        convert(sys.argv[1], "out.avi")


if __name__ == '__main__':
    main()
