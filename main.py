import sys
from video_converter import convert, capture_camera
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(prog='Neural depth map tool')
    parser.add_argument('mode', help='Video source type CAM/FILE')
    parser.add_argument('source', help='Camera number for CAM, file path for FILE')
    parser.add_argument('-a', '--anaglyph', action='store_true', help='Anaglyph output')
    args = parser.parse_args()

    if args.mode == 'vid':
        convert(args.source, "out.avi", args.anaglyph)
    elif args.mode == 'cam':
        capture_camera(int(args.source), args.anaglyph)
    else:
        print('Incorrect mode, allowed: CAM, FILE')


if __name__ == '__main__':
    path, _ = os.path.split(os.path.abspath(__file__))
    os.chdir(path)
    main()
