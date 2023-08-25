import cv2
import os
import sys
import qtvscodestyle

from PyQt6.QtWidgets import QApplication
from dmconvert.postprocessors import create_anaglyph_processor
from dmconvert.converter import DmMediaConverter, DmMediaReader
from dmconvert.readers import DmVideoReader, DmImagesReader, DmCameraReader
from dmconvert.writers import DmScreenWriter, DmImageWriter, DmVideoWriter
from argparse import ArgumentParser

import settings
from ui.main_window import MainWindow
from depthmap_wrappers.models import Models


def use_ui():
    q_app = QApplication(sys.argv)
    stylesheet = qtvscodestyle.load_stylesheet(qtvscodestyle.Theme.SOLARIZED_LIGHT)
    q_app.setStyleSheet(stylesheet)
    window = MainWindow()
    q_app.aboutToQuit.connect(window.prepare_for_exit)
    exit(q_app.exec())


def use_cli():
    parser = ArgumentParser(prog='Neural depth map tool')
    parser.add_argument('mode', type=str, help='Video source type CAM/FILE/IMG')
    parser.add_argument('source', type=str, help='Camera number for CAM, file path for FILE, folder path for IMG')
    parser.add_argument('-t', '--targets', nargs='+', type=str, help='SCREEN, IMAGES, VIDEO')
    parser.add_argument('-a', '--anaglyph', action='store_true', help='Anaglyph output')
    parser.add_argument('-l', '--large', action='store_true', help='Use dpt-large model')
    args = parser.parse_args()

    reader: DmMediaReader
    match args.mode.lower():
        case 'vid':
            reader = DmVideoReader(file_path=args.source)
        case 'cam':
            reader = DmCameraReader(cam_number=args.source)
        case 'img':
            reader = DmImagesReader(directory=args.source)
        case _:
            print('Incorrect mode, allowed: CAM, VID, IMG')
            exit(1)

    model = Models.DEFAULT_SMALL if not args.large else Models.DEFAULT_LARGE

    converter = DmMediaConverter(model, reader)
    converter.preprocessors.append(lambda img: cv2.resize(img, (640, 480), 1, 1, interpolation=cv2.INTER_AREA))

    for target in args.targets:
        match target.lower():
            case 'screen':
                converter.writers.append(DmScreenWriter())
            case 'images':
                converter.writers.append(DmImageWriter('output2', write_concat=True))
            case 'video':
                converter.writers.append(DmVideoWriter('out2.avi'))

    if args.anaglyph:
        converter.postprocessors.append(create_anaglyph_processor(10, 1))

    converter.start()


def apply_settings():
    if settings.AUTODETECT_MODELS:
        Models.autodetect()


def main():
    apply_settings()
    if len(sys.argv) == 1:
        use_ui()
    else:
        use_cli()


if __name__ == '__main__':
    path, _ = os.path.split(os.path.abspath(__file__))
    os.chdir(path)
    main()
