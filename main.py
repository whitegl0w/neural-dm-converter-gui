import cv2
import os
import sys
import qtvscodestyle

from PyQt6.QtWidgets import QApplication
from dmconvert.postprocessors import create_anaglyph_processor
from dmconvert.converter import DmMediaConverter, DmMediaReader
from dmconvert.readers import DmVideoReader, DmImagesReader
from dmconvert.writers import DmScreenWriter, DmImageWriter, DmVideoWriter
from argparse import ArgumentParser
from user_ui.main_window import MainWindow
from models.settings import models, DEFAULT_LARGE, DEFAULT_SMALL


def use_ui():
    qApp = QApplication(sys.argv)
    stylesheet = qtvscodestyle.load_stylesheet(qtvscodestyle.Theme.SOLARIZED_LIGHT)
    qApp.setStyleSheet(stylesheet)
    window = MainWindow()
    qApp.aboutToQuit.connect(window.prepare_for_exit)
    exit(qApp.exec())


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
            reader = DmVideoReader(cam_number=int(args.source))
        case 'img':
            reader = DmImagesReader(directory=args.source)
        case _:
            print('Incorrect mode, allowed: CAM, VID, IMG')
            exit(1)

    model = DEFAULT_SMALL if not args.large else DEFAULT_LARGE

    converter = DmMediaConverter(model, models[model], reader)
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


def main():
    if len(sys.argv) == 1:
        use_ui()
    else:
        use_cli()


if __name__ == '__main__':
    path, _ = os.path.split(os.path.abspath(__file__))
    os.chdir(path)
    main()
