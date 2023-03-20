import cv2
import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow, QLabel, QHBoxLayout, QWidget, QVBoxLayout
from numpy import typing as npt
from .waitingspinnerwidget import QtWaitingSpinner
from dmconvert.converter import DmMediaConverter, DmMediaWriter
from dmconvert.readers import DmVideoReader
from dmconvert.postprocessors import create_anaglyph_processor

models = {
    'dpt_large': "models/dpt_large-midas-2f21e586.pt",
    'midas_v21': "models/model.pt"
}

selected_model = 'midas_v21'


class DmQtWriter(DmMediaWriter):
    def __init__(self, callback):
        self._callback = callback

    def write(self, img: npt.NDArray, dm: npt.NDArray):
        self._callback(img, dm)


class WorkerThread(QThread):
    image: npt.NDArray
    converter: DmMediaConverter = None

    s_image_ready = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def run(self):
        self.converter = DmMediaConverter(selected_model, models[selected_model], DmVideoReader(cam_number=0))
        self.converter.preprocessors.append(lambda img: cv2.resize(img, (640, 480)))
        self.converter.postprocessors.append(create_anaglyph_processor(10, 1))
        self.converter.writers.append(DmQtWriter(lambda img, dm: self.s_image_ready.emit(img, dm)))
        self.converter.start()

    def stop(self):
        if self.converter:
            self.converter.stop()


class MainWindow(QMainWindow):
    s_program_will_finish = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        self.loading = QtWaitingSpinner(self)

        main_widget = QWidget()
        main_layout = QVBoxLayout(self)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        pictures_layout = QHBoxLayout(self)
        main_layout.addLayout(pictures_layout)

        control_panel = ControlWidget()
        main_layout.addWidget(control_panel)

        self.picture_img = QLabel(self)
        self.picture_dm = QLabel(self)
        self.picture_img.setFixedSize(640, 480)
        self.picture_dm.setFixedSize(640, 480)
        pictures_layout.addWidget(self.picture_img)
        pictures_layout.addWidget(self.picture_dm)

        self.worker = WorkerThread(self)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.s_image_ready.connect(self.show_image)
        self.s_program_will_finish.connect(self.worker.stop)
        self.worker.start()
        self.loading.start()

        self.show()

    def show_image(self, img, dm):
        img_h, img_w, img_channel = img.shape
        dm_h, dm_w = dm.shape
        pix_img = QPixmap.fromImage(QImage(img.data, img_w, img_h, img_w * img_channel, QImage.Format.Format_BGR888))
        pix_dm = QPixmap.fromImage(QImage(dm.data, dm_w, dm_h, dm_w, QImage.Format.Format_Grayscale8))
        self.picture_img.setPixmap(pix_img)
        self.picture_dm.setPixmap(pix_dm)
        self.loading.stop()

    def prepare_for_exit(self):
        self.s_program_will_finish.emit()
        self.worker.quit()
        self.worker.wait()


class ControlWidget(QWidget):
    def __init__(self):
        super().__init__()

        
