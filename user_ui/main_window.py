import os
import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import QThread
from PyQt6.QtGui import QImage, QPixmap, QIcon, QColor
from PyQt6.QtWidgets import QMainWindow, QLabel, QHBoxLayout, QWidget, QVBoxLayout, QStackedLayout, QStackedWidget
from numpy import typing as npt

from dmconvert.converter import DmMediaConverter
from dmconvert.readers import DmVideoReader
from dmconvert.writers import DmQtWriter
from .control_panel import ControlPanelWidget
from .settings import POSTPROCESSOR_ELEMENTS, PREPROCESSOR_ELEMENTS
from .waitingspinnerwidget import QtWaitingSpinner
from models.settings import models

selected_model = 'midas_v21'


class WorkerThread(QThread):
    image: npt.NDArray
    converter: DmMediaConverter = None

    s_image_ready = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def run(self):
        reader = DmVideoReader(file_path="D:\\Projects\\Python\\pythonProject\\video_test\\1.mp4")
        self.converter = DmMediaConverter(selected_model, models[selected_model], reader)
        self.converter.writers.append(DmQtWriter(lambda img, dm: self.s_image_ready.emit(img, dm)))
        self.converter.start()

    def stop(self):
        self.converter and self.converter.stop()

    def change_postprocessor(self, new_list):
        self.converter.postprocessors = new_list

    def change_preprocessor(self, new_list):
        self.converter.preprocessors = new_list


class MainWindow(QMainWindow):
    s_program_will_finish = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Depth map конвертер")
        script_dir = os.path.dirname(os.path.realpath(__file__))
        icon = QIcon(os.path.join(script_dir, "icon.ico"))
        self.setWindowIcon(icon)

        # Поток для работы
        self.worker = WorkerThread(self)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.s_image_ready.connect(self.show_image_slot)
        self.s_program_will_finish.connect(self.worker.stop)

        # Основные виджеты
        self.loading_widget = QtWaitingSpinner(self)
        self.stacked_widget = QStackedWidget(self)
        self.main_widget = QWidget(self)
        self.stacked_widget.addWidget(self.loading_widget)
        self.stacked_widget.addWidget(self.main_widget)
        main_layout = QVBoxLayout(self)
        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.stacked_widget)
        # Настройка Layout-ов
        pictures_layout = QHBoxLayout(self)
        panels_layout = QHBoxLayout(self)
        main_layout.addLayout(pictures_layout)
        main_layout.addLayout(panels_layout)
        # Вывод картинок
        self.picture_img = QLabel(self)
        self.picture_dm = QLabel(self)
        self.picture_img.setMinimumSize(640, 480)
        self.picture_dm.setMinimumSize(640, 480)
        self.picture_dm.setScaledContents(True)
        self.picture_img.setScaledContents(True)
        pictures_layout.addWidget(self.picture_img)
        pictures_layout.addWidget(self.picture_dm)
        # Панели управления конвертацией
        postprocessors_panel = ControlPanelWidget("Постпроцессоры", POSTPROCESSOR_ELEMENTS, self)
        preprocessors_panel = ControlPanelWidget("Препроцессоры", PREPROCESSOR_ELEMENTS, self)
        postprocessors_panel.s_control_changed.connect(self.worker.change_postprocessor)
        preprocessors_panel.s_control_changed.connect(self.worker.change_preprocessor)
        postprocessors_panel.setMinimumHeight(300)
        preprocessors_panel.setMinimumHeight(300)
        panels_layout.addWidget(preprocessors_panel)
        panels_layout.addWidget(postprocessors_panel)

        # Запуск работы
        self.worker.start()
        self.loading(True)
        self.show()

    def show_image_slot(self, img, dm):
        img_h, img_w, img_channel = img.shape
        dm_h, dm_w = dm.shape
        pix_img = QPixmap.fromImage(QImage(img.data, img_w, img_h, img_w * img_channel, QImage.Format.Format_BGR888))
        pix_dm = QPixmap.fromImage(QImage(dm.data, dm_w, dm_h, dm_w, QImage.Format.Format_Grayscale8))
        self.picture_img.setPixmap(pix_img)
        self.picture_dm.setPixmap(pix_dm)
        self.loading(False)

    def prepare_for_exit(self):
        self.s_program_will_finish.emit()
        self.worker.quit()
        self.worker.wait()

    def loading(self, flag: bool):
        if flag:
            self.stacked_widget.setCurrentWidget(self.loading_widget)
            self.loading_widget.start()
        else:
            self.stacked_widget.setCurrentWidget(self.main_widget)
            self.loading_widget.stop()
