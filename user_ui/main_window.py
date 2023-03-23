import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow, QLabel, QHBoxLayout, QWidget, QVBoxLayout
from numpy import typing as npt

from dmconvert.converter import DmMediaConverter
from dmconvert.readers import DmVideoReader
from dmconvert.writers import DmQtWriter
from .control_panel import ControlPanelWidget
from .settings import POSTPROCESSOR_ELEMENTS, PREPROCESSOR_ELEMENTS
from .waitingspinnerwidget import QtWaitingSpinner

models = {
    'dpt_large': "models/dpt_large-midas-2f21e586.pt",
    'midas_v21': "models/model.pt"
}

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

        # Поток для работы
        self.worker = WorkerThread(self)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.s_image_ready.connect(self.show_image_slot)
        self.s_program_will_finish.connect(self.worker.stop)

        # Основные виджеты
        self.loading = QtWaitingSpinner(self)
        main_widget = QWidget()
        main_layout = QVBoxLayout(self)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
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
        self.loading.start()
        self.show()

    def show_image_slot(self, img, dm):
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
