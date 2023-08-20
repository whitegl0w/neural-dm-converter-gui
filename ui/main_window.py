import os
from typing import Optional, Type

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import QThread
from PyQt6.QtGui import QImage, QPixmap, QIcon, QAction
from PyQt6.QtWidgets import QMainWindow, QLabel, QHBoxLayout, QWidget, QVBoxLayout, QStackedWidget, QSlider, \
    QPushButton, QToolBar, QDialog, QComboBox, QFileDialog, QLineEdit, QMessageBox
from numpy import typing as npt
from dmconvert.converter import DmMediaConverter, DmMediaReader, DmMediaWriter, DmMediaSeekableReader
from dmconvert.readers import DmCameraReader, DmVideoReader, DmImagesReader
from dmconvert.writers import DmVideoWriter, DmImageWriter, DmCallbackWriter
from models.settings import Models
from .control_panel import ControlPanelWidget
from .settings import POSTPROCESSOR_ELEMENTS, PREPROCESSOR_ELEMENTS
from .waitingspinnerwidget import QtWaitingSpinner

selected_model = Models.DEFAULT_SMALL

READERS_LIST: list[Type[DmMediaReader]] = [DmVideoReader, DmCameraReader, DmImagesReader]
WRITERS_LIST: list[Type[DmMediaWriter]] = [DmVideoWriter, DmImageWriter]


class WorkerThread(QThread):
    image: npt.NDArray
    converter: DmMediaConverter = None

    s_image_ready = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    s_log = QtCore.pyqtSignal(str)

    def run(self):
        # self.reader = DmCameraReader(cam_number=0)  # file_path="video_test/1.mp4")
        # self.converter = DmMediaConverter(selected_model, self.reader)
        # self.converter.writers.append(DmCallbackWriter(lambda img, dm: self.s_image_ready.emit(img, dm)))
        if self.converter:
            self.converter.start()
        else:
            self.s_log.emit('Необходимо задать параметры работы через настройки')

    def set_converter(self, converter: DmMediaConverter):
        self.converter = converter
        self.converter.writers.append(DmCallbackWriter(lambda img, dm: self.s_image_ready.emit(img, dm)))

    def stop(self):
        self.converter and self.converter.stop()

    def change_postprocessor(self, new_list):
        if self.converter:
            self.converter.postprocessors = new_list

    def change_preprocessor(self, new_list):
        if self.converter:
            self.converter.preprocessors = new_list

    def seek_video(self, position: int):
        if isinstance(self.converter.reader, DmMediaSeekableReader):
            print(position)
            self.converter.reader.seek(position)


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
        self.worker.s_image_ready.connect(self.show_image_slot)
        self.worker.s_log.connect(self.log)
        self.s_program_will_finish.connect(self.worker.stop)

        # Меню
        tool_bar = QToolBar("panel")
        self.addToolBar(tool_bar)
        self.m_settings = QAction("Настройки", self)
        self.m_settings.triggered.connect(self.show_settings)
        tool_bar.addAction(self.m_settings)

        # Основные виджеты
        self.loading_widget = QtWaitingSpinner(self)
        self.stacked_widget = QStackedWidget(self)
        self.main_widget = QWidget(self)
        self.stacked_widget.addWidget(self.loading_widget)
        self.stacked_widget.addWidget(self.main_widget)
        main_layout = QVBoxLayout()
        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.stacked_widget)
        # Настройка Layout-ов
        pictures_layout = QHBoxLayout()
        self.panels_layout = QHBoxLayout()
        timeline_layout = QHBoxLayout()
        play_stop_button = QPushButton(self)
        play_stop_button.clicked.connect(self.play)
        play_stop_button.setText("Play/Pause")
        timeline_layout.addWidget(play_stop_button)
        self.seek_widget = QSlider(QtCore.Qt.Orientation.Horizontal, self)
        # self.seek_widget.setMaximum(100)
        self.seek_widget.valueChanged.connect(self.worker.seek_video)
        timeline_layout.addWidget(self.seek_widget)
        main_layout.addLayout(pictures_layout)
        main_layout.addLayout(timeline_layout)
        main_layout.addLayout(self.panels_layout)
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
        self.panels_layout.addWidget(preprocessors_panel)
        self.panels_layout.addWidget(postprocessors_panel)

        self.loading(False)
        self.show()

    def play(self):
        if self.worker.isRunning():
            self.m_settings.setVisible(True)
            self.worker.stop()
            self.worker.quit()
        else:
            self.loading(True)
            self.m_settings.setVisible(False)
            self.worker.start()

    def show_image_slot(self, img, dm):
        img_h, img_w, img_channel = img.shape
        dm_h, dm_w = dm.shape
        pix_img = QPixmap.fromImage(QImage(img.data, img_w, img_h, img_w * img_channel, QImage.Format.Format_BGR888))
        pix_dm = QPixmap.fromImage(QImage(dm.data, dm_w, dm_h, dm_w, QImage.Format.Format_Grayscale8))
        self.picture_img.setPixmap(pix_img)
        self.picture_dm.setPixmap(pix_dm)
        self.loading(False)

    def prepare_for_exit(self):
        if self.worker.isRunning():
            self.s_program_will_finish.emit()
            self.worker.quit()
            self.worker.wait()
        self.worker.deleteLater()

    def loading(self, flag: bool):
        if flag:
            self.stacked_widget.setCurrentWidget(self.loading_widget)
            self.loading_widget.start()
        else:
            self.stacked_widget.setCurrentWidget(self.main_widget)
            self.loading_widget.stop()

    def show_settings(self):
        dialog = SettingsDialog(self)
        dialog.s_apply_settings.connect(self.set_converter)
        dialog.exec()

    @staticmethod
    def log(text: str):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(text)
        msg.setWindowTitle('Info')
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def set_converter(self, converter: DmMediaConverter):
        reader = converter.reader
        if isinstance(reader, DmMediaSeekableReader):
            self.seek_widget.setMaximum(reader.duration)
            print(f"max: {reader.duration}")
        self.worker.set_converter(converter)


class SettingsDialog(QDialog):
    s_apply_settings = QtCore.pyqtSignal(DmMediaConverter)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.readers_mapping = {reader.display_name(): reader for reader in READERS_LIST}
        self.models_mapping = {model.name: model for model in Models}
        self.file_dialog = QFileDialog(self)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        l_model = QLabel("Выбор модели", self)
        main_layout.addWidget(l_model)
        self.cb_model = QComboBox(self)
        self.cb_model.addItems(self.models_mapping.keys())
        main_layout.addWidget(self.cb_model)

        l_reader = QLabel("Выбор источника", self)
        main_layout.addWidget(l_reader)
        self.cb_reader = QComboBox(self)
        self.cb_reader.addItems(self.readers_mapping.keys())
        self.current_reader = self.readers_mapping[self.cb_reader.currentText()]
        self.cb_reader.activated.connect(self.change_reader)
        main_layout.addWidget(self.cb_reader)

        path_layout = QHBoxLayout()
        main_layout.addLayout(path_layout)
        self.le_path = QLineEdit(self)
        path_layout.addWidget(self.le_path)

        self.pb_select_path = QPushButton("Выбрать", self)
        self.pb_select_path.clicked.connect(self.select_path)
        path_layout.addWidget(self.pb_select_path)

        bt_apply = QPushButton("Применить", self)
        bt_apply.clicked.connect(self.apply)
        main_layout.addWidget(bt_apply)

    def select_path(self):
        reader = self.readers_mapping[self.cb_reader.currentText()]
        path: str
        if reader == DmVideoReader:
            path, _ = self.file_dialog.getOpenFileName(self, "Выберите видеофайл", filter="Video (*.avi *.mp4 *.mkv)")
        elif reader == DmImagesReader:
            path = self.file_dialog.getExistingDirectory(self, "Выберите директорию с изображениями")
        else:
            path = ""

        self.le_path.setText(path)

    def change_reader(self, index: int):
        self.current_reader = list(self.readers_mapping.items())[index][1]
        self.pb_select_path.setVisible(self.current_reader in (DmVideoReader, DmImagesReader))

    def apply(self):
        model = self.models_mapping[self.cb_model.currentText()]
        converter = DmMediaConverter(model, self.current_reader(self.le_path.text()))
        print(self.current_reader)
        self.s_apply_settings.emit(converter)
        self.close()
