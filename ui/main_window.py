import os
from pathlib import Path
from typing import Type

import numpy as np
import settings
from PyQt6 import QtCore
from PyQt6.QtCore import QThread
from PyQt6.QtGui import QImage, QPixmap, QIcon, QAction, QIntValidator
from PyQt6.QtWidgets import QMainWindow, QLabel, QHBoxLayout, QWidget, QVBoxLayout, QStackedWidget, QSlider, \
    QPushButton, QToolBar, QDialog, QComboBox, QFileDialog, QLineEdit, QMessageBox
from numpy import typing as npt
from dmconvert.converter import DmMediaConverter, DmMediaReader, DmMediaWriter, DmMediaSeekableReader, DmMediaParams
from dmconvert.readers import DmCameraReader, DmVideoReader, DmImagesReader
from dmconvert.writers import DmVideoWriter, DmImageWriter, DmCallbackWriter
from depthmap_wrappers.models import Models
from .control_panel import ControlPanelWidget
from .processors_settings import POSTPROCESSOR_ELEMENTS, PREPROCESSOR_ELEMENTS
from .waitingspinnerwidget import QtWaitingSpinner

selected_model = None

READERS_LIST: list[Type[DmMediaReader]] = [DmVideoReader, DmCameraReader, DmImagesReader]
READERS_TO_WRITERS_MAP: dict[Type[DmMediaReader], Type[DmMediaWriter]] = {
    DmVideoReader: DmVideoWriter,
    DmImagesReader: DmImageWriter
}


class WorkerThread(QThread):
    image: npt.NDArray
    converter: DmMediaConverter = None

    s_image_ready = QtCore.pyqtSignal(np.ndarray, np.ndarray, int)
    s_log = QtCore.pyqtSignal(str)

    def run(self):
        if self.converter:
            try:
                self.converter.start()
            except Exception as e:
                self.s_log.emit(str(e))
        else:
            self.s_log.emit('Необходимо задать параметры работы через настройки')

    def set_converter(self, converter: DmMediaConverter):
        self.converter = converter

        def ready(img, dm):
            pos = 0
            if isinstance(self.converter.reader, DmMediaSeekableReader):
                pos = self.converter.reader.progress
            self.s_image_ready.emit(img, dm, pos)

        self.converter.writers.append(DmCallbackWriter(ready))

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
        self.seek_widget.valueChanged.connect(self.worker.seek_video)
        timeline_layout.addWidget(self.seek_widget)
        main_layout.addLayout(pictures_layout)
        main_layout.addLayout(timeline_layout)
        main_layout.addLayout(self.panels_layout)
        # Вывод картинок
        self.picture_img = QLabel(self)
        self.picture_dm = QLabel(self)
        self.picture_img.setMinimumSize(320, 240)
        self.picture_dm.setMinimumSize(320, 240)
        self.picture_dm.setScaledContents(True)
        self.picture_img.setScaledContents(True)
        pictures_layout.addWidget(self.picture_img)
        pictures_layout.addWidget(self.picture_dm)
        # Панели управления конвертацией
        postprocessors_panel = ControlPanelWidget("Постпроцессоры", POSTPROCESSOR_ELEMENTS, self)
        preprocessors_panel = ControlPanelWidget("Препроцессоры", PREPROCESSOR_ELEMENTS, self)
        postprocessors_panel.s_control_changed.connect(self.worker.change_postprocessor)
        preprocessors_panel.s_control_changed.connect(self.worker.change_preprocessor)
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

    def show_image_slot(self, img, dm, pos):
        img_h, img_w, img_channel = img.shape
        dm_h, dm_w = dm.shape
        pix_img = QPixmap.fromImage(QImage(img.data, img_w, img_h, img_w * img_channel, QImage.Format.Format_BGR888))
        pix_dm = QPixmap.fromImage(QImage(dm.data, dm_w, dm_h, dm_w, QImage.Format.Format_Grayscale8))
        self.seek_widget.setValue(pos)
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

        l_writer = QLabel("Место сохранения (для вывода только на экран оставить пустым)", self)
        main_layout.addWidget(l_writer)

        path_out_layout = QHBoxLayout()
        main_layout.addLayout(path_out_layout)
        self.le_path_out = QLineEdit(self)
        path_out_layout.addWidget(self.le_path_out)

        self.pb_select_path_out = QPushButton("Выбрать", self)
        self.pb_select_path_out.clicked.connect(self.select_path_out)
        path_out_layout.addWidget(self.pb_select_path_out)

        bt_apply = QPushButton("Применить", self)
        bt_apply.clicked.connect(self.apply)
        main_layout.addWidget(bt_apply)

    def select_path(self):
        path = self.get_path_dialog(save=False)
        self.le_path.setText(path)

    def select_path_out(self):
        path = self.get_path_dialog(save=True)
        self.le_path_out.setText(path)

    def get_path_dialog(self, save: bool):
        reader = self.readers_mapping[self.cb_reader.currentText()]
        path: str
        if reader == DmVideoReader:
            if save:
                path, _ = self.file_dialog.getSaveFileName(self, "Выберите видеофайл",
                                                           filter="Video (*.mp4)")
                if Path(path).suffix.lower() != '.mp4':
                    path = path + '.mp4'
            else:
                path, _ = self.file_dialog.getOpenFileName(self, "Выберите видеофайл",
                                                           filter="Video (*.avi *.mp4 *.mkv)")
        elif reader == DmImagesReader:
            path = self.file_dialog.getExistingDirectory(self, "Выберите директорию")
        else:
            path = ""
        return path

    def change_reader(self, index: int):
        self.current_reader = list(self.readers_mapping.items())[index][1]
        self.__setup_reader_param_line_edit()

    def apply(self):
        try:
            model = self.models_mapping[self.cb_model.currentText()].value
            loader = settings.MODEL_LOADER()
            converter = DmMediaConverter(model, self.current_reader(self.le_path.text()), loader)
            self.__add_writer_if_need(converter)
            self.s_apply_settings.emit(converter)
            self.close()
        except Exception as e:
            MainWindow.log(str(e))

    def __setup_reader_param_line_edit(self):
        self.le_path.clear()
        if self.current_reader == DmCameraReader:
            self.le_path.setPlaceholderText("Введите номер камеры:")
            self.le_path.setValidator(QIntValidator())
        else:
            self.le_path.setPlaceholderText("")
            self.le_path.setValidator(None)

        can_select_path = self.current_reader in (DmVideoReader, DmImagesReader)
        self.pb_select_path.setVisible(can_select_path)
        self.le_path_out.setVisible(can_select_path)
        self.pb_select_path_out.setVisible(can_select_path)

    def __add_writer_if_need(self, converter: DmMediaConverter):
        writer = READERS_TO_WRITERS_MAP.get(self.current_reader)
        if writer is not None and self.le_path_out.text() != "":
            converter.writers.append(writer(self.le_path_out.text()))
