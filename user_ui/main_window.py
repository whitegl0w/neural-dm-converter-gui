from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import QThread, QModelIndex
from PyQt6.QtGui import QImage, QPixmap, QPalette
from PyQt6.QtWidgets import QMainWindow, QLabel, QHBoxLayout, QWidget, QVBoxLayout, QListView, QSpinBox, QSlider, \
    QListWidget, QScrollArea, QCheckBox
from numpy import typing as npt

from .parameters import ControlElement, ControlProperty
from .settings import POSTPROCESSOR_ELEMENTS
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
        self.converter.writers.append(DmQtWriter(lambda img, dm: self.s_image_ready.emit(img, dm)))
        self.converter.start()

    def stop(self):
        self.converter and self.converter.stop()

    def change_postprocessor(self, to_remove, to_add):
        to_remove and self.converter.postprocessors.remove(to_remove)
        to_add and self.converter.postprocessors.append(to_add)

    def change_preprocessor(self, to_remove, to_add):
        to_remove and self.converter.preprocessors.remove(to_remove)
        to_add and self.converter.preprocessors.append(to_add)


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
        control_panel = ControlPanelWidget(POSTPROCESSOR_ELEMENTS, self)
        control_panel.setMinimumHeight(300)
        main_layout.addLayout(pictures_layout)
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

        control_panel.s_control_changed.connect(self.worker.change_postprocessor)

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


class ControlPanelWidget(QWidget):
    s_control_changed = QtCore.pyqtSignal(object, object)

    def __init__(self, elements: list[ControlElement], parent: QWidget = None):
        super().__init__(parent)

        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)
        scroll = QScrollArea(self)
        main_layout.addWidget(scroll)

        scroll_widget = QWidget(self)
        content_layout = QVBoxLayout(self)
        scroll.setWidget(scroll_widget)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidgetResizable(True)
        scroll_widget.setLayout(content_layout)

        for elem in elements:
            sb = ControlElementWidget(elem, self)
            sb.s_control_changed.connect(lambda prev, curr: self.s_control_changed.emit(prev, curr))
            content_layout.addWidget(sb)


class ControlElementWidget(QWidget):
    s_control_changed = QtCore.pyqtSignal(object, object)

    def __init__(self, elem: ControlElement, parent: QWidget = None):
        super().__init__(parent)

        self.elem = elem
        self.props_state = dict((prop.name, prop.min_value) for prop in elem.properties)
        self.last_build: Optional[Callable] = None

        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        label = QLabel(self)
        label.setText(elem.name)
        self.enabled = QCheckBox(self)
        self.enabled.stateChanged.connect(self._raise_control_changed)
        props_layout = QVBoxLayout(self)

        for prop in elem.properties:
            prop_widget = ControlPropertyWidget(prop, self)
            prop_widget.s_prop_has_changed.connect(self._prop_changed_slot)
            props_layout.addWidget(prop_widget)

        main_layout.addWidget(label)
        main_layout.addWidget(self.enabled)
        main_layout.addLayout(props_layout)

    def _prop_changed_slot(self, k, v):
        self.props_state[k] = v
        self._raise_control_changed()

    def _raise_control_changed(self):
        if self.enabled.isChecked():
            new_build = self.elem.builder(**self.props_state)
            self.s_control_changed.emit(self.last_build, new_build)
            self.last_build = new_build
        else:
            self.s_control_changed.emit(self.last_build, None)
            self.last_build = None


class ControlPropertyWidget(QWidget):
    s_prop_has_changed = QtCore.pyqtSignal(str, int)

    def __init__(self, prop: ControlProperty, parent: QWidget = None):
        super().__init__(parent)
        self.prop = prop

        prop_layout = QHBoxLayout(self)

        prop_label = QLabel(self)
        prop_label.setText(prop.caption)
        prop_layout.addWidget(prop_label)

        prop_widget = QSlider(QtCore.Qt.Orientation.Horizontal, self)
        prop_widget.setTickPosition(QSlider.TickPosition.TicksBelow)
        prop_widget.setMinimum(prop.min_value)
        prop_widget.setMaximum(prop.max_value)
        prop_widget.setValue(prop.min_value)
        prop_layout.addWidget(prop_widget)

        self.prop_value = QLabel(self)
        self.prop_value.setText(f"{prop_widget.value()}")
        prop_widget.valueChanged.connect(self._value_changed_slot)
        prop_layout.addWidget(self.prop_value)

        self.setLayout(prop_layout)

    def _value_changed_slot(self, new_value: int):
        self.prop_value.setText(f"{new_value}")
        self.s_prop_has_changed.emit(self.prop.name, new_value)
