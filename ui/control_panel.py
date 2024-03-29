from collections import OrderedDict
from PyQt6 import QtCore
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QHBoxLayout, QWidget, QVBoxLayout, QListView, QSlider, \
    QListWidget, QScrollArea, QAbstractItemView, QListWidgetItem, QGroupBox

from .parameters import ControlElement, ControlProperty


class ControlPanelWidget(QWidget):
    s_control_changed = QtCore.pyqtSignal(list)

    def __init__(self, name: str, elements: list[ControlElement], parent: QWidget = None):
        super().__init__(parent)

        self._active_elements = OrderedDict((elem.name, None) for elem in elements)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        panel_title = QLabel(name, self)
        title_font = QFont()
        title_font.setBold(True)
        panel_title.setFont(title_font)
        main_layout.addWidget(panel_title)

        body_layout = QHBoxLayout()
        main_layout.addLayout(body_layout)

        scroll = QScrollArea(self)
        body_layout.addWidget(scroll, 3)

        scroll_widget = QWidget(self)
        content_layout = QVBoxLayout()
        scroll.setWidget(scroll_widget)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidgetResizable(True)
        scroll_widget.setLayout(content_layout)

        for elem in elements:
            sb = ControlElementWidget(elem, self)
            sb.s_control_changed.connect(self._control_changed_slot)
            content_layout.addWidget(sb)

        self.apply_order = QListWidget(self)
        self.apply_order.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.apply_order.setFlow(QListView.Flow.TopToBottom)
        self.apply_order.setWrapping(True)
        self.apply_order.setResizeMode(QListView.ResizeMode.Adjust)
        self.apply_order.setMovement(QListView.Movement.Snap)
        self.apply_order.model().rowsMoved.connect(self._order_changed_slot)
        body_layout.addWidget(self.apply_order, 1)

        for elem in elements:
            item = QListWidgetItem()
            item.setText(elem.name)
            self.apply_order.addItem(item)

    def _control_changed_slot(self, name, curr):
        self._active_elements[name] = curr
        self._raise_control_changed()

    def _order_changed_slot(self):
        new_order = [self.apply_order.item(i).text() for i in range(self.apply_order.count())]
        active_elements_in_new_order: list = sorted(self._active_elements.items(), key=lambda x: new_order.index(x[0]))
        self._active_elements.clear()
        self._active_elements.update(active_elements_in_new_order)
        self._raise_control_changed()

    def _raise_control_changed(self):
        new_values = [v for v in self._active_elements.values() if v is not None]
        self.s_control_changed.emit(new_values)


class ControlElementWidget(QWidget):
    s_control_changed = QtCore.pyqtSignal(str, object)

    def __init__(self, elem: ControlElement, parent: QWidget = None):
        super().__init__(parent)

        self.elem = elem
        self.props_state = dict((prop.name, prop.min_value) for prop in elem.properties)

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        self.group = QGroupBox(elem.name, self)
        self.group.setCheckable(True)
        self.group.setChecked(False)
        self.group.clicked.connect(self._raise_control_changed)
        main_layout.addWidget(self.group)

        props_layout = QVBoxLayout()
        for prop in elem.properties:
            prop_widget = ControlPropertyWidget(prop, self)
            prop_widget.s_prop_has_changed.connect(self._prop_changed_slot)
            props_layout.addWidget(prop_widget)

        if len(elem.properties) == 0:
            props_layout.addWidget(QLabel("Нет настраиваемых параметров", self))

        self.group.setLayout(props_layout)

    def _prop_changed_slot(self, k, v):
        self.props_state[k] = v
        self._raise_control_changed()

    def _raise_control_changed(self):
        if self.group.isChecked():
            new_build = self.elem.builder(**self.props_state)
            self.s_control_changed.emit(self.elem.name, new_build)
        else:
            self.s_control_changed.emit(self.elem.name, None)
            self.last_build = None


class ControlPropertyWidget(QWidget):
    s_prop_has_changed = QtCore.pyqtSignal(str, int)

    def __init__(self, prop: ControlProperty, parent: QWidget = None):
        super().__init__(parent)
        self.prop = prop

        prop_layout = QHBoxLayout(self)

        prop_label = QLabel(prop.caption, self)
        prop_label.setWordWrap(True)
        prop_layout.addWidget(prop_label, 2)

        prop_widget = QSlider(QtCore.Qt.Orientation.Horizontal, self)
        prop_widget.setTickPosition(QSlider.TickPosition.TicksBelow)
        prop_widget.setMinimum(prop.min_value)
        prop_widget.setMaximum(prop.max_value)
        prop_widget.setValue(prop.min_value)
        prop_layout.addWidget(prop_widget, 6)

        self.prop_value = QLabel(self)
        self.prop_value.setText(f"{prop_widget.value()}")
        prop_widget.valueChanged.connect(self._value_changed_slot)
        prop_layout.addWidget(self.prop_value, 1)

        self.setLayout(prop_layout)

    def _value_changed_slot(self, new_value: int):
        self.prop_value.setText(f"{new_value}")
        self.s_prop_has_changed.emit(self.prop.name, new_value)
