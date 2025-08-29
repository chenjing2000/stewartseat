
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QGroupBox, QLineEdit, QLabel,
                               QComboBox, QPushButton, QDialog,
                               QSpacerItem, QSizePolicy)
from PySide6.QtGui import (QPalette, QColor, QDoubleValidator,
                           QIntValidator, QFontMetrics)

import numpy as np
import control as ct


class TabPage2(QWidget):
    def __init__(self):
        super().__init__()

        tabpage2_layout = QVBoxLayout(self)

        # 基本参数设置
        self.tf_group = QGroupBox("Transfer Function")
        self.tf_layout = QVBoxLayout(self.tf_group)
        tf_layout_1 = QHBoxLayout()

        self.comboboxes = []
        key_label = QLabel("阶次")
        combobox1 = QComboBox()
        combobox1.addItems(["1", "2", "3", "4", "5"])
        combobox1.setCurrentIndex(1)
        combobox1.currentIndexChanged.connect(self.on_combobox1_clicked)
        self.comboboxes.append(combobox1)

        tf_layout_1.addWidget(key_label, 1)
        tf_layout_1.addWidget(combobox1, 1)

        key_label = QLabel("顺序")
        combobox2 = QComboBox()
        combobox2.addItems(["由高到低", "由低到高"])
        self.comboboxes.append(combobox2)

        tf_layout_1.addWidget(key_label, 1)
        tf_layout_1.addWidget(combobox2, 1)

        tf_layout_1.addStretch(2)

        self.tf_layout_2 = QHBoxLayout()

        key_label = QLabel("分子")
        key_label.setFixedWidth(50)
        self.tf_layout_2.addWidget(key_label)

        self.numerator_boxes = []
        self.denominator_boxes = []

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_2, 2, True)
        self.numerator_boxes.append(value_boxes)

        self.tf_layout_3 = QHBoxLayout()

        key_label = QLabel("分母")
        key_label.setFixedWidth(50)
        self.tf_layout_3.addWidget(key_label)

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_3, 3, False)
        self.denominator_boxes.append(value_boxes)

        self.tf_layout.addLayout(tf_layout_1)
        self.tf_layout.addLayout(self.tf_layout_2)
        self.tf_layout.addLayout(self.tf_layout_3)

        tf_layout_4 = QHBoxLayout()

        btn_responses = QPushButton("系统响应")
        btn_responses.clicked.connect(self.btn_responses_clicked)
        tf_layout_4.addWidget(btn_responses)

        self.tf_layout.addLayout(tf_layout_4)

        tabpage2_layout.addWidget(self.tf_group)
        tabpage2_layout.addStretch()

    def set_transfer_function_boxes(self, layout: QHBoxLayout, count: int, isnumerator: bool):

        while layout.count() > 1:
            item = layout.takeAt(1)  # 取出第一个 item
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # 从父容器移除
                widget.deleteLater()    # 释放资源

        if isnumerator:
            layout.addStretch(1)

        value_boxes = []
        for id in range(count):
            value_box = QLineEdit()
            value_box.setAlignment(Qt.AlignmentFlag.AlignRight)

            validator = QDoubleValidator(
                -1e8, 1e8, 2)  # 限定范围，小数点取 2 位
            validator.setNotation(QDoubleValidator.StandardNotation)
            value_box.setValidator(validator)

            value_boxes.append(value_box)
            layout.addWidget(value_box, 2)

        if isnumerator:
            layout.addStretch(1)

        return value_boxes

    def on_combobox1_clicked(self):

        order = self.comboboxes[0].currentIndex() + 1

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_2, order, True)

        self.numerator_boxes = []
        self.numerator_boxes.append(value_boxes)

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_3, order + 1, False)

        self.numerator_boxes = []
        self.denominator_boxes.append(value_boxes)

    def btn_responses_clicked(self):

        order = self.comboboxes[0].currentIndex() + 1

        num = np.zeros(order, dtype=float)
        den = np.zeros(order + 1, dtype=float)

        direction = self.comboboxes[1].currentIndex()
