
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

from MSDChart import *


class TabPage2(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parent = parent

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

        key_label = QLabel("方式")
        combobox2 = QComboBox()
        combobox2.addItems(["普通", "ZPK"])
        combobox2.currentIndexChanged.connect(self.on_combobox2_clicked)
        self.comboboxes.append(combobox2)

        tf_layout_1.addWidget(key_label, 1)
        tf_layout_1.addWidget(combobox2, 1)

        key_label = QLabel("由高至低")
        key_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        key_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        tf_layout_1.addWidget(key_label, 1)

        tf_layout_1.addStretch(1)

        self.tf_layout.addLayout(tf_layout_1)

        self.tf_layout_2 = QHBoxLayout()

        key_label = QLabel("分子")
        key_label.setFixedWidth(50)
        self.tf_layout_2.addWidget(key_label)

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_2, 2, True)
        self.numerator_boxes = value_boxes.copy()

        self.tf_layout.addLayout(self.tf_layout_2)

        self.tf_layout_3 = QHBoxLayout()

        key_label = QLabel("分母")
        key_label.setFixedWidth(50)
        self.tf_layout_3.addWidget(key_label)

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_3, 3, False)
        self.denominator_boxes = value_boxes.copy()

        self.tf_layout.addLayout(self.tf_layout_3)

        tf_layout_4 = QHBoxLayout()

        self.btn_time_responses = QPushButton("时域响应")
        self.btn_time_responses.clicked.connect(
            self.btn_transient_responses_clicked)
        tf_layout_4.addWidget(self.btn_time_responses)

        self.transient_window = None
        self.frequency_window = None

        self.btn_frequency_responses = QPushButton("频域响应")
        self.btn_frequency_responses.clicked.connect(
            self.btn_frequency_responses_clicked)
        tf_layout_4.addWidget(self.btn_frequency_responses)

        self.btn_discrete_control = QPushButton("应用控制")
        tf_layout_4.addWidget(self.btn_discrete_control)

        self.tf_layout.addLayout(tf_layout_4)

        tabpage2_layout.addWidget(self.tf_group)
        tabpage2_layout.addStretch()

        self.Gs = 1.0  # default transfer function

    def set_transfer_function_boxes(self, layout: QHBoxLayout, count: int, isnumerator: bool):

        while layout.count() > 1:
            item = layout.takeAt(1)  # 取出第一个 item
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # 从父容器移除
                widget.deleteLater()    # 释放资源

        if isnumerator:
            layout.addStretch(1)

        boxes = []
        for id in range(count):
            value_box = QLineEdit("")
            value_box.setAlignment(Qt.AlignmentFlag.AlignCenter)

            validator = QDoubleValidator(
                -1e8, 1e8, 2)  # 限定范围，小数点取 2 位
            validator.setNotation(QDoubleValidator.StandardNotation)
            value_box.setValidator(validator)

            boxes.append(value_box)
            layout.addWidget(value_box, 2)

        if isnumerator:
            layout.addStretch(1)

        return boxes

    def on_combobox1_clicked(self):

        order = self.comboboxes[0].currentIndex() + 1

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_2, order, True)

        self.numerator_boxes = value_boxes.copy()

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_3, order + 1, False)

        self.denominator_boxes = value_boxes.copy()

        self.comboboxes[1].setCurrentIndex(0)

    def on_combobox2_clicked(self):

        method = self.comboboxes[1].currentIndex()

        # ZPK 配置方式
        if method == 1:
            order = self.comboboxes[0].currentIndex() + 1
            self.dialog_zpk = ZPKDialog(order=order, parent=self.parent)

            if self.dialog_zpk.exec() == QDialog.Accepted:
                Gs = ct.zpk(
                    self.dialog_zpk.zeros, self.dialog_zpk.poles, self.dialog_zpk.gain)

                num = Gs.num[0][0]
                den = Gs.den[0][0]

                for i, value in enumerate(reversed(num)):
                    self.numerator_boxes[order - 1 - i].setText(str(value))

                for i, value in enumerate(reversed(den)):
                    self.denominator_boxes[order - i].setText(str(value))

    def get_transfer_function_from_boxes(self):

        # 获取传递函数
        order = self.comboboxes[0].currentIndex() + 1

        num = np.zeros(order, dtype=float)
        den = np.zeros(order + 1, dtype=float)

        for id, box in enumerate(self.numerator_boxes):
            box_str = box.text().strip()
            if box_str == "":
                num[id] = 0.0
            else:
                num[id] = float(box_str)

        for id, box in enumerate(self.denominator_boxes):
            box_str = box.text().strip()
            if box_str == "":
                den[id] = 0.0
            else:
                den[id] = float(box_str)

        isGsValid = True

        if np.allclose(num, 0, 0, 1e-6) or np.allclose(den, 0, 0, 1e-6):
            self.parent.status_label.setText(
                "Numerator or denominator is null.")
            isGsValid = False
            return

        self.Gs = ct.TransferFunction(num, den)

        return isGsValid

    def btn_transient_responses_clicked(self):

        if not self.get_transfer_function_from_boxes():
            return

        # 时域响应

        dt = self.parent.dt
        time_stop = float(self.parent.sim_boxes[2].text())

        time = np.arange(0, time_stop, dt)

        # 1. impulse response
        t, y = ct.impulse_response(self.Gs, T=time)

        if self.transient_window is None or not self.transient_window.isVisible():
            self.transient_window = TransientWindow()   # 新建

        self.transient_window.show()
        self.transient_window.raise_()
        self.transient_window.activateWindow()

        plot_widget_i = self.transient_window.plot_widget_1
        plot_widget_j = self.transient_window.plot_widget_2
        plot_widget_k = self.transient_window.plot_widget_3

        for widget in [plot_widget_i, plot_widget_j, plot_widget_k]:
            self.clear_plot_widget(widget)

        plot_widget_i.plot(t, y, pen=pg.mkPen(
            color='r', width=2), name="脉冲响应")

        plot_widget_i.setLabel("left", "displacement (m)")

        # 2. step response
        t, y = ct.step_response(self.Gs, T=time)

        plot_widget_j.plot(t, y, pen=pg.mkPen(
            color='r', width=2), name="阶跃响应")

        plot_widget_j.setLabel("left", "displacement (m)")

        # 3. sine response
        amplitude = float(self.parent.sim_boxes[0].text().strip())
        omega = 2*np.pi*float(self.parent.sim_boxes[1].text().strip())
        x = amplitude * np.sin(omega*time)
        t, y = ct.forced_response(self.Gs, T=time, inputs=x)

        plot_widget_k.plot(t, y, pen=pg.mkPen(
            color='r', width=2), name="正弦响应")

        plot_widget_k.setLabel("left", "displacement (m)")

    def clear_plot_widget(self, plot_widget):
        # 清除子图内的元素
        plot_widget.setTitle("")
        plot_widget.clear()
        if plot_widget.plotItem.legend is not None:
            plot_widget.plotItem.legend.scene().removeItem(plot_widget.plotItem.legend)
            plot_widget.plotItem.legend = None
        plot_widget.plotItem.addLegend()

    def btn_frequency_responses_clicked(self):

        if not self.get_transfer_function_from_boxes():
            return

        mag, phase, omega = ct.frequency_response(self.Gs)

        if self.frequency_window is None or not self.frequency_window.isVisible():
            self.frequency_window = FrequencyWindow()   # 新建

        self.frequency_window.show()
        self.frequency_window.raise_()
        self.frequency_window.activateWindow()

        plot_widget_i = self.frequency_window.plot_widget_1
        plot_widget_j = self.frequency_window.plot_widget_2

        for widget in [plot_widget_i, plot_widget_j]:
            self.clear_plot_widget(widget)

        plot_widget_i.plot(omega, 20*np.log10(mag), pen=pg.mkPen(
            color='r', width=2), name='uncontrol')

        # 添加横轴
        plot_widget_i.addLine(y=0, pen=pg.mkPen(color='k', width=1))

        # Nyquist plot
        realpart = mag * np.cos(phase)
        imagpart = mag * np.sin(phase)

        xdata = np.concatenate([+realpart[::-1], realpart])
        ydata = np.concatenate([-imagpart[::-1], imagpart])

        plot_widget_j.plot(xdata, ydata, pen=pg.mkPen(
            color='r', width=2), name='uncontrol')

        # 添加横轴与纵轴
        plot_widget_j.addLine(x=0, pen=pg.mkPen(color='k', width=1))
        plot_widget_j.addLine(y=0, pen=pg.mkPen(color='k', width=1))
