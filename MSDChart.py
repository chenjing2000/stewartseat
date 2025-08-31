

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QGroupBox, QLineEdit, QLabel,
                               QComboBox, QPushButton, QDialog)
import pyqtgraph as pg
# from OpenGL.GL import *
# from OpenGL.GL.shaders import *

import numpy as np


class TransientWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Transient Response")
        self.setFixedSize(400, 730)
        self.move(630, 100)
        self.show()

        layout = QVBoxLayout(self)
        # Plotting widget
        self.plot_widget_1 = pg.PlotWidget()
        self.plot_widget_1.setTitle(
            "Position vs. Time", color='black', size="10pt")
        self.plot_widget_1.setLabel('left', 'Position (m)', color='black')
        self.plot_widget_1.setLabel('bottom', 'Time (s)', color='black')
        self.plot_widget_1.setBackground('white')
        self.plot_widget_1.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_1.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_widget_1.addLegend(labelTextColor='blue')
        self.plot_curve_11 = self.plot_widget_1.plot(
            [], [], pen=pg.mkPen(color='b', width=2), name='displacement')
        self.plot_curve_12 = self.plot_widget_1.plot(
            [], [], pen=pg.mkPen(color='r', width=2), name='excitation')

        self.coord_label_1 = QLabel("X: , Y: ")

        layout.addWidget(self.plot_widget_1, 1)
        layout.addWidget(self.coord_label_1)

        self.plot_widget_2 = pg.PlotWidget()
        self.plot_widget_2.setTitle(
            "Acceleration vs. Time", color='black', size="10pt")
        self.plot_widget_2.setLabel(
            'left', 'Acceleration (m/s^2)', color='black')
        self.plot_widget_2.setLabel('bottom', 'Time (s)', color='black')
        self.plot_widget_2.setBackground('white')
        self.plot_widget_2.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_2.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_widget_2.addLegend(labelTextColor='blue')
        self.plot_curve_2 = self.plot_widget_2.plot(
            [], [], pen=pg.mkPen(color='b', width=2), name="acceleration")

        self.coord_label_2 = QLabel("X: , Y: ")

        layout.addWidget(self.plot_widget_2, 1)
        layout.addWidget(self.coord_label_2)

        self.plot_widget_3 = pg.PlotWidget()
        self.plot_widget_3.setTitle(
            "Actuator Force vs. Time", color='black', size="10pt")
        self.plot_widget_3.setLabel(
            'left', 'Actuator Force (N)', color='black')
        self.plot_widget_3.setLabel('bottom', 'Time (s)', color='black')
        self.plot_widget_3.setBackground('white')
        self.plot_widget_3.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_3.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_widget_3.addLegend(labelTextColor='blue')
        self.plot_curve_3 = self.plot_widget_3.plot(
            [], [], pen=pg.mkPen(color='b', width=2), name="actuator force")

        self.coord_label_3 = QLabel("X: , Y: ")

        layout.addWidget(self.plot_widget_3, 1)
        layout.addWidget(self.coord_label_3)

        # 连接鼠标移动事件到自定义槽函数
        self.plot_widget_1.scene().sigMouseMoved.connect(
            lambda pos: FrequencyWindow.update_coords(pos, self.plot_widget_1, self.coord_label_1))
        self.plot_widget_2.scene().sigMouseMoved.connect(
            lambda pos: FrequencyWindow.update_coords(pos, self.plot_widget_2, self.coord_label_2))
        self.plot_widget_3.scene().sigMouseMoved.connect(
            lambda pos: FrequencyWindow.update_coords(pos, self.plot_widget_3, self.coord_label_3))

        self.show()


class BodeWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Bode Chart")
        self.setFixedSize(400, 600)
        self.move(630, 100)
        self.show()

        layout = QVBoxLayout(self)
        # Plotting widget
        self.plot_widget_1 = pg.PlotWidget()
        self.plot_widget_1.setLogMode(x=True, y=False)
        self.plot_widget_1.setTitle(
            "Amplitude vs. Frequency", color='black', size="10pt")
        self.plot_widget_1.setLabel('left', 'Amplitude (dB)', color='black')
        self.plot_widget_1.setLabel('bottom', 'Frequency (Hz)', color='black')
        self.plot_widget_1.setBackground('white')
        self.plot_widget_1.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_1.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_widget_1.addLegend(labelTextColor='blue')
        self.plot_curve_11 = self.plot_widget_1.plot(
            [], [], pen=pg.mkPen(color='r', width=2), name='displacement')
        self.plot_curve_12 = self.plot_widget_1.plot(
            [], [], pen=pg.mkPen(color='g', width=2), name='velocity')
        self.plot_curve_13 = self.plot_widget_1.plot(
            [], [], pen=pg.mkPen(color='b', width=2), name='acceleration')

        self.coord_label_1 = QLabel("X: , Y: ")

        layout.addWidget(self.plot_widget_1, 1)
        layout.addWidget(self.coord_label_1)

        self.plot_widget_2 = pg.PlotWidget()
        self.plot_widget_2.setLogMode(x=True, y=False)
        self.plot_widget_2.setTitle(
            "Phase vs. Frequency", color='black', size="10pt")
        self.plot_widget_2.setLabel('left', 'Phase (rad)', color='black')
        self.plot_widget_2.setLabel('bottom', 'Frequency (Hz)', color='black')
        self.plot_widget_2.setBackground('white')
        self.plot_widget_2.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_2.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_widget_2.addLegend(labelTextColor='blue')
        self.plot_curve_21 = self.plot_widget_2.plot(
            [], [], pen=pg.mkPen(color='r', width=2), name='displacement')
        self.plot_curve_22 = self.plot_widget_2.plot(
            [], [], pen=pg.mkPen(color='g', width=2), name='velocity')
        self.plot_curve_23 = self.plot_widget_2.plot(
            [], [], pen=pg.mkPen(color='b', width=2), name='acceleration')

        self.coord_label_2 = QLabel("X: , Y: ")

        layout.addWidget(self.plot_widget_2, 1)
        layout.addWidget(self.coord_label_2)

        # 连接鼠标移动事件到自定义槽函数
        self.plot_widget_1.scene().sigMouseMoved.connect(
            lambda pos: FrequencyWindow.update_coords(pos, self.plot_widget_1, self.coord_label_1))
        self.plot_widget_2.scene().sigMouseMoved.connect(
            lambda pos: FrequencyWindow.update_coords(pos, self.plot_widget_2, self.coord_label_2))

        self.show()


class FrequencyWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Nyquist Chart")
        self.setFixedSize(400, 600)
        self.move(630, 100)
        self.show()

        layout = QVBoxLayout(self)
        # Plotting widget
        self.plot_widget_1 = pg.PlotWidget()
        self.plot_widget_1.setLogMode(x=True, y=False)
        self.plot_widget_1.setTitle(
            "Bode Chart", color='black', size="10pt")
        self.plot_widget_1.setLabel('left', 'Amplitude (dB)', color='black')
        self.plot_widget_1.setLabel('bottom', 'Frequency (Hz)', color='black')
        self.plot_widget_1.setBackground('white')
        self.plot_widget_1.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_1.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_widget_1.addLegend(labelTextColor='blue')

        # 添加横轴
        self.plot_widget_1.addLine(y=0, pen=pg.mkPen(color='k', width=1))

        self.coord_label_1 = QLabel("X: , Y: ")

        layout.addWidget(self.plot_widget_1, 1)
        layout.addWidget(self.coord_label_1)

        self.plot_widget_2 = pg.PlotWidget()
        self.plot_widget_2.setLogMode(x=False, y=False)
        self.plot_widget_2.setTitle(
            "Frequency Chart", color='black', size="10pt")
        self.plot_widget_2.setLabel('left', 'Amplitude (rad)', color='black')
        self.plot_widget_2.setLabel('bottom', 'Frequency (Hz)', color='black')
        self.plot_widget_2.setBackground('white')
        self.plot_widget_2.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_2.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_widget_2.addLegend(labelTextColor='blue')
        self.plot_widget_2.plot([-1], [0], pen=None,
                                symbol="+", name='(-1,j0)')

        # 添加横轴与纵轴
        self.plot_widget_2.addLine(x=0, pen=pg.mkPen(color='k', width=1))
        self.plot_widget_2.addLine(y=0, pen=pg.mkPen(color='k', width=1))

        self.coord_label_2 = QLabel("X: , Y: ")

        layout.addWidget(self.plot_widget_2, 2)
        layout.addWidget(self.coord_label_2)

        # 连接鼠标移动事件到自定义槽函数
        self.plot_widget_1.scene().sigMouseMoved.connect(
            lambda pos: self.update_coords(pos, self.plot_widget_1, self.coord_label_1))
        self.plot_widget_2.scene().sigMouseMoved.connect(
            lambda pos: self.update_coords(pos, self.plot_widget_2, self.coord_label_2))

        self.show()

    @staticmethod
    def update_coords(pos, plot_widget: pg.PlotWidget, label: pg.TextItem):

        if plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = plot_widget.plotItem.vb.mapSceneToView(pos)
            xp, yp = mousePoint.x(), mousePoint.y()

            is_logx = plot_widget.plotItem.ctrl.logXCheck.isChecked()
            if is_logx:
                xp = 10 ** xp

            is_logy = plot_widget.plotItem.ctrl.logYCheck.isChecked()
            if is_logy:
                yp = 10 ** yp

            label.setText(f"X: {xp:.2f}, Y:{yp:.2f}")


class ZPKDialog(QDialog):
    def __init__(self, order: int = 0, parent=None):
        super().__init__(parent=parent)
        self.parent = parent

        self.setWindowTitle("配置零点-极点-增益")
        self.setFixedSize(300, 160)

        layout = QVBoxLayout()

        layout_1 = QHBoxLayout()

        label1 = QLabel("增益")
        self.gain_box = QLineEdit("")
        self.gain_box.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout_1.addWidget(label1)
        layout_1.addStretch(order - 1)
        layout_1.addWidget(self.gain_box, 2)
        layout_1.addStretch(order - 1)

        layout.addLayout(layout_1)

        layout_2 = QHBoxLayout()

        label2 = QLabel("零点")
        layout_2.addWidget(label2)

        self.zero_boxes = self.parent.tabpage2.set_transfer_function_boxes(
            layout_2, order - 1, True)

        layout.addLayout(layout_2)

        layout_3 = QHBoxLayout()

        label3 = QLabel("极点")
        layout_3.addWidget(label3)

        self.pole_boxes = self.parent.tabpage2.set_transfer_function_boxes(
            layout_3, order, False)

        layout.addLayout(layout_3)

        layout_4 = QHBoxLayout()

        btn_zpk = QPushButton("确定")
        btn_zpk.clicked.connect(self.btn_zpk_clicked)

        layout_4.addWidget(btn_zpk)

        layout.addLayout(layout_4)

        self.setStyleSheet(self.parent.styleSheet)

        self.setLayout(layout)

    def btn_zpk_clicked(self):

        if self.gain_box.text().strip() == "":
            self.gain_box.setText("1")
            self.gain = 1.0

        self.gain = float(self.gain_box.text())

        zeros = []

        for box in self.zero_boxes:
            box_text = box.text().strip()
            if box_text != "":
                zeros.append(float(box_text))

        poles = []

        for box in self.pole_boxes:
            box_text = box.text().strip()
            if box_text != "":
                poles.append(float(box_text))

        self.zeros = zeros
        self.poles = poles

        self.accept()
