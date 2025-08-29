
from OpenGL.GL.shaders import *
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QGroupBox, QLineEdit, QLabel,
                               QComboBox, QPushButton, QDialog)
import pyqtgraph as pg
from OpenGL.GL import *


class TimeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transient Response")
        self.setFixedSize(400, 700)
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
        layout.addWidget(self.plot_widget_1, 1)

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
        layout.addWidget(self.plot_widget_2, 1)

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
        layout.addWidget(self.plot_widget_3, 1)


class BodeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bode Chart")
        self.setFixedSize(400, 700)
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
        layout.addWidget(self.plot_widget_1, 1)

        self.plot_widget_2 = pg.PlotWidget()
        self.plot_widget_2.setLogMode(x=True, y=False)
        self.plot_widget_2.setTitle(
            "Phase vs. Frequency", color='black', size="10pt")
        self.plot_widget_2.setLabel('left', 'Amplitude (rad)', color='black')
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
        layout.addWidget(self.plot_widget_2, 1)

        self.plot_widget_3 = pg.PlotWidget()
        self.plot_widget_3.setLogMode(x=True, y=False)
        self.plot_widget_3.setTitle(
            "unControl vs. Control", color='black', size="10pt")
        self.plot_widget_3.setLabel(
            'left', 'Amplitude of Displacement (dB)', color='black')
        self.plot_widget_3.setLabel('bottom', 'Frequency (Hz)', color='black')
        self.plot_widget_3.setBackground('white')
        self.plot_widget_3.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_3.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_widget_3.addLegend(labelTextColor='blue')
        self.plot_curve_31 = self.plot_widget_3.plot(
            [], [], pen=pg.mkPen(color='r', width=2), name='uncontrol')
        self.plot_curve_32 = self.plot_widget_3.plot(
            [], [], pen=pg.mkPen(color='g', width=2), name='control')
        layout.addWidget(self.plot_widget_3, 1)


class NyquistWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nyquist Chart")
        self.setFixedSize(400, 700)
        self.move(630, 100)
        self.show()

        layout = QVBoxLayout(self)
        # Plotting widget
        self.plot_widget_1 = pg.PlotWidget()
        self.plot_widget_1.setLogMode(x=False, y=False)
        self.plot_widget_1.setTitle(
            "Amplitude vs. Frequency", color='black', size="10pt")
        self.plot_widget_1.setLabel('left', 'Im(G(jω))', color='black')
        self.plot_widget_1.setLabel('bottom', 'Re(G(jω))', color='black')
        self.plot_widget_1.setBackground('white')
        self.plot_widget_1.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_1.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_widget_1.addLegend(labelTextColor='blue')
        self.plot_curve_11 = self.plot_widget_1.plot(
            [], [], pen=None, symbol="+", name='critical point')
        self.plot_curve_12 = self.plot_widget_1.plot(
            [], [], pen=pg.mkPen(color='r', width=2), name='uncontrol')
        self.plot_curve_13 = self.plot_widget_1.plot(
            [], [], pen=pg.mkPen(color='g', width=2), name='control')
        layout.addWidget(self.plot_widget_1, 1)
        # 添加横轴与纵轴
        self.plot_widget_1.addLine(x=0, pen=pg.mkPen(color='k', width=1))
        self.plot_widget_1.addLine(y=0, pen=pg.mkPen(color='k', width=1))
