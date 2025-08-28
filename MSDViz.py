# opengl_widget.py

from OpenGL.GL.shaders import *
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QGroupBox, QLineEdit, QLabel,
                               QComboBox, QPushButton, QDialog)
import pyqtgraph as pg
from PySide6.QtGui import QPalette, QColor, QDoubleValidator
from PySide6.QtCore import QTimer, Signal
from OpenGL.GL import *

import numpy as np
import control as ct

from MSDModel import MSDPlant, PIDController, MassSpringDamperGL


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("质量-阻尼-弹簧仿真系统")
        self.setFixedSize(800, 700)
        self.move(100, 100)

        # Main components
        self.msd = MSDPlant(mass=10.0, damping=1.0, stiffness=10.0)
        self.pid = PIDController(kp=100.0, ki=1.0, kd=10.0)
        self.excitation = []
        self.x_series = []
        self.v_series = []
        self.a_series = []
        self.f_series = []

        self.time_stop = 5  # Simulation time period
        self.dt = 0.001
        self.msd.dt = self.dt
        self.pid.dt = self.dt

        self.time = []

        self.step_now = 0
        self.time_now = 0.0

        # --- UI Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: OpenGL widget
        layout_left = QVBoxLayout()
        self.msdviz = MassSpringDamperGL()
        layout_left.addWidget(self.msdviz, 2)  # Take up 2/3 of the space

        # 基本参数设置
        self.params_group = QGroupBox("MSD Parameters")
        self.params_layout = QHBoxLayout(self.params_group)

        default_params = {"质量(kg)": 10, "阻尼(N.s/m)": 40, "刚度": 1000}
        self.msd_params_boxes = []

        for id, (key, value) in enumerate(default_params.items()):
            if id < 2:
                key_label = QLabel(key)
                key_label.setFixedWidth(50)
                self.params_layout.addWidget(key_label)
            else:
                btn_show_info = QPushButton(key)
                btn_show_info.setFixedWidth(50)
                self.params_layout.addWidget(btn_show_info)
                btn_show_info.clicked.connect(self.btn_show_info_clicked)

            value_box = QLineEdit(str(value))
            value_box.setMaximumWidth(100)
            value_box.setAlignment(Qt.AlignmentFlag.AlignRight)

            # 设置只能输入正数（整数或浮点数）
            validator = QDoubleValidator(
                0.0, 1e9, 3, self)  # 范围 [0, 1e9]，小数点后6位
            validator.setNotation(QDoubleValidator.StandardNotation)
            value_box.setValidator(validator)

            self.params_layout.addWidget(value_box)

            self.msd_params_boxes.append(value_box)

        layout_left.addWidget(self.params_group)

        # PID 参数设置
        self.pidset_group = QGroupBox("PID Parameters")
        self.pidset_layout = QHBoxLayout(self.pidset_group)

        default_pid = {"kp:": 1.0, "ki:": 0.0, "kd:": 0.0}
        self.pid_params_boxes = []

        for id, (key, value) in enumerate(default_pid.items()):
            key_label = QLabel(key)
            key_label.setFixedWidth(50)
            self.pidset_layout.addWidget(key_label)

            value_box = QLineEdit(str(value))
            value_box.setMaximumWidth(100)
            value_box.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.pidset_layout.addWidget(value_box)

            self.pid_params_boxes.append(value_box)

        layout_left.addWidget(self.pidset_group)

        main_layout.addLayout(layout_left)

        layout_right = QVBoxLayout()

        # 仿真设置
        self.simulation_group = QGroupBox("仿真设置")
        self.simulation_layout = QVBoxLayout(self.simulation_group)
        simulation_layout_1 = QHBoxLayout()

        self.sim_comboboxes = []
        self.sim_boxes = []

        key_label = QLabel("典型信号")
        key_label.setFixedWidth(50)
        simulation_layout_1.addWidget(key_label)

        combox = QComboBox()
        combox.addItems(["脉冲", "阶跃", "正弦"])
        combox.setMinimumWidth(67)
        simulation_layout_1.addWidget(combox)

        self.sim_comboboxes.append(combox)

        key_label = QLabel("幅值(m)")
        key_label.setFixedWidth(50)
        simulation_layout_1.addWidget(key_label)

        value_box = QLineEdit("0.1")
        value_box.setMaximumWidth(100)
        value_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        simulation_layout_1.addWidget(value_box)

        self.sim_boxes.append(value_box)

        key_label = QLabel("频率(Hz)")
        key_label.setFixedWidth(50)
        simulation_layout_1.addWidget(key_label)

        value_box = QLineEdit("2.0")
        value_box.setMaximumWidth(100)
        value_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        simulation_layout_1.addWidget(value_box)

        self.sim_boxes.append(value_box)

        simulation_layout_2 = QHBoxLayout()

        key_label = QLabel("时长(s)")
        key_label.setFixedWidth(50)
        simulation_layout_2.addWidget(key_label)

        value_box = QLineEdit("5.0")
        value_box.setMaximumWidth(100)
        value_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        simulation_layout_2.addWidget(value_box)

        self.sim_boxes.append(value_box)

        self.time_stop = float(value_box.text())

        key_label = QLabel("对象")
        key_label.setFixedWidth(50)
        simulation_layout_2.addWidget(key_label)

        combox = QComboBox()
        combox.addItems(["位移", "速度", "加速度", "无"])
        combox.setMinimumWidth(67)
        combox.setCurrentIndex(3)
        simulation_layout_2.addWidget(combox)

        self.sim_comboboxes.append(combox)

        self.control_type = combox.currentIndex()

        key_label = QLabel("目标值")
        key_label.setFixedWidth(50)
        simulation_layout_2.addWidget(key_label)

        value_box = QLineEdit("0.0")
        value_box.setMaximumWidth(100)
        value_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        simulation_layout_2.addWidget(value_box)

        self.sim_boxes.append(value_box)

        self.target_value = float(value_box.text())

        simulation_layout_3 = QHBoxLayout()

        btn_bodeplot = QPushButton("bode chart")
        btn_bodeplot.clicked.connect(self.btn_bodeplot_clicked)
        simulation_layout_3.addWidget(btn_bodeplot)

        btn_simulate = QPushButton("simulate")
        btn_simulate.clicked.connect(self.btn_simulate_clicked)
        simulation_layout_3.addWidget(btn_simulate)

        btn_animate = QPushButton("animate")
        btn_animate.clicked.connect(self.btn_animate_clicked)
        simulation_layout_3.addWidget(btn_animate)

        self.simulation_layout.addLayout(simulation_layout_1)
        self.simulation_layout.addLayout(simulation_layout_2)
        self.simulation_layout.addLayout(simulation_layout_3)
        layout_left.addWidget(self.simulation_group)

        self.status_label = QLabel("Ready.")
        layout_left.addWidget(self.status_label)

        # Plotting widget
        self.plot_widget_1 = pg.PlotWidget()
        self.plot_widget_1.setTitle("Position vs. Time", color='black')
        self.plot_widget_1.setLabel('left', 'Position (m)', color='black')
        self.plot_widget_1.setLabel('bottom', 'Time (s)', color='black')
        self.plot_widget_1.setBackground('white')
        self.plot_widget_1.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_1.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_curve_11 = self.plot_widget_1.plot(
            [], [], pen=pg.mkPen(color='b', width=2), name='excitation')
        self.plot_curve_12 = self.plot_widget_1.plot(
            [], [], pen=pg.mkPen(color='r', width=2), name='displacement')
        layout_right.addWidget(self.plot_widget_1, 1)

        self.plot_widget_2 = pg.PlotWidget()
        self.plot_widget_2.setTitle("Acceleration vs. Time", color='black')
        self.plot_widget_2.setLabel(
            'left', 'Acceleration (m/s^2)', color='black')
        self.plot_widget_2.setLabel('bottom', 'Time (s)', color='black')
        self.plot_widget_2.setBackground('white')
        self.plot_widget_2.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_2.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_curve_2 = self.plot_widget_2.plot(
            [], [], pen=pg.mkPen(color='b', width=2))
        layout_right.addWidget(self.plot_widget_2, 1)

        self.plot_widget_3 = pg.PlotWidget()
        self.plot_widget_3.setTitle("Actuator Force vs. Time", color='black')
        self.plot_widget_3.setLabel(
            'left', 'Actuator Force (N)', color='black')
        self.plot_widget_3.setLabel('bottom', 'Time (s)', color='black')
        self.plot_widget_3.setBackground('white')
        self.plot_widget_3.getAxis('left').setPen('black')  # 设置轴线颜色
        self.plot_widget_3.getAxis('bottom').setPen('black')  # 设置轴线颜色
        self.plot_curve_3 = self.plot_widget_3.plot(
            [], [], pen=pg.mkPen(color='b', width=2))
        layout_right.addWidget(self.plot_widget_3, 1)

        main_layout.addLayout(layout_right)

        self._apply_styles()

        # 定时器
        self.timer = QTimer(self)
        self.timer.setInterval(5)  # 每 5 毫秒触发一次
        self.timer.timeout.connect(self.update_simulation)

    def btn_show_info_clicked(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("基本信息")
        dialog.setFixedSize(300, 120)

        layout = QVBoxLayout()

        m = float(self.msd_params_boxes[0].text())
        c = float(self.msd_params_boxes[1].text())
        k = float(self.msd_params_boxes[2].text())

        wn = np.sqrt(k/m)
        zt = c/2/np.sqrt(m*k)
        wd = np.sqrt(1 - zt**2) * wn

        label0 = QLabel(
            f"wn = {wn/(2*np.pi):.2f} Hz, zeta = {zt:.2f}, wd = {wd/(2*np.pi):.2f} Hz.", dialog)

        tr = (np.pi - np.arccos(zt)) / wd
        tp = np.pi / wd
        ts = 4 / (zt*wn)

        Mos = np.exp(- zt * np.pi / np.sqrt(1 - zt**2))

        label1 = QLabel(
            f"tr = {tr:.2f} s, tp = {tp:.2f} s, ts = {ts:.2f} s, Mos = {Mos*100:.1f}%.")

        zs = 1 - 2 * zt**2
        wp = wn * np.sqrt(zs)    # resonant frequency
        Mp = 1/(2 * zt * np.sqrt(1 - zt**2))  # resonant peak

        wb = wn * np.sqrt(zs + np.sqrt(zs**2 + 1))  # bandwidth

        wc = wn * np.sqrt(np.sqrt(1 + 4 * zt**4) - 2 *
                          zt**2)  # crossover frequency

        PM = np.arctan(2 * zt/np.sqrt(- 2 * zt**2 + np.sqrt(1 + 4 * zt**4)))

        label2 = QLabel(
            f"wp = {wp/(2*np.pi):.2f} Hz, Mp = {Mp:.2f}, ")

        label3 = QLabel(
            f"wb = {wb/(2*np.pi):.2f} Hz, wc = {wc/(2*np.pi):.2f} Hz, PM = {PM:.2f}.")

        layout.addWidget(label0)
        layout.addWidget(label1)
        layout.addWidget(label2)
        layout.addWidget(label3)

        self.control_type = self.sim_comboboxes[1].currentIndex()

        if self.control_type == 0:
            pass
        else:
            pass

        dialog.setLayout(layout)
        dialog.exec()  # 阻塞式弹出

    def btn_bodeplot_clicked(self):

        self.msd.m = float(self.msd_params_boxes[0].text())
        self.msd.c = float(self.msd_params_boxes[1].text())
        self.msd.k = float(self.msd_params_boxes[2].text())

        self.pid.kp = float(self.pid_params_boxes[0].text())
        self.pid.ki = float(self.pid_params_boxes[1].text())
        self.pid.kd = float(self.pid_params_boxes[2].text())

    def system_params_initilization(self):

        self.msd.m = float(self.msd_params_boxes[0].text())
        self.msd.c = float(self.msd_params_boxes[1].text())
        self.msd.k = float(self.msd_params_boxes[2].text())

        self.msd.x = 0.0
        self.msd.v = 0.0
        self.msd.a = 0.0

        self.pid.kp = float(self.pid_params_boxes[0].text())
        self.pid.ki = float(self.pid_params_boxes[1].text())
        self.pid.kd = float(self.pid_params_boxes[2].text())

        self.time_stop = float(self.sim_boxes[2].text())
        self.step_now = 0
        self.time_now = 0.0

        dt = self.dt
        self.time = np.arange(0, self.time_stop, dt)
        nt = len(self.time)

        # 获取外部激励信号类型
        signal_type = self.sim_comboboxes[0].currentIndex()

        signal = np.zeros(nt, dtype=float)
        amplitude = float(self.sim_boxes[0].text())

        if signal_type == 1:  # step
            step_start = int(0.1 / dt)
            signal[step_start:] = amplitude

        elif signal_type == 2:  # sine
            omega = 2 * np.pi * float(self.sim_boxes[1].text())
            signal = amplitude * np.sin(omega * self.time)

        else:  # impulse
            impulse_time_period = 0.1
            # 幅值取小点，否则绘图区装不下
            amplitude = amplitude / impulse_time_period

            step_start = int(0.1 / dt)
            step_stop = int((0.1 + impulse_time_period) / dt)
            signal[step_start:step_stop] = amplitude

        self.excitation = signal

        self.control_type = self.sim_comboboxes[1].currentIndex()
        self.target_value = float(self.sim_boxes[3].text())

        self.x_series = np.zeros(nt, dtype=float)
        self.v_series = np.zeros(nt, dtype=float)
        self.a_series = np.zeros(nt, dtype=float)
        self.f_series = np.zeros(nt, dtype=float)

        self.plot_curve_11.setData([], [])
        self.plot_curve_12.setData([], [])
        self.plot_curve_2.setData([], [])
        self.plot_curve_3.setData([], [])

        self.plot_widget_1.setXRange(0, self.time_stop)
        self.plot_widget_2.setXRange(0, self.time_stop)
        self.plot_widget_3.setXRange(0, self.time_stop)

    def btn_simulate_clicked(self):

        self.status_label.setText("Simulation starts.")

        # 停止并重新启动定时器，开始动画
        if self.timer.isActive():
            self.timer.stop()

        self.system_params_initilization()

        for id in range(len(self.time)-1):
            if self.control_type == 0:
                error = self.target_value - self.msd.x
            elif self.control_type == 1:
                error = self.target_value - self.msd.v
            elif self.control_type == 2:
                error = self.target_value - self.msd.a
            else:
                error = 0.0

            force = self.pid.calculate_force(error)

            self.msd.update(self.excitation[id:id+2], force)

            self.x_series[id] = self.msd.x
            self.v_series[id] = self.msd.v
            self.a_series[id] = self.msd.a
            self.f_series[id] = force

        if len(self.x_series) > 0:
            self.plot_curve_11.setData(self.time[:-1], self.excitation[:-1])
            self.plot_curve_12.setData(self.time[:-1], self.x_series[:-1])
            self.plot_curve_2.setData(self.time[:-1], self.a_series[:-1])
            self.plot_curve_3.setData(self.time[:-1], self.f_series[:-1])

        self.status_label.setText("Simulation stops.")

    def btn_animate_clicked(self):
        """
        当点击 'animate' 按钮时调用，用于启动动画仿真。
        """
        self.status_label.setText("Animation starts.")

        self.system_params_initilization()

        # 停止并重新启动定时器，开始动画
        if self.timer.isActive():
            self.timer.stop()
        self.timer.start()

    def update_simulation(self):
        """
        定时器触发的函数，用于驱动动画和图形更新。
        """
        # 检查是否达到仿真时长，如果达到则停止定时器
        if self.step_now >= len(self.time) - 2:
            self.timer.stop()
            self.status_label.setText("Animation stops.")
            return

        # 计算PID控制力

        if self.control_type == 0:
            error = self.target_value - self.msd.x
        elif self.control_type == 1:
            error = self.target_value - self.msd.v
        elif self.control_type == 2:
            error = self.target_value - self.msd.a
        else:
            error = 0.0

        control_force = self.pid.calculate_force(error)

        # 更新物理模型
        self.msd.update(
            self.excitation[self.step_now:self.step_now+2], control_force)

        id = self.step_now
        self.x_series[id] = self.msd.x
        self.v_series[id] = self.msd.v
        self.a_series[id] = self.msd.a
        self.f_series[id] = control_force

        if id % 20 == 0:
            # 更新Pyqtgraph曲线（使用整个数据历史）
            self.plot_curve_11.setData(
                self.time[:id+1], self.excitation[:id+1])
            self.plot_curve_12.setData(self.time[:id+1], self.x_series[:id+1])
            self.plot_curve_2.setData(self.time[:id+1], self.a_series[:id+1])
            self.plot_curve_3.setData(self.time[:id+1], self.f_series[:id+1])

        # 更新OpenGL绘图
        self.msdviz.update_positions(
            self.msd.x, self.excitation[self.step_now + 1])
        self.msdviz.update()

        # 更新时间步
        self.step_now += 1
        self.time_now += self.dt

    def _apply_styles(self):
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        self.setPalette(palette)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 1ex; /* leave space at the top for the title */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left; /* position at the top center */
                padding: 0 3px;
                background-color: #f0f0f0;
            }
            QPushButton {
                padding: 8px;
                border-radius: 5px;
                background-color: #007bff;
                color: white;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QLineEdit, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: none;
                border-radius: 3px;
            }
            QLabel {
                padding: 2px 0;
            }
        """)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
