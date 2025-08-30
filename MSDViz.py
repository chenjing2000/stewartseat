# opengl_widget.py

from OpenGL.GL.shaders import *
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QGroupBox, QLineEdit, QLabel,
                               QComboBox, QPushButton, QDialog,
                               QCheckBox, QFileDialog, QTabWidget)
import pyqtgraph as pg
from PySide6.QtGui import (QPalette, QColor, QDoubleValidator,
                           QIntValidator, QFontMetrics)
from PySide6.QtCore import QTimer, Signal
from OpenGL.GL import *

import numpy as np
import control as ct

from MSDModel import *
from MSDChart import *
from MSDTabs import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("质量-阻尼-弹簧控制系统")
        self.setFixedSize(420, 730)
        self.move(200, 100)

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
        main_layout = QVBoxLayout(central_widget)

        self.msdviz = MassSpringDamperGL()
        main_layout.addWidget(self.msdviz, 2)  # Take up 2/3 of the space

        self.tabpages = QTabWidget()
        self.tabpage1 = QWidget()
        self.tabpage2 = TabPage2(self)

        self.tabpages.addTab(self.tabpage1, "基础控制")
        self.tabpages.addTab(self.tabpage2, "传递函数")

        tabpage1_layout = QVBoxLayout(self.tabpage1)

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
                0.0, 1e9, 2)  # 范围 [0, 1e9]，小数点后2位
            validator.setNotation(QDoubleValidator.StandardNotation)
            value_box.setValidator(validator)

            self.params_layout.addWidget(value_box)

            self.msd_params_boxes.append(value_box)

        tabpage1_layout.addWidget(self.params_group)

        # PID 参数设置
        self.pidset_group = QGroupBox("PID Parameters")
        control_layout = QVBoxLayout(self.pidset_group)
        pidset_layout = QHBoxLayout()

        default_pid = {"kp:": 1.0, "ki:": 0.0, "kd:": 0.0}
        self.ctrl_boxes = []
        self.ctrl_comboboxes = []

        for id, (key, value) in enumerate(default_pid.items()):
            key_label = QLabel(key)
            key_label.setFixedWidth(50)
            pidset_layout.addWidget(key_label)

            value_box = QLineEdit(str(value))
            value_box.setMaximumWidth(100)
            value_box.setAlignment(Qt.AlignmentFlag.AlignRight)
            pidset_layout.addWidget(value_box)

            self.ctrl_boxes.append(value_box)

        control_layout.addLayout(pidset_layout)

        target_layout = QHBoxLayout()

        key_label = QLabel("对象")
        key_label.setFixedWidth(50)
        target_layout.addWidget(key_label, 1)

        combox = QComboBox()
        combox.addItems(["位移", "速度", "加速度", "无"])
        combox.setMinimumWidth(67)
        combox.setCurrentIndex(3)
        target_layout.addWidget(combox, 1)

        self.ctrl_comboboxes.append(combox)

        self.control_type = combox.currentIndex()

        key_label = QLabel("目标值")
        key_label.setFixedWidth(50)
        target_layout.addWidget(key_label, 1)

        value_box = QLineEdit("0.0")
        value_box.setMaximumWidth(100)
        value_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        target_layout.addWidget(value_box, 1)

        self.ctrl_boxes.append(value_box)

        self.target_value = float(value_box.text())

        ctrl_validator = QDoubleValidator(-1e6, 1e6, 2)
        ctrl_validator.setNotation(QDoubleValidator.StandardNotation)

        for box in self.ctrl_boxes:
            box.setValidator(ctrl_validator)

        btn_bodeplot = QPushButton("bode")
        btn_bodeplot.clicked.connect(self.btn_bodeplot_clicked)
        target_layout.addWidget(btn_bodeplot, 1)

        btn_nyquistplot = QPushButton("频谱分析")
        btn_nyquistplot.clicked.connect(self.btn_nyquistplot_clicked)
        target_layout.addWidget(btn_nyquistplot, 1)

        control_layout.addLayout(target_layout)

        tabpage1_layout.addWidget(self.pidset_group)

        main_layout.addWidget(self.tabpages)

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

        sim_validator = QDoubleValidator(0.0, 100.0, 2)
        sim_validator.setNotation(QDoubleValidator.StandardNotation)
        for box in self.sim_boxes:
            box.setValidator(sim_validator)

        self.time_stop = float(value_box.text())

        key_check = QCheckBox("导入")
        key_check.setFixedWidth(50)
        simulation_layout_2.addWidget(key_check)

        btn_input = QPushButton("选择文件")
        btn_input.clicked.connect(self.btn_input_clicked)
        simulation_layout_2.addWidget(btn_input)

        key_label = QLabel("忽略行数")
        key_label.setFixedWidth(50)
        simulation_layout_2.addWidget(key_label)

        value_box = QLineEdit("3")
        value_box.setValidator(QIntValidator(0, 30))
        value_box.setMaximumWidth(100)
        value_box.setAlignment(Qt.AlignmentFlag.AlignRight)
        simulation_layout_2.addWidget(value_box)

        self.sim_boxes.append(value_box)

        self.sim_checkboxes = []
        self.sim_checkboxes.append(key_check)

        simulation_layout_3 = QHBoxLayout()

        btn_simulate = QPushButton("simulate")
        btn_simulate.clicked.connect(self.btn_simulate_clicked)
        simulation_layout_3.addWidget(btn_simulate)

        btn_animate = QPushButton("animate")
        btn_animate.clicked.connect(self.btn_animate_clicked)
        simulation_layout_3.addWidget(btn_animate)

        self.simulation_layout.addLayout(simulation_layout_1)
        self.simulation_layout.addLayout(simulation_layout_2)
        self.simulation_layout.addLayout(simulation_layout_3)
        tabpage1_layout.addWidget(self.simulation_group)

        self.status_label = QLabel("Ready.")
        main_layout.addWidget(self.status_label)

        self._apply_styles()

        # 定时器
        self.timer = QTimer(self)
        self.timer.setInterval(5)  # 每 5 毫秒触发一次
        self.timer.timeout.connect(self.update_simulation)

        self.time_window = None
        self.bode_window = None
        self.nyquist_window = None

    def btn_show_info_clicked(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("基本信息")
        dialog.setFixedSize(300, 200)

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

        Gs = ct.TransferFunction([c, k], [m, c, k])

        kp = float(self.ctrl_boxes[0].text())
        ki = float(self.ctrl_boxes[1].text())
        kd = float(self.ctrl_boxes[2].text())

        self.control_type = self.ctrl_comboboxes[0].currentIndex()
        if self.control_type == 0:
            Gk = ct.TransferFunction([c, k, 0], [m, c + kd, k + kp, ki])
        elif self.control_type == 1:
            Gk = ct.TransferFunction([c, k], [m + kd, c + kp, k + ki])
        elif self.control_type == 2:
            Gk = ct.TransferFunction([c, k], [kd, m + kp, c + ki, k])
        else:
            Gk = ct.TransferFunction([c, k], [m, c, k])

        # 传递函数极点
        poles = ct.poles(Gs)

        poles_str = ", ".join(
            [f"{p.real:.2f}{'+' if p.imag>=0 else ''}{p.imag:.2f}j" for p in poles])
        label4 = QLabel(f"poles = [{poles_str}].")

        poles = ct.poles(Gk)

        poles_str = ", ".join(
            [f"{p.real:.2f}{'+' if p.imag>=0 else ''}{p.imag:.2f}j" for p in poles])
        label5 = QLabel(f"poles = [{poles_str}] for controlled.")

        layout.addWidget(label4)
        layout.addWidget(label5)

        dialog.setLayout(layout)
        dialog.exec()  # 阻塞式弹出

    def btn_input_clicked(self):

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择文件",
            "C:\\",  # 初始路径，可以写 "C:/"
            "文本文件 (*.txt);;CSV 文件 (*.csv)"
        )

        if file_name:
            # 限定显示长度（比如 label 宽度）
            metrics = QFontMetrics(self.status_label.font())
            file_name_short = metrics.elidedText(
                file_name, Qt.ElideMiddle, self.status_label.width())
            self.status_label.setText(file_name_short)

    def system_params_initilization(self):

        self.msd.m = float(self.msd_params_boxes[0].text())
        self.msd.c = float(self.msd_params_boxes[1].text())
        self.msd.k = float(self.msd_params_boxes[2].text())

        self.msd.x = 0.0
        self.msd.v = 0.0
        self.msd.a = 0.0

        self.pid.kp = float(self.ctrl_boxes[0].text())
        self.pid.ki = float(self.ctrl_boxes[1].text())
        self.pid.kd = float(self.ctrl_boxes[2].text())

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

        if self.sim_checkboxes[0].isChecked():
            self.status_label.setText("外部导入文件方法尚未完成。")

        self.excitation = signal

        self.control_type = self.ctrl_comboboxes[0].currentIndex()
        self.target_value = float(self.ctrl_boxes[3].text())

        self.x_series = np.zeros(nt, dtype=float)
        self.v_series = np.zeros(nt, dtype=float)
        self.a_series = np.zeros(nt, dtype=float)
        self.f_series = np.zeros(nt, dtype=float)

        if self.time_window is not None:
            self.time_window.close()

        self.time_window = TransientWindow()
        self.time_window.plot_curve_11.setData([], [])
        self.time_window.plot_curve_12.setData([], [])
        self.time_window.plot_curve_2.setData([], [])
        self.time_window.plot_curve_3.setData([], [])

        self.time_window.plot_widget_1.setXRange(0, self.time_stop)
        self.time_window.plot_widget_2.setXRange(0, self.time_stop)
        self.time_window.plot_widget_3.setXRange(0, self.time_stop)

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
            self.time_window.plot_curve_11.setData(
                self.time[:-1], self.x_series[:-1])
            self.time_window.plot_curve_12.setData(
                self.time[:-1], self.excitation[:-1])
            self.time_window.plot_curve_2.setData(
                self.time[:-1], self.a_series[:-1])
            self.time_window.plot_curve_3.setData(
                self.time[:-1], self.f_series[:-1])

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
            self.time_window.plot_curve_11.setData(
                self.time[:id+1], self.x_series[:id+1])
            self.time_window.plot_curve_12.setData(
                self.time[:id+1], self.excitation[:id+1])
            self.time_window.plot_curve_2.setData(
                self.time[:id+1], self.a_series[:id+1])
            self.time_window.plot_curve_3.setData(
                self.time[:id+1], self.f_series[:id+1])

        # 更新OpenGL绘图
        self.msdviz.mass_motion = self.msd.x
        self.msdviz.base_motion = self.excitation[self.step_now + 1]

        self.msdviz.update()

        # 更新时间步
        self.step_now += 1
        self.time_now += self.dt

    def btn_bodeplot_clicked(self):

        m = float(self.msd_params_boxes[0].text())
        c = float(self.msd_params_boxes[1].text())
        k = float(self.msd_params_boxes[2].text())

        Gs = ct.TransferFunction([c, k], [m, c, k])

        mag, phase, omega = ct.frequency_response(Gs)

        if self.bode_window is not None:
            self.bode_window.close()

        self.bode_window = BodeWindow()
        self.bode_window.plot_curve_11.setData(omega, 20*np.log10(mag))
        self.bode_window.plot_curve_12.setData(omega, 20*np.log10(mag * omega))
        self.bode_window.plot_curve_13.setData(
            omega, 20*np.log10(mag * omega ** 2))

        self.bode_window.plot_curve_21.setData(omega, phase)
        self.bode_window.plot_curve_22.setData(omega, phase + np.pi/2)
        self.bode_window.plot_curve_23.setData(omega, phase + np.pi)

    def closeEvent(self, event):
        # 主窗口关闭前，先关闭子窗口

        windows = [self.time_window,
                   self.bode_window,
                   self.frequency_window,
                   self.tabpage2.transient_window,
                   self.tabpage2.frequency_window]

        for window in windows:
            if window is not None:
                window.close()

        super().closeEvent(event)

    def btn_nyquistplot_clicked(self):

        m = float(self.msd_params_boxes[0].text())
        c = float(self.msd_params_boxes[1].text())
        k = float(self.msd_params_boxes[2].text())

        kp = float(self.ctrl_boxes[0].text())
        ki = float(self.ctrl_boxes[1].text())
        kd = float(self.ctrl_boxes[2].text())

        Gs = ct.TransferFunction([c, k], [m, c, k])

        # 绘图
        self.frequency_window = FrequencyWindow()

        plot_widget_i = self.frequency_window.plot_widget_1
        plot_widget_j = self.frequency_window.plot_widget_2

        mag, phase, omega = ct.frequency_response(Gs)

        plot_widget_i.plot(omega, 20*np.log10(mag), pen=pg.mkPen(
            color='r', width=2), name='uncontrol')

        realpart = mag * np.cos(phase)
        imagpart = mag * np.sin(phase)

        xdata = np.concatenate([+realpart[::-1], realpart])
        ydata = np.concatenate([-imagpart[::-1], imagpart])

        plot_widget_j.plot(xdata, ydata, pen=pg.mkPen(
            color='r', width=2), name='uncontrol')

        self.control_type = self.ctrl_comboboxes[0].currentIndex()
        if self.control_type == 0:
            Gk = ct.TransferFunction([c, k, 0], [m, c + kd, k + kp, ki])
        elif self.control_type == 1:
            Gk = ct.TransferFunction([c, k], [m + kd, c + kp, k + ki])
        elif self.control_type == 2:
            Gk = ct.TransferFunction([c, k], [kd, m + kp, c + ki, k])
        else:
            Gk = ct.TransferFunction([c, k], [m, c, k])

        mag, phase, omega = ct.frequency_response(Gk)

        plot_widget_i.plot(omega, 20*np.log10(mag), pen=pg.mkPen(
            color='g', width=2), name='control')

        realpart = mag * np.cos(phase)
        imagpart = mag * np.sin(phase)

        xdata = np.concatenate([+realpart[::-1], realpart])
        ydata = np.concatenate([-imagpart[::-1], imagpart])

        plot_widget_j.plot(xdata, ydata, pen=pg.mkPen(
            color='g', width=2), name="control")

    def _apply_styles(self):
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        self.setPalette(palette)
        self.styleSheet = """
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
            QPushButton:disabled {
                background-color: #E3E3E3;  /* 灰色背景 */
                color: #A6A6A6;             /* 浅灰文本 */
                border: 1px solid #D0D0D0;  /* 灰色边框 */
            }
            QLineEdit, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: none;
                border-radius: 3px;
            }
            QLabel {
                padding: 2px 0;
            }
            /* 整个 TabWidget 背景 */
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 0px;
            }
            /* Tab bar 区域 */
            QTabBar::tab {
                border: 1px soild #d0d0d0;
                padding: 6px 10px;
                margin: 2px;
                color: #333;
                font-size: 14px;
                border-radius: 6px;
            }
            /* 被选中的 Tab */
            QTabBar::tab:selected {
                background: #0078d4;   /* Windows 11 蓝 */
                color: white;
            }

            /* 未选中时 */
            QTabBar::tab:!selected {
                background: transparent;
                color: #444;
            }
        """
        self.setStyleSheet(self.styleSheet)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
