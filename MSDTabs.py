
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QGroupBox, QLineEdit, QLabel,
                               QComboBox, QPushButton, QDialog,
                               QCheckBox, QFileDialog, QSpacerItem, QSizePolicy)
from PySide6.QtGui import (QDoubleValidator, QIntValidator, QFontMetrics)

import os
import numpy as np
import control as ct
import csv

from MSDChart import *
from MSDModel import DataGroup, MSDPlant, PIDController, MassSpringDamperGL


class TabPage1(QWidget):
    status_changed = Signal(str)
    data_updated = Signal()

    def __init__(self, data: DataGroup,
                 msd: MSDPlant,
                 pid: PIDController,
                 msdviz: MassSpringDamperGL,
                 parent=None):

        super().__init__()

        self.data = data
        self.msd = msd
        self.pid = pid
        self.msdviz = msdviz

        self._setup_ui()

        self.transient_window = None
        self.bode_window = None
        self.frequency_window = None

        self.excitation_file = None

        # 定时器
        self.timer = QTimer(self)
        self.timer.setInterval(10)  # 每 10 毫秒触发一次
        self.timer.timeout.connect(self.update_simulation)

    def _setup_ui(self):

        tabpage1_layout = QVBoxLayout(self)

        # 基本参数设置
        self.params_group = QGroupBox("MSD Parameters")
        self.params_layout = QHBoxLayout(self.params_group)

        default_params = {"质量(kg)": 10, "阻尼(N.s/m)": 40, "刚度": 1000}
        self.msd_boxes = []

        for idx, (key, value) in enumerate(default_params.items()):
            if idx < 2:
                key_label = QLabel(key)
                key_label.setFixedWidth(50)
                self.params_layout.addWidget(key_label)
            else:
                btn_show_infos = QPushButton(key)
                btn_show_infos.setFixedWidth(50)
                self.params_layout.addWidget(btn_show_infos)
                btn_show_infos.clicked.connect(self.btn_show_infos_clicked)

            ebox = self.create_number_box(str(value), 1)

            self.msd_boxes.append(ebox)
            self.params_layout.addWidget(ebox)

        tabpage1_layout.addWidget(self.params_group)

        # PID 参数设置
        self.pidset_group = QGroupBox("PID Parameters")
        pidset_layout = QVBoxLayout(self.pidset_group)
        pidset_layout_1 = QHBoxLayout()

        default_pid = {"kp:": 1.0, "ki:": 0.0, "kd:": 0.0}
        self.pid_boxes = []
        self.pid_comboboxes = []

        for _, (key, value) in enumerate(default_pid.items()):
            key_label = QLabel(key)
            key_label.setFixedWidth(50)
            pidset_layout_1.addWidget(key_label)

            ebox = self.create_number_box(str(value), 0)

            self.pid_boxes.append(ebox)
            pidset_layout_1.addWidget(ebox)

        pidset_layout.addLayout(pidset_layout_1)

        pidset_layout_2 = QHBoxLayout()

        key_label = QLabel("对象")
        key_label.setFixedWidth(50)
        pidset_layout_2.addWidget(key_label)

        combox = QComboBox()
        combox.addItems(["位移", "速度", "加速度", "无"])
        combox.setMinimumWidth(60)
        combox.setCurrentIndex(3)

        self.pid_comboboxes.append(combox)
        pidset_layout_2.addWidget(combox, 10)

        key_label = QLabel("目标值")
        key_label.setFixedWidth(50)
        pidset_layout_2.addWidget(key_label, 10)

        ebox = self.create_number_box("0.0", 0)

        self.pid_boxes.append(ebox)
        pidset_layout_2.addWidget(ebox, 10)

        btn_bodeplot = QPushButton("bode")
        btn_bodeplot.clicked.connect(self.btn_bodeplot_clicked)
        pidset_layout_2.addWidget(btn_bodeplot, 9)

        btn_frequency_analyses = QPushButton("频谱")
        btn_frequency_analyses.clicked.connect(
            self.btn_frequency_analyses_clicked)
        pidset_layout_2.addWidget(btn_frequency_analyses, 9)

        pidset_layout.addLayout(pidset_layout_2)

        tabpage1_layout.addWidget(self.pidset_group)

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
        combox.setMinimumWidth(60)

        self.sim_comboboxes.append(combox)
        simulation_layout_1.addWidget(combox)

        key_label = QLabel("幅值(m)")
        key_label.setFixedWidth(50)
        simulation_layout_1.addWidget(key_label)

        ebox = self.create_number_box("0.1", 1)

        self.sim_boxes.append(ebox)
        simulation_layout_1.addWidget(ebox)

        key_label = QLabel("频率(Hz)")
        key_label.setFixedWidth(50)
        simulation_layout_1.addWidget(key_label)

        ebox = self.create_number_box("2.0", 1)

        self.sim_boxes.append(ebox)
        simulation_layout_1.addWidget(ebox)

        simulation_layout_2 = QHBoxLayout()

        key_label = QLabel("时长(s)")
        key_label.setFixedWidth(50)
        simulation_layout_2.addWidget(key_label)

        ebox = self.create_number_box("5.0", 1)

        self.sim_boxes.append(ebox)
        simulation_layout_2.addWidget(ebox)

        key_check = QCheckBox("导入")
        key_check.setFixedWidth(50)
        simulation_layout_2.addWidget(key_check)

        self.sim_checkboxes = []
        self.sim_checkboxes.append(key_check)

        btn_import_excitation = QPushButton("选择文件")
        btn_import_excitation.clicked.connect(
            self.btn_import_excitation_clicked)
        simulation_layout_2.addWidget(btn_import_excitation)

        key_label = QLabel("忽略行数")
        key_label.setFixedWidth(50)
        simulation_layout_2.addWidget(key_label)

        ebox = QLineEdit("3")
        ebox.setValidator(QIntValidator(0, 30))
        ebox.textEdited.connect(
            lambda: self.data_checking(ebox, False))

        ebox.setMaximumWidth(100)
        ebox.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.sim_boxes.append(ebox)
        simulation_layout_2.addWidget(ebox)

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

    def create_number_box(self, value: str, validator_type: int):
        # validator_type > 0: positive double validator; else: double validator
        #

        lower_bound = 0.0 if validator_type > 0 else -1e8
        validator = QDoubleValidator(lower_bound, 1e8, 2)
        validator.setNotation(QDoubleValidator.StandardNotation)

        ebox = QLineEdit(value)
        ebox.setValidator(validator)
        ebox.textChanged.connect(lambda: self.data_checking(ebox))

        ebox.setMaximumWidth(100)
        ebox.setAlignment(Qt.AlignmentFlag.AlignRight)

        return ebox

    def data_checking(self, line_edit: QLineEdit, is_float: bool = True):
        #
        text = line_edit.text().strip()
        if not text:  # 空字符串单独处理
            line_edit.setProperty("valid", "false")
            line_edit.style().polish(line_edit)
            self.status_changed.emit("Error: Data is empty.")
            return False

        try:
            _ = float(text) if is_float else int(text)
            line_edit.setProperty("valid", "true")
            line_edit.style().polish(line_edit)
            self.status_changed.emit("Message: Data is valid.")
            return True
        except (ValueError, TypeError):
            line_edit.setProperty("valid", "false")
            line_edit.style().polish(line_edit)
            self.status_changed.emit("Error: Data is invalid.")
            return False

    def collect_data(self):

        try:
            self.data.mass = float(self.msd_boxes[0].text())
            self.data.damping = float(self.msd_boxes[1].text())
            self.data.stiffness = float(self.msd_boxes[2].text())

            self.data.kp = float(self.pid_boxes[0].text())
            self.data.ki = float(self.pid_boxes[1].text())
            self.data.kd = float(self.pid_boxes[2].text())

            self.data.control_type = \
                self.pid_comboboxes[0].currentIndex()
            self.data.target_value = \
                float(self.pid_boxes[3].text())

            self.data.signal_type = \
                self.sim_comboboxes[0].currentIndex()
            self.data.amplitude = float(self.sim_boxes[0].text())
            self.data.frequency = float(self.sim_boxes[1].text())
            self.data.time_stop = float(self.sim_boxes[2].text())

            # external excitation
            self.data.is_importing = self.sim_checkboxes[0].isChecked()
            self.data.skiprows = int(self.sim_boxes[3].text())

            self.data_updated.emit()

        except (ValueError) as e:
            self.status_changed.emit(f"Error: Some boxes are empty.")
            return False

        return True

    def get_windows(self):

        windows_list = [self.transient_window, self.frequency_window,
                        self.bode_window]
        return windows_list

    def btn_show_infos_clicked(self):

        if not self.collect_data():  # 更新数据类
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("基本信息")
        dialog.setFixedWidth(300)
        dialog.setMaximumHeight(300)

        layout = QVBoxLayout()

        m, c, k = self.msd.m, self.msd.c, self.msd.k

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

        # crossover frequency
        wc = wn * np.sqrt(np.sqrt(1 + 4 * zt**4) - 2 * zt**2)

        PM = np.arctan(2 * zt/np.sqrt(- 2 * zt**2 + np.sqrt(1 + 4 * zt**4)))

        label2 = QLabel(
            f"wp = {wp/(2*np.pi):.2f} Hz, Mp = {Mp:.2f}, ")

        label3 = QLabel(
            f"wb = {wb/(2*np.pi):.2f} Hz, wc = {wc/(2*np.pi):.2f} Hz, PM = {PM:.2f}.")

        layout.addWidget(label0)
        layout.addWidget(label1)
        layout.addWidget(label2)
        layout.addWidget(label3)

        result = self.get_system_transfer_functions()

        if not result:
            return
        else:
            Gs, Gb = result

        # 传递函数极点
        poles = ct.poles(Gs)

        poles_str = ", ".join(
            [f"{p.real:.2f}{'+' if p.imag>=0 else ''}{p.imag:.2f}j" for p in poles])
        label4 = QLabel(f"open loop poles : [{poles_str}].")
        label4.setWordWrap(True)

        layout.addWidget(label4)

        poles = ct.poles(Gb)

        poles_str = ", ".join(
            [f"{p.real:.2f}{'+' if p.imag>=0 else ''}{p.imag:.2f}j" for p in poles])
        label5 = QLabel(f"closed loop poles : [{poles_str}]")
        label5.setWordWrap(True)

        layout.addWidget(label5)

        dialog.setLayout(layout)
        dialog.exec()  # 阻塞式弹出

    def get_system_transfer_functions(self):

        m = self.msd.m
        c = self.msd.c
        k = self.msd.k

        Gs = ct.tf([c, k], [m, c, k])

        control_type = self.data.control_type

        if control_type < 3:
            kp = self.pid.kp
            ki = self.pid.ki
            kd = self.pid.kd

            Hs = ct.tf([1, 0], [kd, kp, ki])

            if control_type == 1:
                Hx = Hs * ct.tf([1, 0], 1)
            elif control_type == 2:
                Hx = Hs * ct.tf([1, 0, 0], 1)
            else:
                Hx = Hs

            Gf = ct.tf(1, [c, k])  # 力传递函数
            Gb = Hx * Gs / (Hx + Gs * Gf)  # 闭环系统传递函数
            Gb = ct.minreal(Gb, verbose=False)

        else:
            Gb = Gs

        return Gs, Gb

    def btn_import_excitation_clicked(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择文件",
            "C:\\",  # 初始路径，可以写 "C:/"
            "CSV 文件 (*.csv);;文本文件 (*.txt)"
        )

        if filename:
            self.data.filename = filename
            self.status_changed.emit(filename)

    def btn_bodeplot_clicked(self):

        if not self.collect_data():
            return

        Gs, Gb = self.get_system_transfer_functions()

        if self.bode_window is None:
            self.bode_window = BodeWindow()   # 新建

        self.bode_window.show()
        self.bode_window.raise_()
        self.bode_window.activateWindow()

        self.bode_window.clear_plots()

        plot_widget_i = self.bode_window.plot_widget_1
        plot_widget_j = self.bode_window.plot_widget_2

        # 开环频率特性
        mag, phase, omega = ct.frequency_response(Gs)

        plot_widget_i.plot(omega, 20*np.log10(mag), pen=pg.mkPen(
            color='b', width=2), name='位移幅值特性')
        plot_widget_i.plot(omega, 20*np.log10(mag * omega), pen=pg.mkPen(
            color='r', width=2), name='速度幅值特性')
        plot_widget_i.plot(omega, 20*np.log10(mag * omega ** 2), pen=pg.mkPen(
            color='g', width=2), name='加速度幅值特性')

        plot_widget_j.plot(omega, phase, pen=pg.mkPen(
            color='b', width=2), name='位移相位特性')
        plot_widget_j.plot(omega, phase + np.pi/2, pen=pg.mkPen(
            color='r', width=2), name='速度相位特性')
        plot_widget_j.plot(omega, phase + np.pi, pen=pg.mkPen(
            color='g', width=2), name='加速度相位特性')

        # 闭环频率特性
        mag, phase, omega = ct.frequency_response(Gb)

        plot_widget_i.plot(omega, 20*np.log10(mag), pen=pg.mkPen(
            color='b', width=2, style=Qt.DashLine), name='闭环位移幅值特性')
        plot_widget_i.plot(omega, 20*np.log10(mag * omega), pen=pg.mkPen(
            color='r', width=2, style=Qt.DashLine), name='闭环速度幅值特性')
        plot_widget_i.plot(omega, 20*np.log10(mag * omega ** 2), pen=pg.mkPen(
            color='g', width=2, style=Qt.DashLine), name='闭环加速度幅值特性')

        plot_widget_j.plot(omega, phase, pen=pg.mkPen(
            color='b', width=2, style=Qt.DashLine), name='闭环位移相位特性')
        plot_widget_j.plot(omega, phase + np.pi/2, pen=pg.mkPen(
            color='r', width=2, style=Qt.DashLine), name='闭环速度相位特性')
        plot_widget_j.plot(omega, phase + np.pi, pen=pg.mkPen(
            color='g', width=2, style=Qt.DashLine), name='闭环加速度相位特性')

    def btn_frequency_analyses_clicked(self):

        if not self.collect_data():
            return

        Gs, Gb = self.get_system_transfer_functions()

        # 绘图
        if self.frequency_window is None:
            self.frequency_window = FrequencyWindow()   # 新建

        self.frequency_window.show()
        self.frequency_window.raise_()
        self.frequency_window.activateWindow()

        self.frequency_window.clear_plots()

        # 开环\闭环频率特性分析
        self.frequency_window.plot_frequency_figures(Gs, 'b', "open loop")
        self.frequency_window.plot_frequency_figures(Gb, 'r', "closed loop")

        self.frequency_window.add_auxiliary_parts()

    def simulation_data_preparation(self):

        if not self.collect_data():
            return False

        self.msd.reset()

        self.step_now = 0
        self.time_now = 0.0

        dt = self.data.dt
        self.dt = dt
        self.time_stop = self.data.time_stop
        self.target_value = self.data.target_value
        self.control_type = self.data.control_type
        self.signal_type = self.data.signal_type
        self.frequency = self.data.frequency
        self.amplitude = self.data.amplitude

        self.is_importing = self.data.is_importing
        self.filename = self.data.filename
        self.skiprows = self.data.skiprows

        self.excitation = self.get_typical_signal()

        self.time = np.arange(0, self.time_stop, dt)
        nt = len(self.time)

        self.x_series = np.zeros(nt, dtype=float)
        self.v_series = np.zeros(nt, dtype=float)
        self.a_series = np.zeros(nt, dtype=float)
        self.f_series = np.zeros(nt, dtype=float)

        self.status_changed.emit("Initialization completes.")

        return True

    def get_typical_signal(self):

        if self.data.is_importing and self.data.filename:
            self.data.is_reading_success = False

            try:
                self.status_changed.emit(f"Reading {self.data.filename}")
                result = self.read_data_in_csv(self.data.filename,
                                               self.data.skiprows)
                if result:
                    xdata, ydata = result
                else:
                    self.status_changed.emit("Warning: No data is imported.")
                    return None

                if xdata[-1] - xdata[0] < 0.1:
                    self.status_changed.emit("Warning: Time is too short.")
                    return None

                if xdata[-1] - xdata[0] > self.time_stop:
                    mask = xdata <= self.time_stop
                    ydata = ydata[mask]
                    xdata = xdata[mask]
                    self.status_changed.emit(
                        f"Warning: Only {self.time_stop} s WILL run.")

                xdata -= xdata[0]
                time = np.arange(xdata[0], xdata[-1], self.dt)
                signal = np.interp(time, xdata, ydata)

                self.time_stop = xdata[-1]
                self.data.time_stop = xdata[-1]
                self.data_updated.emit()

                self.data.is_reading_success = True
                return signal

            except (Exception, TypeError, IndexError) as e:
                self.status_changed.emit(f"Error:{e}")
                return None

        else:
            tol = 1e-4
            if np.abs(self.data.time_stop - self.data.time_stop_default) > tol:
                self.data.time_stop = self.data.time_stop_default
                self.data_updated.emit()

        dt = self.dt
        self.time = np.arange(0, self.data.time_stop, self.data.dt)
        nt = len(self.time)

        signal = np.zeros(nt, dtype=float)

        if self.signal_type == 1:  # step
            step_start = int(0.1 / dt)
            signal[step_start:] = self.amplitude

        elif self.signal_type == 2:  # sine
            omega = 2 * np.pi * self.frequency
            signal = self.amplitude * np.sin(omega * self.time)

        else:  # impulse
            impulse_time_period = 0.1
            # 幅值取小点，否则绘图区装不下
            amplitude = self.amplitude / impulse_time_period

            step_start = int(0.1 / dt)
            step_stop = int((0.1 + impulse_time_period) / dt)
            signal[step_start:step_stop] = amplitude

        return signal

    def read_data_in_csv(self, file_path, skip_rows=0):
        """
        使用 csv 模块读取文件并提取前两列数据，支持跳过指定行数。

        参数:
        - file_path (str): 要读取的文件路径。
        - skip_rows (int): 从文件开头跳过的行数。默认值为0。

        返回:
        - tuple: 包含两个列表的元组，分别存储时间数据和位置数据。
        """

        if not file_path or not os.path.exists(file_path):
            self.status_changed.emit("Warning: File doesn't exist.")
            self.data.is_reading_success = False
            return None

        xdata = []
        ydata = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)

            # 跳过指定的行数
            for _ in range(skip_rows):
                try:
                    next(reader)
                except StopIteration:
                    self.status_changed.emit(
                        "Warning: Maximum rows have skipped.")
                    return None

            for row in reader:
                # 确保行不为空且至少有两列
                if len(row) >= 2:
                    try:
                        xdata.append(float(row[0]))
                        ydata.append(float(row[1]))
                    except ValueError as e:
                        self.status_changed.emit(
                            f"Warning: Invalid row {row}: {e}.")
                        continue
                else:
                    self.status_changed.emit(
                        "Warning: Invalid row {row} (columns < 2).")

        return np.array(xdata), np.array(ydata)

    def simulation_window_preparation(self):

        if self.transient_window is None:
            self.transient_window = TransientWindow()   # 新建

        self.transient_window.show()
        self.transient_window.raise_()
        self.transient_window.activateWindow()

        self.transient_window.plot_curve_11.setData([], [])
        self.transient_window.plot_curve_12.setData([], [])
        self.transient_window.plot_curve_2.setData([], [])
        self.transient_window.plot_curve_3.setData([], [])

        self.transient_window.plot_widget_1.setXRange(0, self.time_stop)
        self.transient_window.plot_widget_2.setXRange(0, self.time_stop)
        self.transient_window.plot_widget_3.setXRange(0, self.time_stop)

    def btn_simulate_clicked(self):

        if not self.simulation_data_preparation():
            return

        if self.data.is_importing and not self.data.is_reading_success:
            self.status_changed.emit("Data reading failure.")
            return

        self.status_changed.emit("Simulation starts.")

        self.simulation_window_preparation()

        if self.timer.isActive():
            self.timer.stop()

        for idx in range(len(self.time)-1):
            target_list = [self.msd.x,
                           self.msd.v,
                           self.msd.a,
                           0.0]
            error = self.target_value - target_list[self.control_type]

            force = self.pid.calculate_force(error)

            self.msd.update(self.excitation[idx:idx+2], force)

            self.x_series[idx] = self.msd.x
            self.v_series[idx] = self.msd.v
            self.a_series[idx] = self.msd.a
            self.f_series[idx] = force

        # 绘制曲线
        self.transient_window.plot_curve_11.setData(
            self.time[:-1], self.x_series[:-1])
        self.transient_window.plot_curve_12.setData(
            self.time[:-1], self.excitation[:-1])
        self.transient_window.plot_curve_2.setData(
            self.time[:-1], self.a_series[:-1])
        self.transient_window.plot_curve_3.setData(
            self.time[:-1], self.f_series[:-1])

        self.status_changed.emit("Simulation finishes.")

    def btn_animate_clicked(self):
        #

        if not self.simulation_data_preparation():
            return

        if self.data.is_importing and not self.data.is_reading_success:
            self.status_changed.emit("Data reading failure.")
            return

        self.status_changed.emit("Animation starts.")

        self.simulation_window_preparation()

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
            self.status_changed.emit("Animation stops.")
            return

        # 计算PID控制力
        target_list = [self.msd.x,
                       self.msd.v,
                       self.msd.a,
                       0.0]
        error = self.target_value - target_list[self.control_type]

        control_force = self.pid.calculate_force(error)

        # 更新物理模型
        self.msd.update(
            self.excitation[self.step_now:self.step_now+2], control_force)

        idx = self.step_now
        self.x_series[idx] = self.msd.x
        self.v_series[idx] = self.msd.v
        self.a_series[idx] = self.msd.a
        self.f_series[idx] = control_force

        if idx % 20 == 0:
            # 更新Pyqtgraph曲线（使用整个数据历史）
            self.transient_window.plot_curve_11.setData(
                self.time[:idx+1], self.x_series[:idx+1])
            self.transient_window.plot_curve_12.setData(
                self.time[:idx+1], self.excitation[:idx+1])
            self.transient_window.plot_curve_2.setData(
                self.time[:idx+1], self.a_series[:idx+1])
            self.transient_window.plot_curve_3.setData(
                self.time[:idx+1], self.f_series[:idx+1])

        # 更新OpenGL绘图
        self.msdviz.mass_motion = self.msd.x
        self.msdviz.base_motion = self.excitation[self.step_now + 1]

        self.msdviz.update()

        # 更新时间步
        self.step_now += 1
        self.time_now += self.dt


class TabPage2(QWidget):
    status_changed = Signal(str)

    def __init__(self, data: DataGroup,
                 msd: MSDPlant,
                 pid: PIDController,
                 tabpage1: TabPage1,
                 parent=None):

        super().__init__()

        self.data = data
        self.msd = msd
        self.pid = pid
        self.tabpage1 = tabpage1

        self.Hs = 1.0  # default transfer function

        self._setup_ui()

        self.transient_window = None
        self.frequency_window = None
        self.discrete_window = None
        self.transient_window_2 = None
        self.frequency_window_2 = None

    def _setup_ui(self):

        tabpage2_layout = QVBoxLayout(self)

        # 传递函数控制
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

        key_label = QLabel("建模方式")
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

        btn_show_infos = QPushButton("信息")
        btn_show_infos.clicked.connect(self.btn_show_infos_clicked)
        tf_layout_1.addWidget(btn_show_infos, 1)

        self.tf_layout.addLayout(tf_layout_1)

        self.tf_layout_2 = QHBoxLayout()

        key_label = QLabel("分子")
        key_label.setFixedWidth(50)
        self.tf_layout_2.addWidget(key_label)

        eboxes = self.set_transfer_function_boxes(
            self.tf_layout_2, 2, True)
        self.numerator_boxes = eboxes.copy()

        self.tf_layout.addLayout(self.tf_layout_2)

        self.tf_layout_3 = QHBoxLayout()

        key_label = QLabel("分母")
        key_label.setFixedWidth(50)
        self.tf_layout_3.addWidget(key_label)

        eboxes = self.set_transfer_function_boxes(
            self.tf_layout_3, 3, False)
        self.denominator_boxes = eboxes.copy()

        self.tf_layout.addLayout(self.tf_layout_3)

        tf_layout_4 = QHBoxLayout()

        self.btn_time_responses = QPushButton("传递函数时域响应")
        self.btn_time_responses.clicked.connect(
            self.btn_transient_responses_clicked)
        tf_layout_4.addWidget(self.btn_time_responses)

        self.btn_frequency_analyses = QPushButton("传递函数频谱分析")
        self.btn_frequency_analyses.clicked.connect(
            self.btn_frequency_analyses_clicked)
        tf_layout_4.addWidget(self.btn_frequency_analyses)

        self.tf_layout.addLayout(tf_layout_4)

        tf_layout_5 = QHBoxLayout()

        self.btn_discrete_control = QPushButton("离散运动控制")
        self.btn_discrete_control.clicked.connect(
            self.btn_discrete_control_clicked)
        tf_layout_5.addWidget(self.btn_discrete_control)

        self.btn_model_transient_response = QPushButton("闭环时域响应")
        self.btn_model_transient_response.clicked.connect(
            self.btn_model_transient_response_clicked)
        tf_layout_5.addWidget(self.btn_model_transient_response)

        self.btn_model_frequency_analyses = QPushButton("闭环频谱分析")
        self.btn_model_frequency_analyses.clicked.connect(
            self.btn_model_frequency_analyses_clicked)
        tf_layout_5.addWidget(self.btn_model_frequency_analyses)

        self.tf_layout.addLayout(tf_layout_5)

        tabpage2_layout.addWidget(self.tf_group)

        # LQR control
        self.lqr_group = QGroupBox("LQR Control (Bryson's Rule)")
        self.lqr_layout = QVBoxLayout(self.lqr_group)

        lqr_layout_1 = QHBoxLayout()
        self.lqr_boxes = []

        key_label = QLabel("Q")
        key_label.setFixedWidth(50)
        key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lqr_layout_1.addWidget(key_label)

        ebox = self.tabpage1.create_number_box("0.2", 1)
        self.lqr_boxes.append(ebox)
        lqr_layout_1.addWidget(ebox)

        ebox = self.tabpage1.create_number_box("5.0", 1)
        self.lqr_boxes.append(ebox)
        lqr_layout_1.addWidget(ebox)

        key_label = QLabel("R")
        key_label.setFixedWidth(50)
        key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lqr_layout_1.addWidget(key_label)

        ebox = self.tabpage1.create_number_box("1000", 1)
        self.lqr_boxes.append(ebox)
        lqr_layout_1.addWidget(ebox)

        lqr_layout_2 = QHBoxLayout()

        btn_lqr_transient_response = QPushButton("transcient response")
        btn_lqr_transient_response.clicked.connect(
            self.btn_lqr_transient_clicked)
        lqr_layout_2.addWidget(btn_lqr_transient_response)

        self.lqr_layout.addLayout(lqr_layout_1)
        self.lqr_layout.addLayout(lqr_layout_2)
        tabpage2_layout.addWidget(self.lqr_group)

    def get_windows(self):

        windows_list = [self.transient_window, self.frequency_window,
                        self.discrete_window, self.transient_window_2,
                        self.frequency_window_2]
        return windows_list

    def btn_show_infos_clicked(self):

        if not self.simulation_data_preparation():
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("基本信息")
        dialog.setFixedWidth(300)
        dialog.setMaximumHeight(300)

        layout = QVBoxLayout()

        result = self.get_transfer_function_from_boxes()
        if not result:
            return
        else:
            Hs = result

        poles = Hs.poles()

        poles_str = ", ".join(
            [f"{p.real:.2f}{'+' if p.imag>=0 else ''}{p.imag:.2f}j" for p in poles])
        label1 = QLabel(f"transfer function poles : [{poles_str}].")
        label1.setWordWrap(True)
        layout.addWidget(label1)

        Gs, Gb = self.get_system_transfer_functions()

        poles = Gs.poles()

        poles_str = ", ".join(
            [f"{p.real:.2f}{'+' if p.imag>=0 else ''}{p.imag:.2f}j" for p in poles])
        label2 = QLabel(f"open loop poles : [{poles_str}].")
        label2.setWordWrap(True)
        layout.addWidget(label2)

        poles = Gb.poles()

        poles_str = ", ".join(
            [f"{p.real:.2f}{'+' if p.imag>=0 else ''}{p.imag:.2f}j" for p in poles])
        label3 = QLabel(f"closed loop poles : [{poles_str}].")
        label3.setWordWrap(True)
        layout.addWidget(label3)

        dialog.setLayout(layout)
        dialog.exec()  # 阻塞式弹出

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
        for idx in range(count):
            ebox = QLineEdit("")
            ebox.setAlignment(Qt.AlignmentFlag.AlignCenter)

            validator = QDoubleValidator(
                -1e8, 1e8, 2)  # 限定范围，小数点取 2 位
            validator.setNotation(QDoubleValidator.StandardNotation)
            ebox.setValidator(validator)

            boxes.append(ebox)
            layout.addWidget(ebox, 2)

        if isnumerator:
            layout.addStretch(1)

        return boxes

    def on_combobox1_clicked(self):

        order = self.comboboxes[0].currentIndex() + 1

        eboxes = self.set_transfer_function_boxes(
            self.tf_layout_2, order, True)

        self.numerator_boxes = eboxes.copy()

        eboxes = self.set_transfer_function_boxes(
            self.tf_layout_3, order + 1, False)

        self.denominator_boxes = eboxes.copy()

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

        for idx, box in enumerate(self.numerator_boxes):
            box_str = box.text().strip()
            if box_str == "":
                num[idx] = 0.0
            else:
                num[idx] = float(box_str)

        for idx, box in enumerate(self.denominator_boxes):
            box_str = box.text().strip()
            if box_str == "":
                den[idx] = 0.0
            else:
                den[idx] = float(box_str)

        if np.allclose(num, 0, 0, 1e-6) and np.allclose(den, 1, 0, 1e-6):
            self.status_changed.emit(
                "Business cooperation, please contact: dyyydxxx@gmail.com")
            return None

        if np.allclose(num, 0, 0, 1e-6) or np.allclose(den, 0, 0, 1e-6):
            self.status_changed.emit("Numerator or denominator is Null.")
            return None

        Hs = ct.minreal(ct.tf(num, den), verbose=False)

        if len(Hs.num[0][0]) >= len(Hs.den[0][0]):
            self.status_changed.emit("Controller is not strictly proper.")
            return None

        return Hs

    def btn_transient_responses_clicked(self):

        # 0. data preparation
        if not self.simulation_data_preparation():
            return

        result = self.get_transfer_function_from_boxes()
        if not result:
            return
        else:
            Hs = result

        # 时域响应
        # 0. window preparation
        if self.transient_window is None:
            self.transient_window = TransientWindow()   # 新建

        self.transient_window.show()
        self.transient_window.raise_()
        self.transient_window.activateWindow()

        self.transient_window.clear_plots()

        plot_widget_i = self.transient_window.plot_widget_1
        plot_widget_j = self.transient_window.plot_widget_2
        plot_widget_k = self.transient_window.plot_widget_3

        # 1. impulse response
        t, y = ct.impulse_response(Hs, T=self.time)

        plot_widget_i.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="脉冲响应")

        plot_widget_i.setLabel("left", "displacement (m)")

        # 2. step response
        t, y = ct.step_response(Hs, T=self.time)

        plot_widget_j.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="阶跃响应")

        plot_widget_j.setLabel("left", "displacement (m)")

        # 3. sine response
        omega = 2 * np.pi * float(self.tabpage1.sim_boxes[1].text())
        x = np.sin(omega*self.time)
        t, y = ct.forced_response(Hs, T=self.time, inputs=x)

        plot_widget_k.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="正弦响应")

        plot_widget_k.setLabel("left", "displacement (m)")

    def btn_frequency_analyses_clicked(self):

        result = self.get_transfer_function_from_boxes()
        if not result:
            return
        else:
            Hs = result

        # 频域响应
        # 0. window preparation
        if self.frequency_window is None:
            self.frequency_window = FrequencyWindow()   # 新建

        self.frequency_window.show()
        self.frequency_window.raise_()
        self.frequency_window.activateWindow()

        self.frequency_window.clear_plots()

        # 1. frequency response
        self.frequency_window.plot_frequency_figures(
            Hs, 'b', "transfer function")

        self.frequency_window.add_auxiliary_parts()

    def btn_discrete_control_clicked(self):

        result = self.get_transfer_function_from_boxes()
        if not result:
            return
        else:
            Hs = result

        # 0. data preparation
        if not self.simulation_data_preparation():
            return

        if self.data.is_importing and not self.data.is_reading_success:
            self.status_changed.emit("Data reading failure.")
            return

        # 离散时域响应
        # 0. window preparation
        if self.discrete_window is None or not self.discrete_window.isVisible():
            self.discrete_window = TransientWindow()

        self.discrete_window.show()
        self.discrete_window.raise_()
        self.discrete_window.activateWindow()

        self.discrete_window.clear_plots()

        plot_widget_i = self.discrete_window.plot_widget_1
        plot_widget_j = self.discrete_window.plot_widget_2
        plot_widget_k = self.discrete_window.plot_widget_3

        self.status_changed.emit("Simulation starts.")

        # 1. z 变换
        dt = self.dt
        Gz = ct.c2d(Hs, dt, 'tustin')

        num = Gz.num[0][0]
        den = Gz.den[0][0]

        den = den/num[0]
        num = num/num[0]

        na = len(num)
        nb = len(den)

        if len(num) > len(den):
            self.status_changed.emit("G(z) is not strictly proper.")
            return

        nt = len(self.time)

        error_history = np.zeros(nb, dtype=float)
        force_history = np.zeros(na-1, dtype=float)

        self.x_series = np.zeros(nt, dtype=float)
        self.v_series = np.zeros(nt, dtype=float)
        self.a_series = np.zeros(nt, dtype=float)
        self.f_series = np.zeros(nt, dtype=float)

        control_type = self.control_type

        for idx in range(nt-1):
            target_list = [self.msd.x, self.msd.v,
                           self.msd.a, 0.0]
            error = self.target_value - target_list[control_type]

            # 向左滑动平移
            error_history[:-1] = error_history[1:]
            error_history[-1] = error  # 最新元素总在最右边

            force_history[:-1] = force_history[1:]
            force_history[-1] = self.f_series[idx-1]  # 最新元素总在最右边

            force = np.sum(np.flip(den) * error_history) - \
                np.sum(np.flip(num[1:]) * force_history)

            force = max(min(force, 1e5), -1e5)  # 最大输出载荷 100,000 N
            self.f_series[idx] = force

            self.msd.update(
                self.excitation[idx:idx+2], self.f_series[idx])

            self.x_series[idx] = self.msd.x
            self.v_series[idx] = self.msd.v
            self.a_series[idx] = self.msd.a

        # plot figures
        plot_widget_i.plot(self.time[:-1], self.x_series[:-1], pen=pg.mkPen(
            color='b', width=2), name="位移响应")
        plot_widget_i.plot(self.time[:-1], self.excitation[:-1], pen=pg.mkPen(
            color='r', width=2), name="激励信号")
        plot_widget_j.plot(self.time[:-1], self.a_series[:-1], pen=pg.mkPen(
            color='b', width=2), name="加速度响应")
        plot_widget_k.plot(self.time[:-1], self.f_series[:-1], pen=pg.mkPen(
            color='b', width=2), name="控制载荷")

        self.status_changed.emit("Simulation finishes.")

    def get_system_transfer_functions(self):

        m = self.data.mass
        c = self.data.damping
        k = self.data.stiffness

        Gs = ct.tf([c, k], [m, c, k])

        control_type = self.control_type

        if control_type < 3:
            result = self.get_transfer_function_from_boxes()
            if not result:
                return
            else:
                Hs = result

            if control_type == 1:
                Hx = Hs * ct.tf([1, 0], 1)
            elif control_type == 2:
                Hx = Hs * ct.tf([1, 0, 0], 1)
            else:
                Hx = Hs

            Gf = ct.tf(1, [c, k])  # 力传递函数

            Gb = Hx * Gs / (Hx + Gs * Gf)  # 闭环系统传递函数

            Gb = ct.minreal(Gb, verbose=False)

        else:
            Gb = Gs

        return Gs, Gb

    def btn_model_transient_response_clicked(self):

        if not self.simulation_data_preparation():
            return

        # 0. window preparation
        if self.transient_window_2 is None:
            self.transient_window_2 = TransientWindow()   # 新建

        self.transient_window_2.show()
        self.transient_window_2.raise_()
        self.transient_window_2.activateWindow()

        self.transient_window_2.clear_plots()

        plot_widget_i = self.transient_window_2.plot_widget_1
        plot_widget_j = self.transient_window_2.plot_widget_2
        plot_widget_k = self.transient_window_2.plot_widget_3

        time = self.time

        Gs, Gb = self.get_system_transfer_functions()

        t, y = ct.impulse_response(Gs, time)
        plot_widget_i.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="开环系统脉冲响应")

        t, y = ct.impulse_response(Gb, time)
        plot_widget_i.plot(t, y, pen=pg.mkPen(
            color='r', width=2), name="闭环系统脉冲响应")

        plot_widget_i.setLabel("left", "displacement (m)")

        t, y = ct.step_response(Gs, time)
        plot_widget_j.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="开环系统阶跃响应")

        t, y = ct.step_response(Gb, time)
        plot_widget_j.plot(t, y, pen=pg.mkPen(
            color='r', width=2), name="闭环系统阶跃响应")

        plot_widget_j.setLabel("left", "displacement (m)")

        # 3. sine response
        omega = 2 * np.pi * self.frequency
        x = self.amplitude * np.sin(omega*time)

        t, y = ct.forced_response(Gs, T=time, inputs=x)
        plot_widget_k.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="开环系统正弦响应")

        t, y = ct.forced_response(Gb, T=time, inputs=x)
        plot_widget_k.plot(t, y, pen=pg.mkPen(
            color='r', width=2), name="闭环系统正弦响应")

        plot_widget_k.setLabel("left", "displacement (m)")

    def btn_model_frequency_analyses_clicked(self):

        if not self.simulation_data_preparation():
            return

        if self.frequency_window_2 is None:
            self.frequency_window_2 = FrequencyWindow()   # 新建

        self.frequency_window_2.show()
        self.frequency_window_2.raise_()
        self.frequency_window_2.activateWindow()

        self.frequency_window_2.clear_plots()

        Gs, Gb = self.get_system_transfer_functions()

        # 开环\闭环系统频率特性
        self.frequency_window_2.plot_frequency_figures(Gs, 'b', '开环系统频率特性')
        self.frequency_window_2.plot_frequency_figures(Gb, 'r', '闭环系统频率特性')

        self.frequency_window_2.add_auxiliary_parts()

    def simulation_data_preparation(self):

        if not self.tabpage1.collect_data():
            return False

        self.msd.reset()

        self.tabpage1.simulation_data_preparation()

        dt = self.data.dt
        self.dt = dt
        self.time_stop = self.data.time_stop
        self.target_value = self.data.target_value
        self.control_type = self.data.control_type
        self.signal_type = self.data.signal_type
        self.frequency = self.data.frequency
        self.amplitude = self.data.amplitude
        self.excitation = self.tabpage1.excitation

        self.time = np.arange(0, self.time_stop, dt)
        nt = len(self.time)

        self.x_series = np.zeros(nt, dtype=float)
        self.v_series = np.zeros(nt, dtype=float)
        self.a_series = np.zeros(nt, dtype=float)
        self.f_series = np.zeros(nt, dtype=float)

        self.status_changed.emit("System initialization completes.")

        return True

    def btn_lqr_transient_clicked(self):
        pass
