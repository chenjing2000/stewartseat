# opengl_widget.py

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QGroupBox, QLineEdit, QLabel,
                               QComboBox, QPushButton, QDialog)
import pyqtgraph as pg
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QPalette, QColor, QDoubleValidator
from PySide6.QtCore import QTimer, Signal
from OpenGL.GL import *
import numpy as np

from MSDModel import MSDPlant, PIDController


class MassSpringDamperGL(QOpenGLWidget):
    status_message = Signal(str)  # signal to update status

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(400, 400)
        # 视觉比例：把位移（米）映射到屏幕（单位坐标）
        self.meter_to_screen = 40  # 1 m -> 40 px（在内部用归一化再缩放）

        # positions
        self.mass_position_init = 60
        self.mass_motion = 0.0  # state or response
        self.mass_position_actual = self.mass_position_init + self.mass_motion
        self.base_position_init = 0
        self.base_motion = 0.0  # excitation or external input
        self.base_position_actual = self.base_position_init + self.base_motion

        # joints geometry
        self.joints_position = np.array([-15, 0, 15])

        # base geometry
        self.base_width = 50

        # mass geometry
        self.mass_width = 40
        self.mass_height = 20

        self.upper_joint_position = self.mass_position_actual
        self.lower_joint_position = self.base_position_actual

        self.suspension_stroke = self.upper_joint_position - self.lower_joint_position

        # spring geometry
        x_spring = self.joints_position[1]
        self.spring_joint_b = np.array([x_spring, self.upper_joint_position])
        self.spring_joint_a = np.array([x_spring, self.lower_joint_position])
        self.spring_coils = 11
        self.spring_width = 10

        # damper geometry
        x_damper = self.joints_position[0]
        self.damper_joint_b = np.array([x_damper, self.upper_joint_position])
        self.damper_joint_a = np.array([x_damper, self.lower_joint_position])
        self.damper_height = 20
        self.damper_width = 10
        self.damper_thickness = 10

        # actuator geometry
        x_actuator = self.joints_position[2]
        self.actuator_joint_b = np.array(
            [x_actuator, self.upper_joint_position])
        self.actuator_joint_a = np.array(
            [x_actuator, self.lower_joint_position])
        self.actuator_radius = 6

    def update_positions(self, x_mass, x_base):
        self.mass_motion = x_mass * self.meter_to_screen
        self.mass_position_actual = self.mass_position_init + \
            self.mass_motion
        self.base_motion = x_base * self.meter_to_screen
        self.base_position_actual = self.base_position_init + \
            self.base_motion

        self.upper_joint_position = self.mass_position_actual
        self.lower_joint_position = self.base_position_actual

        self.suspension_stroke = self.upper_joint_position - self.lower_joint_position

        # spring geometry
        x_spring = self.joints_position[1]
        self.spring_joint_b = np.array(
            [x_spring, self.upper_joint_position])
        self.spring_joint_a = np.array(
            [x_spring, self.lower_joint_position])

        # damper geometry
        x_damper = self.joints_position[0]
        self.damper_joint_b = np.array(
            [x_damper, self.upper_joint_position])
        self.damper_joint_a = np.array(
            [x_damper, self.lower_joint_position])

        # actuator geometry
        x_actuator = self.joints_position[2]
        self.actuator_joint_b = np.array(
            [x_actuator, self.upper_joint_position])
        self.actuator_joint_a = np.array(
            [x_actuator, self.lower_joint_position])

    def initializeGL(self):
        glClearColor(0.08, 0.09, 0.11, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        glViewport(0, -300, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-90, 90, -40, 140, -1, 1)  # Set up a 2D projection
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        # Draw ground
        wb = self.base_width
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex2f(-wb/2, self.base_position_actual)
        glVertex2f(+wb/2, self.base_position_actual)
        glEnd()

        # Draw mass block
        self.draw_mass()
        self.draw_spring()
        self.draw_damper()

        # Draw controller symbol (circle with arrow)
        self.draw_actuator()

        # Draw datum lines
        self.draw_dashed_lines()

    def draw_mass(self):

        y_mass = self.mass_position_actual
        w = self.mass_width
        h = self.mass_height

        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(2.0)
        glBegin(GL_QUADS)
        glVertex2f(-w/2, y_mass)
        glVertex2f(-w/2, y_mass + h)
        glVertex2f(+w/2, y_mass + h)
        glVertex2f(+w/2, y_mass)
        glEnd()

    def draw_spring(self):

        # 弹簧包含两端直线段与中间的弹簧折线段
        section_height = self.suspension_stroke / 6
        coil_height = (self.suspension_stroke -
                       2 * section_height)/self.spring_coils

        glColor3f(0.8, 0.4, 0.1)  # Orange-brown
        glLineWidth(2.0)

        glBegin(GL_LINE_STRIP)
        # 从下向上绘制
        glVertex2f(self.spring_joint_a[0], self.spring_joint_a[1])
        glVertex2f(self.spring_joint_a[0],
                   self.spring_joint_a[1] + section_height)

        x_direction = 1

        for id in range(self.spring_coils-1):
            x = self.spring_joint_a[0] + x_direction*self.spring_width/2
            y = self.spring_joint_a[1] + section_height + coil_height*(id+1)
            x_direction = -x_direction
            glVertex2f(x, y)

        glVertex2f(self.spring_joint_b[0],
                   self.spring_joint_a[1] + section_height + coil_height*(self.spring_coils))
        glVertex2f(self.spring_joint_b[0], self.spring_joint_b[1])
        glEnd()

    def draw_damper(self):

        section_height = (self.suspension_stroke - self.damper_height)/2

        glColor3f(0.8, 0.4, 0.1)  # Orange-brown
        glLineWidth(2.0)

        glBegin(GL_LINES)
        glVertex2f(self.damper_joint_a[0], self.damper_joint_a[1])
        glVertex2f(self.damper_joint_a[0],
                   self.damper_joint_a[1] + section_height)

        glVertex2f(self.damper_joint_a[0] - self.damper_width/2,
                   self.damper_joint_a[1] + section_height)
        glVertex2f(self.damper_joint_a[0] - self.damper_width/2,
                   self.damper_joint_a[1] + section_height + self.damper_height)

        glVertex2f(self.damper_joint_a[0] + self.damper_width/2,
                   self.damper_joint_a[1] + section_height)
        glVertex2f(self.damper_joint_a[0] + self.damper_width/2,
                   self.damper_joint_a[1] + section_height + self.damper_height)

        glVertex2f(self.damper_joint_a[0] - self.damper_width/2,
                   self.damper_joint_a[1] + section_height)
        glVertex2f(self.damper_joint_a[0] + self.damper_width/2,
                   self.damper_joint_a[1] + section_height)

        glVertex2f(self.damper_joint_a[0] - self.damper_width/2,
                   self.damper_joint_a[1] + section_height + self.damper_thickness)
        glVertex2f(self.damper_joint_a[0] + self.damper_width/2,
                   self.damper_joint_a[1] + section_height + self.damper_thickness)

        glVertex2f(self.damper_joint_a[0],
                   self.damper_joint_a[1] + section_height + self.damper_thickness)
        glVertex2f(self.damper_joint_b[0], self.damper_joint_b[1])
        glEnd()

    def draw_actuator(self):
        # Draw circle
        actuator_center = self.actuator_joint_a[1] + self.suspension_stroke/2

        glColor3f(0.1, 0.6, 0.1)  # Green

        glBegin(GL_LINES)
        glVertex2f(self.actuator_joint_a[0], self.actuator_joint_a[1])
        glVertex2f(self.actuator_joint_a[0],
                   actuator_center - self.actuator_radius)

        glVertex2f(self.actuator_joint_b[0],
                   actuator_center + self.actuator_radius)
        glVertex2f(self.actuator_joint_b[0], self.actuator_joint_b[1])

        angle = 5 / 4 * np.pi
        x = self.actuator_joint_a[0] + \
            self.actuator_radius * np.cos(angle)
        y = actuator_center + \
            self.actuator_radius * np.sin(angle)
        glVertex2f(x, y)

        angle = 1 / 4 * np.pi
        x = self.actuator_joint_a[0] + \
            self.actuator_radius * np.cos(angle)
        y = actuator_center + \
            self.actuator_radius * np.sin(angle)
        glVertex2f(x, y)

        glEnd()

        # 绘制箭头
        arrow_head = np.array([x, y])

        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)

        angle = 5 / 4 * np.pi - 10 * np.pi / 180
        x = arrow_head[0] + self.actuator_radius * np.cos(angle)
        y = arrow_head[1] + self.actuator_radius * np.sin(angle)
        glVertex2f(x, y)

        angle = 5 / 4 * np.pi + 10 * np.pi / 180
        x = arrow_head[0] + self.actuator_radius * np.cos(angle)
        y = arrow_head[1] + self.actuator_radius * np.sin(angle)
        glVertex2f(x, y)

        glEnd()

        glBegin(GL_LINE_LOOP)
        for id in range(36):
            angle = id * 10 * np.pi / 180
            x = self.actuator_joint_a[0] + \
                self.actuator_radius * np.cos(angle)
            y = actuator_center + \
                self.actuator_radius * np.sin(angle)
            glVertex2f(x, y)
        glEnd()

    def draw_dashed_lines(self):

        # 2. 启用虚线模式
        glEnable(GL_LINE_STIPPLE)

        # 3. 设置虚线样式
        # 这里我们使用 factor=1，pattern=0xAAAA
        # 0xAAAA (二进制 1010 1010 1010 1010) 会产生一个等长的点-空隙模式
        glLineStipple(5, 0xAAAA)

        wb = self.base_width
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(2.0)

        glBegin(GL_LINES)
        glVertex2f(-wb/2 - 10, self.base_position_init)
        glVertex2f(+wb/2 + 10, self.base_position_init)

        glVertex2f(-wb/2 - 10, self.mass_position_init + self.mass_height/2)
        glVertex2f(+wb/2 + 10, self.mass_position_init + self.mass_height/2)
        glEnd()

        glDisable(GL_LINE_STIPPLE)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("质量-阻尼-弹簧系统仿真")
        self.setFixedSize(800, 680)
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

        self.step_count = 0

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
        self.timer.setInterval(3)  # 每 2 毫秒触发一次
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

        dialog.setLayout(layout)
        dialog.exec()  # 阻塞式弹出

    def btn_bodeplot_clicked(self):
        pass

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
        self.step_count = 0

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

        self.x_series = []
        self.v_series = []
        self.a_series = []
        self.f_series = []

    def btn_simulate_clicked(self):

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
            self.x_series.append(self.msd.x)
            self.v_series.append(self.msd.v)
            self.a_series.append(self.msd.a)
            self.f_series.append(force)

        if len(self.x_series) > 0:
            self.plot_curve_11.setData(self.time[:-1], self.excitation[:-1])
            self.plot_curve_12.setData(self.time[:-1], np.array(self.x_series))
            self.plot_curve_2.setData(self.time[:-1], np.array(self.a_series))
            self.plot_curve_3.setData(self.time[:-1], np.array(self.f_series))

    def btn_animate_clicked(self):
        """
        当点击 'animate' 按钮时调用，用于启动动画仿真。
        """
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
        if self.step_count >= len(self.time) - 2:
            self.timer.stop()
            print("动画结束。")
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
            self.excitation[self.step_count:self.step_count+2], control_force)

        # 更新OpenGL绘图
        self.msdviz.update_positions(
            self.msd.x, self.excitation[self.step_count + 1])
        self.msdviz.update()

        # # 记录数据
        # self.x_series.append(self.msd.x)
        # self.v_series.append(self.msd.v)
        # self.a_series.append(self.msd.a)
        # self.f_series.append(control_force)

        # # 更新Pyqtgraph曲线（使用整个数据历史）
        # time = self.time[:len(self.x_series)]
        # # self.plot_curve_11.setData(time, self.excitation[:self.step_count + 1])
        # self.plot_curve_12.setData(time, np.array(self.x_series))
        # self.plot_curve_2.setData(time, np.array(self.a_series))
        # self.plot_curve_3.setData(time, np.array(self.f_series))

        # 更新时间步
        self.step_count += 1

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
