# opengl_widget.py

import glfw
from OpenGL.GL.shaders import *
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
import ctypes

import numpy as np
import control as ct

from MSDModel import MSDPlant, PIDController

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec2 aPos;
uniform vec2 uWindowSize;  // 窗口大小
void main()
{
    // 将像素坐标转换到 NDC
    float x = (aPos.x / uWindowSize.x) * 2.0 - 1.0;
    float y = (aPos.y / uWindowSize.y) * 2.0 - 1.0;

    gl_Position = vec4(x, y, 0.0, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main()
{
    FragColor = vec4(uColor, 1.0);
}
"""


class Shaders():
    def __init__(self, vertices: np.ndarray,
                 vertex_shader_source: str,
                 fragment_shader_source: str,
                 color: tuple[float, float, float]):

        self.vertices = vertices
        self.vs_src = vertex_shader_source
        self.fs_src = fragment_shader_source

        # 初始状态
        self.vao = None
        self.vbo = None
        self.shaderProgram = None
        self.wsize_location = -1
        self.color_location = -1
        self.color = color

        self.setup()

    def setup(self):
        """
        初始化所有 OpenGL 资源，包括着色器程序、VAO和VBO。
        """
        self._create_program()
        self._create_objects()

        # 获取 uniform location，只需要执行一次
        glUseProgram(self.shaderProgram)
        self.wsize_location = glGetUniformLocation(
            self.shaderProgram, "uWindowSize")
        if self.wsize_location == -1:
            print("Warning: uWindowSize uniform not found in shader.")

        self.color_location = glGetUniformLocation(
            self.shaderProgram, "uColor")
        if self.color_location == -1:
            print("Warning: uColor uniform not found in shader.")

        glUseProgram(0)

    def set_window_size(self, window_size: tuple[int, int]):

        glUseProgram(self.shaderProgram)

        glUniform2f(self.wsize_location, *window_size)

        glUseProgram(0)

    def _create_program(self):
        vertex_shader = compileShader(self.vs_src, GL_VERTEX_SHADER)
        fragment_shader = compileShader(self.fs_src, GL_FRAGMENT_SHADER)
        self.shaderProgram = compileProgram(vertex_shader, fragment_shader)

    def _create_objects(self):
        """
        创建VAO和VBO，并上传初始顶点数据。
        """
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes,
                     self.vertices, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def update_vertices(self, new_vertices: np.ndarray):
        """
        更新 VBO 中的顶点数据。
        """
        self.vertices = new_vertices
        # 这个更新操作不需要在每次绘制时都做，只在数据变化时才做
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        self.vertices.nbytes, self.vertices)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, mode: int, first: int, count: int):
        """
        绘制组件的一部分。

        参数:
            mode (int): OpenGL 绘制模式 (例如 GL_LINES)。
            first (int): 从 VBO 中第几个顶点开始绘制。
            count (int): 绘制多少个顶点
        """
        glUseProgram(self.shaderProgram)

        glBindVertexArray(self.vao)

        glUniform3f(self.color_location, *self.color)

        glDrawArrays(mode, first, count)

        glBindVertexArray(0)
        glUseProgram(0)


class PartMass():
    def __init__(self, screen_scale: float = 40.0):
        self.screen_scale = screen_scale

        self.width = 40.0
        self.height = 20.0
        self.position = 60.0

        vertices = self.update_vertices()

        color = [0.3, 0.4, 0.5]

        self.shader = Shaders(vertices,
                              vertex_shader_source,
                              fragment_shader_source,
                              color)

    def update_vertices(self, displacement: float = 0.0):

        z = displacement * self.screen_scale
        w = self.width
        h = self.height
        p = self.position

        vertices = np.array([-w/2, p + z,
                             -w/2, p + h + z,
                             +w/2, p + h + z,
                             +w/2, p + z
                             ], dtype=np.float32)

        return vertices

    def draw(self, displacement: float = 0.0):
        """
        更新质量块的位置，并使用 Shaders 类进行绘制。
        """

        vertices = self.update_vertices(displacement)
        self.shader.update_vertices(vertices)
        self.shader.draw(GL_TRIANGLE_FAN, 0, 2)


class PartBase():
    def __init__(self, screen_scale: float = 40.0):
        self.screen_scale = screen_scale

        self.width = 50
        self.position = 0

        vertices = self.update_vertices()

        color = [0.3, 0.4, 0.5]

        self.shader = Shaders(vertices,
                              vertex_shader_source,
                              fragment_shader_source,
                              color)

    def update_vertices(self, displacement: float = 0.0):

        z = displacement * self.screen_scale
        w = self.width
        p = self.position

        vertices = np.array([-w/2, p + z,
                             +w/2, p + z], dtype=np.float32)

        return vertices

    def draw(self, displacement: float = 0.0):
        """
        更新底部的位置，并使用 Shaders 类进行绘制。
        """

        vertices = self.update_vertices(displacement)
        self.shader.update_vertices(vertices)
        self.shader.draw(GL_LINE_STRIP, 0, 2)


class PartDatum():
    def __init__(self, screen_scale: float = 40.0):
        self.screen_scale = screen_scale

        self.width = 60
        self.positions = [0, 60]
        self.section_num = 60

        vertices = self.update_vertices()

        color = [0.3, 0.4, 0.5]

        self.shader = Shaders(vertices,
                              vertex_shader_source,
                              fragment_shader_source,
                              color)

    def update_vertices(self):

        x_step = self.width / (self.section_num + 1)
        vertices = []
        for kd in range(2):
            x_datum = 0.0
            y_datum = self.positions[kd]
            for id in range(self.section_num + 1):
                x_datum += x_step
                vertices.extend([x_datum, y_datum])

        return np.array(vertices, dtype=np.float32)

    def draw(self):
        """
        使用 Shaders 类绘制参考线。
        """

        vertices = self.update_vertices()
        self.shader.update_vertices(vertices)
        self.shader.draw(GL_LINES, 0, self.section_num + 1)
        self.shader.draw(GL_LINES, self.section_num + 1, self.section_num + 1)


class PartSpring():
    def __init__(self, screen_scale: float = 40.0):
        self.screen_scale = screen_scale

        self.spring_center = 0  # x position
        self.spring_width = 10
        self.spring_coils = 11

        self.lower_position = 0.0  # = PartBase.position
        self.upper_position = 60.0  # = PartMass.position

        vertices = self.update_vertices()

        color = [0.3, 0.4, 0.5]

        self.shader = Shaders(vertices,
                              vertex_shader_source,
                              fragment_shader_source,
                              color)

    def update_vertices(self, displacements: np.ndarray = np.array([0.0, 0.0])):
        """
        参数：
            displacements: (2,) array, [lower, upper] displacements.
        """
        zs = displacements * self.screen_scale

        upper_location = self.upper_position + zs[1]
        lower_location = self.lower_position + zs[0]
        spring_stroke = upper_location - lower_location
        spring_shank = spring_stroke / 6
        coil_height = (spring_stroke -
                       2 * spring_shank) / (self.spring_coils + 1)

        vertices = []
        vertices.extend([self.spring_center, lower_location])
        vertices.extend([self.spring_center,
                        lower_location + spring_shank])

        x_direction = 1
        y_spring = lower_location + spring_shank

        for id in range(self.spring_coils - 1):
            x_spring = self.spring_center + x_direction * self.spring_width/2
            x_direction = -x_direction
            y_spring += coil_height
            vertices.extend([x_spring, y_spring])

        vertices.extend([self.spring_center,
                        upper_location - spring_shank])
        vertices.extend([self.spring_center, upper_location])

        return np.array(vertices, dtype=np.float32)

    def draw(self, displacement: float = 0.0):
        """
        更新弹簧的位置，并使用 Shaders 类进行绘制。
        """

        vertices = self.update_vertices(displacement)
        self.shader.update_vertices(vertices)
        self.shader.draw(GL_LINE_STRIP, 0, len(vertices) // 2)


class PartDamper():
    def __init__(self, screen_scale: float = 40.0):
        self.screen_scale = screen_scale

        self.damper_center = -15  # x position
        self.damper_height = 20
        self.damper_width = 10

        self.lower_position = 0.0  # = PartBase.position
        self.upper_position = 60.0  # = PartMass.position

        vertices = self.update_vertices()

        color = [0.3, 0.4, 0.5]

        self.shader = Shaders(vertices,
                              vertex_shader_source,
                              fragment_shader_source,
                              color)

    def update_vertices(self, displacements: np.ndarray = np.array([0.0, 0.0])):
        """
        参数：
            displacements: (2,) array, [lower, upper] displacements.
        """
        zs = displacements * self.screen_scale

        upper_location = self.upper_position + zs[1]
        lower_location = self.lower_position + zs[0]

        center_x = self.damper_center
        center_y = (upper_location + lower_location) / 2

        vertices = []
        vertices.extend([center_x, lower_location])
        vertices.extend([center_x, center_y - self.damper_height / 2])

        vertices.extend([center_x - self.damper_width/2,
                        center_y - self.damper_height / 2])
        vertices.extend([center_x + self.damper_width/2,
                        center_y - self.damper_height / 2])

        vertices.extend([center_x - self.damper_width/2,
                        center_y + self.damper_height/2])
        vertices.extend([center_x + self.damper_width/2,
                        center_y + self.damper_height/2])

        vertices.extend([center_x - self.damper_width/2,
                        center_y - self.damper_height / 2])
        vertices.extend([center_x - self.damper_width/2,
                        center_y + self.damper_height / 2])

        vertices.extend([center_x + self.damper_width/2,
                        center_y - self.damper_height / 2])
        vertices.extend([center_x + self.damper_width/2,
                        center_y + self.damper_height / 2])

        vertices.extend([center_x, center_y])
        vertices.extend([center_x, upper_location])

        return np.array(vertices, dtype=np.float32)

    def draw(self, displacement: float = 0.0):
        """
        更新阻尼的位置，并使用 Shaders 类进行绘制。
        """

        vertices = self.update_vertices(displacement)
        self.shader.update_vertices(vertices)
        self.shader.draw(GL_LINES, 0, len(vertices) // 2)


class PartActuator():
    def __init__(self, screen_scale: float = 40.0):
        self.screen_scale = screen_scale

        self.actuator_center = 15  # x position
        self.actuator_radius = 6

        self.lower_position = 0.0  # = PartBase.position
        self.upper_position = 60.0  # = PartMass.position

        vertices = self.update_vertices()

        color = [0.3, 0.4, 0.5]

        self.shader = Shaders(vertices,
                              vertex_shader_source,
                              fragment_shader_source,
                              color)

    def update_vertices(self, displacements: np.ndarray = np.array([0.0, 0.0])):
        """
        参数：
            displacements: (2,) array, [lower, upper] displacements.
        """
        zs = displacements * self.screen_scale

        upper_location = self.upper_position + zs[1]
        lower_location = self.lower_position + zs[0]

        center_x = self.actuator_center
        center_y = (upper_location + lower_location) / 2

        vertices = []

        # mode : GL_LINES (6 vertices)
        vertices.extend([center_x, lower_location])
        vertices.extend([center_x, center_y - self.actuator_radius])

        vertices.extend([center_x, center_y + self.actuator_radius])
        vertices.extend([center_x, upper_location])

        rho = self.actuator_radius
        vertices.extend([center_x + rho * np.cos(5/4*np.pi),
                        center_y + rho * np.sin(5/4*np.pi)])
        vertices.extend([center_x + rho * np.cos(1/4*np.pi),
                        center_y + rho * np.sin(1/4*np.pi)])

        # mode : GL_LINE_LOOP (30 vertices)

        circle_section_num = 30
        for id in range(circle_section_num):
            alpha = id/circle_section_num * 2 * np.pi
            x_actuator = center_x + rho * np.cos(alpha)
            y_actuator = center_y + rho * np.sin(alpha)
            vertices.extend([x_actuator, y_actuator])

        # mode : GL_LINE_LOOP (3 vertices)

        arrow_head_x = center_x + rho * np.cos(1/4*np.pi)
        arrow_head_y = center_y + rho * np.sin(1/4*np.pi)
        vertices.extend([arrow_head_x, arrow_head_y])

        angle = 5 / 4 * np.pi - 10 * np.pi / 180
        x_actuator = arrow_head_x + self.actuator_radius * np.cos(angle)
        y_actuator = arrow_head_y + self.actuator_radius * np.sin(angle)
        vertices.extend([x_actuator, y_actuator])

        angle = 5 / 4 * np.pi + 10 * np.pi / 180
        x_actuator = arrow_head_x + self.actuator_radius * np.cos(angle)
        y_actuator = arrow_head_y + self.actuator_radius * np.sin(angle)
        vertices.extend([x_actuator, y_actuator])

        return np.array(vertices, dtype=np.float32)

    def draw(self, displacement: float = 0.0):
        """
        更新驱动器的位置，并使用 Shaders 类进行绘制。
        """

        vertices = self.update_vertices(displacement)
        self.shader.update_vertices(vertices)
        self.shader.draw(GL_LINES, 0, 6)
        self.shader.draw(GL_LINE_LOOP, 6, 30)
        self.shader.draw(GL_LINE_LOOP, 36, 3)


class MassSpringDamperGL(QOpenGLWidget):
    status_message = Signal(str)  # signal to update status

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(400, 400)
        # 视觉比例：把位移（米）映射到屏幕（单位坐标）
        self.meter_to_screen = 40  # 1 m -> 40 px（在内部用归一化再缩放）

        # initialization
        self.mass = PartMass(self.meter_to_screen)
        self.mass_shader = Shaders(
            self.mass.vertices, vertex_shader_source, fragment_shader_source, self.mass.bgcolor)

        self.base = PartBase(self.meter_to_screen)
        self.base_shader = Shaders(
            self.base.vertices, vertex_shader_source, fragment_shader_source, self.base.bgcolor)

        self.damper = PartDamper()

    def initializeGL(self):

        glClearColor(0.08, 0.09, 0.11, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.mass_shader.setup()

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

        self.mass_shader.update_vertices(self.mass.update_vertices(x_mass))
        self.mass_shader.draw(GL_LINE_LOOP)

        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        # 绘制质量块
        glColor3f(0.5, 0.5, 0.6)
        glLineWidth(2.0)

        # 绘制弹簧
        glColor3f(0.8, 0.4, 0.1)  # Orange-brown
        glLineWidth(2.0)


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
