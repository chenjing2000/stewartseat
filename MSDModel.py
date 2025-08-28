
from OpenGL.GL.shaders import *
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *

import ctypes

import numpy as np


class MSDPlant:
    def __init__(self, mass: float = 10, damping: float = 40, stiffness: float = 1000, dt: float = 0.001):
        self.m = mass
        self.c = damping
        self.k = stiffness
        self.dt = dt

        self.x = 0.0  # position
        self.v = 0.0  # velocity
        self.a = 0.0  # acceleration

    def update_euler(self, excitation: np.ndarray, control_force: float):

        xa, xe = excitation
        ve = (xe - xa) / self.dt

        uk = control_force
        force = uk - self.k*(self.x - xe) - self.c*(self.v - ve)

        self.a = force / self.m
        self.v += self.a * self.dt
        self.x += self.v * self.dt

    def reset(self):
        self.x = 0.0
        self.v = 0.0
        self.a = 0.0

    def derivative(self, state: np.ndarray, input: np.ndarray, uk: float):
        x, v = state
        xe, ve = input

        force = uk - self.k*(x - xe) - self.c*(v - ve)
        a = force / self.m
        return v, a

    def update(self, excitation: np.ndarray, control_force: float):

        xa, xe = excitation
        ve = (xe - xa) / self.dt

        dt = self.dt
        uk = control_force

        x, v = self.x, self.v
        f = self.derivative
        k1v, k1a = f((x, v), [xe, ve], uk)
        k2v, k2a = f((x + 0.5 * dt * k1v, v + 0.5 * dt * k1a), [xe, ve], uk)
        k3v, k3a = f((x + 0.5 * dt * k2v, v + 0.5 * dt * k2a), [xe, ve], uk)
        k4v, k4a = f((x + dt * k3v, v + dt * k3a), [xe, ve], uk)

        self.x += (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        self.v += (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)

        force = uk - self.k*(self.x - xe) - self.c*(self.v - ve)
        self.a = force / self.m


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral = 0.0
        self.error_prev = 0.0

    def calculate_force(self, error: float):
        # 抗积分饱和：限制积分项
        self.integral += error * self.dt
        self.integral = max(min(self.integral, 1e4), -1e4)
        derivative = (error - self.error_prev) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.error_prev = error
        return output


vertex_shader_source = """
#version 330 core
layout (location = 0) in vec2 aPos;
uniform vec2 uWindowSize;  // 窗口大小
void main()
{
    // 将像素坐标转换到 NDC
    float x = (aPos.x / uWindowSize.x) * 4.0 - 0.0;
    float y = (aPos.y / uWindowSize.y) * 4.0 - 0.5;

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

    def setup(self):
        """
        初始化着色器程序、VAO/VBO 参数。
        初始化 uWindowSize 与 uColor 的存储位置。
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

    def draw(self, mode: int, first: int, count: int, line_width: float = 3.0):
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
        glLineWidth(line_width)

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

        color = [0.8, 0.4, 0.4]

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
        self.shader.draw(GL_TRIANGLE_FAN, 0, 4)


class PartBase():
    def __init__(self, screen_scale: float = 40.0):
        self.screen_scale = screen_scale

        self.width = 50
        self.position = 0

        vertices = self.update_vertices()

        color = [0.3, 0.5, 0.6]

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


class PartDatums():
    def __init__(self, screen_scale: float = 40.0):
        self.screen_scale = screen_scale

        self.width = 80
        self.positions = [0, 70]  # 参考线的位置
        self.section_num = 20

        vertices = self.update_vertices()

        color = [0.7, 0.3, 0.3]

        self.shader = Shaders(vertices,
                              vertex_shader_source,
                              fragment_shader_source,
                              color)

    def update_vertices(self):

        x_step = self.width / (self.section_num + 1)
        vertices = []
        for kd in range(2):
            x_datum = -self.width / 2
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
        self.shader.draw(GL_LINES, 0, self.section_num + 1, 1.5)
        self.shader.draw(GL_LINES,
                         self.section_num + 1, self.section_num + 1, 1.5)


class PartSpring():
    def __init__(self, screen_scale: float = 40.0):
        self.screen_scale = screen_scale

        self.spring_center = 0  # x position
        self.spring_width = 10
        self.spring_coils = 11

        self.lower_position = 0.0  # = PartBase.position
        self.upper_position = 60.0  # = PartMass.position

        vertices = self.update_vertices()

        color = [0.7, 0.8, 0.6]

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
                       2 * spring_shank) / (self.spring_coils)

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

    def draw(self, displacements: np.ndarray = np.array([0.0, 0.0])):
        """
        更新弹簧的位置，并使用 Shaders 类进行绘制。
        """

        vertices = self.update_vertices(displacements)
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

        color = [0.8, 0.8, 0.5]

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
                        center_y])
        vertices.extend([center_x + self.damper_width/2,
                        center_y])

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

    def draw(self, displacements: np.ndarray = np.array([0.0, 0.0])):
        """
        更新阻尼的位置，并使用 Shaders 类进行绘制。
        """

        vertices = self.update_vertices(displacements)
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

        color = [0.9, 0.6, 0.2]

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

    def draw(self, displacements: np.ndarray = np.array([0.0, 0.0])):
        """
        更新驱动器的位置，并使用 Shaders 类进行绘制。
        """

        vertices = self.update_vertices(displacements)
        self.shader.update_vertices(vertices)
        self.shader.draw(GL_LINES, 0, 6)
        self.shader.draw(GL_LINE_LOOP, 6, 30)
        self.shader.draw(GL_LINE_LOOP, 36, 3)


class MassSpringDamperGL(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(400, 400)
        # 视觉比例：把位移（米）映射到屏幕（单位坐标）
        self.meter_to_screen = 40  # 1 m -> 40 px（在内部用归一化再缩放）

        self.datums = PartDatums(self.meter_to_screen)
        self.mass = PartMass(self.meter_to_screen)
        self.damper = PartDamper(self.meter_to_screen)
        self.spring = PartSpring(self.meter_to_screen)
        self.actuator = PartActuator(self.meter_to_screen)
        self.base = PartBase(self.meter_to_screen)

        # 记录位移
        self.mass_motion = 0.0
        self.base_motion = 0.0

    def initializeGL(self):
        # 打印 OpenGL 版本
        # print("OpenGL Version:", glGetString(GL_VERSION).decode())

        glClearColor(0.3, 0.3, 0.4, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 调用部件的 setup 方法
        self.datums.shader.setup()
        self.mass.shader.setup()
        self.damper.shader.setup()
        self.spring.shader.setup()
        self.actuator.shader.setup()
        self.base.shader.setup()

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

        window_size = [width, height]
        self.datums.shader.set_window_size(window_size)
        self.mass.shader.set_window_size(window_size)
        self.damper.shader.set_window_size(window_size)
        self.spring.shader.set_window_size(window_size)
        self.actuator.shader.set_window_size(window_size)
        self.base.shader.set_window_size(window_size)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        self.datums.draw()
        self.mass.draw(self.mass_motion)
        self.damper.draw(np.array([self.base_motion, self.mass_motion]))
        self.spring.draw(np.array([self.base_motion, self.mass_motion]))
        self.actuator.draw(np.array([self.base_motion, self.mass_motion]))
        self.base.draw(self.base_motion)

        self.update()  # 告诉 PyQt 重绘窗口
