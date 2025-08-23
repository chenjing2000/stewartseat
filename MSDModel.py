
from OpenGL.GL import *
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Signal

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


class MassSpringDamperGLX(QOpenGLWidget):
    status_message = Signal(str)  # signal to update status

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(400, 400)
        # 视觉比例：把位移（米）映射到屏幕（单位坐标）
        self.meter_to_screen = 40  # 1 m -> 40 px（在内部用归一化再缩放）

        # positions
        self.mass_position_init = 60
        self.mass_motion = 0.0  # state or response
        self.base_position_init = 0
        self.base_motion = 0.0  # excitation or external input

        # joints geometry
        self.joints_position = np.array([-15, 0, 15])

        # base geometry
        self.base_width = 50

        # mass geometry
        self.mass_height = 20
        self.mass_width = 40

        # spring geometry
        self.spring_coils = 11
        self.spring_width = 10

        # damper geometry
        self.damper_height = 20
        self.damper_width = 10
        self.damper_thickness = 10

        # actuator geometry
        self.actuator_radius = 6

        self.update_positions(0, 0)

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
