
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
