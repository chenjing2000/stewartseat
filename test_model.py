
import numpy as np
from stewart_model import BodyModel, StewartModel
import matplotlib.pyplot as plt
from loguru import logger


if __name__ == "__main__":

    body = BodyModel()

    # 信号参数
    frequency = 1.  # 频率(Hz)
    amplitude = 0.1  # 振幅
    dt = 0.02  # 采样周期(秒)
    num_samples = 1000  # 采样点数

    # 使用采样周期生成时间轴
    t = np.arange(0, num_samples) * dt

    # 生成虚拟的座椅侧倾或俯仰信号
    # 实际中，座椅侧倾或俯仰信号由实验数据给出，替换掉这个数据即可
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    seat_motion = np.vstack((np.zeros(len(t), dtype=float),
                            np.zeros(len(t), dtype=float),
                            0.1*signal,
                            0.1*signal,
                            0.2*signal,
                            np.zeros(len(t), dtype=float))).T
    # seat_motion[0, [0, 1, 5]] = np.array([0.1, -0.1, -0.01])

    butt_motion_series = np.zeros((num_samples, 3), dtype=float)
    butt_velocity_series = np.zeros((num_samples, 3), dtype=float)
    butt_acceleration_series = np.zeros((num_samples, 3), dtype=float)

    for id in range(num_samples):
        body._body_dynamics(seat_motion[id], np.zeros(2, dtype=float), dt)
        butt_motion_series[id] = body.butt_motion
        butt_velocity_series[id] = body.butt_velocity
        butt_acceleration_series[id] = body.butt_acceleration

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(t, butt_acceleration_series)

    plt.subplot(3, 1, 2)
    plt.plot(t, butt_velocity_series)

    plt.subplot(3, 1, 3)
    plt.plot(t, butt_motion_series)

    plt.show()
