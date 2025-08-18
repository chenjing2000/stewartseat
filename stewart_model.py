# 本程序为本人思考之结晶，不能给任何人，尤其是公司！！！
# 座椅悬架是弹簧阻尼器，在底座中心初始坐标系下建模。
import numpy as np
from loguru import logger

# 导入初始化配置 (请确保 'init.py' 文件存在且包含这些常量)
from init import (BASE_RADIUS, BASE_JOINTS_ANGLE, SEAT_RADIUS,
                  SEAT_JOINTS_ANGLE, SEAT_CENTER_HEIGHT, SEAT_MASS, SEAT_PARAMS,
                  SEAT_INERTIA, STROKE_LIMITS, BODY_MASS,
                  CUSHION_THICKNESS, BODY_CENTER_HEIGHT, BODY_INERTIA,
                  CUSHION_RADIUS, CUSHION_PARAMS, CUSHION_JOINTS_ANGLE,
                  MOTION_LIMITS, MAX_ACCELERATIONS
                  )


class BodyModel:

    def __init__(self, body_mass: float = BODY_MASS,
                 cushion_thickness: float = CUSHION_THICKNESS,
                 body_center_height: float = BODY_CENTER_HEIGHT,
                 body_inertia: list[list[float]] = BODY_INERTIA,
                 cushion_params: list[float] = CUSHION_PARAMS,
                 ):
        """
        研究人体动力学时，将车辆坐标系平移到座椅（运动时）中心为基准坐标系，
        从而简化人体坐标的计算。
        值得注意的是，座椅运动时有 6 个自由度发生了位移，但车辆坐标系并不随
        座椅转动 (phi, theta, psi) 而转动，只随座椅平动 (x, y, z) 而平
        动，故没有产生科氏力。若需计算人体相对于底座中心处全局坐标系的位移，
        平动位移只需加上座椅中心处的位移即可，转动位移不需要变换，因为这两
        个坐标系都是车辆坐标系的平移，并未发生相对转动。
        座椅之上有坐垫，坐垫之上是人体。将坐垫考虑为 1 个垂直于座椅的悬架，
        这个悬架包含垂向、侧倾（等效为卷簧）、俯仰（等效为卷簧）三个方向的弹
        簧减振器。
        设人体屁股中心处 x, y 坐标与座椅中心处 x, y 坐标相同，且人体的 yaw
        姿态与座椅的 yaw 姿态相同。人体相对于座椅只有 z, roll, pitch 三个
        方向的位移，所以人体的姿态用 3 个坐标 (z, roll, pitch) 来表示。
        """

        self.body_mass = body_mass
        self.cushion_thickness = cushion_thickness
        self.body_center_height = body_center_height
        self.body_inertia = np.array(body_inertia, dtype=float)

        self.cushion_params = np.array(cushion_params)

        self.body_center_local = np.array(
            [0, 0, self.body_center_height], dtype=float)

        self.butt_inertia = self._get_inertia_matrix(self.body_mass,
                                                     self.body_inertia,
                                                     self.body_center_local)

        # 记录上一时刻的坐垫等效悬架变形量
        self.strokes_previous = np.zeros(3, dtype=float)

        # 在（运动的）座椅中心坐标系下，人体质心相对于质心初始位置的位移
        self.butt_motion = np.zeros(3, dtype=float)
        self.butt_velocity = np.zeros(3, dtype=float)
        self.butt_acceleration = np.zeros(3, dtype=float)

    @staticmethod
    def _get_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
        """
        根据给定的欧拉角 [roll, pitch, yaw] (弧度) 计算旋转矩阵。
        采用 XYZ 顺序旋转 (Roll-Pitch-Yaw)。

        参数:
            euler_angles: 包含 [roll, pitch, yaw] 的数组 (弧度)。

        返回:
            np.ndarray: (3, 3) 旋转矩阵。
        """

        roll, pitch, yaw = euler_angles

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [+np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, +np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, +np.sin(roll), np.cos(roll)]
        ])

        # 旋转顺序: X (Roll) -> Y (Pitch) -> Z (Yaw)
        return Rz @ Ry @ Rx

    @staticmethod
    def _get_joints_local_position(radius, angles):
        """
        根据铰点半径与角度分布计算其在局部平面下的位置
        """
        logger.debug(f"radius = {radius}, angles = {angles}")
        local_position = np.array([[radius * np.cos(x), radius * np.sin(x), 0.0]
                                  for x in angles])
        return local_position

    @staticmethod
    def _get_inertia_matrix(mass: float, inertia: np.ndarray, radius: np.ndarray) -> np.ndarray:
        """
        根据平移定理，计算质心不在坐标系原点时的转动惯量矩阵。
        """

        inertia_offset = inertia.copy()
        inertia_offset += mass * \
            (radius @ radius.T * np.eye(3) - np.outer(radius, radius))

        return inertia_offset

    def _body_dynamics(self, seat_motion: np.ndarray, external_acceleration: np.ndarray, dt: float):
        """
        目的：当输入为座椅运动位移与加速度时，计算人体的动态响应，并返回坐垫对
        座椅的作用力与力矩。

        参数：
            seat_motion: (3,) 数组，在全局坐标系下，座椅中心相对于其初始位置
                的位移。
            external_acceleration = [ax, ay]: (2,) 数组，ax 与 ay 表示由于
                车体转向、加/减速等运动导致人体受到的加速度，这些加速度都是水
                平的，故只有纵向与侧向两个分量。惯性载荷的 z 向分量为零，故仅
                考虑 x, y。
            dt: 采样时间。

        返回：
            force_on_seat: 座椅中心处受到的来自坐垫的力，仅包含 
                (z, phi, theta) 方向分量；
            external_forces: 人体对座椅在 x, y 方向的惯性力。
        """

        forces_on_butt, external_forces = \
            self._get_cushion_forces(seat_motion, external_acceleration, dt)

        self._get_butt_acceleration(forces_on_butt)

        self.butt_velocity += self.butt_acceleration*dt
        # 在（运动时）座椅中心处坐标系下，人体质心相对于质心初始位置的位移
        self.butt_motion += self.butt_velocity*dt

        forces_on_seat = -forces_on_butt

        return forces_on_seat, external_forces

    def _get_cushion_forces(self, seat_motion: np.ndarray, external_acceleration: np.ndarray, dt: float):
        """
        根据座椅状态与人体屁股状态，计算座垫等效悬架对人体与座椅的作用力。

        参数：
            seat_motion: (3,) 数组，在全局坐标系下，座椅中心相对于其初始位置
                的位移。
            external_acceleration = [ax, ay]: (2,) 数组，ax 与 ay 表示由于
                车体转向、加/减速等运动导致人体受到的加速度，这些加速度都是水
                平的，故只有纵向与侧向两个分量。设外部施加的 z 向加速度为零。
            dt: 采样时间。

        返回：
            forces_on_butt: (3,) 数组，人体屁股中心处受到的（在 z, phi, 
                theta）方向的力与力矩；
            external_forces: (3,) 数组，是外部加速度产生的惯性力（作用在质
                心），而 forces_on_butt 包含了将这个力转移到人体屁股中心后产
                生的力矩。同时，external_forces 还需要返回，以便在更高层的
                仿真代码中将它们施加到座椅模型上。
        """

        mb = self.body_mass
        cb, kb, cr, kr, cp, kp = self.cushion_params

        # 1. 计算垂向、侧倾与俯仰方向的作用力
        suspensions_stroke = self.butt_motion - seat_motion
        suspensions_velocity = (suspensions_stroke - self.strokes_previous)/dt

        forces_on_butt = np.array([kb, kr, kp]) * suspensions_stroke + \
            np.array([cb, cr, cp]) * suspensions_velocity

        self.strokes_previous = suspensions_stroke
        # 2. 计算外部 x, y 方向加速度对人体的力矩（人体外部的力传递至座椅）
        external_forces = mb * external_acceleration

        euler_angles = np.append(self.butt_motion[1:], 0)
        Rb = self._get_rotation_matrix(euler_angles)
        body_center_local_actual = self.body_center_local @ Rb.T

        # 3. 将质心处所受载荷传递至人体屁股中心处
        forces_on_butt[1:] += np.cross(body_center_local_actual,
                                       np.append(external_forces, 0))[:2]

        return forces_on_butt, external_forces

    def _get_butt_acceleration(self, forces_on_butt: np.ndarray):
        """
        根据屁股中心的受到的载荷，求屁股中心处在 (z, phi, theta) 方向的加速度。

        参数：
            forces_on_butt: (3,) 数组，仅包含 (z, phi, theta) 方向分量。
            dt: 采样时间。
        """

        mb = self.body_mass

        # 计算人体平动与转动的加速度
        # 注意：forces_on_butt 前须有负号
        butt_acceleration = np.zeros(3, dtype=float)
        butt_acceleration[0] = -forces_on_butt[0] / mb
        butt_acceleration[1:] = np.linalg.solve(
            self.butt_inertia[:2, :2], -forces_on_butt[1:])

        self.butt_acceleration = butt_acceleration


class StewartModel(BodyModel):

    def __init__(self, base_radius: float = BASE_RADIUS,
                 base_joints_angle: list[float] = BASE_JOINTS_ANGLE,
                 seat_radius: float = SEAT_RADIUS,
                 seat_joints_angle: list[float] = SEAT_JOINTS_ANGLE,
                 seat_center_height: float = SEAT_CENTER_HEIGHT,
                 seat_mass: float = SEAT_MASS,
                 seat_params: list[float] = SEAT_PARAMS,
                 seat_inertia: list[list[float]] = SEAT_INERTIA,
                 stroke_limits: list[float] = STROKE_LIMITS,
                 motion_limits: list[list[float]] = MOTION_LIMITS,
                 max_accelerations: list[float] = MAX_ACCELERATIONS,
                 body_mass: float = BODY_MASS,
                 body_inertia: list[float] = BODY_INERTIA
                 ):
        """
        假设：全局坐标系是将车辆坐标系平移至底座中心点的坐标系，与初始状态下的
        底座坐标系重合，当底座发生振动时，底座坐标系随底座中心运动，但全局坐标
        系仍保持在初始位置。相对于全局坐标系，底座只会发生 z, roll, pitch 向
        振动，不会发生 x, y, psi 方向运动，后三个方向的运动，将全局转化为加速
        度反向施加到座椅与人体上。也就是说，视角总是处在底座中心处，看不到地面
        坐标系上车体的运动，只能感知到车辆在水平面内受到的加速度。垂向、侧倾、
        俯仰的位移是假设可观测的（其中，z 实际上是不可观测的），加速度是可观测
        的，但地面坐标系下的车辆位移或横摆是不可观测的。

        座椅、底座的中心坐标系：这两个坐标系是为了方便计算铰接点位置而存在的，
        x-y-z 轴与车辆坐标系对齐，跟随座椅或底座的中心点运动。初始状态底座中心
        坐标系与全局坐标系重合，底座振动时 z 值是相对于全局坐标系变化的；座椅
        中心坐标系的 x-y-z 都是相对于定义在底座中心处的全局坐标系的。

        坐标 (x, y, z, phi, theta, psi) 均相对于刚定义的全局坐标系，都不是相
        对于地面绝对坐标系。其中，默认全局坐标系与底座初始状态时的底座中心坐标
        系重合，且底座不会相对于该坐标系发生 (x, y, psi) 移动，即底座总是有 
        (x = 0, y = 0, psi = 0)（即使在车辆运动时），但底座中心的其它坐标是可
        以运动的。

        不过，座椅是可以相对于全局坐标系发生运动的，即座椅的 (x, y, z, ...) 均
        是相对于全局坐标系的。也就是说，(x, y) 是指座椅相对于底座中心的水平运
        运，psi 是座椅相对于全局坐标系的横摆转动，也是相对于底座坐标系的 z 轴
        转动。

        这样做的目的，是将 (x, y, psi) 从地面绝对坐标系中脱离出来，变成相对于全
        局坐标系的值，从而变成传感器可测量的。然后，将车辆受到的侧向加速度施加到
        人体与座椅之上，从而实现在局部坐标系中对全局运动的模拟。

        初始化 Stewart 模型。

        参数:
            base_radius (float): 底座（下板）连接点的半径 (m)。
            base_joints_angle (list): 底座连接点在 XY 平面上的角度 (度)。
            seat_radius (float): 座椅（上板）连接点的半径 (m)。
            seat_joints_angle (list): 座椅连接点在 XY 平面上的角度 (度)。
            seat_center_height (float): 座椅中心相对于全局坐标系的垂直高度 (m)。
            stroke_limits (list): 悬架的行程限制 [min_stroke, max_stroke] (m)。
            body_center_height (float): 集中质量质心相对于座椅中心的高度 (m)。
        """

        self.base_radius = base_radius
        self.base_joints_angle = np.array(base_joints_angle, dtype=float)

        # 座椅参数
        self.seat_radius = seat_radius
        self.seat_joints_angle = np.array(seat_joints_angle, dtype=float)

        self.seat_center_height = seat_center_height
        self.seat_mass = seat_mass
        self.seat_params = np.array(seat_params)
        self.seat_inertia = np.array(seat_inertia)

        self.base_joints_local = BodyModel._get_joints_local_position(
            self.base_radius, self.base_joints_angle)
        # base_motion is used as external input, that's why it is not stated here.
        # self.base_joints_actual = self.base_joints_local + base_motion[:3]
        self.base_joints_actual_previous = self.base_joints_local

        # 全局坐标系是车辆坐标系的副本，只不过是将后者的 x-y-psi 跟随底座中心运动
        self.seat_center_local = np.array(
            [0, 0, self.seat_center_height], dtype=float)  # 相对于全局坐标系的位置

        self.seat_joints_local = BodyModel._get_joints_local_position(
            self.seat_radius, self.seat_joints_angle)
        self.seat_joints_actual_previous = self.seat_joints_local + \
            np.array([0, 0, self.seat_center_height])

        # 全局坐标系下，座椅中心相对于其初始位置的位移量
        self.seat_motion = np.zeros(3, dtype=float)
        self.seat_velocity = np.zeros(3, dtype=float)
        self.seat_acceleration = np.zeros(3, dtype=float)

        # 默认 6 个悬架的初始长度都是相等的。
        suspensions_vector = self.seat_center_local + \
            self.seat_joints_local - self.base_joints_local

        suspensions_length = np.linalg.norm(suspensions_vector, axis=1)
        self.suspensions_length_nominal = suspensions_length
        self.suspensions_direction = suspensions_vector / \
            suspensions_length[:, np.newaxis]
        self.suspensions_stroke = np.zeros(6, dtype=float)
        self.suspensions_velocity = np.zeros(6, dtype=float)

    def _stewart_seat_dynamics(self, base_motion: np.ndarray, forces_on_seat_from_body: np.ndarray, dt: float):
        """
        根据座椅和底座的位姿信息，计算悬架变形量及端点间相对速度，为下一步计算
        悬架对座椅的支撑力打下基础。

        参数:
            base_motion (np.ndarray): (3,) 数组，底座在全局坐标系下的  
                (z, roll, pitch) 位移量，因为 (x, y, psi) 是假设为零的，故
                忽略。
            external_acceleration = [ax, ay]: (2,) 数组，表示由于车体转向、
                加减速等运动导致人体受到的加速度，这些加速度都是水平的，故只有
                纵向与侧向两个分量。
            dt: 采样周期。
        """

        forces_on_seat = self._get_suspensions_forces(base_motion, dt)

        # 注意符号
        forces_on_seat += -forces_on_seat_from_body

        # 7. 计算座椅平动与转动响应
        self.seat_acceleration[0] = forces_on_seat[0] / self.seat_mass

        self.seat_acceleration[1:] = np.linalg.solve(
            self.seat_inertia[:2, :2], forces_on_seat[1:])

        # 更新座椅中心的位置
        self.seat_velocity += self.seat_acceleration*dt
        self.seat_motion += self.seat_velocity*dt

    def _get_suspensions_forces(self, base_motion: np.ndarray, dt: float):
        """
        根据座椅和底座的位姿信息，计算座椅悬架对座椅的作用力。

        参数:
            base_motion: (3,) 数组，底座在全局坐标系下的位移量，由于全局坐标
            系固定在底座上，其 x, y, yaw 分量均为零，故其仅含 z, roll, pitch
            分量。
            dt: 采样周期。

        返回：
            forces_on_seat: (3,) 数组，座椅中心受到的力与力矩。
        """

        seat_rotation = np.append(self.seat_motion[1:], 0)
        base_rotation = np.append(base_motion[1:], 0)  # 底座的 roll-pitch-yaw

        # 1. 获取座椅铰点的位置
        Rs = BodyModel._get_rotation_matrix(seat_rotation)

        seat_joints_actual = self.seat_joints_local @ Rs.T
        seat_joints_actual[:, 2] += self.seat_motion[0] + \
            self.seat_center_height

        seat_joints_velocity = (
            seat_joints_actual - self.seat_joints_actual_previous) / dt
        self.seat_joints_actual_previous = seat_joints_actual

        # 2. 获取底座铰点的位置
        Rb = BodyModel._get_rotation_matrix(base_rotation)

        base_joints_actual = self.base_joints_local @ Rb.T
        base_joints_actual[:, 2] += base_motion[0]

        base_joints_velocity = (
            base_joints_actual - self.base_joints_actual_previous) / dt
        self.base_joints_actual_previous = base_joints_actual

        relative_joints_velocity = seat_joints_velocity - base_joints_velocity

        # 3. 计算悬架长度
        suspensions_vector = seat_joints_actual - base_joints_actual
        suspensions_length = np.linalg.norm(suspensions_vector, axis=1)
        logger.debug(f"suspension length : {suspensions_length}")
        suspensions_direction = suspensions_vector / \
            suspensions_length[:, np.newaxis]

        # 4. 获取悬架伸长量和端点间相对速度
        self.suspensions_stroke = suspensions_length - \
            self.suspensions_length_nominal

        # 6. 计算座椅受到的力
        cs, ks = self.seat_params

        # 计算座椅悬架力作用点相对于（运动时）座椅中心的位置矢量
        momentum_radius = self.seat_joints_local @ Rs.T

        suspensions_force = np.zeros(6, dtype=float)
        suspensions_force_vector = np.zeros((6, 3), dtype=float)
        forces_on_seat = np.zeros(3, dtype=float)

        for id in range(6):
            self.suspensions_velocity[id] = np.dot(
                suspensions_direction[id], relative_joints_velocity[id])
            suspensions_force[id] = ks*self.suspensions_stroke[id] + \
                cs*self.suspensions_velocity[id]
            suspensions_force_vector[id] = suspensions_force[id] * \
                suspensions_direction[id]

            # 悬架伸长时对座椅有向下的力，故须增加负号
            forces_on_seat[0] += -suspensions_force_vector[id, 2]
            forces_on_seat[1:] += np.cross(
                momentum_radius[id], -suspensions_force_vector[id])[:2]

        return forces_on_seat


if __name__ == "__mainx__":

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
    seat_motion = np.vstack((0.1*signal,
                            0.1*signal,
                            0.2*signal)).T
    # seat_motion[0, [0, 1, 5]] = np.array([0.1, -0.1, -0.01])

    butt_motion_series = np.zeros((num_samples, 3), dtype=float)
    butt_velocity_series = np.zeros((num_samples, 3), dtype=float)
    butt_acceleration_series = np.zeros((num_samples, 3), dtype=float)

    for id in range(num_samples):
        forces_on_seat, external_forces = body._body_dynamics(
            seat_motion[id], np.zeros(2, dtype=float), dt)
        butt_motion_series[id] = body.butt_motion
        butt_velocity_series[id] = body.butt_velocity
        butt_acceleration_series[id] = body.butt_acceleration

    plt.figure()

    plt.subplot(2, 3, 1)
    plt.plot(t, seat_motion[:, 0], 'r-', label='z seat')
    plt.plot(t, butt_motion_series[:, 0], 'g-', label='z body')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(t, seat_motion[:, 1], 'r-', label='roll seat')
    plt.plot(t, butt_motion_series[:, 1], 'g-', label='roll body')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(t, seat_motion[:, 2], 'r-', label='pitch seat')
    plt.plot(t, butt_motion_series[:, 2], 'g-', label='pitch body')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(t, butt_motion_series)

    plt.subplot(2, 3, 5)
    plt.plot(t, butt_velocity_series)

    plt.subplot(2, 3, 6)
    plt.plot(t, butt_acceleration_series)

    plt.show()


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # 信号参数
    frequency = 1.  # 频率(Hz)
    amplitude = 0.02  # 振幅
    dt = 0.02  # 采样周期(秒)
    num_samples = 3000  # 采样点数

    # 使用采样周期生成时间轴
    t = np.arange(0, num_samples) * dt

    # 生成虚拟的座椅侧倾或俯仰信号
    # 实际中，座椅侧倾或俯仰信号由实验数据给出，替换掉这个数据即可
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    base_motion_test = np.vstack((signal,
                                  signal,
                                  signal)).T

    logger.debug(f"base shape = {base_motion_test.shape}")

    # 初始化一个对象，用于座椅侧倾或俯仰控制测试
    hexa = StewartModel()
    body = BodyModel()

    seat_motion_series = np.zeros((num_samples, 3), dtype=float)
    seat_velocity_series = np.zeros((num_samples, 3), dtype=float)
    seat_acceleration_series = np.zeros((num_samples, 3), dtype=float)

    forces = np.zeros((len(signal), 3))
    moments = np.zeros((len(signal), 3))

    butt_motion_series = np.zeros((num_samples, 3), dtype=float)
    butt_velocity_series = np.zeros((num_samples, 3), dtype=float)
    butt_acceleration_series = np.zeros((num_samples, 3), dtype=float)

    external_acceleration = np.array([0, 0])
    ft = np.zeros(3, dtype=float)

    for id in range(len(base_motion_test)-1):
        forces_on_seat, external_forces = body._body_dynamics(
            seat_motion_series[id-1], np.zeros(2, dtype=float), dt)
        butt_motion_series[id] = body.butt_motion
        butt_velocity_series[id] = body.butt_velocity
        butt_acceleration_series[id] = body.butt_acceleration

        hexa._stewart_seat_dynamics(base_motion_test[id], forces_on_seat, dt)
        seat_motion_series[id] = hexa.seat_motion

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 3, 1)
    plt.plot(t, base_motion_test[:, 0], 'b-', label='z base')
    plt.plot(t, seat_motion_series[:, 0], 'r-', label='z seat')
    plt.plot(t, butt_motion_series[:, 0], 'g-', label='z body')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(t, base_motion_test[:, 1], 'b-', label='roll base')
    plt.plot(t, seat_motion_series[:, 1], 'r-', label='roll seat')
    plt.plot(t, butt_motion_series[:, 1], 'g-', label='roll body')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(t, base_motion_test[:, 2], 'b-', label='pitch base')
    plt.plot(t, seat_motion_series[:, 2], 'r-', label='pitch seat')
    plt.plot(t, butt_motion_series[:, 2], 'g-', label='pitch body')
    plt.legend()

    plt.legend()
    plt.grid(True)
    plt.show()
