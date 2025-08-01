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
                 cushion_radius: float = CUSHION_RADIUS,
                 cushion_params: list[float] = CUSHION_PARAMS,
                 cushion_joints_angle: list[float] = CUSHION_JOINTS_ANGLE,
                 ):
        """
        研究人体动力学时，将车辆坐标系平移到座椅（运动时）中心为基准坐标系，
        从而简化人体坐标的计算。
        值得注意的是，座椅运动时有 6 个自由度发生了位移，但车辆坐标系并不随
        座椅转动 (phi, theta, psi) 而转动，只随座椅平动 (x, y, z) 而平
        动，故没有产生科氏力。若需计算人体相对于底座中心处全局坐标系的位移，
        平动位移只需加上座椅中心处的位移即可，转动位移不需要变换，因为这两
        个坐标系都是车辆坐标系的平移，并未发生相对转动。
        座椅之上有坐垫，坐垫之上是人体。将坐垫考虑为 6 个垂直于座椅的悬架，
        发生侧倾俯仰时，就不一定会垂直于人体屁股所在的平面了。
        """

        self.body_mass = body_mass
        self.cushion_thickness = cushion_thickness
        self.body_center_height = body_center_height
        self.body_inertia = np.array(body_inertia)
        self.cushion_radius = cushion_radius
        self.cushion_params = np.array(cushion_params)
        # 将坐垫等效为 6 根弹簧阻尼部件
        self.cushion_joints_angle = np.array(
            [np.radians(x) for x in cushion_joints_angle])

        self.cushion_seat_joints_local = self._get_joints_local_position(
            self.cushion_radius, self.cushion_joints_angle)

        self.buttocks_joints_local = self._get_joints_local_position(
            self.cushion_radius, self.cushion_joints_angle)

        self.body_to_buttocks_local = np.array(
            [0, 0, -self.body_center_height])
        self.seat_to_body_local = np.array(
            [0, 0, self.cushion_thickness + self.body_center_height])

        # 记录上一时刻的坐垫等效悬架变形量
        self.suspensions_stroke = np.zeros(6, dtype=float)
        self.suspensions_velocity = np.zeros(6, dtype=float)

        # 在（运动的）座椅中心坐标系下，人体质心相对于质心初始位置的位移
        self.body_motion = np.zeros(6, dtype=float)
        self.body_velocity = np.zeros(6, dtype=float)
        self.body_acceleration = np.zeros(6, dtype=float)

    @staticmethod
    def _get_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
        """
        根据给定的欧拉角 [roll, pitch, yaw] (弧度) 计算旋转矩阵。
        采用 XYZ 顺序旋转 (Roll-Pitch-Yaw)。

        参数:
            euler_angles (np.ndarray): 包含 [roll, pitch, yaw] 的 NumPy 数组 (弧度)。

        返回:
            np.ndarray: (3, 3) 旋转矩阵。
        """

        roll, pitch, yaw = euler_angles

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # 旋转顺序: X (Roll) -> Y (Pitch) -> Z (Yaw)
        return Rz @ Ry @ Rx

    @staticmethod
    def _get_joints_local_position(radius, angles):
        """
        根据铰点半径与角度分布计算其在局部平面下的位置
        """
        local_position = np.array([[radius*np.cos(x), radius*np.sin(x), 0.0]
                                  for x in angles])
        return local_position

    def _body_dynamics(self, seat_motion: np.ndarray, external_acceleration: np.ndarray, dt: float):
        """
        目的：当输入为座椅运动位移与加速度时，计算人体的动态响应，并返回坐垫对
        座椅的作用力与力矩。

        参数：
            seat_motion: (6,) 数组，在全局坐标系下，座椅中心相对于其初始位置的位
                移。
            external_acceleration = [ax, ay, 0]: (3,) 数组，ax 与 ay 表示由于
                车体转向、加/减速等运动导致人体受到的加速度，这些加速度都是水平的，
                故只有纵向与侧向两个分量。其第 3 个分量为零，是为了计算方便。
            dt: 采样时间。

        返回：
            force_on_seat: 座椅中心处受到的来自坐垫的力；
            momentum_on_seat: 座椅中心处受到的来自坐垫的力矩。

        注意：
            body_motion: 人体质心在（运动的）局部坐标系（车辆坐标系平移至座椅中
                心处坐标系）下，相对于质心初始位置的位移；也就是说，人体质心先相
                对于（运动的）座椅处车辆坐标系平移，人体再相对于人体质心旋转（旋
                转是绝对的）。理解这一点，对于求解屁股中心处位移非常重要。
            本函数采用 Euler-Forward 积分法进行数值计算。
        """

        # 1. 计算坐垫等效悬架在坐椅上的铰点的位置
        seat_translation = seat_motion[:3]  # 座椅平动位移 (全局)
        seat_rotation = seat_motion[3:]  # 座椅转动欧拉角 (全局)

        Rs = self._get_rotation_matrix(seat_rotation)
        # 计算坐垫下表面铰点（即座椅上的铰点）在（初始）座椅坐标系下的实际位置
        # 坐垫坐标相对于初始状态时座椅中心
        cushion_seat_joints_actual = seat_translation + \
            self.cushion_seat_joints_local @ Rs.T

        # 2. 计算坐垫等效悬架在屁股上的铰点的位置
        # body_motion 是人体质心相对于（运动时）座椅中心的位移
        body_translation = self.body_motion[:3]
        body_rotation = self.body_motion[3:]

        Rb = self._get_rotation_matrix(body_rotation)
        # 计算坐垫上表面铰点（即屁股下的铰点）在（初始）座椅坐标系下的实际位置
        buttocks_joints_actual = seat_translation + \
            self.seat_to_body_local + body_translation + \
            self.body_to_buttocks_local @ Rb.T + self.buttocks_joints_local @ Rb.T

        # 3. 计算坐垫等效悬架的变形量
        suspensions_length = np.linalg.norm(
            buttocks_joints_actual-cushion_seat_joints_actual, axis=1)

        suspensions_stroke_current = suspensions_length - self.cushion_thickness
        self.suspensions_velocity = (suspensions_stroke_current -
                                     self.suspensions_stroke) / dt
        self.suspensions_stroke = suspensions_stroke_current

        # 4. 计算坐垫等效悬架的方向向量
        suspensions_direction = (
            buttocks_joints_actual - cushion_seat_joints_actual)

        suspensions_direction /= suspensions_length[:, np.newaxis]

        # 5. 计算人体受到的力
        mb = self.body_mass
        cb, kb = self.cushion_params

        suspensions_force = kb*self.suspensions_stroke + cb*self.suspensions_velocity
        suspensions_force_vector = np.zeros((6, 3), dtype=float)

        # 计算坐垫悬架力作用点相对于（运动时）人体质心的位置矢量
        momentum_radius = self.body_to_buttocks_local @ Rb.T + \
            self.buttocks_joints_local @ Rb.T

        force_on_body = np.zeros(3, dtype=float)
        momentum_on_body = np.zeros(3, dtype=float)
        for i in range(6):
            suspensions_force_vector[i] = suspensions_force[i] * \
                suspensions_direction[i]
            # 将坐垫悬架作用力移动至人体质心处
            force_on_body += suspensions_force_vector[i]
            momentum_on_body += np.cross(
                momentum_radius[i], suspensions_force_vector[i])

        force_on_body += mb*external_acceleration  # 添加外部惯性力

        # 6. 计算人体平动与转动响应
        self.body_acceleration[:3] = force_on_body[:3] / mb

        self.body_acceleration[3:] = np.linalg.solve(
            self.body_inertia, momentum_on_body)
        self.body_velocity += self.body_acceleration*dt
        # 在（运动时）座椅中心处坐标系下，人体质心相对于质心初始位置的位移
        self.body_motion += self.body_velocity*dt

        # 7. 计算坐垫等效悬架对座椅中心的作用力
        momentum_radius = buttocks_joints_actual - seat_translation

        force_on_seat = np.zeros(3, dtype=float)
        momentum_on_seat = np.zeros(3, dtype=float)
        for i in range(6):
            force_on_seat += -suspensions_force_vector[i]
            momentum_on_seat += np.cross(
                momentum_radius[i], -suspensions_force_vector[i])

        return force_on_seat, momentum_on_seat

    def _body_dynamics_with_runge_kutta4(self, seat_motion: np.ndarray, external_acceleration: np.ndarray, dt: float):
        """
        目的：当输入为座椅运动位移与加速度时，计算人体的动态响应，并返回坐垫对座
        椅的作用力与力矩。
        本函数与 self.body_dynamics() 的唯一区别是采用了 runge-kutta 数值积分
        方法。

        参数：
            seat_motion: (6,) 数组，在全局坐标系下，座椅中心相对于其初始位置的位
                移。
            external_acceleration = [ax, ay, 0]: (3,) 数组，ax 与 ay 表示由于
                车体转向、加/减速等运动导致人体受到的加速度，这些加速度都是水平的，
                故只有纵向与侧向两个分量。其第 3 个分量为零，是为了计算方便。
            dt: 采样时间。

        返回：
            force_on_seat: 座椅中心处受到的来自坐垫的力；
            momentum_on_seat: 座椅中心处受到的来自坐垫的力矩。

        注意：
            body_motion: 人体质心在（运动的）局部坐标系（车辆坐标系平移至座椅中
                心处坐标系）下，相对于质心初始位置的位移；也就是说，人体质心先相
                对于（运动的）座椅处车辆坐标系平移，人体再相对于人体质心旋转（旋
                转是绝对的）。理解这一点，对于求解屁股中心处位移非常重要。
        """

        # 保存当前状态，用于 RK4 (K1 评估点)
        body_motion_current = self.body_motion.copy()
        body_velocity_current = self.body_velocity.copy()
        # 复制悬架状态，用于 RK4 步骤中的阻尼计算和最终更新
        suspensions_stroke_current = self.suspensions_stroke.copy()

        # 1. 利用 runge-kutta4 计算人体动态响应
        v1, a1 = self._get_body_state_derivative(
            seat_motion,
            body_motion_current,
            body_velocity_current,
            external_acceleration,
            suspensions_stroke_current,
            dt)

        v2, a2 = self._get_body_state_derivative(
            seat_motion,
            body_motion_current + v1*dt/2,
            body_velocity_current + a1*dt/2,
            external_acceleration,
            suspensions_stroke_current,
            dt/2)

        v3, a3 = self._get_body_state_derivative(
            seat_motion,
            body_motion_current + v2*dt/2,
            body_velocity_current + a2*dt/2,
            external_acceleration,
            suspensions_stroke_current,
            dt/2)

        v4, a4 = self._get_body_state_derivative(
            seat_motion,
            body_motion_current + v3*dt,
            body_velocity_current + a3*dt,
            external_acceleration,
            suspensions_stroke_current,
            dt)

        self.body_motion += (v1 + 2*v2 + 2*v3 + v4) * dt / 6
        self.body_velocity += (a1 + 2*a2 + 2*a3 + a4) * dt / 6

        # 2. 重新计算新状态下坐垫等效悬架的行程与方向 （以下内容与 self.body_dynamics 相同）
        seat_translation = seat_motion[:3]  # 座椅平动位移 (全局)
        seat_rotation = seat_motion[3:]  # 座椅转动欧拉角 (全局)

        Rs = self._get_rotation_matrix(seat_rotation)
        # 计算坐垫下表面铰点（即座椅上的铰点）在（初始）座椅坐标系下的实际位置
        # 坐垫坐标相对于初始状态时座椅中心
        cushion_seat_joints_actual = seat_translation + \
            self.cushion_seat_joints_local @ Rs.T

        # 3. 计算坐垫等效悬架在屁股上的铰点的位置
        # body_motion 是人体质心相对于（运动时）座椅中心的位移
        body_translation = self.body_motion[:3]
        body_rotation = self.body_motion[3:]

        Rb = self._get_rotation_matrix(body_rotation)
        # 计算坐垫上表面铰点（即屁股下的铰点）在（初始）座椅坐标系下的实际位置
        buttocks_joints_actual = seat_translation + \
            self.seat_to_body_local + body_translation + \
            self.body_to_buttocks_local @ Rb.T + self.buttocks_joints_local @ Rb.T

        # 4. 计算坐垫等效悬架的变形量
        suspensions_length = np.linalg.norm(
            buttocks_joints_actual-cushion_seat_joints_actual, axis=1)

        suspensions_stroke_current = suspensions_length - self.cushion_thickness
        self.suspensions_velocity = (suspensions_stroke_current -
                                     self.suspensions_stroke) / dt
        self.suspensions_stroke = suspensions_stroke_current

        # 5. 计算坐垫等效悬架的方向向量
        suspensions_direction = (
            buttocks_joints_actual - cushion_seat_joints_actual)
        suspensions_direction /= suspensions_length[:, np.newaxis]

        # 6. 计算坐垫等效悬架对屁股上铰点的作用力
        cb, kb = self.cushion_params

        suspensions_force = kb*self.suspensions_stroke + cb*self.suspensions_velocity
        suspensions_force_vector = suspensions_force[:, np.newaxis] * \
            suspensions_direction
        # 删除了与人体有关的动力学计算，因为这部分已经在龙格库塔算法中完成了

        # 7. 计算坐垫等效悬架对座椅中心的作用力
        momentum_radius = buttocks_joints_actual - seat_translation

        force_on_seat = np.zeros(3, dtype=float)
        momentum_on_seat = np.zeros(3, dtype=float)
        for i in range(6):
            force_on_seat += -suspensions_force_vector[i]
            momentum_on_seat += np.cross(
                momentum_radius[i], -suspensions_force_vector[i])

        return force_on_seat, momentum_on_seat

    def _get_body_state_derivative(self, seat_motion: np.ndarray, body_motion: np.ndarray, body_velocity: np.ndarray, external_acceleration: np.ndarray, suspensions_stroke: np.ndarray, dt: float):
        """
        本函数是为了进行龙格-库塔法计算而设计的，其目的是获得当前状态的微分。

        参数：
            seat_motion: (6,) 数组；
            body_motion: (6,) 数组；
            body_velocity: (6,) 数组；
            external_accleration: (3,) 数组；
            dt: 采样时间。

        返回：
            body_velocity: (6,) 数组；
            body_acceleration: (6,) 数组；
            suspensions_stroke: (6,) 数组；
            suspensions_force_vector: (6,3) 数组。
        """

        # 1. 计算坐垫等效悬架在坐椅上的铰点的位置
        seat_translation = seat_motion[:3]  # 座椅平动位移 (全局)
        seat_rotation = seat_motion[3:]  # 座椅转动欧拉角 (全局)

        Rs = self._get_rotation_matrix(seat_rotation)
        # 计算坐垫下表面铰点（即座椅上的铰点）在（初始）座椅坐标系下的实际位置
        # 坐垫坐标相对于初始状态时座椅中心
        cushion_seat_joints_actual = seat_translation + \
            self.cushion_seat_joints_local @ Rs.T

        # 2. 计算坐垫等效悬架在屁股上的铰点的位置
        # body_motion 是人体质心相对于（运动时）座椅中心的位移
        body_translation = body_motion[:3]
        body_rotation = body_motion[3:]

        Rb = self._get_rotation_matrix(body_rotation)
        # 计算坐垫上表面铰点（即屁股下的铰点）在（初始）座椅坐标系下的实际位置
        buttocks_joints_actual = seat_translation + \
            self.seat_to_body_local + body_translation + \
            self.body_to_buttocks_local @ Rb.T + self.buttocks_joints_local @ Rb.T

        # 3. 计算坐垫等效悬架的变形量
        suspensions_length = np.linalg.norm(
            buttocks_joints_actual-cushion_seat_joints_actual, axis=1)

        suspensions_stroke_current = suspensions_length - self.cushion_thickness
        suspensions_velocity = (suspensions_stroke_current -
                                suspensions_stroke) / dt
        suspensions_stroke = suspensions_stroke_current

        # 4. 计算坐垫等效悬架的方向向量
        suspensions_direction = (
            buttocks_joints_actual - cushion_seat_joints_actual)
        suspensions_direction /= suspensions_length[:, np.newaxis]

        # 5. 计算人体受到的力
        mb = self.body_mass
        cb, kb = self.cushion_params

        suspensions_force = kb*suspensions_stroke + cb*suspensions_velocity
        suspensions_force_vector = suspensions_force[:, np.newaxis] * \
            suspensions_direction

        # 计算坐垫悬架力作用点相对于（运动时）人体质心的位置矢量
        momentum_radius = self.body_to_buttocks_local @ Rb.T + \
            self.buttocks_joints_local @ Rb.T

        force_on_body = np.zeros(3, dtype=float)
        momentum_on_body = np.zeros(3, dtype=float)
        for i in range(6):
            # 将坐垫悬架作用力移动至人体质心处
            force_on_body += suspensions_force_vector[i]
            momentum_on_body += np.cross(
                momentum_radius[i], suspensions_force_vector[i])

        force_on_body += mb*external_acceleration  # 添加外部惯性力

        # 7. 计算人体平动与转动响应
        body_acceleration = np.zeros(6, dtype=float)
        body_acceleration[:3] = force_on_body[:3] / mb
        body_acceleration[3:] = np.linalg.solve(
            self.body_inertia, momentum_on_body)

        return body_velocity, body_acceleration


class StewartModel(BodyModel):

    def __init__(self, base_radius: float = BASE_RADIUS,
                 base_joints_angle: list[int] = BASE_JOINTS_ANGLE,
                 seat_radius: float = SEAT_RADIUS,
                 seat_joints_angle: list[int] = SEAT_JOINTS_ANGLE,
                 seat_center_height: float = SEAT_CENTER_HEIGHT,
                 seat_mass: float = SEAT_MASS,
                 seat_params: list[float] = SEAT_PARAMS,
                 seat_inertia: list[list[float]] = SEAT_INERTIA,
                 stroke_limits: list[float] = STROKE_LIMITS,
                 motion_limits: list[list[float]] = MOTION_LIMITS,
                 max_accelerations: list[float] = MAX_ACCELERATIONS,
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
        系重合，且底座不会相对于该坐标系发生 (x, y， psi) 移动，即底座总是有 
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
        self.body = BodyModel()

        self.base_radius = base_radius
        self.base_joints_angle = np.array(
            [np.radians(x) for x in base_joints_angle])

        # 座椅参数
        self.seat_radius = seat_radius
        self.seat_joints_angle = np.array(
            [np.radians(x) for x in seat_joints_angle])
        self.seat_center_height = seat_center_height
        self.seat_mass = seat_mass
        self.seat_params = np.array(seat_params)
        self.seat_inertia = np.array(seat_inertia)

        self.stroke_limits = np.array(stroke_limits)

        self.motion_limits = np.array(motion_limits)
        self.max_accelerations = np.array(max_accelerations)

        self.base_joints_local = BodyModel._get_joints_local_position(
            self.base_radius, self.base_joints_angle)
        # base_motion is used as external input, that's why it is not stated here.
        # self.base_joints_actual = self.base_joints_local + base_motion[:3]
        self.base_joints_actual = self.base_joints_local

        # 全局坐标系是车辆坐标系的副本，只不过是将后者的 x-y-psi 跟随底座中心运动
        self.seat_center_local = np.array(
            [0, 0, self.seat_center_height], dtype=float)  # 相对于全局坐标系的位置
        self.seat_center_actual = self.seat_center_local

        self.seat_joints_local = BodyModel._get_joints_local_position(
            self.seat_radius, self.seat_joints_angle)
        self.seat_joints_actual = self.seat_joints_local + \
            self.seat_center_actual

        self.seat_acceleration = np.zeros(6, dtype=float)
        self.seat_velocity = np.zeros(6, dtype=float)
        # 全局坐标系下，座椅中心相对于其初始位置的位移量
        self.seat_motion = np.zeros(6, dtype=float)

        # 默认 6 个悬架的初始长度都是相等的。
        base_joints_to_seat_joints = self.seat_center_local + \
            self.seat_joints_local - self.base_joints_local
        self.suspensions_length_nominal = np.linalg.norm(
            base_joints_to_seat_joints[0])

        # 当前的悬架状态
        self.suspensions_direction = base_joints_to_seat_joints
        self.suspensions_direction /= np.linalg.norm(
            self.suspensions_direction, axis=1)[:, np.newaxis]
        self.suspensions_stroke = np.zeros(6, dtype=float)
        self.suspensions_velocity = np.zeros(6, dtype=float)

    def padding_base_motion_signal(self, base_motion: np.ndarray) -> np.ndarray:
        """
        目的：将输入信号的维度由 3 维的转换为 6 维，以便于后续计算。
        这个函数应该在仿真开始前处理一次数据，而不是在每次迭代中调用。
        在 if __name__ == "__main__": 块中调用是正确的做法。

        参数：
        base_motion: (N, 3) 数组，或 (N, 6) 数组。如果是 (N, 3)，表示 [z, roll, pitch]。

        返回：
            base_motion_padded: (N, 6) 数组，表示 [x, y, z, roll, pitch, yaw]。
        """
        if base_motion.shape[1] == 3:
            # 在第0列和第1列前面填充2个0（x, y），在最后一列后面填充1个0（yaw）
            base_motion_padded = np.pad(
                base_motion, ((0, 0), (2, 1)), 'constant', constant_values=0)
        elif base_motion.shape[1] == 6:
            # 如果已经是6维，则直接返回，可能是已经处理过的
            base_motion_padded = base_motion
            logger.info(
                "base_motion is already 6-dimensional. No padding applied.")
        else:
            raise ValueError(
                f"Input base_motion should have 3 or 6 columns, but got {base_motion.shape[1]}.")

        return base_motion_padded

    def _stewart_seat_dynamics(self, base_motion: np.ndarray, external_acceleration: np.ndarray, dt: float):
        """
        根据座椅和底座的位姿信息，计算悬架变形量及端点间相对速度，为下一步计算悬
        架对座椅的支撑力打下基础。

        参数:
            base_motion (np.ndarray): (3,) 数组，底座在全局坐标系下的 
            (z, roll, pitch) 位移量，因为 (x, y, psi) 是假设为零的，故忽略。
            external_acceleration = [ax, ay]: (2,) 数组，表示由于车体转向、加/
            减速等运动导致人体受到的加速度，这些加速度都是水平的，故只有纵向与
            侧向两个分量。
            dt: 采样周期。
        """
        # 增加 z 向加速度分量，将加速度改为 (3,) 数组，以简化计算
        if (len(external_acceleration) == 2):
            external_acceleration = np.pad(
                external_acceleration, (0, 1), 'constant')
        else:
            logger.error(
                "External_acceleration has only 2 components in x, y directions.\n")

        base_translation = base_motion[:3]
        base_rotation = base_motion[3:]  # 底座的 roll-pitch-yaw

        seat_translation = self.seat_motion[:3]
        seat_rotation = self.seat_motion[3:]
        logger.debug(
            f"base_m = {base_motion}, seat_m = {self.seat_motion}")
        # 1. 获取座椅铰点的位置
        Rs = BodyModel._get_rotation_matrix(seat_rotation)

        self.seat_joints_actual = self.seat_center_local + \
            seat_translation + self.seat_joints_local @ Rs.T

        # 2. 获取底座铰点的位置
        Rb = BodyModel._get_rotation_matrix(base_rotation)

        self.base_joints_actual = base_translation + self.base_joints_local @ Rb.T

        # 3. 计算悬架长度
        suspensions_length = np.linalg.norm(
            self.seat_joints_actual - self.base_joints_actual, axis=1)

        # 4. 获取悬架伸长量和端点间相对速度
        suspensions_stroke_current = suspensions_length - self.suspensions_length_nominal
        self.suspensions_velocity = (
            suspensions_stroke_current-self.suspensions_stroke) / dt
        self.suspensions_stroke = suspensions_stroke_current

        # 5. 计算悬架的方向向量
        suspensions_direction = (
            self.seat_joints_actual - self.base_joints_actual)
        self.suspensions_direction /= np.linalg.norm(
            suspensions_direction, axis=1)[:, np.newaxis]

        # 6. 计算座椅受到的力
        ms = self.seat_mass
        cs, ks = self.seat_params

        suspension_force = ks*self.suspensions_stroke+cs*self.suspensions_velocity
        suspension_force_vector = np.zeros((6, 3), dtype=float)

        # 计算座椅悬架力作用点相对于（运动时）座椅中心的位置矢量
        momentum_radius = self.seat_joints_local @ Rs.T

        force_on_seat = np.zeros(3, dtype=float)
        momentum_on_seat = np.zeros(3, dtype=float)

        for i in range(6):
            suspension_force_vector[i] = suspension_force[i] * \
                suspensions_direction[i]
            # 将座椅悬架作用力移动至座椅质心处
            force_on_seat += suspension_force_vector[i]
            momentum_on_seat += np.cross(
                momentum_radius[i], suspension_force_vector[i])
        logger.debug(f"momentum_on_seat = {momentum_on_seat}")
        force_on_seat_from_body, momentum_on_seat_from_body = self.body._body_dynamics(
            self.seat_motion, external_acceleration, dt)
        logger.debug(f"force_on_seat = {force_on_seat_from_body}")
        logger.debug(f"momentum_on_seat_body = {momentum_on_seat_from_body}")

        # force_on_seat += force_on_seat_from_body
        # momentum_on_seat += momentum_on_seat_from_body

        force_on_seat += ms*external_acceleration  # 添加外部加速度载荷

        # 7. 计算座椅平动与转动响应
        self.seat_acceleration[:3] = force_on_seat[:3] / ms

        self.seat_acceleration[3:] = np.linalg.solve(
            self.seat_inertia, momentum_on_seat)
        self.seat_velocity += self.seat_acceleration*dt
        # 更新座椅中心的位置
        self.seat_motion += self.seat_velocity*dt

    def is_valid_suspensions_length(self, lengths: np.ndarray) -> np.bool_:
        """
        检查计算出的悬架长度是否在允许的行程范围内。

        参数:
            lengths (np.ndarray): (6,) 数组，悬架的当前长度。

        返回:
            bool: 如果所有长度都在有效范围内则为 True，否则为 False。
        """
        min_length, max_length = self.suspensions_length_nominal + self.stroke_limits
        is_valid = np.all((lengths >= min_length) & (lengths <= max_length))
        return is_valid

    def reset_platform(self):
        """
        重置座椅状态，将座椅连接点和底座连接点的位置恢复到初始状态。
        """
        self._initialize_joints_position()


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # 初始化一个对象，用于座椅侧倾或俯仰控制测试
    hexa = StewartModel()

    # 信号参数
    frequency = 1.  # 频率(Hz)
    amplitude = 0.2  # 振幅
    dt = 0.002  # 采样周期(秒)
    num_samples = 1000  # 采样点数

    # 使用采样周期生成时间轴
    t = np.arange(0, num_samples) * dt

    # 生成虚拟的座椅侧倾或俯仰信号
    # 实际中，座椅侧倾或俯仰信号由实验数据给出，替换掉这个数据即可
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    base_motion_test = np.vstack((0.*signal,
                                  0.*signal,
                                  0.*signal)).T
    base_motion_test = hexa.padding_base_motion_signal(base_motion_test)
    base_motion_test[:, 3:] *= np.pi/180
    logger.debug(f"base = {base_motion_test.shape}")

    seat_motion_series = np.zeros_like(base_motion_test)

    for i in range(len(base_motion_test)):
        logger.debug(f"base_motion = {base_motion_test[i]}")
        hexa._stewart_seat_dynamics(base_motion_test[i], np.array([0, 0]), dt)
        seat_motion_series[i] = hexa.seat_motion

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 3, 1)
    plt.plot(t, base_motion_test[:, 0], 'b-', label='x base')
    plt.plot(t, seat_motion_series[:, 0], 'r-', label='x seat')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(t, base_motion_test[:, 1], 'b-', label='y base')
    plt.plot(t, seat_motion_series[:, 1], 'r-', label='y seat')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(t, base_motion_test[:, 2], 'b-', label='z base')
    plt.plot(t, seat_motion_series[:, 2], 'r-', label='z seat')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(t, base_motion_test[:, 3], 'b-', label='roll base')
    plt.plot(t, seat_motion_series[:, 3], 'r-', label='roll seat')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(t, base_motion_test[:, 4], 'b-', label='pitch base')
    plt.plot(t, seat_motion_series[:, 4], 'r-', label='pitch seat')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(t, base_motion_test[:, 5], 'b-', label='yaw base')
    plt.plot(t, seat_motion_series[:, 5], 'r-', label='yaw seat')
    plt.legend()

    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__mainx__":
    import pandas as pd
    import matplotlib.pyplot as plt

    # 初始化一个对象，用于平台侧倾或俯仰控制测试
    hexa = StewartModel()

    # 使用Pandas读取CSV文件
    df = pd.read_csv("C:\\pycheche\\testStewartPlatform\\beamng_acc_cleaned_osim_lp03_resample.csv",
                     skiprows=0, header=0)

    # 获取列名(不包含第一行标题)
    columns = df.columns

    # 提取时间列和数据列
    t = df[columns[0]].values.astype(float)
    base_motion_test = df[columns[3:6]].values.astype(float)
    base_motion_test = hexa.padding_base_motion_signal(base_motion_test)
    base_motion_test[:, 3:] *= np.pi/180
    logger.debug(f"base = {base_motion_test.shape}")
    dt = 0.02

    seat_motion_series = np.zeros_like(base_motion_test)
    body_motion_series = np.zeros_like(base_motion_test)

    for i in range(len(base_motion_test)):
        logger.debug(f"base_motion = {base_motion_test[i]}")
        # hexa.body._body_dynamics(base_motion_test[i], np.array([0, 0, 0]))
        hexa.body._body_dynamics_with_runge_kutta4(
            base_motion_test[i], np.array([0, 0, 0,]), dt)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 3, 1)
    plt.plot(t, base_motion_test[:, 0], 'b-', label='x base')
    plt.plot(t, seat_motion_series[:, 0], 'r-', label='x seat')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(t, base_motion_test[:, 1], 'b-', label='y base')
    plt.plot(t, seat_motion_series[:, 1], 'r-', label='y seat')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(t, base_motion_test[:, 2], 'b-', label='z base')
    plt.plot(t, seat_motion_series[:, 2], 'r-', label='z seat')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(t, base_motion_test[:, 3], 'b-', label='roll base')
    plt.plot(t, seat_motion_series[:, 3], 'r-', label='roll seat')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(t, base_motion_test[:, 4], 'b-', label='pitch base')
    plt.plot(t, seat_motion_series[:, 4], 'r-', label='pitch seat')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(t, base_motion_test[:, 5], 'b-', label='yaw base')
    plt.plot(t, seat_motion_series[:, 5], 'r-', label='yaw seat')
    plt.legend()

    plt.legend()
    plt.grid(True)
    plt.show()
