"""
几何参数配置文件
包含所有与平台、电动缸等几何相关的默认参数
"""

# 底座参数
BASE_RADIUS = 0.19  # 基座半径 (m)
BASE_JOINTS_ANGLE = [30, 90, 150, 210, 270, 330]

# 座椅参数
SEAT_RADIUS = 0.14  # 平台半径 (m)
# [a+b for a in [60, 180, 300] for b in [-5, 5]]
SEAT_JOINTS_ANGLE = [a+b for a in [60, 180, 300] for b in [-5, 5]]
SEAT_CENTER_HEIGHT = 0.10  # 平台中心高度 (m)，相对于基座
SEAT_MASS = 20
SEAT_PARAMS = [300, 1000]  # 将人体质量集成到座椅上，但高度不在座椅中心，而在上方
SEAT_INERTIA = [[100, 0, 0], [0, 100, 0], [0, 0, 10]]  # 将人体转动惯量集成到座椅上

# 电动缸参数
STROKE_LIMITS = [-0.2, 0.2]  # 电缸最大伸缩行程 (m)

# 人体参数（忽略平台质量对于运动的影响）
BODY_MASS = 80.
BODY_INERTIA = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]  # 人体相对于人体质心的转动惯量
CUSHION_THICKNESS = 0.05  # 屁股中心至座椅中心的高度，即坐垫的厚度
BODY_CENTER_HEIGHT = 0.40  # 人体与座椅集成质量的质心高度 (m)，相对于屁股中心的高度
CUSHION_RADIUS = 0.14
CUSHION_JOINTS_ANGLE = [id*60 for id in range(6)]
# 坐垫的弹簧参数，分别为垂向、侧倾、俯仰方向的弹簧刚度和阻尼系数
CUSHION_PARAMS = [500, 2000, 600, 2000, 600, 2000]

# 座椅运动极限位置
MOTION_LIMITS = [[-0.10, -0.10, -0.02, -20, -20, -20],
                 [0.10, 0.10, 0.08, 20, 20, 20]]

# 极限相对加速度与极限角加速度
MAX_ACCELERATIONS = [20, 9000]

# 控制器参数
KW_PARAMS = [[0.9, -0.1257, 0.0],
             [0.9, -0.1257, 0.0],
             [0.9, -0.1257, 0.0]]
KV_PARAMS = [8., 3., 2.]
