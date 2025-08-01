import numpy as np
import OpenGL.GL as GL
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram
from PySide6.QtGui import QMatrix4x4, QVector3D
from PySide6.QtCore import QPointF, QSize, QTimer, Signal, Qt
from PySide6.QtWidgets import QLabel
import ctypes  # For ctypes.c_void_p
from hexapod_model import Hexapod  # 假设 hexapod 模型在 hexapod_model.py 中定义

# --- ShapeRenderer 类定义 (与之前相同) ---


class ShapeRenderer:
    def __init__(self, shader_program: QOpenGLShaderProgram,
                 vertex_data: np.ndarray, index_data: np.ndarray):
        self.shader_program = shader_program
        self.indices_count = len(index_data)

        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.ebo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertex_data.nbytes,
                        vertex_data, GL.GL_STATIC_DRAW)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER,
                        index_data.nbytes, index_data, GL.GL_STATIC_DRAW)

        # Position attribute (layout (location = 0))
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * vertex_data.itemsize, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        # Normal attribute (layout (location = 1))
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 *
                                 vertex_data.itemsize, ctypes.c_void_p(3 * vertex_data.itemsize))
        GL.glEnableVertexAttribArray(1)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self, model_matrix: QMatrix4x4, object_color: np.ndarray):
        model_loc = self.shader_program.uniformLocation("model")
        self.shader_program.setUniformValue(model_loc, model_matrix)
        object_color_loc = self.shader_program.uniformLocation("objectColor")
        self.shader_program.setUniformValue(
            object_color_loc, object_color[0], object_color[1], object_color[2])

        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices_count,
                          GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)

    def __del__(self):
        if GL.glIsVertexArray(self.vao):
            GL.glDeleteVertexArrays(1, [self.vao])
        if GL.glIsBuffer(self.vbo):
            GL.glDeleteBuffers(1, [self.vbo])
        if GL.glIsBuffer(self.ebo):
            GL.glDeleteBuffers(1, [self.ebo])
# --- 结束 ShapeRenderer 类定义 ---


# VERTEX_SHADER_SOURCE 和 FRAGMENT_SHADER_SOURCE 保持不变
VERTEX_SHADER_SOURCE = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;

    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos; // Camera position

void main()
{
    // Ambient
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
"""


class OpenGLHexapodViz(QOpenGLWidget):
    status_message = Signal(str)

    def __init__(self, hexapod_model: Stewart, parent=None):  # 注意类型提示改为 StewartModel
        super().__init__(parent)
        self.hexapod_model = hexapod_model
        self.hexapod_model.viz_widget = self  # 将可视化 widget 绑定到模型

        self.status_label = QLabel("Ready.")
        self.is_animation_playing = False
        self.animation_timer_playback = QTimer(self)
        self.animation_timer_playback.timeout.connect(
            self._playback_animation_frame)
        self.animation_playback_speed_ms = 30
        self.animation_current_frame = 0
        # Base platform pose (fixed for Stewart)
        self.animation_base_data = None
        self.animation_top_platform_data = None  # Top platform pose
        self.animation_max_frames = 0

        self.shader_program = None
        self.sphere_renderer = None
        self.cylinder_renderer = None
        self.cone_renderer = None
        # 我们将使用 CylinderRenderer 来绘制 Stewart 平台的腿和平台本身（如果需要厚度）
        # 或者直接使用 sphere_renderer 来绘制关节
        self.plate_renderer = None  # 专门用于绘制平台，例如厚度很小的立方体或圆柱体

        self._init_gl_resources = False

        # 相机参数
        self.camera_position = np.array([0.5, -0.5, 0.5], dtype=np.float32)
        self.camera_target = np.array(
            [0.0, 0.0, 0.1], dtype=np.float32)  # 相机目标点，略微抬高
        self.camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # 光源位置
        self.light_position = np.array([0.5, 0.5, 1.0], dtype=np.float32)

        # 鼠标控制参数
        self.last_mouse_pos = QPointF()
        self.camera_yaw = -45.0
        self.camera_pitch = -30.0
        self.camera_distance = 1.0  # 相机距离目标点的距离

    def initializeGL(self):
        self.makeCurrent()

        if not self._init_gl_resources:
            self._compile_shaders()
            self._setup_shape_renderers()
            self._init_gl_resources = True

        GL.glClearColor(0.2, 0.3, 0.3, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_CULL_FACE)  # 开启背面剔除
        GL.glCullFace(GL.GL_BACK)  # 剔除背面

        self._update_camera_position()

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.shader_program.bind()

        projection = self._get_projection_matrix()
        self.shader_program.setUniformValue("projection", projection)

        view = self._get_view_matrix()
        self.shader_program.setUniformValue("view", view)
        self.shader_program.setUniformValue("viewPos", self.camera_position[0],
                                            self.camera_position[1],
                                            self.camera_position[2])
        self.shader_program.setUniformValue("lightPos", self.light_position[0],
                                            self.light_position[1],
                                            self.light_position[2])
        self.shader_program.setUniformValue("lightColor", 1.0, 1.0, 1.0)

        # 绘制绝对坐标系 (在世界原点)
        model_matrix_sphere = QMatrix4x4()
        model_matrix_sphere.translate(0.0, 0.0, 0.0)  # 世界原点球体
        self.sphere_renderer.draw(
            model_matrix_sphere, np.array([0.7, 0.7, 0.7]))

        axis_len = 0.2
        cylinder_radius = 0.008
        cone_radius = 0.02
        cone_height = axis_len * 0.25

        self._draw_axis_with_arrow(np.array(
            [1.0, 0.0, 0.0]), axis_len, cylinder_radius, cone_radius, cone_height, np.array([1.0, 0.0, 0.0]))
        self._draw_axis_with_arrow(np.array(
            [0.0, 1.0, 0.0]), axis_len, cylinder_radius, cone_radius, cone_height, np.array([0.0, 1.0, 0.0]))
        self._draw_axis_with_arrow(np.array(
            [0.0, 0.0, 1.0]), axis_len, cylinder_radius, cone_radius, cone_height, np.array([0.0, 0.0, 1.0]))

        # --- 绘制 Stewart 模型 ---
        self._draw_stewart_model()

        self.shader_program.release()
        self.update()

    def _draw_stewart_model(self):
        base_points_global, top_points_global = self.hexapod_model.get_platform_points_global()
        leg_lengths = self.hexapod_model.leg_lengths

        # 1. 绘制底部平台
        # 底部平台通常是固定的，或只做平移。
        # 我们直接使用 StewartModel 中的 base_pose 来构建底部平台的变换矩阵。
        base_x, base_y, base_z, base_roll, base_pitch, base_yaw = self.hexapod_model.base_pose
        base_platform_matrix = QMatrix4x4()
        base_platform_matrix.translate(base_x, base_y, base_z)
        base_platform_matrix.rotate(np.degrees(base_yaw), 0, 0, 1)
        base_platform_matrix.rotate(np.degrees(base_pitch), 0, 1, 0)
        base_platform_matrix.rotate(np.degrees(base_roll), 1, 0, 0)

        # 假设底部平台是一个圆盘或者多边形，我们可以用一个扁平的圆柱体或立方体来近似
        # 由于 ShapeRenderer 的圆柱体是沿Z轴，中心在原点
        plate_thickness = 0.01
        base_plate_scale = QMatrix4x4()
        base_plate_scale.scale(StewartModel.BASE_PLATFORM_RADIUS * 2,
                               StewartModel.BASE_PLATFORM_RADIUS * 2, plate_thickness)  # 放大到实际尺寸
        self.plate_renderer.draw(
            base_platform_matrix * base_plate_scale, np.array([0.4, 0.4, 0.4]))  # 灰色底部平台

        # 2. 绘制顶部平台
        top_x, top_y, top_z, top_roll, top_pitch, top_yaw = self.hexapod_model.top_platform_pose
        top_platform_matrix = QMatrix4x4()
        top_platform_matrix.translate(top_x, top_y, top_z)
        top_platform_matrix.rotate(np.degrees(top_yaw), 0, 0, 1)
        top_platform_matrix.rotate(np.degrees(top_pitch), 0, 1, 0)
        top_platform_matrix.rotate(np.degrees(top_roll), 1, 0, 0)

        top_plate_scale = QMatrix4x4()
        top_plate_scale.scale(StewartModel.TOP_PLATFORM_RADIUS * 2,
                              StewartModel.TOP_PLATFORM_RADIUS * 2, plate_thickness)
        self.plate_renderer.draw(
            top_platform_matrix * top_plate_scale, np.array([0.8, 0.2, 0.2]))  # 红色顶部平台

        # 3. 绘制六条腿 (作动器)
        leg_radius = 0.005  # 腿的半径

        for i in range(StewartModel.NUM_LEGS):
            p1 = base_points_global[i]
            p2 = top_points_global[i]
            current_leg_length = leg_lengths[i]

            # 计算腿的中心点
            center_x = (p1[0] + p2[0]) / 2.0
            center_y = (p1[1] + p2[1]) / 2.0
            center_z = (p1[2] + p2[2]) / 2.0

            # 计算腿的方向向量
            direction = p2 - p1
            norm_direction = direction / np.linalg.norm(direction)

            # 计算从Z轴到腿方向的旋转
            z_axis = np.array([0.0, 0.0, 1.0])
            rotation_matrix_np = np.identity(4, dtype=np.float32)
            if not np.allclose(norm_direction, z_axis):
                rotation_axis = np.cross(z_axis, norm_direction)
                angle_rad = np.arccos(np.dot(z_axis, norm_direction))
                if np.linalg.norm(rotation_axis) > 1e-6:
                    axis_normalized = rotation_axis / \
                        np.linalg.norm(rotation_axis)
                    rotation_matrix_np = self._create_rotation_matrix_np(
                        angle_rad, axis_normalized)
                elif np.dot(z_axis, norm_direction) < 0:  # 180度翻转
                    rotation_matrix_np = self._create_rotation_matrix_np(
                        np.pi, np.array([1.0, 0.0, 0.0]))

            rotation_matrix = QMatrix4x4(rotation_matrix_np.flatten())

            # 构建腿的模型矩阵
            leg_model_matrix = QMatrix4x4()
            leg_model_matrix.translate(center_x, center_y, center_z)  # 平移到腿的中心
            leg_model_matrix = leg_model_matrix * rotation_matrix  # 应用旋转
            leg_model_matrix.scale(
                leg_radius, leg_radius, current_leg_length)  # 缩放

            # 绘制腿
            # 根据腿的长度，给不同的颜色，模拟长度变化
            if current_leg_length > StewartModel.LEG_MAX_LENGTH * 0.95 or current_leg_length < StewartModel.LEG_MIN_LENGTH * 1.05:
                # 接近极限长度时显示红色
                self.cylinder_renderer.draw(
                    leg_model_matrix, np.array([1.0, 0.0, 0.0]))
            else:
                self.cylinder_renderer.draw(
                    leg_model_matrix, np.array([0.2, 0.5, 0.8]))  # 蓝色腿

            # 绘制连接关节的球体 (可选)
            joint_sphere_radius = 0.008
            # 底部关节球
            base_joint_matrix = QMatrix4x4()
            base_joint_matrix.translate(p1[0], p1[1], p1[2])
            base_joint_matrix.scale(
                joint_sphere_radius, joint_sphere_radius, joint_sphere_radius)
            self.sphere_renderer.draw(
                base_joint_matrix, np.array([1.0, 1.0, 0.0]))  # 黄色关节

            # 顶部关节球
            top_joint_matrix = QMatrix4x4()
            top_joint_matrix.translate(p2[0], p2[1], p2[2])
            top_joint_matrix.scale(joint_sphere_radius,
                                   joint_sphere_radius, joint_sphere_radius)
            self.sphere_renderer.draw(
                top_joint_matrix, np.array([0.0, 1.0, 1.0]))  # 青色关节

    def _compile_shaders(self):
        self.shader_program = QOpenGLShaderProgram(self)
        self.shader_program.addShaderFromSourceCode(
            QOpenGLShader.Vertex, VERTEX_SHADER_SOURCE)
        self.shader_program.addShaderFromSourceCode(
            QOpenGLShader.Fragment, FRAGMENT_SHADER_SOURCE)
        if not self.shader_program.link():
            print("Shader linking failed:", self.shader_program.log())
        else:
            print("Shader program linked successfully.")

    def _setup_shape_renderers(self):
        sphere_data = self._generate_sphere_data(
            radius=1.0, slices=32, stacks=32)
        self.sphere_renderer = ShapeRenderer(
            self.shader_program, sphere_data['vertices'], sphere_data['indices'])
        cylinder_data = self._generate_cylinder_data(
            radius=1.0, height=1.0, segments=32)
        self.cylinder_renderer = ShapeRenderer(
            self.shader_program, cylinder_data['vertices'], cylinder_data['indices'])
        cone_data = self._generate_cone_data(
            radius=1.0, height=1.0, segments=32)
        self.cone_renderer = ShapeRenderer(
            self.shader_program, cone_data['vertices'], cone_data['indices'])
        # Plate renderer (可以是一个非常扁的圆柱体或立方体)
        plate_data = self._generate_cylinder_data(
            radius=0.5, height=1.0, segments=32)  # 单位圆柱体
        self.plate_renderer = ShapeRenderer(
            self.shader_program, plate_data['vertices'], plate_data['indices'])

    def _draw_axis_with_arrow(self, axis_color: np.ndarray, length: float,
                              cylinder_radius: float, cone_radius: float, cone_height: float,
                              direction: np.ndarray,
                              base_transform_matrix: QMatrix4x4 = None):
        if base_transform_matrix is None:
            base_transform_matrix = QMatrix4x4()

        z_axis = np.array([0.0, 0.0, 1.0])
        norm_direction = direction / np.linalg.norm(direction)

        rotation_matrix_np = np.identity(4, dtype=np.float32)
        if not np.allclose(norm_direction, z_axis):
            rotation_axis = np.cross(z_axis, norm_direction)
            angle_rad = np.arccos(np.dot(z_axis, norm_direction))
            if np.linalg.norm(rotation_axis) > 1e-6:
                axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_matrix_np = self._create_rotation_matrix_np(
                    angle_rad, axis_normalized)
            elif np.dot(z_axis, norm_direction) < 0:
                rotation_matrix_np = self._create_rotation_matrix_np(
                    np.pi, np.array([1.0, 0.0, 0.0]))

        rotation_matrix = QMatrix4x4(rotation_matrix_np.flatten())

        cylinder_height_actual = length - cone_height
        cylinder_scale_matrix = QMatrix4x4()
        cylinder_scale_matrix.scale(
            cylinder_radius, cylinder_radius, cylinder_height_actual)

        model_cylinder = base_transform_matrix * \
            rotation_matrix * cylinder_scale_matrix
        if self.cylinder_renderer is not None:
            self.cylinder_renderer.draw(model_cylinder, axis_color)

        cone_translate_matrix = QMatrix4x4()
        cone_translate_matrix.translate(0.0, 0.0, cylinder_height_actual)
        cone_scale_matrix = QMatrix4x4()
        cone_scale_matrix.scale(cone_radius, cone_radius, cone_height)

        model_cone = base_transform_matrix * rotation_matrix * \
            cone_translate_matrix * cone_scale_matrix
        if self.cone_renderer is not None:
            self.cone_renderer.draw(model_cone, axis_color)

    # 几何体数据生成函数 (与之前相同，或者经过调整以中心在原点)
    def _generate_sphere_data(self, radius, slices, stacks):
        vertices = []
        indices = []
        for i in range(stacks + 1):
            phi = np.pi / stacks * i
            for j in range(slices + 1):
                theta = 2 * np.pi / slices * j
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                vertices.extend([x, y, z])
                norm = np.linalg.norm([x, y, z])
                if norm > 0:
                    vertices.extend([x / norm, y / norm, z / norm])
                else:
                    vertices.extend([0.0, 0.0, 0.0])
        for i in range(stacks):
            for j in range(slices):
                p0 = i * (slices + 1) + j
                p1 = p0 + slices + 1
                p2 = p1 + 1
                p3 = p0 + 1
                indices.extend([p0, p1, p2])
                indices.extend([p0, p2, p3])
        return {'vertices': np.array(vertices, dtype=np.float32),
                'indices': np.array(indices, dtype=np.uint32)}

    def _generate_cylinder_data(self, radius, height, segments):
        vertices = []
        indices = []
        # Top center and bottom center
        top_center_idx = 0
        bottom_center_idx = 1
        vertices.extend([0.0, 0.0, height / 2.0, 0.0, 0.0, 1.0])  # Top center
        vertices.extend([0.0, 0.0, -height / 2.0, 0.0,
                        0.0, -1.0])  # Bottom center

        # Body and caps
        start_idx = 2
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Top ring vertex for cap
            vertices.extend([x, y, height / 2.0])
            vertices.extend([0.0, 0.0, 1.0])  # Normal for top cap (vertical)

            # Top ring vertex for side
            vertices.extend([x, y, height / 2.0])
            # Normal for side (radial)
            vertices.extend([np.cos(angle), np.sin(angle), 0.0])

            # Bottom ring vertex for cap
            vertices.extend([x, y, -height / 2.0])
            # Normal for bottom cap (vertical)
            vertices.extend([0.0, 0.0, -1.0])

            # Bottom ring vertex for side
            vertices.extend([x, y, -height / 2.0])
            # Normal for side (radial)
            vertices.extend([np.cos(angle), np.sin(angle), 0.0])

        # Indices for bottom cap
        for i in range(segments):
            idx1 = start_idx + i * 4 + 2  # Bottom ring (cap normal)
            idx2 = start_idx + (i + 1) * 4 + 2
            indices.extend([bottom_center_idx, idx2, idx1])

        # Indices for top cap
        for i in range(segments):
            idx1 = start_idx + i * 4 + 0  # Top ring (cap normal)
            idx2 = start_idx + (i + 1) * 4 + 0
            indices.extend([top_center_idx, idx1, idx2])

        # Indices for cylinder sides
        for i in range(segments):
            p0 = start_idx + i * 4 + 1  # Top ring (side normal)
            p1 = start_idx + i * 4 + 3  # Bottom ring (side normal)
            p2 = start_idx + (i + 1) * 4 + 3
            p3 = start_idx + (i + 1) * 4 + 1
            indices.extend([p0, p1, p2])
            indices.extend([p0, p2, p3])
        return {'vertices': np.array(vertices, dtype=np.float32),
                'indices': np.array(indices, dtype=np.uint32)}

    def _generate_cone_data(self, radius, height, segments):
        vertices = []
        indices = []
        tip_idx = 0
        # Tip vertex (normal points up)
        vertices.extend([0.0, 0.0, height, 0.0, 0.0, 1.0])
        base_center_idx = 1
        # Base center (normal points down)
        vertices.extend([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

        start_idx = 2
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Base ring vertex (for cap)
            vertices.extend([x, y, 0.0])
            vertices.extend([0.0, 0.0, -1.0])  # Normal for base cap

            # Side vertex
            vertices.extend([x, y, 0.0])
            # Side normal calculation
            norm_x = np.cos(angle)
            norm_y = np.sin(angle)
            norm_z = radius / height  # This creates a 'flatter' normal for lighting effect
            normal_vec = np.array([norm_x, norm_y, norm_z])
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
            vertices.extend(normal_vec.tolist())

        # Indices for base cap
        for i in range(segments):
            idx1 = start_idx + i * 2  # Base ring vertex (cap normal)
            idx2 = start_idx + (i + 1) * 2
            indices.extend([base_center_idx, idx2, idx1])

        # Indices for cone sides
        for i in range(segments):
            idx1_side = start_idx + i * 2 + 1  # Side vertex
            idx2_side = start_idx + (i + 1) * 2 + 1
            indices.extend([tip_idx, idx1_side, idx2_side])
        return {'vertices': np.array(vertices, dtype=np.float32),
                'indices': np.array(indices, dtype=np.uint32)}

    def _create_translation_matrix_np(self, x, y, z):
        matrix = np.identity(4, dtype=np.float32)
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z
        return matrix

    def _create_scale_matrix_np(self, sx, sy, sz):
        matrix = np.identity(4, dtype=np.float32)
        matrix[0, 0] = sx
        matrix[1, 1] = sy
        matrix[2, 2] = sz
        return matrix

    def _create_rotation_matrix_np(self, angle_rad, axis: np.ndarray):
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        x, y, z = axis / np.linalg.norm(axis)

        matrix = np.identity(4, dtype=np.float32)
        matrix[0, 0] = c + x*x*(1-c)
        matrix[0, 1] = x*y*(1-c) - z*s
        matrix[0, 2] = x*z*(1-c) + y*s
        matrix[1, 0] = y*x*(1-c) + z*s
        matrix[1, 1] = c + y*y*(1-c)
        matrix[1, 2] = y*z*(1-c) - x*s
        matrix[2, 0] = z*x*(1-c) - y*s
        matrix[2, 1] = z*y*(1-c) + x*s
        matrix[2, 2] = c + z*z*(1-c)
        return matrix

    def _get_projection_matrix(self):
        fov_deg = 45.0
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        near = 0.1
        far = 100.0

        proj_matrix = QMatrix4x4()
        proj_matrix.perspective(fov_deg, aspect, near, far)
        return proj_matrix

    def _get_view_matrix(self):
        view_matrix = QMatrix4x4()
        view_matrix.lookAt(QVector3D(*self.camera_position),
                           QVector3D(*self.camera_target),
                           QVector3D(*self.camera_up))
        return view_matrix

    def _update_camera_position(self):
        offset_x = self.camera_distance * \
            np.cos(np.radians(self.camera_yaw)) * \
            np.cos(np.radians(self.camera_pitch))
        offset_y = self.camera_distance * \
            np.sin(np.radians(self.camera_yaw)) * \
            np.cos(np.radians(self.camera_pitch))
        offset_z = self.camera_distance * np.sin(np.radians(self.camera_pitch))

        self.camera_position[0] = self.camera_target[0] + offset_x
        self.camera_position[1] = self.camera_target[1] + offset_y
        self.camera_position[2] = self.camera_target[2] + offset_z

        if self.camera_pitch > 90 or self.camera_pitch < -90:
            self.camera_up = np.array([0.0, 0.0, -1.0])
        else:
            self.camera_up = np.array([0.0, 0.0, 1.0])

        self.update()

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event):
        dx = event.position().x() - self.last_mouse_pos.x()
        dy = event.position().y() - self.last_mouse_pos.y()

        if event.buttons() == Qt.MouseButton.LeftButton:
            self.camera_yaw += dx * 0.2
            self.camera_pitch -= dy * 0.2
            self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch))
            self._update_camera_position()
        elif event.buttons() == Qt.MouseButton.RightButton:
            self._pan_camera(dx, dy)
        self.last_mouse_pos = event.position()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.camera_distance -= delta * 0.001 * self.camera_distance
        self.camera_distance = max(0.1, self.camera_distance)
        self._update_camera_position()

    def _pan_camera(self, dx: float, dy: float):
        inv_result = self._get_view_matrix().inverted()
        if not inv_result[1]:
            return
        inv_view_matrix = inv_result[0]

        camera_right = np.array([inv_view_matrix.column(
            0).x(), inv_view_matrix.column(0).y(), inv_view_matrix.column(0).z()])
        camera_right = camera_right / np.linalg.norm(camera_right)

        camera_up_screen = np.array([inv_view_matrix.column(
            1).x(), inv_view_matrix.column(1).y(), inv_view_matrix.column(1).z()])
        camera_up_screen = camera_up_screen / np.linalg.norm(camera_up_screen)

        pan_speed = self.camera_distance * 0.002

        pan_amount_x = dx * pan_speed
        pan_amount_y = dy * pan_speed

        self.camera_target += camera_right * pan_amount_x
        self.camera_target -= camera_up_screen * pan_amount_y

        self._update_camera_position()

    def _draw_text(self, x, y, z, text):
        pass

    # 修改参数名为 top_platform_pose_data
    def load_animation_data(self, base_pose_data, top_platform_pose_data):
        self.animation_base_data = base_pose_data
        self.animation_top_platform_data = top_platform_pose_data
        self.animation_max_frames = len(base_pose_data)
        self.status_message.emit(
            f"Animation loaded with {self.animation_max_frames} frames.")
        self.animation_current_frame = 0
        self.is_animation_playing = False
        self._playback_animation_frame()

    def start_animation(self):
        if self.animation_max_frames > 0:
            self.is_animation_playing = True
            self.animation_timer_playback.start(
                self.animation_playback_speed_ms)
            self.status_message.emit("Animation started.")

    def pause_animation(self):
        self.is_animation_playing = False
        self.animation_timer_playback.stop()
        self.status_message.emit("Animation paused.")

    def stop_animation(self):
        self.pause_animation()
        self.animation_current_frame = 0
        if self.animation_max_frames > 0:
            self._playback_animation_frame()
        self.status_message.emit("Animation stopped.")

    def set_animation_speed(self, ms):
        self.animation_playback_speed_ms = ms
        if self.is_animation_playing:
            self.animation_timer_playback.stop()
            self.animation_timer_playback.start(
                self.animation_playback_speed_ms)
        self.status_message.emit(f"Animation speed set to {ms} ms/frame.")

    def _playback_animation_frame(self):
        if self.animation_base_data is None or self.animation_current_frame >= self.animation_max_frames:
            self.stop_animation()
            return

        base_pose = self.animation_base_data[self.animation_current_frame] if self.animation_base_data is not None else None
        top_platform_pose = self.animation_top_platform_data[
            self.animation_current_frame] if self.animation_top_platform_data is not None else None
        if base_pose is not None and top_platform_pose is not None:
            self.hexapod_model.update_pose(base_pose, top_platform_pose)
            self.update()

        self.animation_current_frame += 1
        if self.animation_current_frame >= self.animation_max_frames:
            self.stop_animation()
            self.status_message.emit("Animation finished.")
