
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QGroupBox, QLineEdit, QLabel,
                               QComboBox, QPushButton, QDialog,
                               QSpacerItem, QSizePolicy)
from PySide6.QtGui import (QPalette, QColor, QDoubleValidator,
                           QIntValidator, QFontMetrics)

import numpy as np
import control as ct

from MSDChart import *


class TabPage2(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parent = parent

        tabpage2_layout = QVBoxLayout(self)

        # 基本参数设置
        self.tf_group = QGroupBox("Transfer Function")
        self.tf_layout = QVBoxLayout(self.tf_group)
        tf_layout_1 = QHBoxLayout()

        self.comboboxes = []
        key_label = QLabel("阶次")
        combobox1 = QComboBox()
        combobox1.addItems(["1", "2", "3", "4", "5"])
        combobox1.setCurrentIndex(1)
        combobox1.currentIndexChanged.connect(self.on_combobox1_clicked)
        self.comboboxes.append(combobox1)

        tf_layout_1.addWidget(key_label, 1)
        tf_layout_1.addWidget(combobox1, 1)

        key_label = QLabel("建模方式")
        combobox2 = QComboBox()
        combobox2.addItems(["普通", "ZPK"])
        combobox2.currentIndexChanged.connect(self.on_combobox2_clicked)
        self.comboboxes.append(combobox2)

        tf_layout_1.addWidget(key_label, 1)
        tf_layout_1.addWidget(combobox2, 1)

        key_label = QLabel("由高至低")
        key_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        key_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        tf_layout_1.addWidget(key_label, 1)

        btn_show_infos = QPushButton("信息")
        btn_show_infos.clicked.connect(self.btn_show_infos_clicked)
        tf_layout_1.addWidget(btn_show_infos, 1)

        self.tf_layout.addLayout(tf_layout_1)

        self.tf_layout_2 = QHBoxLayout()

        key_label = QLabel("分子")
        key_label.setFixedWidth(50)
        self.tf_layout_2.addWidget(key_label)

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_2, 2, True)
        self.numerator_boxes = value_boxes.copy()

        self.tf_layout.addLayout(self.tf_layout_2)

        self.tf_layout_3 = QHBoxLayout()

        key_label = QLabel("分母")
        key_label.setFixedWidth(50)
        self.tf_layout_3.addWidget(key_label)

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_3, 3, False)
        self.denominator_boxes = value_boxes.copy()

        self.tf_layout.addLayout(self.tf_layout_3)

        tf_layout_4 = QHBoxLayout()

        self.btn_time_responses = QPushButton("传递函数时域响应")
        self.btn_time_responses.clicked.connect(
            self.btn_transient_responses_clicked)
        tf_layout_4.addWidget(self.btn_time_responses)

        self.transient_window = None
        self.frequency_window = None
        self.discrete_window = None
        self.transient_window_2 = None
        self.frequency_window_2 = None

        self.btn_frequency_analyses = QPushButton("传递函数频谱分析")
        self.btn_frequency_analyses.clicked.connect(
            self.btn_frequency_analyses_clicked)
        tf_layout_4.addWidget(self.btn_frequency_analyses)

        self.tf_layout.addLayout(tf_layout_4)

        tf_layout_5 = QHBoxLayout()

        self.btn_discrete_control = QPushButton("离散运动控制")
        self.btn_discrete_control.clicked.connect(
            self.btn_discrete_control_clicked)
        tf_layout_5.addWidget(self.btn_discrete_control)

        self.btn_model_transient_response = QPushButton("闭环时域响应")
        self.btn_model_transient_response.clicked.connect(
            self.btn_model_transient_response_clicked)
        tf_layout_5.addWidget(self.btn_model_transient_response)

        self.btn_model_frequency_analyses = QPushButton("闭环频谱分析")
        self.btn_model_frequency_analyses.clicked.connect(
            self.btn_model_frequency_analyses_clicked)
        tf_layout_5.addWidget(self.btn_model_frequency_analyses)

        self.tf_layout.addLayout(tf_layout_5)

        tabpage2_layout.addWidget(self.tf_group)
        tabpage2_layout.addStretch()

        self.Hs = 1.0  # default transfer function

    def get_windows(self):

        windows_list = [self.transient_window, self.frequency_window,
                        self.discrete_window, self.transient_window_2,
                        self.frequency_window_2]
        return windows_list

    def btn_show_infos_clicked(self):

        dialog = QDialog(self)
        dialog.setWindowTitle("基本信息")
        dialog.setFixedWidth(300)
        dialog.setMaximumHeight(700)

        layout = QVBoxLayout()

        if not self.get_transfer_function_from_boxes():
            return

        poles = self.Hs.poles()

        poles_str = ", ".join(
            [f"{p.real:.2f}{'+' if p.imag>=0 else ''}{p.imag:.2f}j" for p in poles])
        label1 = QLabel(f"transfer function poles : [{poles_str}].")
        label1.setWordWrap(True)
        layout.addWidget(label1)

        Gs, Gb = self.get_system_transfer_functions()

        poles = Gs.poles()

        poles_str = ", ".join(
            [f"{p.real:.2f}{'+' if p.imag>=0 else ''}{p.imag:.2f}j" for p in poles])
        label2 = QLabel(f"open loop poles : [{poles_str}].")
        label2.setWordWrap(True)
        layout.addWidget(label2)

        poles = Gb.poles()

        poles_str = ", ".join(
            [f"{p.real:.2f}{'+' if p.imag>=0 else ''}{p.imag:.2f}j" for p in poles])
        label3 = QLabel(f"closed loop poles : [{poles_str}].")
        label3.setWordWrap(True)
        layout.addWidget(label3)

        dialog.setLayout(layout)
        dialog.exec()  # 阻塞式弹出

    def set_transfer_function_boxes(self, layout: QHBoxLayout, count: int, isnumerator: bool):

        while layout.count() > 1:
            item = layout.takeAt(1)  # 取出第一个 item
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # 从父容器移除
                widget.deleteLater()    # 释放资源

        if isnumerator:
            layout.addStretch(1)

        boxes = []
        for id in range(count):
            value_box = QLineEdit("")
            value_box.setAlignment(Qt.AlignmentFlag.AlignCenter)

            validator = QDoubleValidator(
                -1e8, 1e8, 2)  # 限定范围，小数点取 2 位
            validator.setNotation(QDoubleValidator.StandardNotation)
            value_box.setValidator(validator)

            boxes.append(value_box)
            layout.addWidget(value_box, 2)

        if isnumerator:
            layout.addStretch(1)

        return boxes

    def on_combobox1_clicked(self):

        order = self.comboboxes[0].currentIndex() + 1

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_2, order, True)

        self.numerator_boxes = value_boxes.copy()

        value_boxes = self.set_transfer_function_boxes(
            self.tf_layout_3, order + 1, False)

        self.denominator_boxes = value_boxes.copy()

        self.comboboxes[1].setCurrentIndex(0)

    def on_combobox2_clicked(self):

        method = self.comboboxes[1].currentIndex()

        # ZPK 配置方式
        if method == 1:
            order = self.comboboxes[0].currentIndex() + 1
            self.dialog_zpk = ZPKDialog(order=order, parent=self.parent)

            if self.dialog_zpk.exec() == QDialog.Accepted:
                Gs = ct.zpk(
                    self.dialog_zpk.zeros, self.dialog_zpk.poles, self.dialog_zpk.gain)

                num = Gs.num[0][0]
                den = Gs.den[0][0]

                for i, value in enumerate(reversed(num)):
                    self.numerator_boxes[order - 1 - i].setText(str(value))

                for i, value in enumerate(reversed(den)):
                    self.denominator_boxes[order - i].setText(str(value))

    def get_transfer_function_from_boxes(self):

        # 获取传递函数
        order = self.comboboxes[0].currentIndex() + 1

        num = np.zeros(order, dtype=float)
        den = np.zeros(order + 1, dtype=float)

        for id, box in enumerate(self.numerator_boxes):
            box_str = box.text().strip()
            if box_str == "":
                num[id] = 0.0
            else:
                num[id] = float(box_str)

        for id, box in enumerate(self.denominator_boxes):
            box_str = box.text().strip()
            if box_str == "":
                den[id] = 0.0
            else:
                den[id] = float(box_str)

        if np.allclose(num, 0, 0, 1e-6) or np.allclose(den, 0, 0, 1e-6):
            self.parent.status_label.setText(
                "Numerator or denominator is Null.")
            return False

        self.Hs = ct.minreal(ct.tf(num, den))

        return True

    def btn_transient_responses_clicked(self):

        if not self.get_transfer_function_from_boxes():
            return

        # 时域响应
        dt = self.parent.dt
        time_stop = float(self.parent.sim_boxes[2].text())

        time = np.arange(0, time_stop, dt)

        # 1. impulse response
        t, y = ct.impulse_response(self.Hs, T=time)

        if self.transient_window is None or not self.transient_window.isVisible():
            self.transient_window = TransientWindow()   # 新建

        self.transient_window.show()
        self.transient_window.raise_()
        self.transient_window.activateWindow()

        plot_widget_i = self.transient_window.plot_widget_1
        plot_widget_j = self.transient_window.plot_widget_2
        plot_widget_k = self.transient_window.plot_widget_3

        for widget in [plot_widget_i, plot_widget_j, plot_widget_k]:
            self.clear_plot_widget(widget)

        plot_widget_i.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="脉冲响应")

        plot_widget_i.setLabel("left", "displacement (m)")

        # 2. step response
        t, y = ct.step_response(self.Hs, T=time)

        plot_widget_j.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="阶跃响应")

        plot_widget_j.setLabel("left", "displacement (m)")

        # 3. sine response
        amplitude = float(self.parent.sim_boxes[0].text())
        omega = 2*np.pi*float(self.parent.sim_boxes[1].text())
        x = amplitude * np.sin(omega*time)
        t, y = ct.forced_response(self.Hs, T=time, inputs=x)

        plot_widget_k.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="正弦响应")

        plot_widget_k.setLabel("left", "displacement (m)")

    def clear_plot_widget(self, plot_widget):
        # 清除子图内的元素
        plot_widget.setTitle("")
        plot_widget.clear()
        if plot_widget.plotItem.legend is not None:
            plot_widget.plotItem.legend.scene().removeItem(plot_widget.plotItem.legend)
            plot_widget.plotItem.legend = None
        plot_widget.plotItem.addLegend()

    def btn_frequency_analyses_clicked(self):

        if not self.get_transfer_function_from_boxes():
            return

        mag, phase, omega = ct.frequency_response(self.Hs)

        if self.frequency_window is None or not self.frequency_window.isVisible():
            self.frequency_window = FrequencyWindow()   # 新建

        self.frequency_window.show()
        self.frequency_window.raise_()
        self.frequency_window.activateWindow()

        plot_widget_i = self.frequency_window.plot_widget_1
        plot_widget_j = self.frequency_window.plot_widget_2

        for widget in [plot_widget_i, plot_widget_j]:
            self.clear_plot_widget(widget)

        plot_widget_i.plot(omega, 20*np.log10(mag), pen=pg.mkPen(
            color='b', width=2), name='uncontrol')

        # 添加横轴
        plot_widget_i.addLine(y=0, pen=pg.mkPen(color='k', width=1))

        # Nyquist plot
        realpart = mag * np.cos(phase)
        imagpart = mag * np.sin(phase)

        xdata = np.concatenate([+realpart[::-1], realpart])
        ydata = np.concatenate([-imagpart[::-1], imagpart])

        plot_widget_j.plot(xdata, ydata, pen=pg.mkPen(
            color='b', width=2), name='uncontrol')

        # 添加横轴与纵轴
        plot_widget_j.addLine(x=0, pen=pg.mkPen(color='k', width=1))
        plot_widget_j.addLine(y=0, pen=pg.mkPen(color='k', width=1))

    def btn_discrete_control_clicked(self):

        if not self.get_transfer_function_from_boxes():
            return

        if self.discrete_window is None or not self.discrete_window.isVisible():
            self.discrete_window = TransientWindow()   # 新建

        self.discrete_window.show()
        self.discrete_window.raise_()
        self.discrete_window.activateWindow()

        plot_widget_i = self.discrete_window.plot_widget_1
        plot_widget_j = self.discrete_window.plot_widget_2
        plot_widget_k = self.discrete_window.plot_widget_3

        plot_widget_i.plotItem.clear()
        plot_widget_j.plotItem.clear()
        plot_widget_k.plotItem.clear()

        self.parent.status_label.setText("Simulation starts.")

        dt = self.parent.dt
        Gz = ct.c2d(self.Hs, dt, 'tustin')

        num = Gz.num[0][0]
        den = Gz.den[0][0]

        den = den/num[0]
        num = num/num[0]

        na = len(num)
        nb = len(den)

        if len(num) > len(den):
            self.parent.status_label.setText(
                "Order of num must be less than that of den in G(s).")
            return

        self.system_params_initialization()

        nt = len(self.time)

        error_history = np.zeros(nb, dtype=float)
        force_history = np.zeros(na-1, dtype=float)

        self.x_series = np.zeros(nt, dtype=float)
        self.v_series = np.zeros(nt, dtype=float)
        self.a_series = np.zeros(nt, dtype=float)
        self.f_series = np.zeros(nt, dtype=float)

        control_type = self.parent.ctrl_comboboxes[0].currentIndex()

        for id in range(nt-1):
            target_list = [self.parent.msd.x, self.parent.msd.v,
                           self.parent.msd.a, 0.0]
            error = self.target_value - target_list[control_type]

            # 向左滑动平移
            error_history[:-1] = error_history[1:]
            error_history[-1] = error  # 最新元素总在最右边

            force_history[:-1] = force_history[1:]
            force_history[-1] = self.f_series[id-1]  # 最新元素总在最右边

            force = np.sum(np.flip(den) * error_history) - \
                np.sum(np.flip(num[1:]) * force_history)

            force = max(min(force, 1e5), -1e5)  # 最大输出载荷 100,000 N
            self.f_series[id] = force

            self.parent.msd.update(self.excitation[id:id+2], self.f_series[id])

            self.x_series[id] = self.parent.msd.x
            self.v_series[id] = self.parent.msd.v
            self.a_series[id] = self.parent.msd.a

        # plot figures
        plot_widget_i.plot(self.time[:-1], self.x_series[:-1], pen=pg.mkPen(
            color='b', width=2), name="位移响应")
        plot_widget_i.plot(self.time[:-1], self.excitation[:-1], pen=pg.mkPen(
            color='r', width=2), name="激励信号")
        plot_widget_j.plot(self.time[:-1], self.a_series[:-1], pen=pg.mkPen(
            color='b', width=2), name="加速度响应")
        plot_widget_k.plot(self.time[:-1], self.f_series[:-1], pen=pg.mkPen(
            color='b', width=2), name="控制载荷")

        self.parent.status_label.setText("Simulation finishes.")

    def get_system_transfer_functions(self):

        m = float(self.parent.msd_params_boxes[0].text())
        c = float(self.parent.msd_params_boxes[1].text())
        k = float(self.parent.msd_params_boxes[2].text())

        Gs = ct.tf([c, k], [m, c, k])

        control_type = self.parent.ctrl_comboboxes[0].currentIndex()

        if control_type < 3:
            if not self.get_transfer_function_from_boxes():
                return Gs, Gs

            Hx = self.Hs

            numx = Hx.num[0][0]
            denx = Hx.den[0][0]

            if control_type == 1:
                denx.extend([0])

            if control_type == 2:
                denx.extend([0, 0])

            Hx = ct.tf(numx, denx)

            Gf = ct.tf(1, [c, k])  # 力传递函数

            Gb = Hx * Gs / (Hx + Gs * Gf)  # 闭环系统传递函数

            Gb = ct.minreal(Gb, verbose=False)

        else:
            Gb = Gs

        return Gs, Gb

    def btn_model_transient_response_clicked(self):

        self.system_params_initialization()

        if self.transient_window_2 is None or not self.transient_window_2.isVisible():
            self.transient_window_2 = TransientWindow()   # 新建

        self.transient_window_2.show()
        self.transient_window_2.raise_()
        self.transient_window_2.activateWindow()

        plot_widget_i = self.transient_window_2.plot_widget_1
        plot_widget_j = self.transient_window_2.plot_widget_2
        plot_widget_k = self.transient_window_2.plot_widget_3

        for widget in [plot_widget_i, plot_widget_j, plot_widget_k]:
            self.clear_plot_widget(widget)

        dt = self.parent.dt
        time_stop = float(self.parent.sim_boxes[2].text())
        time = np.arange(0, time_stop, dt)

        Gs, Gb = self.get_system_transfer_functions()

        t, y = ct.impulse_response(Gs, time)
        plot_widget_i.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="开环系统脉冲响应")

        t, y = ct.impulse_response(Gb, time)
        plot_widget_i.plot(t, y, pen=pg.mkPen(
            color='r', width=2), name="闭环系统脉冲响应")

        plot_widget_i.setLabel("left", "displacement (m)")

        t, y = ct.step_response(Gs, time)
        plot_widget_j.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="开环系统阶跃响应")

        t, y = ct.step_response(Gb, time)
        plot_widget_j.plot(t, y, pen=pg.mkPen(
            color='r', width=2), name="闭环系统阶跃响应")

        plot_widget_j.setLabel("left", "displacement (m)")

        # 3. sine response
        amplitude = float(self.parent.sim_boxes[0].text())
        omega = 2*np.pi*float(self.parent.sim_boxes[1].text())
        x = amplitude * np.sin(omega*time)

        t, y = ct.forced_response(Gs, T=time, inputs=x)
        plot_widget_k.plot(t, y, pen=pg.mkPen(
            color='b', width=2), name="开环系统正弦响应")

        t, y = ct.forced_response(Gb, T=time, inputs=x)
        plot_widget_k.plot(t, y, pen=pg.mkPen(
            color='r', width=2), name="闭环系统正弦响应")

        plot_widget_k.setLabel("left", "displacement (m)")

    def btn_model_frequency_analyses_clicked(self):

        self.system_params_initialization()

        if self.frequency_window_2 is None or not self.frequency_window_2.isVisible():
            self.frequency_window_2 = FrequencyWindow()   # 新建

        self.frequency_window_2.show()
        self.frequency_window_2.raise_()
        self.frequency_window_2.activateWindow()

        plot_widget_i = self.frequency_window_2.plot_widget_1
        plot_widget_j = self.frequency_window_2.plot_widget_2

        for widget in [plot_widget_i, plot_widget_j]:
            self.clear_plot_widget(widget)

        Gs, Gb = self.get_system_transfer_functions()

        # 1. 开环系统频率特性
        mag, phase, omega = ct.frequency_response(Gs)

        plot_widget_i.plot(omega, 20*np.log10(mag), pen=pg.mkPen(
            color='b', width=2), name='开环系统频率特性')

        # 添加横轴
        plot_widget_i.addLine(y=0, pen=pg.mkPen(color='k', width=1))

        # Nyquist plot
        realpart = mag * np.cos(phase)
        imagpart = mag * np.sin(phase)

        xdata = np.concatenate([+realpart[::-1], realpart])
        ydata = np.concatenate([-imagpart[::-1], imagpart])

        plot_widget_j.plot(xdata, ydata, pen=pg.mkPen(
            color='b', width=2), name='开环系统频率特性')

        # 2. 闭环系统频率特性
        mag, phase, omega = ct.frequency_response(Gb)

        plot_widget_i.plot(omega, 20*np.log10(mag), pen=pg.mkPen(
            color='r', width=2), name='闭环系统频率特性')

        # 添加横轴
        plot_widget_i.addLine(y=0, pen=pg.mkPen(color='k', width=1))

        # Nyquist plot
        realpart = mag * np.cos(phase)
        imagpart = mag * np.sin(phase)

        xdata = np.concatenate([+realpart[::-1], realpart])
        ydata = np.concatenate([-imagpart[::-1], imagpart])

        plot_widget_j.plot(xdata, ydata, pen=pg.mkPen(
            color='r', width=2), name='闭环系统频率特性')

        # 添加横轴与纵轴
        plot_widget_j.addLine(x=0, pen=pg.mkPen(color='k', width=1))
        plot_widget_j.addLine(y=0, pen=pg.mkPen(color='k', width=1))

    def system_params_initialization(self):

        self.parent.msd.m = float(self.parent.msd_params_boxes[0].text())
        self.parent.msd.c = float(self.parent.msd_params_boxes[1].text())
        self.parent.msd.k = float(self.parent.msd_params_boxes[2].text())

        self.parent.msd.x = 0.0
        self.parent.msd.v = 0.0
        self.parent.msd.a = 0.0

        self.time_stop = float(self.parent.sim_boxes[2].text())

        dt = self.parent.dt
        self.time = np.arange(0, self.time_stop, dt)
        nt = len(self.time)

        # 获取外部激励信号类型
        signal_type = self.parent.sim_comboboxes[0].currentIndex()

        signal = np.zeros(nt, dtype=float)
        amplitude = float(self.parent.sim_boxes[0].text())

        if signal_type == 1:  # step
            step_start = int(0.1 / dt)
            signal[step_start:] = amplitude

        elif signal_type == 2:  # sine
            omega = 2 * np.pi * float(self.parent.sim_boxes[1].text())
            signal = amplitude * np.sin(omega * self.time)

        else:  # impulse
            impulse_time_period = 0.1
            # 幅值取小点，否则绘图区装不下
            amplitude = amplitude / impulse_time_period

            step_start = int(0.1 / dt)
            step_stop = int((0.1 + impulse_time_period) / dt)
            signal[step_start:step_stop] = amplitude

        if self.parent.sim_checkboxes[0].isChecked():
            self.parent.status_label.setText("外部导入文件方法尚未完成。")

        self.excitation = signal

        self.target_value = float(self.parent.ctrl_boxes[3].text())

        self.x_series = np.zeros(nt, dtype=float)
        self.v_series = np.zeros(nt, dtype=float)
        self.a_series = np.zeros(nt, dtype=float)
        self.f_series = np.zeros(nt, dtype=float)
