# opengl_widget.py


from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QTabWidget)
from PySide6.QtGui import QIcon


from MSDModel import *
from MSDChart import *
from MSDTabs import *


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("质量-阻尼-弹簧控制系统")
        self.setWindowIcon(QIcon("icon/rocket-launch.png"))
        self.setFixedSize(420, 730)
        self.move(200, 100)

        self.data = DataGroup()
        self.msd = MSDPlant(self.data)
        self.pid = PIDController(self.data)
        self.lqr = LQRController(self.data)

        self._setup_ui()

        self.tabpage1.status_changed.connect(self.update_status_infos)
        self.tabpage2.status_changed.connect(self.update_status_infos)

        self.tabpage1.data_updated.connect(
            lambda: self.msd.update_parameters())
        self.tabpage1.data_updated.connect(
            lambda: self.pid.update_parameters())
        self.tabpage1.data_updated.connect(
            lambda: self.lqr.update_parameters())

    def _setup_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.msdviz = MassSpringDamperGL()
        main_layout.addWidget(self.msdviz, 2)  # Take up 2/3 of the space

        self.tabpages = QTabWidget()
        self.tabpage1 = TabPage1(self.data, self.msd, self.pid, self.msdviz)
        self.tabpage2 = TabPage2(
            self.data, self.msd, self.pid, self.lqr, self.tabpage1)

        self.tabpages.addTab(self.tabpage1, "基础控制")
        self.tabpages.addTab(self.tabpage2, "进阶控制")

        main_layout.addWidget(self.tabpages)

        self._apply_styles()
        self.statusBar().setFixedHeight(20)

    def update_status_infos(self, message):
        self.statusBar().showMessage(message, 4000)

    def closeEvent(self, event):
        # 主窗口关闭前，先关闭子窗口

        for window in self.tabpage1.get_windows():
            if window is not None:
                window.close()

        for window in self.tabpage2.get_windows():
            if window is not None:
                window.close()

        super().closeEvent(event)

    def _apply_styles(self):
        self.setStyleSheet("""
            MainWindow {
                background-color: #f0f0f0;
            }
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
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QLineEdit[valid="false"]{
                border: 1px solid red;
                background-color: #ffebeb;
            }

            /* 下拉按钮样式 */
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 13px;                /* 按钮宽度 */
                border-left: 1px solid #CCCCCC;  /* 分隔线 */
                background-color: #60A5FA;  /* 淡青色背景 */
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            /* 下拉箭头样式 */
            QComboBox::down-arrow {
                width: 8px;   /* 箭头宽度 */
                height: 8px;  /* 箭头高度 */
            }
            /* 下拉列表样式 */
            QComboBox QAbstractItemView {
                border: 1px solid #D1D5DB;
                border-radius: 4px;
                background-color: #FFFFFF;
                selection-background-color: #3B82F6;
                selection-color: #333333;
            }

            QLabel {
                padding: 2px 0;
            }
            /* 整个 TabWidget 背景 */
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                padding: 0px;
            }
            /* Tab bar 区域 */
            QTabBar::tab {
                border: 1px solid #d0d0d0;
                padding: 6px 10px;
                margin: 2px;
                color: #333;
                font-size: 14px;
                border-radius: 6px;
            }
            /* 被选中的 Tab */
            QTabBar::tab:selected {
                background: #0078d4;   /* Windows 11 蓝 */
                color: white;
            }
            /* 未选中时 */
            QTabBar::tab:!selected {
                background: transparent;
                color: #444;
            }
            /*
            * 复选框指示器 (未选中状态)
            */
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #999999; /* 边框颜色 */
                border-radius: 4px; /* 轻微的圆角 */
                background-color: #f0f0f0; /* 背景颜色 */
            }
            /*
            * 复选框指示器 (鼠标悬停状态)
            */
            QCheckBox::indicator:hover {
                border: 1px solid #5d5d5d;
                background-color: #e0e0e0;
            }
            /*
            * 复选框指示器 (选中状态)
            */
            QCheckBox::indicator:checked {
                border: 1px solid #0078d4; /* 边框变为蓝色 */
                background-color: #0078d4; /* 背景变为蓝色 */
                image: url(icon/check.png); /* 添加一个对勾图标 */
            }
            """)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
