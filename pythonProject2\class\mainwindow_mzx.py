"""PCB缺陷检测系统"""
import sys
import os
import cv2
import numpy as np
import json
import time
import threading
import subprocess
from datetime import datetime
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtWidgets import (QMainWindow, QLabel, QComboBox, QPushButton,
                             QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QTextEdit, QWidget, QMessageBox, QProgressBar,
                             QSplitter, QListWidget, QDialog, QDoubleSpinBox)
from PyQt5.QtGui import QImage, QPixmap, QFont

# 依赖检查（自动安装缺失依赖）
def check_dependencies():
    """检查并自动安装必要的Python依赖包"""
    required_packages = ['PyQt5', 'opencv-python', 'numpy']
    for package in required_packages:
        pkg_name = 'cv2' if package == 'opencv-python' else package.replace('-', '_')
        try:
            __import__(pkg_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            __import__(pkg_name)

check_dependencies()

# YOLO导入标记
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO = None

class DetectionHistory:
    """检测历史管理类"""
    def __init__(self, max_history=50):
        """初始化历史记录配置"""
        self.history_file = "detection_history.json"
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.load_history()

    def load_history(self):
        """加载本地保存的检测历史记录"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.history = deque(data[-self.max_history:], maxlen=self.max_history)

    def save_history(self):
        """将当前检测历史保存到本地文件"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.history), f, ensure_ascii=False, indent=2)

    def add_record(self, record):
        """添加一条新的检测历史记录"""
        self.history.append(record)
        self.save_history()

    def get_recent(self, count=10):
        """获取最近的N条检测历史记录（倒序）"""
        return list(self.history)[-count:][::-1]

    def clear_history(self):
        """清空所有检测历史记录及本地文件"""
        self.history.clear()
        if os.path.exists(self.history_file):
            os.remove(self.history_file)

class SimulatedDetector:
    """模拟缺陷检测器（无YOLO模型时使用）"""
    def __init__(self, model_name="模拟模型"):
        """初始化模拟检测器的缺陷类别和置信度阈值"""
        self.class_names = {0:"短路",1:"开路",2:"焊锡桥接",3:"焊锡不足",4:"异物残留",5:"线路划痕"}
        self.confidence_threshold = 0.5
        self.model_name = model_name

    def detect(self, image):
        """模拟缺陷检测，生成随机缺陷数据"""
        h, w = image.shape[:2]
        detections = []
        np.random.seed(int(time.time()))
        for i in range(np.random.randint(1, 5)):
            x1 = np.random.randint(50, w - 150)
            y1 = np.random.randint(50, h - 150)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(50, 150)
            class_id = np.random.randint(0, len(self.class_names))
            detections.append({
                'class_id': class_id,
                'class_name': self.class_names[class_id],
                'confidence': np.random.uniform(0.6, 0.95),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'center': [float((x1+x2)/2), float((y1+y2)/2)],
                'width': float(x2-x1),
                'height': float(y2-y1)
            })
        return detections

    def draw_detections(self, image, detections):
        """在图像上绘制缺陷检测框和标签"""
        if image is None or len(image.shape) < 3:
            return image
        img_copy = image.copy()
        colors = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,255,255)]
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            color = colors[det['class_id'] % len(colors)]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_copy, (x1, y1-th-10), (x1+tw, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return img_copy

class YOLODetector:
    """YOLO实际缺陷检测器"""
    def __init__(self, model_path=None, device='cpu'):
        """初始化YOLO模型路径、设备和缺陷类别"""
        self.model = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path) if model_path else "未知模型"
        self.device = device
        self.class_names = {0:"短路",1:"开路",2:"焊锡桥接",3:"焊锡不足",4:"异物残留",5:"线路划痕"}
        self.confidence_threshold = 0.5
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path):
        """加载YOLO预训练模型"""
        if not YOLO_AVAILABLE or YOLO is None:
            self.model = None
            return
        self.model = YOLO(model_path)
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names

    def detect(self, image):
        """使用YOLO模型执行缺陷检测"""
        if self.model is None:
            return SimulatedDetector(self.model_name).detect(image)
        results = self.model(source=image, conf=self.confidence_threshold, device=self.device, verbose=False)
        detections = []
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls_id = int(box.cls[0])
                    detections.append({
                        'class_id': cls_id,
                        'class_name': self.class_names.get(cls_id, f"缺陷_{cls_id}"),
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float((x1+x2)/2), float((y1+y2)/2)],
                        'width': float(x2-x1),
                        'height': float(y2-y1)
                    })
        return detections

    def draw_detections(self, image, detections):
        """调用模拟检测器的绘制方法，统一绘制样式"""
        return SimulatedDetector(self.model_name).draw_detections(image, detections)

class VideoDetectionThread(QThread):
    """视频检测独立线程（避免阻塞UI）"""
    frame_ready = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)

    def __init__(self, detector, video_path, frame_interval=1.0):
        """初始化视频检测线程的检测器、视频路径和抽帧间隔"""
        super().__init__()
        self.detector = detector
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.is_running = True
        self.is_paused = False
        self.all_detections = []

    def run(self):
        """线程主执行函数：执行视频帧读取和缺陷检测"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit([])
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, int(cap.get(cv2.CAP_PROP_FPS) * self.frame_interval))
        frame_count = 0
        while self.is_running and cap.isOpened():
            while self.is_paused and self.is_running:
                time.sleep(0.1)
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            detections = self.detector.detect(frame)
            self.all_detections.extend(detections)
            self.frame_ready.emit(self.detector.draw_detections(frame, detections))
            if total_frames > 0:
                self.progress.emit(min(int((frame_count/total_frames)*100), 100))
            time.sleep(0.03)
        cap.release()
        self.finished.emit(self.all_detections)

    def pause(self):
        """暂停视频检测"""
        self.is_paused = True

    def resume(self):
        """恢复视频检测"""
        self.is_paused = False

    def stop(self):
        """停止视频检测并终止线程"""
        self.is_running = False
        self.is_paused = False

class PCBMainWindowComplete(QMainWindow):
    """PCB缺陷检测系统主窗口"""
    def __init__(self):
        """初始化窗口变量、UI和核心组件"""
        super().__init__()
        self.detector = None
        self.current_image = None
        self.current_video_path = None
        self.camera_active = False
        self.cap = None
        self.camera_timer = QTimer()
        self.video_thread = None
        self.models_dir = "./class/models"
        if not os.path.exists(self.models_dir):
            self.models_dir = "./models" if os.path.exists("./models") else "models"
        self.video_playing = False
        self.video_paused = False
        self.init_ui()
        self.init_output_directories()
        self.history_manager = DetectionHistory()
        self.connect_signals()
        self.load_model_list()
        if self.model_combo.count() == 0:
            self.status_label.setText("无模型，使用模拟检测器")
            self.detector = SimulatedDetector("模拟模型")
        else:
            self.model_combo.setCurrentIndex(0)
            QtCore.QTimer.singleShot(100, self.load_selected_model)
        self.update_history_display()

    def init_output_directories(self):
        """初始化输出目录（截图、检测结果）"""
        self.output_dir = "./output"
        self.screenshots_dir = os.path.join(self.output_dir, "screenshots")
        self.results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.status_label.setText(f"输出目录: {os.path.abspath(self.output_dir)}")

    def init_ui(self):
        """构建系统主界面布局和控件"""
        self.setWindowTitle("24人工智能3-1班马子欣实训项目---PCB缺陷检测系统")
        self.setGeometry(100, 100, 1400, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        # 原始图像组
        source_group = QGroupBox("原始图像/视频")
        source_layout = QVBoxLayout(source_group)
        self.source_label = QLabel("选择图像/打开摄像头")
        self.source_label.setAlignment(Qt.AlignCenter)
        self.source_label.setMinimumSize(500, 350)
        self.source_label.setStyleSheet("QLabel{border:2px dashed #ccc;background:#f8f8f8;color:#666;font-size:14px;}")
        source_layout.addWidget(self.source_label)
        # 原始图像按钮布局
        source_btn_layout = QHBoxLayout()
        self.browse_image_btn = QPushButton("上传图像")
        self.browse_video_btn = QPushButton("上传视频")
        self.camera_btn = QPushButton("打开摄像头")
        self.save_source_btn = QPushButton("保存原始截图")
        self.show_output_btn = QPushButton("显示输出目录")
        source_btn_layout.addWidget(self.browse_image_btn)
        source_btn_layout.addWidget(self.browse_video_btn)
        source_btn_layout.addWidget(self.camera_btn)
        source_btn_layout.addWidget(self.save_source_btn)
        source_btn_layout.addWidget(self.show_output_btn)
        source_layout.addLayout(source_btn_layout)
        left_layout.addWidget(source_group)
        # 检测结果组
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout(result_group)
        self.result_label = QLabel("检测结果显示区")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(500, 350)
        self.result_label.setStyleSheet("QLabel{border:2px dashed #ccc;background:#f8f8f8;color:#666;font-size:14px;}")
        result_layout.addWidget(self.result_label)
        # 结果按钮布局
        result_btn_layout = QHBoxLayout()
        self.save_result_btn = QPushButton("保存结果截图")
        self.fullscreen_btn = QPushButton("全屏查看")
        self.save_result_custom_btn = QPushButton("自定义保存")
        result_btn_layout.addWidget(self.save_result_btn)
        result_btn_layout.addWidget(self.fullscreen_btn)
        result_btn_layout.addWidget(self.save_result_custom_btn)
        result_layout.addLayout(result_btn_layout)
        left_layout.addWidget(result_group)
        # 右侧控制区
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        # 模型选择组
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        model_layout.addWidget(QLabel("选择模型:"))
        model_layout.addWidget(self.model_combo)
        self.refresh_models_btn = QPushButton("刷新模型")
        model_layout.addWidget(self.refresh_models_btn)
        right_layout.addWidget(model_group)
        # 检测设置组
        detect_group = QGroupBox("检测设置")
        detect_layout = QVBoxLayout(detect_group)
        # 阈值布局
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("置信度阈值:"))
        self.threshold_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 99)
        self.threshold_slider.setValue(50)
        self.threshold_label = QLabel("0.50")
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        detect_layout.addLayout(threshold_layout)
        # 模式布局
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("检测模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["单张图像", "视频文件", "实时摄像头"])
        mode_layout.addWidget(self.mode_combo)
        detect_layout.addLayout(mode_layout)
        # 检测按钮
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.setMinimumHeight(40)
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        detect_layout.addWidget(self.detect_btn)
        detect_layout.addWidget(self.stop_btn)
        detect_layout.addWidget(self.progress_bar)
        right_layout.addWidget(detect_group)
        # 视频控制组
        video_group = QGroupBox("视频控制")
        video_layout = QVBoxLayout(video_group)
        frame_interval_layout = QHBoxLayout()
        frame_interval_layout.addWidget(QLabel("抽帧间隔(秒):"))
        self.frame_interval_spin = QDoubleSpinBox()
        self.frame_interval_spin.setRange(0.1, 10.0)
        self.frame_interval_spin.setValue(1.0)
        self.frame_interval_spin.setSingleStep(0.5)
        frame_interval_layout.addWidget(self.frame_interval_spin)
        video_layout.addLayout(frame_interval_layout)
        # 视频按钮布局
        video_btn_layout = QHBoxLayout()
        self.play_video_btn = QPushButton("播放")
        self.pause_video_btn = QPushButton("暂停")
        self.stop_video_btn = QPushButton("停止")
        self.play_video_btn.setEnabled(False)
        self.pause_video_btn.setEnabled(False)
        self.stop_video_btn.setEnabled(False)
        video_btn_layout.addWidget(self.play_video_btn)
        video_btn_layout.addWidget(self.pause_video_btn)
        video_btn_layout.addWidget(self.stop_video_btn)
        video_layout.addLayout(video_btn_layout)
        right_layout.addWidget(video_group)
        # 历史记录组
        history_group = QGroupBox("检测历史")
        history_layout = QVBoxLayout(history_group)
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(200)
        history_layout.addWidget(self.history_list)
        history_btn_layout = QHBoxLayout()
        self.clear_history_btn = QPushButton("清空历史")
        self.refresh_history_btn = QPushButton("刷新")
        history_btn_layout.addWidget(self.clear_history_btn)
        history_btn_layout.addWidget(self.refresh_history_btn)
        history_layout.addLayout(history_btn_layout)
        right_layout.addWidget(history_group)
        # 检测统计组
        stats_group = QGroupBox("检测统计")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlainText("等待检测...")
        stats_layout.addWidget(self.stats_text)
        right_layout.addWidget(stats_group)
        # 组装布局
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 400])
        main_layout.addWidget(splitter)
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)

    def connect_signals(self):
        """连接控件信号与对应槽函数"""
        self.browse_image_btn.clicked.connect(self.browse_image)
        self.browse_video_btn.clicked.connect(self.browse_video)
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.save_source_btn.clicked.connect(self.save_source_screenshot)
        self.save_result_btn.clicked.connect(self.save_result_screenshot)
        self.save_result_custom_btn.clicked.connect(lambda: self.save_custom_screenshot(self.result_label, "检测结果"))
        self.show_output_btn.clicked.connect(self.show_output_directory)
        self.detect_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.fullscreen_btn.clicked.connect(self.view_fullscreen)
        self.refresh_models_btn.clicked.connect(self.load_model_list)
        self.clear_history_btn.clicked.connect(self.clear_history)
        self.refresh_history_btn.clicked.connect(self.update_history_display)
        self.model_combo.currentIndexChanged.connect(self.load_selected_model)
        self.play_video_btn.clicked.connect(self.play_video)
        self.pause_video_btn.clicked.connect(self.pause_video)
        self.stop_video_btn.clicked.connect(self.stop_video)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.camera_timer.timeout.connect(self.update_camera)

    def update_history_display(self):
        """更新历史记录列表显示"""
        self.history_list.clear()
        for record in self.history_manager.get_recent(20):
            item_text = f"{record.get('timestamp','未知')} - {record.get('mode','未知')} - {record.get('filename','未知')} - 缺陷:{record.get('defect_count',0)}"
            self.history_list.addItem(item_text)

    def add_to_history(self, mode, filename, defect_count, processing_time=0):
        """添加检测记录到历史管理"""
        self.history_manager.add_record({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': mode,
            'filename': filename,
            'defect_count': defect_count,
            'processing_time': f"{processing_time:.2f}s",
            'model': self.model_combo.currentText() if self.model_combo.count() > 0 else "模拟模型"
        })
        self.update_history_display()

    def clear_history(self):
        """弹窗确认并清空检测历史"""
        if QMessageBox.question(self, "确认", "清空所有历史记录？", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.history_manager.clear_history()
            self.update_history_display()
            self.status_label.setText("历史记录已清空")

    def save_source_screenshot(self):
        """保存原始图像到截图目录"""
        if self.source_label.pixmap() and not self.source_label.pixmap().isNull():
            filename = f"source_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self.screenshots_dir, filename)
            if self.source_label.pixmap().save(filepath, "JPEG"):
                QMessageBox.information(self, "成功", f"保存至:\n{filepath}")
                self.status_label.setText(f"已保存原始截图: {filename}")
                self.add_to_history("截图保存", filename, 0)
                os.startfile(self.screenshots_dir) if sys.platform == "win32" else subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", self.screenshots_dir])
            else:
                QMessageBox.critical(self, "错误", "保存失败")
        else:
            QMessageBox.warning(self, "警告", "无可用图像")

    def save_result_screenshot(self):
        """保存检测结果图像到结果目录"""
        if self.result_label.pixmap() and not self.result_label.pixmap().isNull():
            filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self.results_dir, filename)
            if self.result_label.pixmap().save(filepath, "JPEG"):
                QMessageBox.information(self, "成功", f"保存至:\n{filepath}")
                self.status_label.setText(f"已保存结果: {filename}")
                self.add_to_history("结果保存", filename, 0)
                os.startfile(self.results_dir) if sys.platform == "win32" else subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", self.results_dir])
            else:
                QMessageBox.critical(self, "错误", "保存失败")
        else:
            QMessageBox.warning(self, "警告", "无可用结果")

    def save_custom_screenshot(self, label, file_type="截图"):
        """自定义路径保存图像"""
        if label.pixmap() and not label.pixmap().isNull():
            default_filename = f"{file_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            file_path, _ = QFileDialog.getSaveFileName(self, f"保存{file_type}", os.path.join(self.output_dir, default_filename), "JPEG(*.jpg);;PNG(*.png)")
            if file_path:
                if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path += '.jpg'
                save_format = "PNG" if file_path.lower().endswith('.png') else "JPEG"
                if label.pixmap().save(file_path, save_format):
                    QMessageBox.information(self, "成功", f"保存至:\n{file_path}")
                    self.status_label.setText(f"已保存: {os.path.basename(file_path)}")
                    self.add_to_history("自定义保存", os.path.basename(file_path), 0)
                    os.startfile(os.path.dirname(file_path)) if sys.platform == "win32" else subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", os.path.dirname(file_path)])
                else:
                    QMessageBox.critical(self, "错误", "保存失败")
        else:
            QMessageBox.warning(self, "警告", "无可用图像")

    def show_output_directory(self):
        """显示输出目录信息并提供打开目录功能"""
        output_path = os.path.abspath(self.output_dir)
        if os.path.exists(output_path):
            screenshot_count = len([f for f in os.listdir(self.screenshots_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            result_count = len([f for f in os.listdir(self.results_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            dir_info = f"截图数: {screenshot_count} | 结果数: {result_count} | 总数: {screenshot_count+result_count}"
            msg = QMessageBox(self)
            msg.setWindowTitle("输出目录")
            msg.setText(f"路径:\n{output_path}")
            msg.setInformativeText(dir_info)
            open_btn = msg.addButton("打开目录", QMessageBox.ActionRole)
            msg.addButton("关闭", QMessageBox.RejectRole)
            msg.exec_()
            if msg.clickedButton() == open_btn:
                os.startfile(output_path) if sys.platform == "win32" else subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", output_path])
        else:
            QMessageBox.warning(self, "错误", "目录不存在")

    def load_model_list(self):
        """加载模型目录下的所有可用模型文件"""
        self.model_combo.clear()
        if not os.path.exists(self.models_dir):
            self.model_combo.addItem("无可用模型")
            return
        model_files = [f for f in os.listdir(self.models_dir) if f.lower().endswith(('.pt', '.pth', '.onnx'))]
        if not model_files:
            self.model_combo.addItem("无可用模型")
            self.status_label.setText("未找到模型文件")
        else:
            for model_file in sorted(model_files):
                self.model_combo.addItem(model_file)
            self.status_label.setText(f"找到 {len(model_files)} 个模型")

    def load_selected_model(self):
        """加载下拉框选中的YOLO模型"""
        if self.model_combo.count() == 0 or self.model_combo.currentText() == "无可用模型":
            return
        model_path = os.path.join(self.models_dir, self.model_combo.currentText())
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "错误", "模型文件不存在")
            return
        if not YOLO_AVAILABLE:
            self.detector = SimulatedDetector(self.model_combo.currentText())
            self.status_label.setText(f"YOLO不可用，使用模拟检测: {self.model_combo.currentText()}")
            return
        self.detector = YOLODetector(model_path, device='cpu')
        self.status_label.setText(f"模型加载成功: {self.model_combo.currentText()}")
        self.add_to_history("模型加载", self.model_combo.currentText(), 0)

    def update_threshold_label(self, value):
        """更新置信度阈值显示并同步到检测器"""
        threshold = value / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        if self.detector:
            self.detector.confidence_threshold = threshold

    def browse_image(self):
        """打开文件选择器选择并加载单张图像"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.load_image(file_path)

    def browse_video(self):
        """打开文件选择器选择视频文件并预览首帧"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.current_video_path = file_path
            filename = os.path.basename(file_path)
            self.status_label.setText(f"已选择视频: {filename}")
            self.preview_video(file_path)
            self.play_video_btn.setEnabled(True)
            self.add_to_history("视频选择", filename, 0)

    def preview_video(self, video_path):
        """预览视频文件的第一帧图像"""
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.display_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.source_label)
                self.current_image = frame
        cap.release()

    def load_image(self, file_path):
        """加载指定路径的图像文件并显示"""
        img = cv2.imread(file_path)
        if img is not None:
            self.current_image = img
            self.display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.source_label)
            filename = os.path.basename(file_path)
            self.status_label.setText(f"已加载: {filename}")
            self.add_to_history("图像加载", filename, 0)
        else:
            QMessageBox.critical(self, "错误", "加载失败")

    def display_image(self, cv_image, label):
        """将OpenCV图像转换为Qt图像并在指定标签显示"""
        h, w, ch = cv_image.shape
        q_img = QImage(cv_image.data, w, h, ch*w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def toggle_camera(self):
        """切换摄像头的开启/关闭状态"""
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "错误", "无法打开摄像头")
                return
            self.camera_active = True
            self.camera_btn.setText("关闭摄像头")
            self.camera_timer.start(30)
            self.status_label.setText("摄像头已开启")
            self.add_to_history("摄像头", "开启", 0)
        else:
            self.camera_active = False
            self.camera_timer.stop()
            self.cap.release()
            self.cap = None
            self.camera_btn.setText("打开摄像头")
            self.source_label.clear()
            self.source_label.setText("选择图像/打开摄像头")
            self.status_label.setText("摄像头已关闭")
            self.add_to_history("摄像头", "关闭", 0)

    def update_camera(self):
        """定时更新摄像头画面并执行实时检测"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_image(frame_rgb, self.source_label)
                self.current_image = frame
                if self.mode_combo.currentText() == "实时摄像头" and self.detector is not None:
                    self.detect_camera_frame(frame)

    def detect_camera_frame(self, frame):
        """对摄像头单帧图像执行缺陷检测并更新结果"""
        detections = self.detector.detect(frame)
        result_rgb = cv2.cvtColor(self.detector.draw_detections(frame, detections), cv2.COLOR_BGR2RGB)
        self.display_image(result_rgb, self.result_label)
        if detections:
            stats_text = f"实时检测\n时间: {datetime.now().strftime('%H:%M:%S')}\n缺陷数: {len(detections)}个\n"
            defect_stats = {}
            for det in detections:
                defect_stats[det['class_name']] = defect_stats.get(det['class_name'], 0) + 1
            for cls, cnt in defect_stats.items():
                stats_text += f"  {cls}: {cnt}个\n"
            self.stats_text.setPlainText(stats_text)

    def start_detection(self):
        """根据选择的模式启动对应的缺陷检测流程"""
        if self.detector is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        mode = self.mode_combo.currentText()
        if mode == "单张图像":
            if self.current_image is None:
                QMessageBox.warning(self, "警告", "请先选择图像")
                return
            threading.Thread(target=self.detect_image_thread).start()
        elif mode == "视频文件":
            if not self.current_video_path:
                QMessageBox.warning(self, "警告", "请先选择视频")
                return
            self.play_video()
        elif mode == "实时摄像头":
            if not self.camera_active:
                QMessageBox.warning(self, "警告", "请先打开摄像头")
                return
            self.status_label.setText("实时检测已启动")

    def detect_image_thread(self):
        """独立线程执行单张图像缺陷检测，避免阻塞UI"""
        start_time = time.time()
        detections = self.detector.detect(self.current_image)
        result_rgb = cv2.cvtColor(self.detector.draw_detections(self.current_image, detections), cv2.COLOR_BGR2RGB)
        QtCore.QMetaObject.invokeMethod(self, "update_image_result", QtCore.Qt.QueuedConnection,
                                      QtCore.Q_ARG(np.ndarray, result_rgb), QtCore.Q_ARG(list, detections),
                                      QtCore.Q_ARG(float, time.time()-start_time))

    @QtCore.pyqtSlot(np.ndarray, list, float)
    def update_image_result(self, result_image, detections, processing_time):
        """更新单张图像检测结果显示和统计信息"""
        self.display_image(result_image, self.result_label)
        self.update_stats(detections, processing_time)
        filename = "摄像头画面" if self.camera_active else "图像文件"
        self.status_label.setText(f"检测完成，缺陷{len(detections)}个，耗时{processing_time:.2f}秒")
        self.add_to_history("图像检测", filename, len(detections), processing_time)

    def play_video(self):
        """启动视频缺陷检测线程"""
        self.video_thread = VideoDetectionThread(self.detector, self.current_video_path, self.frame_interval_spin.value())
        self.video_thread.frame_ready.connect(self.on_video_frame)
        self.video_thread.progress.connect(self.update_progress)
        self.video_thread.finished.connect(self.on_video_finished)
        self.play_video_btn.setEnabled(False)
        self.pause_video_btn.setEnabled(True)
        self.stop_video_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.video_playing = True
        self.video_thread.start()
        filename = os.path.basename(self.current_video_path)
        self.status_label.setText(f"检测中: {filename}")
        self.add_to_history("视频检测", filename, 0)

    def pause_video(self):
        """切换视频检测的暂停/继续状态"""
        if self.video_thread and self.video_playing:
            if not self.video_paused:
                self.video_thread.pause()
                self.pause_video_btn.setText("继续")
                self.video_paused = True
                self.status_label.setText("已暂停")
            else:
                self.video_thread.resume()
                self.pause_video_btn.setText("暂停")
                self.video_paused = False
                self.status_label.setText("检测中...")

    def stop_video(self):
        """停止视频检测并重置控件状态"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        self.video_playing = False
        self.video_paused = False
        self.play_video_btn.setEnabled(True)
        self.pause_video_btn.setEnabled(False)
        self.stop_video_btn.setEnabled(False)
        self.pause_video_btn.setText("暂停")
        self.progress_bar.setVisible(False)
        self.status_label.setText("视频检测已停止")

    def on_video_frame(self, frame):
        """更新视频检测过程中的单帧结果显示"""
        self.display_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.result_label)

    def on_video_finished(self, detections):
        """视频检测完成后更新统计信息和控件状态"""
        self.video_playing = False
        self.video_paused = False
        self.play_video_btn.setEnabled(True)
        self.pause_video_btn.setEnabled(False)
        self.stop_video_btn.setEnabled(False)
        self.pause_video_btn.setText("暂停")
        self.progress_bar.setVisible(False)
        if detections:
            stats_text = f"视频检测完成\n总缺陷数: {len(detections)}\n"
            defect_stats = {}
            for det in detections:
                defect_stats[det['class_name']] = defect_stats.get(det['class_name'], 0) + 1
            for cls, cnt in defect_stats.items():
                stats_text += f"  {cls}: {cnt}个\n"
            self.stats_text.setPlainText(stats_text)
            self.status_label.setText(f"检测完成，缺陷{len(detections)}个")
            self.add_to_history("视频检测完成", os.path.basename(self.current_video_path), len(detections))
        else:
            self.status_label.setText("检测完成，无缺陷")

    def update_progress(self, value):
        """更新视频检测进度条数值"""
        self.progress_bar.setValue(value)

    def update_stats(self, detections, processing_time):
        """更新检测统计信息文本框"""
        stats_text = f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n处理时间: {processing_time:.2f}秒\n缺陷数: {len(detections)}个\n"
        if detections:
            defect_stats = {}
            for det in detections:
                defect_stats[det['class_name']] = defect_stats.get(det['class_name'], 0) + 1
            stats_text += "缺陷分布:\n"
            for cls, cnt in defect_stats.items():
                stats_text += f"  {cls}: {cnt}个\n"
            stats_text += f"平均置信度: {sum(d['confidence'] for d in detections)/len(detections):.2%}"
        self.stats_text.setPlainText(stats_text)

    def stop_detection(self):
        """停止当前正在执行的所有检测任务"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("检测已停止")

    def view_fullscreen(self):
        """全屏查看检测结果图像"""
        if self.result_label.pixmap() and not self.result_label.pixmap().isNull():
            dialog = QDialog(self)
            dialog.setWindowTitle("全屏查看")
            dialog.setWindowState(Qt.WindowFullScreen)
            layout = QVBoxLayout(dialog)
            label = QLabel()
            label.setPixmap(self.result_label.pixmap().scaled(dialog.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            layout.addWidget(label)
            dialog.exec_()
        else:
            QMessageBox.information(self, "提示", "无结果可查看")

    def closeEvent(self, event):
        """窗口关闭前清理资源（线程、摄像头）"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        if self.camera_active:
            self.camera_timer.stop()
            self.cap.release()
        event.accept()

def main():
    """程序主入口，创建应用并显示主窗口"""
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    app.setStyle("Fusion")
    window = PCBMainWindowComplete()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()