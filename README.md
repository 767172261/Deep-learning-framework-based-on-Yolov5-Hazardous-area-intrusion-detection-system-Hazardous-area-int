基于Yolov5的危险区域闯入检测系统
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e5fae4f0cd274a379d635a529f67a854.png)

可以自己划定禁停区域，检测人类类别，检测危险区域闯入的人员个数，显示当前违禁状态。
按钮有四种功能：
选择权重文件
初始化模型
暂停与继续
结束检测
检测类型有三种：
图片检测
视频检测
摄像头实时检测
显示检测结果，位置信息，置信度等，准确率高，UI界面美观
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6ff50372729e4c5d8514f5f2a749e680.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/893a979091bd491fa75c18dae134b31b.png)
 **基于 YOLOv5 的危险区域闯入检测系统** 的完整实现代码， UI 界面功能（图片/视频/摄像头检测、自定义禁停区、人员检测、状态提示等），并附带详细注释和使用说明。

---

## 🎯 功能概述

| 功能 | 说明 |
|------|------|
| ✅ 自定义禁停区域 | 使用鼠标在图像上绘制多边形区域 |
| ✅ 检测人类类别 | 仅识别 `person` 类别 |
| ✅ 实时计数 | 统计进入禁停区的人员数量 |
| ✅ 状态提示 | 显示“危险！！！”或“安全” |
| ✅ 三种检测模式 | 图片、视频、摄像头实时 |
| ✅ 模型加载 | 支持 `.pt` 权重文件选择 |
| ✅ 结果可视化 | 显示框、置信度、坐标 |
| ✅ UI 美观简洁 | PyQt5 实现 |

---

## 📁 文件结构建议

```
intrusion_detection_system/
├── main.py                     # 主程序入口
├── ui_mainwindow.py            # UI 界面代码（由 Qt Designer 生成）
├── detect.py                   # YOLO 推理核心函数
├── weights/                    # 存放 yolov5s.pt 或自定义模型
│   └── yolov5s.pt
└── resources/                  # 图标资源（可选）
```

---

## 🧩 一、依赖安装

```bash
pip install torch torchvision opencv-python PyQt5 numpy
```

> 若需 GPU 加速，请确保已安装 CUDA 和 PyTorch-GPU 版本。

---

## 🖼️ 二、UI 界面设计（`.ui` 文件）

使用 **Qt Designer** 创建界面，保存为 `mainwindow.ui`，然后用以下命令生成 Python 代码：

```bash
pyuic5 mainwindow.ui -o ui_mainwindow.py
```

### 示例 UI 布局（关键控件）：
- `label_camera`：显示摄像头画面
- `pushButton_select_weights`：选择权重文件
- `pushButton_init_model`：初始化模型
- `pushButton_pause_resume`：暂停/继续
- `pushButton_stop`：结束检测
- `radioButton_image`：图片检测
- `radioButton_video`：视频检测
- `radioButton_camera`：摄像头检测
- `textBrowser_result`：显示检测结果
- `label_status`：状态提示（危险 / 安全）
- `label_count`：当前闯入人数
- `label_region`：禁停区域标记（可拖动）

---

## 🔧 三、核心代码（`main.py`）

```python
# main.py
import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.uic import loadUi
import torch
from detect import detect_person_in_region

class IntrusionDetectionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的危险区域闯入检测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 加载 UI
        self.ui = loadUi("ui_mainwindow.py", self)  # 替换为你的 .ui 文件路径

        # 初始化变量
        self.model = None
        self.cap = None
        self.is_running = False
        self.region_points = []  # 禁停区域顶点坐标
        self.current_frame = None
        self.intruder_count = 0

        # 连接按钮信号
        self.pushButton_select_weights.clicked.connect(self.select_weights)
        self.pushButton_init_model.clicked.connect(self.init_model)
        self.pushButton_pause_resume.clicked.connect(self.toggle_pause_resume)
        self.pushButton_stop.clicked.connect(self.stop_detection)

        # 检测类型切换
        self.radioButton_image.toggled.connect(self.on_detection_type_changed)
        self.radioButton_video.toggled.connect(self.on_detection_type_changed)
        self.radioButton_camera.toggled.connect(self.on_detection_type_changed)

        # 鼠标事件用于绘制区域
        self.label_camera.mousePressEvent = self.on_mouse_press
        self.label_camera.mouseMoveEvent = self.on_mouse_move
        self.label_camera.mouseReleaseEvent = self.on_mouse_release

        # 定时器刷新画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 刷新一次

        # 默认显示图片检测
        self.radio_button_selected = "image"

    def select_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "", "PyTorch 模型 (*.pt)")
        if file_path:
            self.weights_path = file_path
            self.statusBar().showMessage(f"已选择权重文件: {os.path.basename(file_path)}")

    def init_model(self):
        if not hasattr(self, 'weights_path'):
            QMessageBox.warning(self, "错误", "请先选择权重文件！")
            return

        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights_path, force_reload=True)
            self.model.conf = 0.3  # 置信度阈值
            self.model.classes = [0]  # 只检测 person (class 0)
            self.statusBar().showMessage("模型初始化成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")

    def toggle_pause_resume(self):
        if self.is_running:
            self.timer.stop()
            self.pushButton_pause_resume.setText("继续")
            self.is_running = False
        else:
            self.timer.start(30)
            self.pushButton_pause_resume.setText("暂停")
            self.is_running = True

    def stop_detection(self):
        self.timer.stop()
        self.is_running = False
        self.pushButton_pause_resume.setText("暂停")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.textBrowser_result.clear()
        self.label_status.setText("状态：安全")
        self.label_count.setText("危险区域闯入人员：0")

    def on_detection_type_changed(self):
        sender = self.sender()
        if sender.isChecked():
            self.radio_button_selected = sender.objectName().split('_')[1]
            self.reset_capture()

    def reset_capture(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_frame = None
        self.region_points = []
        self.textBrowser_result.clear()
        self.label_status.setText("状态：安全")
        self.label_count.setText("危险区域闯入人员：0")

    def on_mouse_press(self, event):
        if event.button() == Qt.LeftButton and self.radio_button_selected == "image":
            x, y = event.pos().x(), event.pos().y()
            self.region_points.append((x, y))
            self.draw_region_on_frame()

    def on_mouse_move(self, event):
        if event.buttons() & Qt.LeftButton and self.radio_button_selected == "image":
            x, y = event.pos().x(), event.pos().y()
            self.region_points[-1] = (x, y)
            self.draw_region_on_frame()

    def on_mouse_release(self, event):
        if event.button() == Qt.LeftButton and self.radio_button_selected == "image":
            if len(self.region_points) >= 3:
                self.draw_region_on_frame()
            else:
                self.region_points.pop()

    def draw_region_on_frame(self):
        if not self.current_frame or len(self.region_points) < 3:
            return
        frame_copy = self.current_frame.copy()
        pts = np.array(self.region_points, dtype=np.int32)
        cv2.polylines(frame_copy, [pts], True, (0, 255, 0), 2)
        self.display_frame(frame_copy)

    def update_frame(self):
        if not self.is_running or not self.model:
            return

        if self.radio_button_selected == "image":
            if not self.current_frame:
                self.load_image()
            else:
                self.process_frame(self.current_frame)
        elif self.radio_button_selected == "video":
            if not self.cap:
                self.cap = cv2.VideoCapture("test.mp4")  # 替换为你的视频路径
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.process_frame(frame)
            else:
                self.cap.release()
                self.cap = None
        elif self.radio_button_selected == "camera":
            if not self.cap:
                self.cap = cv2.VideoCapture(0)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.process_frame(frame)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png)")
        if file_path:
            self.current_frame = cv2.imread(file_path)
            self.display_frame(self.current_frame)

    def process_frame(self, frame):
        if not self.region_points:
            self.display_frame(frame)
            return

        # 调用检测函数
        results = detect_person_in_region(frame, self.model, self.region_points)
        detected_people = results['detected_people']
        intruders = results['intruders']

        # 更新统计
        self.intruder_count = len(intruders)
        self.label_count.setText(f"危险区域闯入人员：{self.intruder_count}")

        # 更新状态
        if self.intruder_count > 0:
            self.label_status.setText("状态：危险！！！")
            self.label_status.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.label_status.setText("状态：安全")
            self.label_status.setStyleSheet("color: green; font-weight: bold;")

        # 显示检测结果
        self.textBrowser_result.clear()
        for i, obj in enumerate(detected_people):
            pos = tuple(obj['position'])
            conf = obj['confidence']
            class_name = obj['class']
            self.textBrowser_result.append(f"({pos[0]}, {pos[1]}), ({pos[2]}, {pos[3]}) Obj: {class_name} Conf: {conf:.2f}")

        # 可视化结果
        vis_frame = frame.copy()
        for person in detected_people:
            x1, y1, x2, y2 = person['position']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_frame, f"{person['class']} {person['confidence']:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 绘制禁停区
        pts = np.array(self.region_points, dtype=np.int32)
        cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 2)
        cv2.fillPoly(vis_frame, [pts], (64, 64, 64), cv2.LINE_AA)

        # 标记闯入者
        for intruder in intruders:
            x1, y1, x2, y2 = intruder['position']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(vis_frame, "INTRUDER", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.display_frame(vis_frame)

    def display_frame(self, frame):
        h, w = frame.shape[:2]
        qimage = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.label_camera.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


def detect_person_in_region(frame, model, region_points):
    # 将图像转换为 RGB 并推理
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    detected_people = []
    intruders = []

    for box in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        class_id = int(cls)
        class_name = model.names[class_id]

        if class_id == 0:  # person
            position = (int(x1), int(y1), int(x2), int(y2))
            detected_people.append({
                'position': position,
                'confidence': float(conf),
                'class': class_name
            })

            # 判断是否在区域内
            if is_point_in_polygon((x1 + x2) // 2, (y1 + y2) // 2, region_points):
                intruders.append({
                    'position': position,
                    'confidence': float(conf),
                    'class': class_name
                })

    return {
        'detected_people': detected_people,
        'intruders': intruders
    }


def is_point_in_polygon(x, y, points):
    """判断点是否在多边形内"""
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = IntrusionDetectionSystem()
    window.show()
    sys.exit(app.exec_())
```

---

## 🧠 四、YOLO 推理模块（`detect.py`）

```python
# detect.py
import torch
import cv2

def detect_person_in_region(frame, model, region_points):
    # 推理
    results = model(frame)
    
    detected_people = []
    intruders = []

    for box in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        class_id = int(cls)
        class_name = model.names[class_id]

        if class_id == 0:  # person
            position = (int(x1), int(y1), int(x2), int(y2))
            detected_people.append({
                'position': position,
                'confidence': float(conf),
                'class': class_name
            })

            # 判断是否在禁停区内
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if is_point_in_polygon(center_x, center_y, region_points):
                intruders.append({
                    'position': position,
                    'confidence': float(conf),
                    'class': class_name
                })

    return {
        'detected_people': detected_people,
        'intruders': intruders
    }

def is_point_in_polygon(x, y, points):
    """判断点是否在多边形内"""
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
```

---

## 📌 五、使用说明

1. **准备环境**：
   ```bash
   pip install torch torchvision opencv-python PyQt5
   ```

2. **下载 YOLOv5 模型**：
   - 下载 `yolov5s.pt` 至 `weights/` 目录
   - 或训练自己的模型（推荐使用 COCO 数据集）

3. **运行程序**：
   ```bash
   python main.py
   ```

4. **操作流程**：
   - 点击“选择Weights” → 选择 `yolov5s.pt`
   - 点击“初始化模型”
   - 选择检测类型（图片/视频/摄像头）
   - 在图像上点击鼠标左键绘制禁停区（至少3个点）
   - 点击“开始检测”，系统自动识别并报警

---


