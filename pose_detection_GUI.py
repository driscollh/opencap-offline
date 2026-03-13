# ==============================================================================
# File: pose_detection_GUI.py
# Author: Harry G. Driscoll
# Date: Jan 2026
#
# License: Distributed under the Apache 2.0 License
# ==============================================================================

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, 
                             QCheckBox, QScrollArea, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QBrush

# --- COLORS FOR IDS ---
COLORS = [
    (255, 0, 0),    # ID 0: Red
    (0, 255, 0),    # ID 1: Green
    (0, 0, 255),    # ID 2: Blue
    (255, 255, 0),  # ID 3: Yellow
    (0, 255, 255),  # ID 4: Cyan
    (255, 0, 255)   # ID 5: Magenta
]

class TimelineWidget(QWidget):
    """
    Custom Widget that draws stacked bars for each subject.
    Clicking a bar jumps the video to that frame.
    """
    frame_jump_requested = pyqtSignal(int)

    def __init__(self, total_frames, tracking_data):
        super().__init__()
        self.total_frames = total_frames
        self.tracking_data = tracking_data # {frame_idx: [id, id, ...]}
        self.setMinimumHeight(150)
        self.active_ids = sorted(list(set(
            pid for frame in tracking_data.values() for pid in frame
        )))
        
        # Row height for each subject lane
        self.row_height = 30
        self.margin_left = 60 # Space for labels (ID 0, ID 1...)

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        # Draw background
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

        # Calculate pixels per frame
        px_per_frame = (w - self.margin_left) / self.total_frames

        for row_idx, subject_id in enumerate(self.active_ids):
            # 1. Draw Label
            y_pos = row_idx * self.row_height + 10
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(5, y_pos + 20, f"Subject {subject_id}")

            # 2. Draw Bars
            color_rgb = COLORS[subject_id % len(COLORS)]
            painter.setBrush(QBrush(QColor(*color_rgb)))
            painter.setPen(Qt.NoPen)

            # Optimisation: In a real app, combine consecutive frames into one rect
            # Here we draw simple strips for demonstration
            for frame_idx, subjects_in_frame in self.tracking_data.items():
                if subject_id in subjects_in_frame:
                    x_pos = self.margin_left + (frame_idx * px_per_frame)
                    painter.drawRect(int(x_pos), int(y_pos), int(max(1, px_per_frame)), 20)

    def mousePressEvent(self, event):
        # Jump video to click position
        x = event.x()
        if x > self.margin_left:
            timeline_width = self.width() - self.margin_left
            click_ratio = (x - self.margin_left) / timeline_width
            target_frame = int(click_ratio * self.total_frames)
            self.frame_jump_requested.emit(target_frame)

class SubjectSelectorApp(QMainWindow):
    def __init__(self, video_path, tracking_data, total_frames):
        super().__init__()
        self.setWindowTitle("Subject Exclusion Interface")
        self.resize(1000, 800)

        self.video_path = video_path
        self.total_frames = total_frames
        self.tracking_data = tracking_data # Format: { frame_idx: [(id, bbox), ...] }
        self.current_frame_idx = 0
        self.playing = False
        self.valid_ids = set() # Which IDs to keep

        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 1. Video Display (Using Label)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label)

        # 2. Controls (Play/Pause/Slider)
        controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_btn)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, total_frames - 1)
        self.slider.sliderMoved.connect(self.set_frame)
        controls_layout.addWidget(self.slider)
        layout.addLayout(controls_layout)

        # 3. Timeline Widget
        # We process tracking_data slightly to fit TimelineWidget expected format
        simple_track_data = {}
        unique_ids = set()
        for f, detections in tracking_data.items():
            simple_track_data[f] = [d['id'] for d in detections]
            for d in detections: unique_ids.add(d['id'])

        self.timeline = TimelineWidget(total_frames, simple_track_data)
        self.timeline.frame_jump_requested.connect(self.set_frame)
        layout.addWidget(self.timeline)

        # 4. Exclusion Checkboxes
        chk_layout = QHBoxLayout()
        self.checkboxes = {}
        for uid in sorted(list(unique_ids)):
            chk = QCheckBox(f"Keep Subject {uid}")
            chk.setChecked(True)
            # Match color to timeline
            r,g,b = COLORS[uid % len(COLORS)]
            chk.setStyleSheet(f"color: rgb({r},{g},{b}); font-weight: bold;")
            chk_layout.addWidget(chk)
            self.checkboxes[uid] = chk
        
        btn_confirm = QPushButton("Confirm Selection")
        btn_confirm.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px;")
        btn_confirm.clicked.connect(self.export_selection)
        chk_layout.addWidget(btn_confirm)
        layout.addLayout(chk_layout)

        # Setup Video Capture
        self.cap = cv2.VideoCapture(video_path)
        
        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Initial Frame
        self.update_image()

    def toggle_play(self):
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.playing = True
            self.timer.start(30) # ~30 FPS
            self.play_btn.setText("Pause")

    def next_frame(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.slider.setValue(self.current_frame_idx)
            self.update_image()
        else:
            self.toggle_play()

    def set_frame(self, frame_idx):
        self.current_frame_idx = frame_idx
        self.update_image()

    def update_image(self):
        # 1. Set Video Pos
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret: return

        # 2. Draw Bounding Boxes
        if self.current_frame_idx in self.tracking_data:
            detections = self.tracking_data[self.current_frame_idx]
            for det in detections:
                pid = det['id']
                bbox = det['bbox'] # [x, y, w, h]
                
                # Color based on ID
                color = COLORS[pid % len(COLORS)]
                
                # Draw Box
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"ID {pid}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 3. Convert to Qt Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 4. Display
        # Scale to fit label while keeping aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def export_selection(self):
        final_ids = []
        for uid, chk in self.checkboxes.items():
            if chk.isChecked():
                final_ids.append(uid)
        
        print(f"User selected to KEEP IDs: {final_ids}")
        self.close()

# --- MOCK DATA GENERATOR FOR TESTING ---
def create_dummy_data():
    # Simulates tracking data for 300 frames
    data = {}
    for f in range(300):
        frame_dets = []
        # Person 0: Walking across frame 0-300
        if 0 <= f <= 300:
            frame_dets.append({'id': 0, 'bbox': [100 + f, 100, 50, 100]})
        
        # Person 1: Walks in at frame 100, leaves at 200 (The "Contamination")
        if 100 <= f <= 200:
            frame_dets.append({'id': 1, 'bbox': [500 - f, 150, 60, 110]})
            
        data[f] = frame_dets
    return data

if __name__ == "__main__":
    # Create a dummy black video for testing purposes
    # In your real code, you pass the actual file path
    dummy_video_path = "temp_dummy.avi"
    out = cv2.VideoWriter(dummy_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 480))
    for _ in range(300):
        out.write(np.zeros((480, 640, 3), dtype=np.uint8))
    out.release()

    # Run App
    app = QApplication(sys.argv)
    tracking_data = create_dummy_data()
    window = SubjectSelectorApp(dummy_video_path, tracking_data, 300)
    window.show()
    sys.exit(app.exec_())