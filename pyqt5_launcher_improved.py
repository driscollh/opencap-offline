# ==============================================================================
# File: pyqt5_launcher_improved.py
# Author: Harry G. Driscoll
# Date: Jan 2026
#
# OpenCap Portable Pro - Motion Capture Processing Launcher
# 
# A comprehensive GUI application for managing motion capture video processing,
# calibration, and biomechanical analysis workflows.
# 
# License: Distributed under the Apache 2.0 License
# ==============================================================================

import sys
import os
import shutil
import cv2
import numpy as np
import subprocess
import json
import yaml
import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

# --- FORCE PYQT5 ---
os.environ["QT_API"] = "pyqt5"

# --- MATPLOTLIB CRASH FIX ---
import matplotlib
matplotlib.use('Agg')

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTreeWidget, QTreeWidgetItem, QFrame, QSplitter,
    QSlider, QStyle, QFileDialog, QMessageBox, QDialog, QLineEdit, QFormLayout,
    QCheckBox, QScrollArea, QTextEdit, QRadioButton, QButtonGroup, QGridLayout,
    QSizePolicy, QProgressBar, QShortcut, QMenuBar, QMenu, QAction, QStatusBar
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal as Signal, QThread, QSize, QProcess, QSettings,
    QRunnable, QThreadPool, pyqtSlot, QRect
)
from PyQt5.QtGui import (
    QImage, QPixmap, QIcon, QColor, QFont, QPalette, QKeySequence, QPainter, QBrush
)

from PyQt5.QtCore import QDateTime

# --- 3D ENGINE ---
from pyvistaqt import QtInteractor
import pyvista as pv

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('opencap_launcher.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

class Config:
    """Application configuration constants"""
    
    # UI Dimensions
    WINDOW_WIDTH = 1600
    WINDOW_HEIGHT = 950
    TREE_WIDTH = 300
    VIDEO_CONTAINER_WIDTH = 600
    LOG_HEIGHT = 100
    HEADER_HEIGHT = 60
    BUTTON_HEIGHT = 40
    LOGO_MAX_WIDTH = 150
    LOGO_MAX_HEIGHT = 40
    
    PLAYBACK_FPS = 30
    MIN_VIDEO_SIZE = (200, 150)
    
    # 3D Visualization Defaults (METERS)
    SKELETON_JOINT_RADIUS = 0.015
    SKELETON_BONE_RADIUS = 0.008
    
    # --- CHANGED: Larger Grid for full-scale models ---
    GRID_SIZE = 6
    GRID_OPACITY = 0.3
    
    MAX_FRAMES_LOAD = 10000
    
    VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi')
    OVERLAY_EXTENSION = ('.avi', '.mp4')
    MARKER_EXTENSION = '.trc'
    MESH_EXTENSION = ('.vtp', '.vtk', '.obj', '.stl')
    
    # Default Session Values
    DEFAULT_HEIGHT = 1.75
    DEFAULT_WEIGHT = 70.0
    DEFAULT_CHECKERBOARD_ROWS = 4
    DEFAULT_CHECKERBOARD_COLS = 5
    DEFAULT_SQUARE_SIZE = 35.0
    DEFAULT_NUM_CAMERAS = 2
    
    # Trial Type Defaults
    DEFAULT_DYNAMIC_NAME = "walking_1"
    
    # Validation Limits
    MIN_HEIGHT = 0.5
    MAX_HEIGHT = 3.0
    MIN_WEIGHT = 20.0
    MAX_WEIGHT = 500.0
    MIN_CAMERAS = 1
    MAX_CAMERAS = 10
    MIN_CHECKERBOARD_DIM = 2
    
    # UI Settings
    STATUS_MESSAGE_TIMEOUT = 5000  # milliseconds
    MAX_RECENT_SESSIONS = 10

# --- CONFIG ---
class Colors:
    # Dark Mode Colors
    BG = "#0d0d0d"
    PANEL = "#1f1f1f"
    ACCENT = "#6184D8"
    TEXT = "#E0E0E0"
    BORDER = "#333333"
    INPUT_BG = "#2b2b2b"
    DISABLED_BG = "#202020"
    
    # Light Mode Colors
    LIGHT_BG = "#f2f1ec"
    LIGHT_PANEL = "#ffffff"
    LIGHT_TEXT = "#222222"
    LIGHT_BORDER = "#cccccc"
    LIGHT_INPUT = "#ffffff"
    LIGHT_DISABLED = "#f0f0f0"  # <--- NEW: Light Grey for disabled inputs

# Trial Types
class TrialType(Enum):
    """Enumeration of trial types"""
    INTRINSICS = "intrinsics"
    CALIBRATION = "calibration"
    NEUTRAL = "neutral"
    DYNAMIC = "dynamic"

# File Types
class FileType(Enum):
    """Enumeration of file types"""
    RAW_VIDEO = "raw"
    OVERLAY_VIDEO = "overlay"
    MARKER_DATA = "trc"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- COLORS FOR TRACKING ---
TRACK_COLORS = [
    (255, 0, 0),    # ID 0: Red
    (0, 255, 0),    # ID 1: Green
    (0, 0, 255),    # ID 2: Blue
    (255, 255, 0),  # ID 3: Yellow
    (0, 255, 255),  # ID 4: Cyan
    (255, 0, 255)   # ID 5: Magenta
]

# =============================================================================
# STYLESHEETS
# =============================================================================

class Styles:
    DARK = f"""
    QMainWindow, QDialog, QMessageBox {{ background-color: {Colors.BG}; }}
    QWidget {{ color: {Colors.TEXT}; font-family: 'Segoe UI', sans-serif; font-size: 13px; }}
    
    /* ADDED DOT: .QFrame ensures this only affects frames, not labels */
    .QFrame {{ background-color: {Colors.PANEL}; border-radius: 4px; }}
    QSplitter::handle {{ background-color: {Colors.BORDER}; }}

    /* SPECIFIC FRAMES - DARK MODE */
    .QFrame#HeaderFrame {{
        background-color: {Colors.BG};
        border-bottom: 1px solid {Colors.BORDER};
        border-radius: 0px;
    }}
    .QFrame#ImportPanel {{
        background-color: {Colors.PANEL};
        border: none;
    }}

    /* MENU BAR */
    QMenuBar {{ background-color: {Colors.BG}; color: {Colors.TEXT}; }}
    QMenuBar::item {{ background-color: transparent; }}
    QMenuBar::item:selected {{ background-color: {Colors.ACCENT}; }}
    QMenu {{ background-color: {Colors.PANEL}; border: 1px solid {Colors.BORDER}; }}
    QMenu::item {{ padding: 5px 20px; }}
    QMenu::item:selected {{ background-color: {Colors.ACCENT}; }}

    /* BUTTONS */
    QPushButton {{ 
        background-color: #3a3a3a; color: white; border: 1px solid #444;
        border-radius: 4px; padding: 6px 12px; font-weight: bold; 
    }}
    QPushButton:hover {{ background-color: {Colors.ACCENT}; border-color: {Colors.ACCENT}; }}
    QPushButton:disabled {{ background-color: #222; color: #555; border-color: #333; }}
    QPushButton#AccentButton {{ background-color: {Colors.ACCENT}; border: 1px solid {Colors.ACCENT}; }}
    QPushButton#AccentButton:hover {{ background-color: #4f6cb3; }}

    /* INPUTS & DIALOGS */
    QLineEdit, QSpinBox {{ 
        background-color: {Colors.INPUT_BG}; border: 1px solid #444; padding: 4px; border-radius: 4px; color: white;
    }}
    QComboBox {{ 
        background-color: #2b2b2b; 
        border: 1px solid #444; 
        padding: 4px; 
        border-radius: 4px; 
        color: white; 
    }}

    QComboBox QAbstractItemView {{
        background-color: #2b2b2b;
        color: white;
        selection-background-color: #222222;
        selection-color: white;
        border: 1px solid #444;
        outline: 0px;
    }}

    QScrollArea, QScrollArea QWidget {{
        background-color: #1a1a1a; /* Slightly lighter than main BG */
        border: none;
    }}

    QTextEdit {{ background-color: #111; border: 1px solid #333; font-family: 'Consolas', monospace; color: #eee; }}
    
    QTreeWidget {{ background-color: {Colors.PANEL}; border: none; color: #EEE; outline: 0; }}
    QTreeWidget::item:selected {{ background-color: #333; color: {Colors.ACCENT}; }}
    
    QHeaderView::section {{ background-color: {Colors.PANEL}; color: {Colors.TEXT}; border: none; }}
    """

    LIGHT = f"""
    QMainWindow, QDialog, QMessageBox {{ background-color: {Colors.LIGHT_BG}; }}
    QWidget {{ color: {Colors.LIGHT_TEXT}; font-family: 'Segoe UI', sans-serif; font-size: 13px; }}
    
    /* ADDED DOT: .QFrame prevents borders from appearing on Labels/Logos */
    .QFrame {{ background-color: {Colors.LIGHT_PANEL}; border-radius: 4px; border: 1px solid {Colors.LIGHT_BORDER}; }}
    QSplitter::handle {{ background-color: {Colors.LIGHT_BORDER}; }}
    QLabel {{ color: {Colors.LIGHT_TEXT}; border: none; }}

    /* SPECIFIC FRAMES - LIGHT MODE */
    .QFrame#HeaderFrame {{
        background-color: {Colors.LIGHT_PANEL};
        border-bottom: 1px solid {Colors.LIGHT_BORDER};
        border-radius: 0px;
        border-top: none; border-left: none; border-right: none;
    }}
    .QFrame#ImportPanel {{
        background-color: {Colors.LIGHT_PANEL};
        border: 1px solid {Colors.LIGHT_BORDER};
    }}

    /* MENU BAR */
    QMenuBar {{ background-color: {Colors.LIGHT_BG}; color: {Colors.LIGHT_TEXT}; }}
    QMenuBar::item {{ background-color: transparent; }}
    QMenuBar::item:selected {{ background-color: {Colors.ACCENT}; color: white; }}
    QMenu {{ background-color: {Colors.LIGHT_PANEL}; border: 1px solid {Colors.LIGHT_BORDER}; }}
    QMenu::item {{ padding: 5px 20px; color: {Colors.LIGHT_TEXT}; }}
    QMenu::item:selected {{ background-color: {Colors.ACCENT}; color: white; }}

    /* BUTTONS */
    QPushButton {{ 
        background-color: #e0e0e0; color: #000; border: 1px solid #bbb;
        border-radius: 4px; padding: 6px 12px; font-weight: bold; 
    }}
    QPushButton:hover {{ background-color: #d0d0d0; border-color: #aaa; }}
    QPushButton:disabled {{ background-color: #f0f0f0; color: #aaa; border-color: #ddd; }}
    QPushButton#AccentButton {{ background-color: {Colors.ACCENT}; color: white; border: 1px solid {Colors.ACCENT}; }}
    QPushButton#AccentButton:hover {{ background-color: #4f6cb3; }}

    /* INPUTS */
    QLineEdit, QSpinBox {{ 
        background-color: {Colors.LIGHT_INPUT}; border: 1px solid {Colors.LIGHT_BORDER}; padding: 4px; border-radius: 4px; color: {Colors.LIGHT_TEXT};
    }}
    QComboBox {{ 
        background-color: white; border: 1px solid #cccccc; padding: 4px; 
        border-radius: 4px; color: #222222;
    }}
    QComboBox QAbstractItemView {{
    background-color: white;
    color: #222222;
    selection-background-color: #6184D8;
    }}
    QTextEdit {{ background-color: {Colors.LIGHT_INPUT}; border: 1px solid {Colors.LIGHT_BORDER}; font-family: 'Consolas', monospace; color: {Colors.LIGHT_TEXT}; }}
    
    /* TREE WIDGET */
    QTreeWidget {{ background-color: {Colors.LIGHT_PANEL}; border: none; color: {Colors.LIGHT_TEXT}; outline: 0; }}
    QTreeWidget::item:selected {{ background-color: #e0e0e0; color: #000; }}
    
    QHeaderView::section {{ background-color: {Colors.LIGHT_PANEL}; color: {Colors.LIGHT_TEXT}; border: none; }}
    """

# =============================================================================
# UTILITY CLASSES
# =============================================================================

class InputValidator:
    """Validates and sanitizes user inputs"""
    
    @staticmethod
    def sanitize_trial_name(name: str) -> str:
        """
        Sanitize trial name to prevent injection attacks.
        Allows: alphanumeric, underscore, hyphen
        """
        if not name:
            return ""
        
        # Remove any dangerous characters
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '', name)
        
        # Ensure it doesn't start with special chars
        safe_name = re.sub(r'^[_\-]+', '', safe_name)
        
        return safe_name
    
    @staticmethod
    def validate_video_path(path: str) -> Tuple[bool, str]:
        """Returns (is_valid, error_message)"""
        if not path:
            return False, "Path is empty"
        
        path_obj = Path(path)
        
        if not path_obj.exists():
            return False, f"File does not exist: {path}"
        
        if not path_obj.is_file():
            return False, f"Path is not a file: {path}"
        
        ext = path_obj.suffix.lower()
        if ext not in Config.VIDEO_EXTENSIONS:
            return False, f"Invalid video format: {ext}"
        
        return True, ""
    
    @staticmethod
    def validate_directory(path: str) -> Tuple[bool, str]:
        """Returns (is_valid, error_message)"""
        if not path:
            return False, "Path is empty"
        
        path_obj = Path(path)
        
        if not path_obj.exists():
            return False, f"Directory does not exist: {path}"
        
        if not path_obj.is_dir():
            return False, f"Path is not a directory: {path}"
        
        return True, ""
    
    @staticmethod
    def validate_float(value: str, min_val: float, max_val: float, 
                      field_name: str) -> Tuple[bool, str, Optional[float]]:
        """
        Validate float input.
        Returns (is_valid, error_message, parsed_value)
        """
        try:
            float_val = float(value)
            if float_val < min_val or float_val > max_val:
                return False, f"{field_name} must be between {min_val} and {max_val}", None
            return True, "", float_val
        except ValueError:
            return False, f"{field_name} must be a valid number", None
    
    @staticmethod
    def validate_int(value: str, min_val: int, max_val: int, 
                    field_name: str) -> Tuple[bool, str, Optional[int]]:
        """
        Validate integer input.
        Returns (is_valid, error_message, parsed_value)
        """
        try:
            int_val = int(value)
            if int_val < min_val or int_val > max_val:
                return False, f"{field_name} must be between {min_val} and {max_val}", None
            return True, "", int_val
        except ValueError:
            return False, f"{field_name} must be a valid integer", None

class IntrinsicsWorker(QThread):
    """
    Background thread for camera calibration.
    Supports calibrating all cameras or a single specific camera.
    """
    finished = Signal(bool, str)  # Emits (Success Status, Message)

    def __init__(self, session_name: str, target_cam: Optional[str] = None):
        """
        Initialize the worker.
        
        Args:
            session_name: The name of the active session folder.
            target_cam: Name of specific camera (e.g., 'Cam0') or None for all.
        """
        super().__init__()
        self.session_name = session_name
        self.target_cam = target_cam

    def run(self):
        """Execute the calibration logic."""
        try:
            # Import the calibration logic locally to ensure it's fresh
            import generate_intrinsics
            
            if self.target_cam:
                # Logic for a single camera
                logger.info(f"Starting intrinsics for {self.target_cam} in {self.session_name}")
                generate_intrinsics.calibrate_camera(self.session_name, self.target_cam)
                msg = f"Intrinsics for {self.target_cam} complete."
            else:
                # Standard logic for all cameras
                logger.info(f"Starting intrinsics for all cameras in {self.session_name}")
                generate_intrinsics.calibrate_session(self.session_name)
                msg = "Full session intrinsics calibration complete."
            
            self.finished.emit(True, msg)
            
        except Exception as e:
            logger.error(f"IntrinsicsWorker error: {str(e)}", exc_info=True)
            self.finished.emit(False, f"Calibration Error: {str(e)}")

class MetadataEditorDialog(QDialog):
    """Popup to edit ALL metadata with a safety reset to the session's initial state."""
    def __init__(self, parent, session_path):
        super().__init__(parent)
        self.setWindowTitle("Advanced Metadata Editor")
        self.resize(600, 750)
        self.session_path = Path(session_path)
        self.metadata_file = self.session_path / "sessionMetadata.yaml"
        self.inputs = {}
        
        # 1. Capture the "Original" state as a snapshot before any edits occur
        self.original_data = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.original_data = yaml.safe_load(f)
        
        self.current_data = self.original_data.copy()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Scroll area for the form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.form_layout = QFormLayout(content)
        
        # Populate the fields based on current_data
        self._populate_fields()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        
        # RESET BUTTON
        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.setToolTip("Revert all fields to the original session creation values.")
        btn_reset.clicked.connect(self.reset_to_original)
        
        # SAVE & CANCEL
        btn_save = QPushButton("Save Metadata")
        btn_save.setObjectName("AccentButton")
        btn_save.clicked.connect(self.save_metadata)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_reset)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

    def _populate_fields(self):
        """Clears and rebuilds form rows based on the stored data dictionary."""
        # Clear existing rows if any
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.inputs = {}
        for key, value in self.current_data.items():
            if isinstance(value, dict):
                edit = QLineEdit(json.dumps(value))
                edit.setToolTip("Nested data (JSON format)")
            else:
                edit = QLineEdit(str(value))
            
            self.form_layout.addRow(f"<b>{key}:</b>", edit)
            self.inputs[key] = edit

    def reset_to_original(self):
        """Restores the UI fields to the snapshot taken when the dialog opened."""
        reply = QMessageBox.question(
            self, "Confirm Reset", 
            "Are you sure you want to discard all changes and return to the original session values?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.current_data = self.original_data.copy()
            self._populate_fields()

    def save_metadata(self):
        """Validates all fields and writes back to sessionMetadata.yaml."""
        updated_dict = {}
        for key, edit in self.inputs.items():
            text = edit.text().strip()
            # Determine type based on the original data structure
            orig = self.original_data.get(key)
            
            try:
                if isinstance(orig, dict):
                    updated_dict[key] = json.loads(text)
                elif isinstance(orig, int):
                    updated_dict[key] = int(text)
                elif isinstance(orig, float):
                    updated_dict[key] = float(text)
                elif isinstance(orig, bool):
                    updated_dict[key] = text.lower() == 'true'
                else:
                    updated_dict[key] = text
            except Exception as e:
                QMessageBox.critical(self, "Formatting Error", f"Invalid input for {key}: {e}")
                return

        with open(self.metadata_file, 'w') as f:
            yaml.dump(updated_dict, f, default_flow_style=False, sort_keys=False)
        self.accept()

class PlatformUtils:
    """Platform-specific utilities"""
    
    @staticmethod
    def is_windows() -> bool:
        return sys.platform.startswith('win')
    
    @staticmethod
    def is_mac() -> bool:
        return sys.platform == 'darwin'
    
    @staticmethod
    def is_linux() -> bool:
        return sys.platform.startswith('linux')
    
    @staticmethod
    def get_video_backend() -> int:
        """Get optimal video backend for platform"""
        if PlatformUtils.is_windows():
            return cv2.CAP_DSHOW
        elif PlatformUtils.is_mac():
            return cv2.CAP_AVFOUNDATION
        else:
            return cv2.CAP_V4L2  # Linux

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_trial_sort_key(trial_name: str) -> int:
    """
    Get sorting priority for trial names.
    
    Args:
        trial_name: Name of the trial
        
    Returns:
        Integer sort key (lower = higher priority)
    """
    t = trial_name.lower()
    if "intrinsic" in t:
        return 0
    if "calibration" in t:
        return 1
    if "neutral" in t or "static" in t:
        return 2
    return 3

# =============================================================================
# NEW: SUBJECT SELECTION WIDGETS
# =============================================================================

class TimelineWidget(QWidget):
    """Draws stacked bars for each detected subject ID over time."""
    frame_jump_requested = Signal(int)

    def __init__(self, total_frames, tracking_data):
        super().__init__()
        self.total_frames = max(1, total_frames)
        self.tracking_data = tracking_data # {frame_idx: [id, id...]}
        self.setMinimumHeight(120)
        
        # Identify all unique IDs
        all_ids = set()
        for f_data in self.tracking_data.values():
            for pid in f_data: all_ids.add(pid)
        self.active_ids = sorted(list(all_ids))
        
        self.row_height = 25
        self.margin_left = 70 

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

        if not self.active_ids: return

        px_per_frame = (w - self.margin_left) / self.total_frames

        for row_idx, subject_id in enumerate(self.active_ids):
            y_pos = row_idx * (self.row_height + 5) + 10
            
            # Label
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(5, y_pos + 18, f"Subject {subject_id}")

            # Draw Bars
            color_rgb = TRACK_COLORS[subject_id % len(TRACK_COLORS)]
            painter.setBrush(QBrush(QColor(*color_rgb)))
            painter.setPen(Qt.NoPen)

            # Optimization: Draw distinct strips
            start_frame = -1
            for f in range(self.total_frames):
                has_subject = (f in self.tracking_data) and (subject_id in self.tracking_data[f])
                
                if has_subject and start_frame == -1:
                    start_frame = f
                elif not has_subject and start_frame != -1:
                    # Draw segment
                    x = self.margin_left + (start_frame * px_per_frame)
                    width = (f - start_frame) * px_per_frame
                    painter.drawRect(int(x), int(y_pos), int(max(1, width)), int(self.row_height))
                    start_frame = -1
            
            # Catch trailing segment
            if start_frame != -1:
                x = self.margin_left + (start_frame * px_per_frame)
                width = (self.total_frames - start_frame) * px_per_frame
                painter.drawRect(int(x), int(y_pos), int(max(1, width)), int(self.row_height))

    def mousePressEvent(self, event):
        x = event.x()
        if x > self.margin_left:
            ratio = (x - self.margin_left) / (self.width() - self.margin_left)
            frame = int(ratio * self.total_frames)
            self.frame_jump_requested.emit(frame)

class SubjectSelectorDialog(QDialog):
    """Popup Window for reviewing and excluding subjects."""
    def __init__(self, parent, video_path, tracking_file):
        super().__init__(parent)
        self.setWindowTitle("Review Detected Subjects")
        self.resize(1000, 700)
        
        self.video_path = video_path
        self.tracking_file = tracking_file
        self.excluded_ids = []
        self.selection_made = False
        
        # Load Tracking Data
        with open(tracking_file, 'r') as f:
            raw_data = json.load(f)
            
        self.total_frames = raw_data['total_frames']
        self.tracks = raw_data['tracks'] # {frame_str: [{'id': 0, 'bbox': []}]}
        
        # Convert keys to int for easier handling
        self.tracks = {int(k): v for k, v in self.tracks.items()}

        self._init_ui()
        self._load_video()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # GPU Selection
        layout.addWidget(QLabel("Select GPU:"))
        self.gpu_combo = QComboBox()
        self.gpu_map = self._get_gpu_info()
        self.gpu_combo.addItems(list(self.gpu_map.keys()))
        layout.addWidget(self.gpu_combo)

        # --- DYNAMIC DLC CHECK FOR POSE ESTIMATOR ---
        layout.addWidget(QLabel("Pose Estimator:"))
        self.pose_combo = QComboBox()
        
        # Identify the DLC path relative to the launcher
        base_path = os.path.dirname(os.path.abspath(__file__))
        dlc_path = os.path.join(base_path, "Blackwell_RTMPose")
        
        # Build the choices list
        pose_choices = ["OpenPose"]
        if os.path.exists(dlc_path):
            pose_choices.append("RTMPose")
            self.pose_combo.setToolTip("Blackwell DLC detected: RTMPose Unlocked.")
        else:
            self.pose_combo.setToolTip("DLC missing: RTMPose is disabled for this installation.")

        self.pose_combo.addItems(pose_choices)
        layout.addWidget(self.pose_combo)
        # --------------------------------------------
        
        # Resolution Selection
        layout.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        layout.addWidget(self.res_combo)

        # Connect the estimator change to the resolution/complexity logic
        self.pose_combo.currentTextChanged.connect(self._update_res_options)
        self._update_res_options(self.pose_combo.currentText())

        # 1. Video Player
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #444;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label)

        # 2. Timeline
        simple_tracks = {}
        unique_ids = set()
        for f, dets in self.tracks.items():
            ids = [d['id'] for d in dets]
            simple_tracks[f] = ids
            for i in ids: unique_ids.add(i)
            
        self.timeline = TimelineWidget(self.total_frames, simple_tracks)
        self.timeline.frame_jump_requested.connect(self.seek_frame)
        layout.addWidget(self.timeline)

        # 3. Controls (Play/Pause + Checkboxes)
        ctrl_layout = QHBoxLayout()
        
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.btn_play)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.sliderMoved.connect(self.seek_frame)
        ctrl_layout.addWidget(self.slider)
        
        layout.addLayout(ctrl_layout)

        # 4. Checkboxes for Exclusion
        chk_frame = QFrame()
        chk_layout = QHBoxLayout(chk_frame)
        chk_layout.addWidget(QLabel("<b>Deselect to Exclude:</b>"))
        
        self.checkboxes = {}
        for uid in sorted(list(unique_ids)):
            # Default Checked = Keep
            chk = QCheckBox(f"Subject {uid}")
            chk.setChecked(True) 
            c = TRACK_COLORS[uid % len(TRACK_COLORS)]
            chk.setStyleSheet(f"color: rgb({c[0]},{c[1]},{c[2]}); font-weight: bold;")
            chk_layout.addWidget(chk)
            self.checkboxes[uid] = chk
            
        btn_confirm = QPushButton("Confirm Selection")
        btn_confirm.setObjectName("AccentButton") # Uses launcher style
        btn_confirm.clicked.connect(self.confirm_selection)
        chk_layout.addWidget(btn_confirm)
        
        layout.addWidget(chk_frame)

        # Internal State
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.current_frame = 0
        self.playing = False
        self.cap = None

    def _load_video(self):
        if not os.path.exists(self.video_path):
            self.video_label.setText(f"Error: Video not found\n{self.video_path}")
            return
        self.cap = cv2.VideoCapture(self.video_path)
        self.show_frame(0)

    def toggle_play(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.btn_play.setText("Play")
        else:
            self.timer.start(33) # ~30fps
            self.playing = True
            self.btn_play.setText("Pause")

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.slider.setValue(self.current_frame)
            self.show_frame(self.current_frame)
        else:
            self.toggle_play()

    def seek_frame(self, frame_idx):
        self.current_frame = frame_idx
        self.slider.setValue(frame_idx)
        self.show_frame(frame_idx)

    def show_frame(self, idx):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret: return

        # Draw Bounding Boxes
        if idx in self.tracks:
            for det in self.tracks[idx]:
                uid = det['id']
                bbox = det['bbox'] # [x, y, w, h]
                color = TRACK_COLORS[uid % len(TRACK_COLORS)]
                
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, f"ID {uid}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert to Qt
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def confirm_selection(self):
        # Identify unchecked IDs
        self.excluded_ids = []
        for uid, chk in self.checkboxes.items():
            if not chk.isChecked():
                self.excluded_ids.append(uid)
        
        self.selection_made = True
        self.accept()
        
    def closeEvent(self, event):
        if self.cap: self.cap.release()
        super().closeEvent(event)

# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class VideoLoader(QThread):
    finished = Signal(list, str) 
    
    def __init__(self, path, target_size):
        super().__init__()
        self.path = path
        self.target_size = target_size 
        self.frames = []
        self._is_cancelled = False  # <--- NEW FLAG
        
    def cancel(self):
        """Safely signal the thread to stop."""
        self._is_cancelled = True

    def run(self):
        if not self.path or not os.path.exists(self.path):
            self.finished.emit([], "error")
            return

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.finished.emit([], "error")
            return
            
        while not self._is_cancelled:  # <--- CHECK FLAG HERE
            ret, frame = cap.read()
            if not ret:
                break
                
            resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            pix = QPixmap.fromImage(qimg).copy()
            self.frames.append(pix)
            
        cap.release()
        
        # Only emit success if we weren't interrupted
        if not self._is_cancelled:
            self.finished.emit(self.frames, self.path)

class DualVideoPlayer(QWidget):
    """
    Synchronized dual video player with RAM buffering for smooth 60FPS.
    """
    frame_changed = Signal(int) 
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        self._init_state()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Video Area
        video_layout = QHBoxLayout()
        
        # Fixed 9:16 Portrait Size
        self.fixed_size = (280, 498)
        
        self.raw_label = QLabel("Raw Video")
        self.raw_label.setAlignment(Qt.AlignCenter)
        self.raw_label.setStyleSheet("background-color: black; border: 1px solid #333;")
        self.raw_label.setFixedSize(*self.fixed_size)
        
        self.overlay_label = QLabel("Overlay")
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("background-color: black; border: 1px solid #333;")
        self.overlay_label.setFixedSize(*self.fixed_size)
        
        video_layout.addWidget(self.raw_label)
        video_layout.addWidget(self.overlay_label)
        layout.addLayout(video_layout)
        
        # Controls
        controls = QHBoxLayout()
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(40)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False) # Disable until loaded
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.seek)
        self.slider.setEnabled(False)
        
        controls.addWidget(self.play_button)
        controls.addWidget(self.slider)
        layout.addLayout(controls)
        
        # Loading Label (Overlay on top, optional, but helpful)
        self.status_lbl = QLabel("No Video", self)
        self.status_lbl.setStyleSheet("color: white; background: rgba(0,0,0,0.5); padding: 5px;")
        self.status_lbl.adjustSize()
        self.status_lbl.move(10, 10)
    
    def _init_state(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._timer_tick)
        self.playing = False
        self.total_frames = 0
        self.current_frame = 0
        self.target_fps = 60.0
        
        # RAM Caches
        self.raw_cache = []
        self.overlay_cache = []
        self.loaders = []

    def load(self, raw_path: Optional[str], overlay_path: Optional[str] = None) -> None:
        self.stop()
        self.raw_cache = []
        self.overlay_cache = []
        self.play_button.setEnabled(False)
        self.slider.setEnabled(False)
        self.raw_label.clear()
        self.overlay_label.clear()
        self.raw_label.setText("Loading...")
        self.status_lbl.setText("Buffering videos to RAM...")
        self.status_lbl.show()
        
        # Start Raw Loader
        if raw_path and os.path.exists(raw_path):
            loader1 = VideoLoader(raw_path, self.fixed_size)
            loader1.finished.connect(self._on_raw_loaded)
            self.loaders.append(loader1)
            loader1.start()
            
        # Start Overlay Loader
        if overlay_path and os.path.exists(overlay_path):
            loader2 = VideoLoader(overlay_path, self.fixed_size)
            loader2.finished.connect(self._on_overlay_loaded)
            self.loaders.append(loader2)
            loader2.start()

    def _on_raw_loaded(self, frames, path):
        self.raw_cache = frames
        self._check_loading_complete()

    def _on_overlay_loaded(self, frames, path):
        self.overlay_cache = frames
        self._check_loading_complete()

    def _check_loading_complete(self):
        # We assume loading is done when raw is ready (overlay is optional)
        if self.raw_cache:
            self.total_frames = len(self.raw_cache)
            self.slider.setRange(0, max(0, self.total_frames - 1))
            self.play_button.setEnabled(True)
            self.slider.setEnabled(True)
            self.status_lbl.hide()
            self.show_frame(0)
            
            # Clean up threads
            self.loaders = []

    def show_frame(self, idx: int) -> None:
        if self.total_frames == 0: return
        
        # Clamp index
        idx = max(0, min(idx, self.total_frames - 1))
        self.current_frame = idx
        self.frame_changed.emit(idx)
        
        # 1. Show Raw (Instant from RAM)
        if idx < len(self.raw_cache):
            self.raw_label.setPixmap(self.raw_cache[idx])
            
        # 2. Show Overlay (Instant from RAM)
        if idx < len(self.overlay_cache):
            self.overlay_label.setPixmap(self.overlay_cache[idx])
        elif self.overlay_cache: # Handle case where overlay might be shorter
            pass 

    def toggle_play(self) -> None:
        if self.total_frames == 0: return
        
        self.playing = not self.playing
        if self.playing:
            self.play_button.setText("||")
            self.timer.start(int(1000 / self.target_fps)) # Exact interval for 60fps
        else:
            self.play_button.setText("▶")
            self.timer.stop()

    def _timer_tick(self) -> None:
        """
        Simple incrementer. Since data is in RAM, we don't need time-checks 
        to skip frames. We can just play every single frame sequentially.
        """
        if self.total_frames == 0: return

        next_frame = (self.current_frame + 1) % self.total_frames
        
        self.slider.blockSignals(True)
        self.slider.setValue(next_frame)
        self.slider.blockSignals(False)
        self.show_frame(next_frame)

    def seek(self, val: int) -> None:
        self.show_frame(val)
    
    def stop(self) -> None:
        self.playing = False
        self.timer.stop()
        self.play_button.setText("▶")
        
        # Safely stop any active loaders
        for loader in self.loaders:
            if loader.isRunning():
                loader.cancel()  # Tell it to stop looping
                loader.wait()    # Block for a few milliseconds until it cleanly exits
        self.loaders = []
    
    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)

class BoneRegistry:
    """Manages loading of external bone meshes."""
    def __init__(self, geometry_path: Path):
        self.geometry_path = geometry_path
        self.meshes = {}
        self._scan_meshes()
    
    def _scan_meshes(self):
        if not self.geometry_path.exists(): return
        for f in self.geometry_path.iterdir():
            if f.suffix.lower() in Config.MESH_EXTENSION:
                try:
                    self.meshes[f.stem.lower()] = pv.read(str(f))
                except Exception:
                    pass

    def get_mesh(self, bone_name: str) -> Optional[pv.PolyData]:
        key = bone_name.lower()
        if key in self.meshes:
            return self.meshes[key].copy()
        return None

class SkeletonViewer3D(QtInteractor):
    def __init__(self, parent=None, geometry_path=None):
        super().__init__(parent)
        self.geometry_path = geometry_path
        self.bone_registry = None
        if self.geometry_path:
            self.bone_registry = BoneRegistry(self.geometry_path)
            
        self._init_scene()
        self._init_state()
    
    def _init_scene(self):
        self.set_background(Colors.BG)
        plane = pv.Plane(i_size=Config.GRID_SIZE, j_size=Config.GRID_SIZE)
        self.add_mesh(plane, color=Colors.INPUT_BG, opacity=Config.GRID_OPACITY, show_edges=True)
        self.enable_eye_dome_lighting()
    
    def _init_state(self):
        self.skel_actors = {}
        self.markers = None
        self.bone_pairs = []

    def load_trc(self, path: str):
        self.clear_skeleton()
        if not path or not Path(path).exists(): return
        
        try:
            with open(path, 'r') as f: lines = f.readlines()
            if len(lines) < 6: return
            
            # Header
            header_meta = lines[2].strip().split('\t')
            num_markers = int(header_meta[3])
            raw_names = lines[3].strip().split('\t')
            names = [n.strip() for n in raw_names[2:] if n.strip()]
            
            # Data
            data = []
            max_lines = min(len(lines[5:]), Config.MAX_FRAMES_LOAD)
            for line in lines[5:5+max_lines]:
                if line.strip():
                    parts = line.strip().split('\t')
                    row = [float(x) if x else 0.0 for x in parts[2:]]
                    data.append(row)
            
            self.markers = np.array(data)
            if self.markers.shape[1] >= num_markers * 3:
                self.markers = self.markers[:, :num_markers*3]
                self.markers = self.markers.reshape((len(data), num_markers, 3))
            
            # Unit Detection
            subset = np.abs(self.markers[:100])
            if subset.size > 0 and np.percentile(subset, 95) > 10.0:
                logger.info("Detected MM units. Scaling to Meters.")
                self.markers *= 0.001
            
            # Build Hierarchy
            self.bone_pairs = self._get_bone_map(names)
            logger.info(f"Skeleton Loaded: {num_markers} markers. {len(self.bone_pairs)} segments.")

            # Draw Joints
            for i, pos in enumerate(self.markers[0]):
                sphere = pv.Sphere(radius=Config.SKELETON_JOINT_RADIUS, center=(0,0,0))
                actor = self.add_mesh(
                    sphere, color=Colors.ACCENT, reset_camera=False, render=False
                )
                actor.SetPosition(self._swizzle(pos))
                self.skel_actors[f'joint_{i}'] = actor
            
            self.update_frame(0)
            self.reset_camera()
            self.render()
            
        except Exception as e:
            logger.error(f"TRC Load Error: {e}", exc_info=True)

    def _find_marker_index(self, target_names: List[str], available_names: Dict[str, int]) -> int:
        for target in target_names:
            if target.lower() in available_names:
                return available_names[target.lower()]
        return -1

    def _get_bone_map(self, names: List[str]) -> List[Tuple]:
        """
        Returns definition tuples: 
        (StartIdx, EndIdx, MeshName, AxisIndex, FixedScale, RotOffset, [OrientL, OrientR])
        """
        name_map = {name.lower(): i for i, name in enumerate(names)}
        
        # --- MARKER GROUPS ---
        hips_center = ['V.Sacral', 'L5_study', 'midHip', 'L.PSIS_study']
        neck = ['Neck', 'C7_study']
        head = ['HeadTop', 'Head', 'R.Head', 'L.Head', 'Neck']
        
        # Shoulders (For Rib Orientation)
        r_sh = ['r_shoulder_study', 'RShoulder']
        l_sh = ['L_shoulder_study', 'LShoulder']
        
        # Pelvis Width (For Pelvis Orientation)
        r_asis = ['r.ASIS_study', 'R.ASIS']
        l_asis = ['L.ASIS_study', 'L.ASIS']

        # Knee/Elbows (For Limb Orientation - "Knee Pointing" vector)
        r_knee = ['r_knee_study', 'RKnee']
        l_knee = ['L_knee_study', 'LKnee']

        definitions = [
            # --- TORSO (Oriented by Shoulders) ---
            # This locks the ribs to face forward relative to the shoulder line
            (hips_center, neck, 'hat_spine', 1, False, (0,0,0), l_sh, r_sh),
            (hips_center, neck, 'hat_ribs_scap', 1, False, (0,0,0), l_sh, r_sh),
            (neck, head, 'hat_skull', 1, True, (-90,0,0), None, None), # Skull keeps manual offset
            (neck, head, 'hat_jaw', 1, True, (-90,0,0), None, None),

            # --- PELVIS (Oriented by ASIS) ---
            # Anchored Center->Spine, Oriented by Left/Right ASIS
            (hips_center, neck, 'sacrum', 1, True, (90,0,0), l_asis, r_asis),
            (hips_center, neck, 'l_pelvis', 1, True, (90,0,0), l_asis, r_asis),
            (hips_center, neck, 'r_pelvis', 1, True, (90,0,0), l_asis, r_asis),

            # --- RIGHT LEG ---
            (['RHJC_study', 'RHip'], r_knee, 'r_femur', 1, False, (0,0,0), None, None),
            (r_knee, ['RAnkle', 'r_ankle_study'], 'r_tibia', 1, False, (0,0,0), None, None),
            (r_knee, ['RAnkle', 'r_ankle_study'], 'r_fibula', 1, False, (0,0,0), None, None),
            (r_knee, r_knee, 'r_patella', 1, True, (0,0,0), None, None), # Fixed
            
            # --- RIGHT FOOT ---
            (['RAnkle', 'r_ankle_study'], ['r_toe_study', 'RBigToe'], 'r_foot', 0, True, (0,90,0), None, None),
            (['RAnkle', 'r_ankle_study'], ['r_toe_study', 'RBigToe'], 'r_talus', 0, False, (0,90,0), None, None),
            (['r_toe_study', 'RBigToe'], ['r_toe_study', 'RBigToe'], 'r_bofoot', 0, True, (0,90,0), None, None),

            # --- LEFT LEG ---
            (['LHJC_study', 'LHip'], l_knee, 'l_femur', 1, False, (0,0,0), None, None),
            (l_knee, ['LAnkle', 'L_ankle_study'], 'l_tibia', 1, False, (0,0,0), None, None),
            (l_knee, ['LAnkle', 'L_ankle_study'], 'l_fibula', 1, False, (0,0,0), None, None),
            (l_knee, l_knee, 'l_patella', 1, True, (0,0,0), None, None),
            
            # --- LEFT FOOT ---
            (['LAnkle', 'L_ankle_study'], ['L_toe_study', 'LBigToe'], 'l_foot', 0, True, (0,90,0), None, None),
            (['LAnkle', 'L_ankle_study'], ['L_toe_study', 'LBigToe'], 'l_talus', 0, False, (0,90,0), None, None),
            (['L_toe_study', 'LBigToe'], ['L_toe_study', 'LBigToe'], 'l_bofoot', 0, True, (0,90,0), None, None),

            # --- ARMS ---
            (['RShoulder', 'r_shoulder_study'], ['RElbow', 'r_lelbow_study'], 'humerus_rv', 1, False, (0,0,0), None, None),
            (['RElbow', 'r_lelbow_study'], ['RWrist', 'r_lwrist_study'], 'radius_rv', 1, False, (0,0,0), None, None),
            (['RElbow', 'r_lelbow_study'], ['RWrist', 'r_lwrist_study'], 'ulna_rv', 1, False, (0,0,0), None, None),
            (['RWrist', 'r_lwrist_study'], ['RHand', 'r_finger_study'], 'metacarpal3_rvs', 1, False, (0,0,0), None, None),

            (['LShoulder', 'L_shoulder_study'], ['LElbow', 'L_lelbow_study'], 'humerus_lv', 1, False, (0,0,0), None, None),
            (['LElbow', 'L_lelbow_study'], ['LWrist', 'L_lwrist_study'], 'radius_lv', 1, False, (0,0,0), None, None),
            (['LElbow', 'L_lelbow_study'], ['LWrist', 'L_lwrist_study'], 'ulna_lv', 1, False, (0,0,0), None, None),
            (['LWrist', 'L_lwrist_study'], ['LHand', 'L_finger_study'], 'metacarpal3_lvs', 1, False, (0,0,0), None, None),
        ]
        
        pairs = []
        for p in definitions:
            # Unpack variable length tuple (support optional orientation markers)
            start_opts, end_opts, b_name, axis, fixed, rot, *orient = p
            
            s_idx = self._find_marker_index(start_opts, name_map)
            e_idx = self._find_marker_index(end_opts, name_map)
            
            # Orientation Indices (Left/Right)
            o_l, o_r = (-1, -1)
            if orient and orient[0] and orient[1]:
                o_l = self._find_marker_index(orient[0], name_map)
                o_r = self._find_marker_index(orient[1], name_map)

            if s_idx != -1:
                final_end = e_idx if e_idx != -1 else s_idx
                pairs.append((s_idx, final_end, b_name, axis, fixed, rot, o_l, o_r))

        return pairs

    def _apply_orientation_transform(self, mesh, start, end, orient_l, orient_r, axis_idx):
        """
        Computes a rigid transform matrix to align the mesh to a coordinate frame defined 
        by 3 points: Start, End, and Orientation(L/R).
        This PREVENTS the 'spinning' artifact during flexion.
        """
        # 1. Primary Axis (Longitudinal) - OpenSim Y
        # Vector from Start to End
        y_vec = end - start
        len_y = np.linalg.norm(y_vec)
        if len_y < 0.0001: return mesh # Degenerate
        y_vec /= len_y

        # 2. Secondary Axis (Lateral) - OpenSim Z
        # Vector from Left to Right (e.g. Left Shoulder to Right Shoulder)
        # Note: OpenSim Z is "Right". So L -> R is +Z.
        z_vec = orient_r - orient_l
        len_z = np.linalg.norm(z_vec)
        
        if len_z < 0.0001:
            # Fallback if orientation markers missing/degenerate
            # Assume arbitrary Z if Y is vertical-ish
            z_vec = np.array([1, 0, 0]) 
        else:
            z_vec /= len_z

        # 3. Tertiary Axis (Forward) - OpenSim X
        # X = Y cross Z ?? No.
        # OpenSim: Y is Up. Z is Right. X is Forward.
        # Cross(Up, Right) = Cross(Y, Z) = X (Forward).
        x_vec = np.cross(y_vec, z_vec)
        x_len = np.linalg.norm(x_vec)
        
        if x_len < 0.0001:
            # Fallback
            x_vec = np.array([0, 1, 0])
        else:
            x_vec /= x_len

        # 4. Re-Orthogonalize Z (Z = X cross Y)
        # Ensures 90 degree angles
        z_vec = np.cross(x_vec, y_vec)
        z_vec /= np.linalg.norm(z_vec)

        # 5. Construct Rotation Matrix (3x3)
        # Columns are the new axes [X, Y, Z]
        # OpenSim mesh assumes Y=Up, Z=Right, X=Fwd.
        # We want to map:
        #   Mesh X -> x_vec
        #   Mesh Y -> y_vec
        #   Mesh Z -> z_vec
        
        rot_mat = np.eye(4)
        rot_mat[0:3, 0] = x_vec
        rot_mat[0:3, 1] = y_vec
        rot_mat[0:3, 2] = z_vec
        rot_mat[0:3, 3] = start # Translation

        # 6. Apply Transform
        # First, apply transform to a copy of the mesh
        # Note: PyVista 'transform' applies 4x4 matrix
        mesh.transform(rot_mat, inplace=True)
        return mesh

    def _align_mesh(self, mesh, start, end, b_name, axis_idx, is_fixed, rot_offset, o_l, o_r):
        """Aligns mesh using either Basic Vector or Full Orientation Frame."""
        
        # --- 1. Unit Fix (MM to M) ---
        bounds = mesh.bounds
        native_len = abs(bounds[2*axis_idx+1] - bounds[2*axis_idx])
        if native_len > 10.0:
            mesh.scale([0.001, 0.001, 0.001], inplace=True)
            native_len *= 0.001
        if native_len < 0.0001: native_len = 1.0

        # --- 2. Scaling ---
        if not is_fixed:
            target_len = np.linalg.norm(end - start)
            scale_ratio = target_len / native_len
            scales = [scale_ratio, scale_ratio, scale_ratio]
            # Ensure thickness isn't absurd for wide bones (Pelvis)
            # If using Orientation (o_l/o_r), we often want Uniform scaling to preserve shape
            mesh.scale(scales, inplace=True)
        
        # --- 3. Manual Pre-Rotation (Fix Native Mesh Orientation) ---
        if rot_offset != (0,0,0):
            mesh.rotate_x(rot_offset[0], inplace=True)
            mesh.rotate_y(rot_offset[1], inplace=True)
            mesh.rotate_z(rot_offset[2], inplace=True)

        # --- 4. Alignment Strategy ---
        
        # STRATEGY A: Full Orientation (Torso/Pelvis)
        # If we have Left/Right orientation markers (e.g. Shoulders/ASIS), use them.
        # This prevents the "Spinning Ribs" bug.
        if o_l is not None and o_r is not None:
             # Important: We must ensure the Mesh origin is (0,0,0) before transforming
             # But the mesh is already centered at its joint origin.
             # The _apply_orientation_transform handles Translation to 'start'.
             return self._apply_orientation_transform(mesh, start, end, o_l, o_r, axis_idx)

        # STRATEGY B: Standard Stick-Figure Alignment (Limbs)
        # Use existing rotate_vector logic for limbs where roll is less critical
        else:
            vec = end - start
            target_len = np.linalg.norm(vec)
            if target_len < 0.0001: 
                mesh.translate(start, inplace=True)
                return mesh
            
            # Standard Vector Alignment
            direction_sign = 1.0
            # OpenSim Y=Up. Limbs grow DOWN.
            if axis_idx == 1 and not ('hat' in b_name or 'spine' in b_name): 
                direction_sign = -1.0 

            source_vec = np.zeros(3)
            source_vec[axis_idx] = direction_sign
            
            target_vec = vec / target_len
            rot_axis = np.cross(source_vec, target_vec)
            dot_val = np.dot(source_vec, target_vec)
            
            if np.linalg.norm(rot_axis) < 0.001:
                if dot_val < 0:
                    flip = np.array([1,0,0]) if axis_idx != 0 else np.array([0,1,0])
                    mesh.rotate_vector(vector=flip, angle=180, point=[0,0,0], inplace=True)
            else:
                angle_deg = np.degrees(np.arccos(np.clip(dot_val, -1.0, 1.0)))
                mesh.rotate_vector(vector=rot_axis, angle=angle_deg, point=[0,0,0], inplace=True)

            # Foot Fix (Feet need 90 deg pitch relative to the vector)
            if 'foot' in b_name or 'talus' in b_name or 'bofoot' in b_name:
                mesh.rotate_vector(vector=target_vec, angle=90, point=[0,0,0], inplace=True)

            mesh.translate(start, inplace=True)
            return mesh

    def _update_bone_geometry(self, i, start, end, b_name, axis, fixed, rot, o_l_idx, o_r_idx, frame):
        new_mesh = None
        if self.bone_registry:
            custom_mesh = self.bone_registry.get_mesh(b_name)
            if custom_mesh:
                try:
                    # Get Orientation Marker Positions if indices exist
                    o_l_pos, o_r_pos = None, None
                    if o_l_idx != -1 and o_r_idx != -1:
                        o_l_pos = np.array(self._swizzle(frame[o_l_idx]))
                        o_r_pos = np.array(self._swizzle(frame[o_r_idx]))

                    new_mesh = self._align_mesh(
                        custom_mesh, start, end, b_name, axis, fixed, rot, o_l_pos, o_r_pos
                    )
                except Exception as e:
                    pass

        if new_mesh is None and not fixed:
            line = pv.Line(start, end)
            new_mesh = line.tube(radius=Config.SKELETON_BONE_RADIUS)

        actor_key = f'bone_{i}'
        if actor_key in self.skel_actors:
            self.skel_actors[actor_key].mapper.dataset.DeepCopy(new_mesh)
        else:
            color = "#DDDDDD" if new_mesh and new_mesh.n_points > 100 else "#EEEEEE"
            self.skel_actors[actor_key] = self.add_mesh(
                new_mesh, color=color, reset_camera=False, render=False
            )

    def update_frame(self, idx: int):
        if self.markers is None: return
        try:
            frame = self.markers[idx % len(self.markers)]
            for i, pos in enumerate(frame):
                key = f'joint_{i}'
                if  key in self.skel_actors:
                    self.skel_actors[key].SetPosition(self._swizzle(pos))
            
            for i, p in enumerate(self.bone_pairs):
                # Unpack tuple (Start, End, Name, Axis, Fixed, Rot, OL, OR)
                p1, p2, b_name, axis, fixed, rot, o_l, o_r = p
                
                start = np.array(self._swizzle(frame[p1]))
                end = np.array(self._swizzle(frame[p2]))
                
                #self._update_bone_geometry(i, start, end, b_name, axis, fixed, rot, o_l, o_r, frame)
            
            self.render()
        except Exception as e:
            logger.error(f"Frame Update Error: {e}", exc_info=True)

    def _swizzle(self, p):
        # OpenSim: X=Fwd, Y=Up, Z=Right
        # PyVista: X=Fwd, Z=Up, Y=Left (Negate Z to avoid mirror)
        return (float(p[0]), -float(p[2]), float(p[1]))

    def clear_skeleton(self):
        for k in list(self.skel_actors.keys()):
            self.remove_actor(self.skel_actors[k])
        self.skel_actors = {}
        self.markers = None

# =============================================================================
# DIALOG WINDOWS
# =============================================================================

class PipelineConfigDialog(QDialog):
    """
    Configuration dialog for pipeline execution.
    
    Allows user to select GPU, resolution preset, and which trials to process.
    """
    
    def __init__(self, parent, session_path: str):
        super().__init__(parent)
        self.session_path = Path(session_path)
        self.setWindowTitle("Pipeline Configuration")
        self.resize(400, 500)
        self.setStyleSheet(parent.styleSheet())
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # GPU Selection
        layout.addWidget(QLabel("Select GPU:"))
        self.gpu_combo = QComboBox()
        self.gpu_combo.setToolTip("Choose GPU for processing (requires CUDA)")
        self.gpu_map = self._get_gpu_info()
        self.gpu_combo.addItems(list(self.gpu_map.keys()))
        layout.addWidget(self.gpu_combo)

        # --- NEW: Pose Estimator Selection ---
        layout.addWidget(QLabel("Pose Estimator:"))
        self.pose_combo = QComboBox()
        self.pose_combo.setToolTip("Select the AI model for pose estimation")
        self.pose_combo.addItems(["OpenPose", "RTMPose"])
        layout.addWidget(self.pose_combo)
        # -------------------------------------
        
        # Resolution Selection
        layout.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        self.res_combo.setToolTip("Select processing resolution preset")
        self.res_combo.addItems(["1x368", "1x736", "1x736_2scales","736x1 (Landscape)"])
        layout.addWidget(self.res_combo)

        # Inside _init_ui, after creating pose_combo:
        self.pose_combo.currentTextChanged.connect(self._update_res_options)
        self._update_res_options(self.pose_combo.currentText())
        
        # Trial Selection
        layout.addWidget(QLabel("Select Trials to Process:"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.trial_content = QWidget()
        self.trial_layout = QVBoxLayout(self.trial_content)
        
        self.checks: Dict[str, QCheckBox] = {}
        self._populate_trials()
        
        scroll.setWidget(self.trial_content)
        layout.addWidget(scroll)
        
        # Buttons
        btn_box = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_cancel.setToolTip("Cancel pipeline execution")
        
        btn_run = QPushButton("RUN PIPELINE")
        btn_run.setObjectName("AccentButton")
        btn_run.clicked.connect(self.accept)
        btn_run.setToolTip("Start processing selected trials")
        
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_run)
        layout.addLayout(btn_box)

    def _update_res_options(self, estimator):
        """Changes resolution options or model complexity based on estimator."""
        self.res_combo.clear()
        if estimator.lower() == "openpose":
            # Standard OpenPose resolution presets
            self.res_combo.addItems(["1x368", "1x736", "1x736_2scales", "736x1 (Landscape)"])
            self.res_combo.setEnabled(True)
        else:
            # RTMPose Model Complexity Toggles
            # 'rtmpose-m' is the Balanced/Fast model
            # 'rtmpose-l' is the High Accuracy model
            self.res_combo.addItems(["RTMPose-m (Fast/Balanced)", "RTMPose-l (High Accuracy)"])
            self.res_combo.setEnabled(True) # Re-enable so the user can choose

    def _get_gpu_info(self) -> Dict[str, str]:
        """
        Query available GPUs using nvidia-smi.
        
        Returns:
            Dictionary mapping GPU names to indices
        """
        try:
            cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
            result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            gpu_names = result.decode().strip().split('\n')
            gpu_map = {name.strip(): str(i) for i, name in enumerate(gpu_names)}
            logger.info(f"Detected {len(gpu_map)} GPU(s)")
            return gpu_map
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"GPU detection failed: {e}. Using CPU fallback.")
            return {"CPU / Default": "0"}

    def _populate_trials(self) -> None:
        """Scan session for available trials"""
        media_path = self.session_path / "Videos" / "Cam0" / "InputMedia"
        
        if not media_path.exists():
            logger.warning(f"InputMedia path not found: {media_path}")
            return
        
        trials = [
            d.name for d in media_path.iterdir() 
            if d.is_dir()
        ]
        
        # 1. Filter out calibration trials
        valid_trials = [t for t in trials if t.lower() not in ["intrinsics", "calibration"]]
        
        # 2. Custom Sort: Force 'neutral' to the top (Priority 0), then sort the rest alphabetically (Priority 1)
        sorted_trials = sorted(valid_trials, key=lambda x: (0 if x.lower() == 'neutral' else 1, x.lower()))
        
        for trial_name in sorted_trials:
            checkbox = QCheckBox(trial_name)
            checkbox.setChecked(True)
            checkbox.setToolTip(f"Process trial: {trial_name}")
            
            # Make the neutral trial bold so it stands out as the anchor trial
            if trial_name.lower() == 'neutral':
                checkbox.setStyleSheet("font-weight: bold;")
                
            self.trial_layout.addWidget(checkbox)
            self.checks[trial_name] = checkbox
        
        if not self.checks:
            label = QLabel("No trials available for processing")
            label.setStyleSheet("color: #888;")
            self.trial_layout.addWidget(label)

    def get_data(self) -> Dict[str, Any]:
        """Get selected configuration."""
        selected_trials = [
            InputValidator.sanitize_trial_name(t) 
            for t, chk in self.checks.items() 
            if chk.isChecked()
        ]
        
        return {
            "gpu": self.gpu_map[self.gpu_combo.currentText()],
            "res": self.res_combo.currentText(),
            "pose_estimator": self.pose_combo.currentText().lower(), # Adds 'openpose' or 'rtmpose'
            "trials": selected_trials
        }


class CreateSessionDialog(QDialog):
    """
    Dialog for creating new capture sessions.
    
    Collects subject information, checkerboard settings, and camera count.
    Validates all inputs before accepting.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Session")
        # Apply the current style from the parent window
        if parent:
            self.setStyleSheet(parent.styleSheet())
        self.resize(450, 600)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # --- Subject Information ---
        form = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Session_2024_01")
        self.name_edit.setToolTip("Unique identifier for this session")
        
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("e.g., Subject_001")
        self.subject_edit.setToolTip("Subject identifier for this capture session")

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["m", "f"]) 
        
        self.height_edit = QLineEdit(str(Config.DEFAULT_HEIGHT))
        self.height_edit.setToolTip(f"Subject height in meters ({Config.MIN_HEIGHT}-{Config.MAX_HEIGHT}m)")
        
        self.weight_edit = QLineEdit(str(Config.DEFAULT_WEIGHT))
        self.weight_edit.setToolTip(f"Subject weight in kilograms ({Config.MIN_WEIGHT}-{Config.MAX_WEIGHT}kg)")
        
        self.tag_combo = QComboBox()
        self.tag_combo.addItems([
            "Healthy", "Unimpaired", "Impaired", "Exo-assisted", "Other"
        ])
        self.tag_combo.currentTextChanged.connect(self._handle_tag_change)

        self.placement_combo = QComboBox()
        self.placement_combo.addItems(["Vertical (Wall)", "Horizontal (Ground)"])
        
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Portrait (Default)", "Landscape"])
        self.orientation_combo.setToolTip("Portrait: Phone held vertically.\nLandscape: Phone held horizontally.")
        
        form.addRow("Session Name:*", self.name_edit)
        form.addRow("Subject ID:*", self.subject_edit)
        form.addRow("Gender:*", self.gender_combo)
        form.addRow("Height (m):*", self.height_edit)
        form.addRow("Weight (kg):*", self.weight_edit)
        form.addRow("Tag:", self.tag_combo)
        
        layout.addLayout(form)
        
        # --- Camera & Calibration Settings ---
        layout.addWidget(QLabel("<b>Camera & Calibration Settings</b>"))
        form_calib = QFormLayout()

        # 1. Camera Type (Priority 2)
        self.cam_type_combo = QComboBox()
        self.cam_type_combo.addItems(["iPhone", "Android", "Other", "Mixed"])
        self.cam_type_combo.setToolTip("Select the type of camera device being used")

        # 2. Checkerboard Placement (Priority 1)
        self.placement_combo = QComboBox()
        self.placement_combo.addItems(["Vertical (Wall)", "Horizontal (Ground)"])
        self.placement_combo.setToolTip(
            "Vertical: Board matches gravity (standard).\n"
            "Horizontal: Board is flat on ground (good for 360° setups)."
        )

        self.rows_edit = QLineEdit(str(Config.DEFAULT_CHECKERBOARD_ROWS))
        self.cols_edit = QLineEdit(str(Config.DEFAULT_CHECKERBOARD_COLS))
        self.size_edit = QLineEdit(str(Config.DEFAULT_SQUARE_SIZE))
        self.cams_edit = QLineEdit(str(Config.DEFAULT_NUM_CAMERAS))
        
        form_calib.addRow("Camera Type:", self.cam_type_combo)
        form_calib.addRow("Camera Orientation:", self.orientation_combo)
        form_calib.addRow("Camera Nums:", self.cams_edit)
        form_calib.addRow("Board Placement:", self.placement_combo)
        form_calib.addRow("Rows:", self.rows_edit)
        form_calib.addRow("Cols:", self.cols_edit)
        form_calib.addRow("Square Size (mm):", self.size_edit)
        

        layout.addLayout(form_calib)
        
        # Required fields note
        note_label = QLabel("* Required fields")
        note_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(note_label)
        
        # Buttons
        btn_box = QHBoxLayout()
        ok_btn = QPushButton("Create")
        ok_btn.setObjectName("AccentButton")
        ok_btn.clicked.connect(self._validate_and_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_box.addWidget(cancel_btn)
        btn_box.addWidget(ok_btn)
        layout.addLayout(btn_box)
    
    def _validate_and_accept(self) -> None:
        """Validate all inputs before accepting dialog"""
        errors = []
        
        # Session name
        if not self.name_edit.text().strip():
            errors.append("Session name cannot be empty")
        
        # Subject ID
        if not self.subject_edit.text().strip():
            errors.append("Subject ID cannot be empty")
        
        # Height validation
        is_valid, error, _ = InputValidator.validate_float(
            self.height_edit.text(),
            Config.MIN_HEIGHT,
            Config.MAX_HEIGHT,
            "Height"
        )
        if not is_valid:
            errors.append(error)
        
        # Weight validation
        is_valid, error, _ = InputValidator.validate_float(
            self.weight_edit.text(),
            Config.MIN_WEIGHT,
            Config.MAX_WEIGHT,
            "Weight"
        )
        if not is_valid:
            errors.append(error)
        
        # Rows validation
        is_valid, error, _ = InputValidator.validate_int(
            self.rows_edit.text(),
            Config.MIN_CHECKERBOARD_DIM,
            100,
            "Rows"
        )
        if not is_valid:
            errors.append(error)
        
        # Columns validation
        is_valid, error, _ = InputValidator.validate_int(
            self.cols_edit.text(),
            Config.MIN_CHECKERBOARD_DIM,
            100,
            "Columns"
        )
        if not is_valid:
            errors.append(error)
        
        # Square size validation
        is_valid, error, _ = InputValidator.validate_float(
            self.size_edit.text(),
            1.0,
            1000.0,
            "Square size"
        )
        if not is_valid:
            errors.append(error)
        
        # Cameras validation
        is_valid, error, _ = InputValidator.validate_int(
            self.cams_edit.text(),
            Config.MIN_CAMERAS,
            Config.MAX_CAMERAS,
            "Number of cameras"
        )
        if not is_valid:
            errors.append(error)
        
        # Show errors if any
        if errors:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please correct the following errors:\n\n" + "\n".join(f"• {e}" for e in errors)
            )
            return
        
        self.accept()

    def get_data(self) -> Dict[str, Any]:
        """
        Get validated form data.
        
        Returns:
            Dictionary with all session parameters
        """
        return {
            "name": self.name_edit.text().strip(),
            "subject": self.subject_edit.text().strip(),
            "gender": self.gender_combo.currentText(),
            "height": float(self.height_edit.text()),
            "weight": float(self.weight_edit.text()),
            "tag": self.tag_combo.currentText(),
            "cam_type": self.cam_type_combo.currentText(),
            "placement": self.placement_combo.currentText(),
            "rows": int(self.rows_edit.text()),
            "cols": int(self.cols_edit.text()),
            "size": float(self.size_edit.text()),
            "cams": int(self.cams_edit.text()),
            "orientation": self.orientation_combo.currentText()
        }

    def _handle_tag_change(self, text):
        """If 'Other' is selected, prompt for a custom string."""
        if text == "Other":
            from PyQt5.QtWidgets import QInputDialog
            custom_tag, ok = QInputDialog.getText(
                self, 
                "Custom Subject Tag", 
                "Enter specific subject condition/tag:",
                text=""
            )
            
            if ok and custom_tag.strip():
                # Temporarily add the custom tag to the combo and select it
                self.tag_combo.addItem(custom_tag.strip())
                self.tag_combo.setCurrentText(custom_tag.strip())
            else:
                # If user cancelled, revert to a default like 'healthy'
                self.tag_combo.setCurrentIndex(0)

class MixedCameraDialog(QDialog):
    """Dialogue to assign device types to individual cameras."""
    def __init__(self, parent, num_cameras):
        super().__init__(parent)
        self.setWindowTitle("Mixed Camera Setup")
        layout = QVBoxLayout(self)
        self.combos = {}

        form = QFormLayout()
        for i in range(num_cameras):
            combo = QComboBox()
            combo.addItems(["iPhone", "Android", "Other"])
            cam_name = f"Cam{i}"
            form.addRow(f"{cam_name} Device:", combo)
            self.combos[cam_name] = combo
        
        layout.addLayout(form)
        
        btn = QPushButton("Confirm Assignments")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def get_config(self):
        # Maps user-friendly names to the internal strings main.py expects
        mapping = {
            "iPhone": "iPhone_Auto_Detect",
            "Android": "Android_Generic",
            "Other": "Generic_Webcam"
        }
        return {cam: mapping[cb.currentText()] for cam, cb in self.combos.items()}

# =============================================================================
# MAIN WINDOW
# =============================================================================

class OpenCapPro(QMainWindow):
    """
    Main application window for OpenCap Portable Pro.
    
    Provides complete interface for session management, video import,
    pipeline execution, and results visualization.
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize paths
        self.app_path = Path(__file__).parent.resolve()
        self.data_dir = self.app_path / "Data"
        self.data_dir.mkdir(exist_ok=True)
        self.script_path = self.app_path / "reprocessOffline.py"

        # SET APP ICON
        icon_path = self.app_path / "app_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        # Initialize settings (Use ONE consistent registry key)
        self.settings = QSettings('OpenCap', 'LauncherPro')
        
        # Initialize process
        self.process: Optional[QProcess] = None
        
        # Setup window
        self.setWindowTitle("OpenCap Portable Pro")

        # --- UNIFIED THEME LOADING ---
        # Default to dark mode for fresh installs
        saved_theme = self.settings.value('theme', 'dark') 
        
        if saved_theme == 'dark':
            self.setStyleSheet(Styles.DARK)
            self.current_theme = 'dark'
        else:
            self.setStyleSheet(Styles.LIGHT)
            self.current_theme = 'light'
        
        # Build UI (Now everything will inherit the correct theme from the start)
        self._init_ui()
        self._setup_shortcuts()
        self._restore_geometry()
        
        # Ensure 3D viewer background matches the loaded theme
        if self.current_theme == 'light' and hasattr(self, 'skeleton_viewer'):
            self.skeleton_viewer.set_background("#e0e0e0")
        elif self.current_theme == 'dark' and hasattr(self, 'skeleton_viewer'):
            self.skeleton_viewer.set_background(Colors.BG)
        
        # Initial data load
        self.cam_buttons: Dict[str, Dict[str, Any]] = {}

        # DEBOUNCE TIMER
        self.tree_click_timer = QTimer()
        self.tree_click_timer.setSingleShot(True)
        self.tree_click_timer.timeout.connect(self._execute_tree_click)
        self.pending_click_data = None

        self.refresh_sessions()
        
        logger.info("OpenCap Portable Pro initialized")

    def _handle_stdout(self):
        """Read output from the running pipeline process and update the GUI."""
        if not self.process:
            return
            
        # Read all available data from the process
        data = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        
        # Process the output line by line
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
                
            # 1. Update Progress Bar
            if line.startswith("%%PROGRESS:"):
                try:
                    val = int(line.split(":", 1)[1].strip())
                    self.progress_bar.setValue(val)
                except ValueError:
                    pass
                    
            # 2. Update Status Label
            elif line.startswith("%%STATUS:"):
                msg = line.split(":", 1)[1].strip()
                self.progress_label.setText(msg)
                # Let's make status updates bold in the log so they stand out
                self.log_box.append(f"<span style='color: {Colors.ACCENT};'><b>{msg}</b></span>")
                
            # 3. Standard Output (FFmpeg, OpenPose, OpenSim logs)
            else:
                self.log_box.append(line)
                
        # Auto-scroll the log box to the bottom so you can see the latest output
        scrollbar = self.log_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _init_ui(self):
        """Initialize all UI components with Vertical Splitter for Resizability"""
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 1. Fixed Header Area
        self._create_menu_bar()
        self._create_header()
        
        # 2. Controls Area (Import + Pipeline buttons)
        self.controls_container = QWidget()
        controls_layout = QVBoxLayout(self.controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        self._create_import_panel()
        self._create_pipeline_strip()
        self.main_layout.addWidget(self.controls_container)

        # 3. VERTICAL SPLITTER: Dashboard (Top) vs Log Area (Bottom)
        # This allows you to resize the height of the log area
        self.v_splitter = QSplitter(Qt.Vertical)
        
        # Add the existing horizontal dashboard splitter to the top of the vertical one
        self._create_dashboard()
        self.v_splitter.addWidget(self.splitter)
        
        # Add the new log area content to the bottom
        self.log_container = QWidget()
        self._create_log_area_content()
        self.v_splitter.addWidget(self.log_container)
        
        # Initial sizes: 75% for visualization, 25% for logs
        self.v_splitter.setSizes([700, 250])
        
        self.main_layout.addWidget(self.v_splitter, 1) # '1' ensures it expands to fill space
        
        # 4. Status Bar (Fixed at bottom)
        self._create_status_bar()
        
        # FIX: Allow window to be resized in both directions
        self.setMinimumSize(1200, 800) 

    def _create_log_area_content(self):
        """Refactored Log Area to live inside the Vertical Splitter"""
        log_layout = QVBoxLayout(self.log_container)
        # Increased bottom margin (20) to prevent clipping in full-screen
        log_layout.setContentsMargins(5, 5, 5, 20) 
        log_layout.setSpacing(5)
        
        # Progress section
        progress_container = QWidget()
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_label = QLabel("Idle")
        self.progress_label.setStyleSheet("color: #888; font-size: 11px; border: none;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(15)
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        log_layout.addWidget(progress_container)
        
        # Log output - Removed fixed height so it respects the splitter
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Process output will appear here...")
        log_layout.addWidget(self.log_box)
    
    def _create_menu_bar(self):
        """Create menu bar with shortcuts"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        new_action = QAction('&New Session...', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.open_new_session_dialog)
        new_action.setToolTip("Create a new capture session")
        file_menu.addAction(new_action)
        
        refresh_action = QAction('&Refresh Sessions', self)
        refresh_action.setShortcut('Ctrl+R')
        refresh_action.triggered.connect(self.refresh_sessions)
        refresh_action.setToolTip("Refresh session list from disk")
        file_menu.addAction(refresh_action)
        
        file_menu.addSeparator()
        
        # Recent sessions submenu
        self.recent_menu = file_menu.addMenu('Recent Sessions')
        self._update_recent_menu()
        
        file_menu.addSeparator()
        
        quit_action = QAction('&Quit', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # --- FIX 7: VIEW MENU FOR THEME ---
        view_menu = menubar.addMenu('&View')
        
        theme_action = QAction('&Toggle Dark/Light Mode', self)
        theme_action.setShortcut('Ctrl+T')
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)
        # ----------------------------------
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # --- FIX 7: THEME TOGGLE FUNCTION ---
    def toggle_theme(self):
        """Switch between Dark and Light stylesheets"""
        current_sheet = self.styleSheet()
        
        if "background-color: #f0f0f0" in current_sheet:
            # Switch to Dark
            self.setStyleSheet(Styles.DARK)
            if hasattr(self, 'skeleton_viewer'):
                self.skeleton_viewer.set_background(Colors.BG)
            self.settings.setValue('theme', 'dark')
        else:
            # Switch to Light
            self.setStyleSheet(Styles.LIGHT)
            if hasattr(self, 'skeleton_viewer'):
                self.skeleton_viewer.set_background("#e0e0e0") 
            self.settings.setValue('theme', 'light')
            
        # Refresh dynamic elements
        self._update_logo()
        self._update_trial_name_input()

    def _update_logo(self):
        """Update logo based on current theme"""
        # Determine which file to use
        current_theme = self.settings.value('theme', 'dark')
        
        if current_theme == 'light':
            logo_file = "opencap_logo_light.png"
        else:
            logo_file = "opencap_logo_dark.png"
            
        logo_path = self.app_path / logo_file
        
        # Load and scale the image
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            scaled = pixmap.scaled(
                Config.LOGO_MAX_WIDTH,
                Config.LOGO_MAX_HEIGHT,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.logo_label.setPixmap(scaled)
        else:
            # Fallback text if image missing
            self.logo_label.setText("OpenCap")
            color = Colors.ACCENT if current_theme == 'dark' else Colors.ACCENT
            self.logo_label.setStyleSheet(
                f"font-size: 20px; color: {color}; font-weight: bold; border: none;"
            )

    def _create_header(self):
        """Create header with logo and session selector"""
        header = QFrame()
        header.setFixedHeight(Config.HEADER_HEIGHT)
        header.setObjectName("HeaderFrame")
        
        header_layout = QHBoxLayout(header)
        
        # --- LEFT SIDE: Logo and New Session ---
        self.logo_label = QLabel()
        self._update_logo()
        header_layout.addWidget(self.logo_label)

        # New session button
        new_btn = QPushButton("+ New Session")
        new_btn.setObjectName("AccentButton")
        new_btn.clicked.connect(self.open_new_session_dialog)
        new_btn.setToolTip("Create new session (Ctrl+N)")
        header_layout.addWidget(new_btn)

        header_layout.addStretch()

        # --- RIGHT SIDE: Metadata, Refresh, and Selector ---
        
        # Session selector
        self.session_combo = QComboBox()
        self.session_combo.setFixedWidth(200)
        self.session_combo.currentTextChanged.connect(self.on_session_change)
        
        self.session_combo.setToolTip("Select active session")
        header_layout.addWidget(self.session_combo)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_sessions)
        refresh_btn.setToolTip("Refresh session list (Ctrl+R)")
        header_layout.addWidget(refresh_btn)
        
        self.edit_meta_btn = QPushButton("Edit Metadata")
        self.edit_meta_btn.clicked.connect(self.open_metadata_editor)
        header_layout.addWidget(self.edit_meta_btn)
        
        self.main_layout.addWidget(header)

    def open_metadata_editor(self):
        session = self.session_combo.currentText()
        if not session: return
        
        dlg = MetadataEditorDialog(self, self.data_dir / session)
        if dlg.exec_() == QDialog.Accepted:
            self._update_status("Metadata updated successfully.", success=True)
            # Refresh UI elements that might depend on metadata
            self.on_session_change(session)
    
    def _create_import_panel(self):
        """Create trial import interface"""
        panel = QFrame()
        panel.setObjectName("ImportPanel") # <--- THIS IS THE KEY FIX
        panel_layout = QVBoxLayout(panel)
        
        # Top row: Trial type selection
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel(
            "IMPORT TRIAL:", 
            styleSheet="font-weight:bold; color:#888;"
        ))
        
        # Radio buttons for trial types
        self.type_group = QButtonGroup(self)
        self.type_buttons: Dict[str, QRadioButton] = {}
        
        for text, trial_type in [
            ("Intrinsics", TrialType.INTRINSICS),
            ("Calibration", TrialType.CALIBRATION),
            ("Neutral", TrialType.NEUTRAL),
            ("Dynamic", TrialType.DYNAMIC)
        ]:
            radio = QRadioButton(text)
            if trial_type == TrialType.CALIBRATION:
                radio.setChecked(True)
            
            radio.setToolTip(self._get_trial_type_tooltip(trial_type))
            self.type_group.addButton(radio)
            top_row.addWidget(radio)
            self.type_buttons[trial_type.value] = radio
            radio.toggled.connect(self._update_trial_name_input)
        
        # Trial name input
        self.trial_name_edit = QLineEdit()
        self.trial_name_edit.setPlaceholderText("Trial Name")
        self.trial_name_edit.setToolTip(
            "Custom name for dynamic trials.\n"
            "Examples: walking_1, running_fast, jump_test"
        )
        top_row.addWidget(self.trial_name_edit)
        
        panel_layout.addLayout(top_row)
        
        # Camera slots (dynamic)
        self.cam_slot_layout = QGridLayout()
        panel_layout.addLayout(self.cam_slot_layout)
        
        # Import button
        import_btn = QPushButton("EXECUTE IMPORT")
        import_btn.clicked.connect(self.run_import)
        import_btn.setToolTip("Import selected videos into session structure (Ctrl+I)")
        panel_layout.addWidget(import_btn)
        
        self.main_layout.addWidget(panel)
        self._update_trial_name_input()
    
    def _get_trial_type_tooltip(self, trial_type: TrialType) -> str:
        """Get tooltip text for trial type"""
        tooltips = {
            TrialType.INTRINSICS: "Intrinsic calibration images for camera parameters",
            TrialType.CALIBRATION: "Extrinsic calibration trial with checkerboard",
            TrialType.NEUTRAL: "Neutral pose trial for model initialization",
            TrialType.DYNAMIC: "Dynamic movement trial (walking, running, etc.)"
        }
        return tooltips.get(trial_type, "")
    
    def _create_pipeline_strip(self):
        """Create pipeline control buttons with Research Mode toggle"""
        from PyQt5.QtWidgets import QStackedWidget
        
        strip = QWidget()
        strip_layout = QVBoxLayout(strip)
        strip_layout.setContentsMargins(20, 10, 20, 10)
        
        # --- TOGGLE SWITCH ---
        toggle_layout = QHBoxLayout()
        self.research_mode_cb = QCheckBox("Research Mode (Granular Controls)")
        self.research_mode_cb.setStyleSheet("font-weight: bold; color: #888;")
        self.research_mode_cb.toggled.connect(self._toggle_research_mode)
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.research_mode_cb)
        strip_layout.addLayout(toggle_layout)
        
        # --- STACKED WIDGET ---
        self.pipeline_stack = QStackedWidget()
        
        # 1. CLINICAL PAGE
        self.clinical_page = QWidget()
        clin_layout = QHBoxLayout(self.clinical_page)
        clin_layout.setContentsMargins(0, 0, 0, 0)
        
        calibration_btn_clin = QPushButton("1. RUN INTRINSICS")
        calibration_btn_clin.setFixedHeight(Config.BUTTON_HEIGHT)
        calibration_btn_clin.clicked.connect(self.run_intrinsics)
        
        pipeline_btn = QPushButton("2. RUN PIPELINE")
        pipeline_btn.setObjectName("AccentButton") # Keeps the bright color
        pipeline_btn.setFixedHeight(Config.BUTTON_HEIGHT)
        pipeline_btn.clicked.connect(lambda: self.run_pipeline(step="all"))
        
        clin_layout.addWidget(calibration_btn_clin)
        clin_layout.addWidget(pipeline_btn)
        
        # 2. RESEARCH PAGE
        self.research_page = QWidget()
        res_layout = QHBoxLayout(self.research_page)
        res_layout.setContentsMargins(0, 0, 0, 0)
        
        calibration_btn_res = QPushButton("1. Run Intrinsics")
        calibration_btn_res.setFixedHeight(Config.BUTTON_HEIGHT)
        calibration_btn_res.clicked.connect(self.run_intrinsics)
        
        # --- NEW: Dedicated Extrinsics Button ---
        btn_extrinsics = QPushButton("2. Calibrate Extrinsics")
        btn_extrinsics.setFixedHeight(Config.BUTTON_HEIGHT)
        btn_extrinsics.clicked.connect(lambda: self.run_pipeline(step="calibrate"))
        
        # --- CHANGED: Pose Estimator now acts as Step 3 ---
        btn_pose = QPushButton("3. Run Pose")
        btn_pose.setFixedHeight(Config.BUTTON_HEIGHT)
        btn_pose.clicked.connect(lambda: self.run_pipeline(step="pose"))
        
        btn_kinematics = QPushButton("4. Triangulate & OpenSim")
        btn_kinematics.setFixedHeight(Config.BUTTON_HEIGHT)
        btn_kinematics.clicked.connect(lambda: self.run_pipeline(step="kinematics"))
        
        res_layout.addWidget(calibration_btn_res)
        res_layout.addWidget(btn_extrinsics)
        res_layout.addWidget(btn_pose)
        res_layout.addWidget(btn_kinematics)
        
        # Add pages to stack
        self.pipeline_stack.addWidget(self.clinical_page)
        self.pipeline_stack.addWidget(self.research_page)
        
        strip_layout.addWidget(self.pipeline_stack)
        self.main_layout.addWidget(strip)

    def _toggle_research_mode(self, checked):
        """Swaps the visible button panel"""
        if checked:
            self.pipeline_stack.setCurrentIndex(1)
            self._update_status("Research Mode Enabled: Granular execution unlocked.")
        else:
            self.pipeline_stack.setCurrentIndex(0)
            self._update_status("Clinical Mode Enabled: Automated pipeline locked in.")
    
    def _create_dashboard(self):
        """Create main visualization dashboard"""
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left: Trial tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setFixedWidth(Config.TREE_WIDTH)
        self.tree.itemClicked.connect(self.on_tree_click)
        self.tree.setToolTip("Click trial to load video and 3D visualization")

        # --- NEW: Enable Right-Click Context Menu ---
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_tree_context_menu)
        # --------------------------------------------
        
        self.splitter.addWidget(self.tree)
        
        # Center: 3D viewer
        # --- CHANGED: Point to 'opencap-portable/Geometry' (app_path/Geometry) ---
        geometry_folder = self.app_path / "Geometry"
        self.skeleton_viewer = SkeletonViewer3D(geometry_path=geometry_folder)
        self.splitter.addWidget(self.skeleton_viewer)
        
        # Right: Video player
        self.video_container = QWidget()
        self.video_container.setFixedWidth(Config.VIDEO_CONTAINER_WIDTH)
        video_layout = QVBoxLayout(self.video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        preview_label = QLabel("PREVIEW")
        preview_label.setStyleSheet("font-weight:bold; color:#888; padding:5px;")
        video_layout.addWidget(preview_label)
        
        self.video_player = DualVideoPlayer()
        self.video_player.frame_changed.connect(self.skeleton_viewer.update_frame)

        video_layout.addWidget(self.video_player)
        self.splitter.addWidget(self.video_container)
        
        self.splitter.setSizes([300, 600, 600])
        self.main_layout.addWidget(self.splitter, 1)
    
    def _create_log_area(self):
        """Create log output area with progress bar"""
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(5, 5, 5, 5)
        log_layout.setSpacing(5)
        
        # Progress section
        progress_container = QWidget()
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_label = QLabel("Idle")
        self.progress_label.setStyleSheet("color: #888; font-size: 11px;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(20)
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        log_layout.addWidget(progress_container)
        
        # Log output
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(Config.LOG_HEIGHT)
        self.log_box.setPlaceholderText("Process output will appear here...")
        log_layout.addWidget(self.log_box)
        
        self.main_layout.addWidget(log_container)
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Session label (permanent)
        self.session_status_label = QLabel()
        self.status_bar.addPermanentWidget(self.session_status_label)
        
        self.status_bar.showMessage("Ready")
    
    def _setup_shortcuts(self):
        """Configure keyboard shortcuts"""
        # Import
        QShortcut(QKeySequence("Ctrl+I"), self, self.run_import)
        
        # Pipeline
        QShortcut(QKeySequence("Ctrl+Shift+P"), self, self.run_pipeline)
    
    def _restore_geometry(self):
        """Restore window geometry from settings"""
        geometry = self.settings.value('window/geometry')
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
        
        splitter_state = self.settings.value('splitter/state')
        if splitter_state:
            self.splitter.restoreState(splitter_state)
    
    # -------------------------------------------------------------------------
    # SESSION MANAGEMENT
    # -------------------------------------------------------------------------
    
    def refresh_sessions(self) -> None:
        """Refresh session list from disk"""
        logger.info("Refreshing sessions")
        current = self.session_combo.currentText()
        self.session_combo.clear()
        
        if self.data_dir.exists():
            sessions = sorted([
                d.name for d in self.data_dir.iterdir() 
                if d.is_dir()
            ])
            self.session_combo.addItems(sessions)
            
            if current in sessions:
                self.session_combo.setCurrentText(current)
            
            logger.info(f"Found {len(sessions)} session(s)")
        
        self._update_status("Sessions refreshed")
    
    def on_session_change(self, session_name: str) -> None:
        """Handle session selection change"""
        if not session_name:
            return
        
        logger.info(f"Session changed to: {session_name}")
        self._add_to_recent(session_name)
        self.refresh_cam_slots(session_name)
        self.refresh_tree(session_name)
        
        self._update_status(f"Loaded session: {session_name}")
        self.session_status_label.setText(f"Session: {session_name}")
    
    def refresh_cam_slots(self, session_name: str) -> None:
        """Refresh camera import slots with Multi-select and Clear support"""
        # Clear existing slots
        for i in reversed(range(self.cam_slot_layout.count())):
            widget = self.cam_slot_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        video_root = self.data_dir / session_name / "Videos"
        cameras = []
        if video_root.exists():
            cameras = sorted([d.name for d in video_root.iterdir() if d.is_dir() and d.name.startswith("Cam")])
        
        self.cam_buttons = {}
        for i, cam in enumerate(cameras):
            label = QLabel(f"{cam}:")
            path_label = QLabel("No files selected")
            path_label.setStyleSheet("color: gray;")
            
            # Browse Button
            browse_btn = QPushButton("Browse")
            browse_btn.clicked.connect(lambda checked, c=cam, l=path_label: self._browse_file(c, l))
            
            # Clear Button (NEW)
            clear_btn = QPushButton("Clear")
            clear_btn.setFixedWidth(60)
            clear_btn.clicked.connect(lambda checked, c=cam, l=path_label: self._clear_selection(c, l))
            
            self.cam_slot_layout.addWidget(label, i, 0)
            self.cam_slot_layout.addWidget(path_label, i, 1)
            self.cam_slot_layout.addWidget(browse_btn, i, 2)
            self.cam_slot_layout.addWidget(clear_btn, i, 3) # Add to column 3
            
            self.cam_buttons[cam] = {"paths": [], "label": path_label}

        
        logger.info(f"Created {len(cameras)} camera slot(s)")
    
    def _browse_file(self, camera: str, label: QLabel) -> None:
        """Browse for one or more video files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, f"Select Videos for {camera}", "", 
            "Video Files (*.mp4 *.avi *.mov);;All Files (*.*)"
        )
        
        if file_paths:
            valid_paths = []
            filenames = []
            for path in file_paths:
                is_valid, error = InputValidator.validate_video_path(path)
                if is_valid:
                    valid_paths.append(path)
                    filenames.append(Path(path).name)
                else:
                    self.log_box.append(f"Skipping {Path(path).name}: {error}")

            if valid_paths:
                self.cam_buttons[camera]["paths"] = valid_paths
                # Update text immediately
                display_text = ", ".join(filenames)
                if len(display_text) > 55:
                    display_text = f"{len(filenames)} files: {filenames[0]}..."
                
                label.setText(display_text)
                label.setStyleSheet(f"color: {Colors.ACCENT}; font-weight: bold;")

    def _clear_selection(self, camera: str, label: QLabel) -> None:
        """Resets the selection for a specific camera"""
        self.cam_buttons[camera]["paths"] = []
        label.setText("No files selected")
        label.setStyleSheet("color: gray; font-weight: normal;")
        logger.info(f"Cleared selection for {camera}")
    
    def refresh_tree(self, session_name: str):
        # --- 1. SAVE EXPANSION STATE ---
        expanded_paths = set()
        
        # Helper function to recursively record which folders are open
        def save_state(item, current_path):
            path = f"{current_path}/{item.text(0)}" if current_path else item.text(0)
            if item.isExpanded():
                expanded_paths.add(path)
            for i in range(item.childCount()):
                save_state(item.child(i), path)
                
        # Only save state if the tree actually has items in it
        was_populated = self.tree.topLevelItemCount() > 0
        if was_populated:
            for i in range(self.tree.topLevelItemCount()):
                save_state(self.tree.topLevelItem(i), "")

        # Now we can safely clear the tree
        self.tree.clear()
        
        video_root = self.data_dir / session_name / "Videos"
        if not video_root.exists(): return
        
        # --- LOGGING START ---
        self.log_box.append(f"\n--- Scanning Session: {session_name} ---")
        
        cameras = sorted([d.name for d in video_root.iterdir() if d.name.startswith("Cam")])
        
        for cam in cameras:
            cam_item = QTreeWidgetItem([cam])
            cam_item.setIcon(0, self.style().standardIcon(QStyle.SP_DriveCDIcon))
            self.tree.addTopLevelItem(cam_item)
            
            cam_root = video_root / cam
            media_path = cam_root / "InputMedia"
            
            # Find all folders starting with "OutputVideos"
            output_candidates = sorted([
                d.name for d in cam_root.iterdir()
                if d.is_dir() and (
                    d.name.startswith("OutputVideos") or 
                    d.name.startswith("OutputMedia") or 
                    d.name.startswith("OutputJsons")
                )
            ], reverse=True)
            
            # Debug: Tell us if it even found the OutputVideos folder
            self.log_box.append(f"[{cam}] Found Output Folders: {output_candidates}")
            
            if media_path.exists():
                trials = sorted([t.name for t in media_path.iterdir() if t.is_dir()], key=get_trial_sort_key)
                
                for trial_name in trials:
                    trial_path = media_path / trial_name
                    
                    # 1. TRC Path
                    trc_path = self.data_dir / session_name / "MarkerData" / "PreAugmentation" / f"{trial_name}.trc"
                    
                    # 2. Overlay Search
                    overlay_path = None
                    trc_path = None # Initialize here
                    
                    for out_folder in output_candidates:
                        res_string = out_folder.replace("OutputVideos_", "").replace("OutputMedia_", "")
                        
                        potential_trc = self.data_dir / session_name / "MarkerData" / res_string / "PreAugmentation" / f"{trial_name}.trc"
                        exact_vid_path = cam_root / out_folder / trial_name / f"{trial_name}_overlay.avi"
                        
                        if exact_vid_path.exists():
                            overlay_path = str(exact_vid_path)
                            trc_path = str(potential_trc) if potential_trc.exists() else None
                            break
                        else:
                            search_dir = cam_root / out_folder / trial_name
                            if search_dir.exists():
                                overlays = [f for f in search_dir.iterdir() if f.suffix.lower() in Config.OVERLAY_EXTENSION]
                                if overlays:
                                    overlay_path = str(overlays[0])
                                    self.log_box.append(f"  -> FOUND OVERLAY (Fallback): {overlay_path}")
                                    break
                    
                    # 3. Create Trial Folder Item
                    trial_item = QTreeWidgetItem([trial_name])
                    trial_item.setIcon(0, self.style().standardIcon(QStyle.SP_DirIcon))
                    cam_item.addChild(trial_item)
                    
                    # 4. Get the raw video paths
                    videos = sorted([f for f in trial_path.iterdir() if f.suffix.lower() in Config.VIDEO_EXTENSIONS])
                    if not videos:
                        continue
                        
                    raw_vid_path = str(videos[0])
                    
                    # 5. Add a "Raw Only" clickable item for EVERY video found
                    for vid in videos:
                        raw_item = QTreeWidgetItem([f"Raw Video: {vid.name}"])
                        raw_item.setIcon(0, self.style().standardIcon(QStyle.SP_FileIcon))
                        raw_item.setData(0, Qt.UserRole, {
                            "type": "video",
                            "path": str(vid),
                            "overlay": None,
                            "trc": None
                        })
                        trial_item.addChild(raw_item)
                    
                    # 6. Add a clickable item for EVERY resolution it finds!
                    for out_folder in output_candidates:
                        res_string = out_folder.replace("OutputVideos_", "").replace("OutputMedia_", "").replace("OutputJsons_", "")
                        exact_vid_path = cam_root / out_folder / trial_name / f"{trial_name}_overlay.avi"
                        
                        # REMOVED hardcoded "OpenPose_" here so it finds RTMPose directories seamlessly
                        potential_trc = self.data_dir / session_name / "MarkerData" / res_string / "PreAugmentation" / f"{trial_name}.trc"
                        
                        if exact_vid_path.exists():
                            proc_item = QTreeWidgetItem([f"Overlay: {res_string}"])
                            proc_item.setIcon(0, self.style().standardIcon(QStyle.SP_MediaPlay))
                            proc_item.setData(0, Qt.UserRole, {
                                "type": "video",
                                "path": raw_vid_path,
                                "overlay": str(exact_vid_path),
                                "trc": str(potential_trc) if potential_trc.exists() else None
                            })
                            trial_item.addChild(proc_item)
                        
            # Set default expansion for first-time loads
            cam_item.setExpanded(True)
            
        # --- 2. RESTORE EXPANSION STATE ---
        # If the tree had items before we refreshed, re-apply the user's specific state
        if was_populated:
            def restore_state(item, current_path):
                path = f"{current_path}/{item.text(0)}" if current_path else item.text(0)
                item.setExpanded(path in expanded_paths)
                for i in range(item.childCount()):
                    restore_state(item.child(i), path)
                    
            for i in range(self.tree.topLevelItemCount()):
                restore_state(self.tree.topLevelItem(i), "")
            
        # Scroll log to bottom so you see the latest check
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())
    
    def on_tree_click(self, item, col):
        """Intercepts the click and starts a tiny delay to prevent spam-crashing."""
        data = item.data(0, Qt.UserRole)
        if not data: return
        
        if data["type"] == "video":
            self.pending_click_data = data
            # Restart the 200ms timer. If clicked again quickly, the timer resets.
            self.tree_click_timer.start(200) 
            self._update_status("Loading media...", warning=True)

    def _execute_tree_click(self):
        """Actually loads the video and skeleton after the user stops clicking."""
        data = self.pending_click_data
        if not data: return
        
        logger.info(f"Loading media from path: {data['path']}")
        
        # 1. Load Skeleton
        if data.get("trc"):
            self.skeleton_viewer.load_trc(data["trc"])
        else:
            self.skeleton_viewer.clear_skeleton()
        
        # 2. Load Video & Overlay
        self.video_player.blockSignals(True)
        overlay_path = data.get("overlay")
        self.video_player.load(data["path"], overlay_path)
        self.video_player.blockSignals(False)
        
        # 3. Manually sync the first frame
        if self.video_player.total_frames > 0:
            self.video_player.show_frame(0)
            
        self._update_status("Media loaded successfully.", success=True)
    
    # -------------------------------------------------------------------------
    # TRIAL IMPORT
    # -------------------------------------------------------------------------
    
    def _update_trial_name_input(self) -> None:
        """Update trial name input based on selection and theme"""
        # Get selected trial type
        selected = [
            k for k, v in self.type_buttons.items() 
            if v.isChecked()
        ][0]
        
        is_dynamic = (selected == TrialType.DYNAMIC.value)
        
        # Check current theme
        current_theme = self.settings.value('theme', 'dark')
        is_dark = (current_theme != 'light')
        
        if is_dynamic:
            self.trial_name_edit.setText(Config.DEFAULT_DYNAMIC_NAME)
            self.trial_name_edit.setEnabled(True)
            
            # Active Styling (Editable)
            bg = Colors.INPUT_BG if is_dark else Colors.LIGHT_INPUT
            text = "white" if is_dark else Colors.LIGHT_TEXT
            border = "#444" if is_dark else Colors.LIGHT_BORDER
            
            self.trial_name_edit.setStyleSheet(
                f"color: {text}; background-color: {bg}; border: 1px solid {border};"
            )
        else:
            self.trial_name_edit.setText(selected)
            self.trial_name_edit.setEnabled(False)
            
            # Disabled Styling (Read-only)
            bg = Colors.DISABLED_BG if is_dark else Colors.LIGHT_DISABLED
            text = "gray" 
            border = "#333" if is_dark else Colors.LIGHT_BORDER
            
            self.trial_name_edit.setStyleSheet(
                f"color: {text}; background-color: {bg}; border: 1px solid {border};"
            )
    
    def run_import(self) -> None:
        """Execute trial import with proper sanitization and conflict resolution"""
        session = self.session_combo.currentText()
        trial_name = self.trial_name_edit.text().strip()

        # 1. Basic Validation
        files_staged = any(data.get("paths") for data in self.cam_buttons.values())
        if not files_staged or not session or not trial_name:
            QMessageBox.warning(self, "Input Error", "Ensure session is selected, trial is named, and files are browsed.")
            return

        # 2. Sanitize Name
        final_name = InputValidator.sanitize_trial_name(trial_name)
        
        # 3. Check for Existing Files (Conflict Resolution)
        conflict_found = False
        for cam, data in self.cam_buttons.items():
            if data.get("paths"):
                cam_dest_dir = self.data_dir / session / "Videos" / cam / "InputMedia" / final_name
                if cam_dest_dir.exists() and any(f.suffix.lower() in Config.VIDEO_EXTENSIONS for f in cam_dest_dir.iterdir() if f.is_file()):
                    conflict_found = True
                    break
        
        overwrite_all = False
        suffix_mode = False
        
        if conflict_found:
            reply = QMessageBox.question(
                self, 
                "Folder Already Exists", 
                f"Videos already exist in the '{final_name}' folder.\n\nDo you want to overwrite them?\n\n• Yes: Delete old videos and replace.\n• No: Keep old videos and append a suffix (_1, _2) to the new ones.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Yes:
                overwrite_all = True
            else:
                suffix_mode = True

        # 4. Perform Import
        try:
            files_copied = 0
            for cam, data in self.cam_buttons.items():
                if data.get("paths"):
                    dest_dir = self.data_dir / session / "Videos" / cam / "InputMedia" / final_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Clear existing videos if overwriting so we don't mix .mov and .mp4
                    if overwrite_all:
                        for f in dest_dir.iterdir():
                            if f.is_file() and f.suffix.lower() in Config.VIDEO_EXTENSIONS:
                                f.unlink()
                                
                    for idx, file_path in enumerate(data["paths"]):
                        source = Path(file_path)
                        extension = source.suffix
                        
                        base_filename = f"{final_name}_{idx}" if len(data["paths"]) > 1 else final_name
                        
                        if suffix_mode:
                            counter = 1
                            dest_file = dest_dir / f"{base_filename}_{counter}{extension}"
                            while dest_file.exists():
                                counter += 1
                                dest_file = dest_dir / f"{base_filename}_{counter}{extension}"
                        else:
                            dest_file = dest_dir / f"{base_filename}{extension}"
                            
                        shutil.copy(str(source), str(dest_file))
                        files_copied += 1
            
            QMessageBox.information(self, "Success", f"Imported {files_copied} files into '{final_name}'")
            self.refresh_tree(session)
            self._update_status(f"Imported trial: {final_name}", success=True)

        except Exception as e:
            logger.error(f"Import failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Import failed: {str(e)}")
            self.refresh_tree(session)
    
    # -------------------------------------------------------------------------
    # PIPELINE EXECUTION
    # -------------------------------------------------------------------------
    
    def run_intrinsics(self):
        """Triggers the calibration and manages button state"""
        session = self.session_combo.currentText()
        if not session:
            QMessageBox.warning(self, "Warning", "Please select a session first.")
            return

        from PyQt5.QtWidgets import QInputDialog
        cams = list(self.cam_buttons.keys())
        choices = ["All Cameras"] + cams
        
        cam_choice, ok = QInputDialog.getItem(
            self, "Calibrate Intrinsics", "Select Camera to Calibrate:", choices, 0, False
        )
        
        if ok:
            target = None if cam_choice == "All Cameras" else cam_choice
            self.intrinsics_btn = self.sender()
            self.intrinsics_btn.setEnabled(False)
            
            # Start worker with the specific target
            self.worker = IntrinsicsWorker(session, target_cam=target)
            self.worker.finished.connect(self._on_intrinsics_finished)
            self.worker.start()
        
        # 1. Disable the button to prevent multiple simultaneous runs
        # We store a reference to the button to re-enable it later
        self.intrinsics_btn = self.sender() 
        self.intrinsics_btn.setEnabled(False)
        self.intrinsics_btn.setText("CALIBRATING...")
        
        self.log_box.append(f"\n>>> Starting Intrinsics Calibration for: {session}")
        
        # 2. Start background worker
        self.worker = IntrinsicsWorker(session)
        self.worker.finished.connect(self._on_intrinsics_finished)
        self.worker.start()

    def _on_intrinsics_finished(self, success, message):
        """Rectifies the 'greyed-out' issue by re-enabling the button"""
        # 3. Reset Button State so it can be pressed again
        if hasattr(self, 'intrinsics_btn'):
            self.intrinsics_btn.setEnabled(True)
            self.intrinsics_btn.setText("1. RUN INTRINSICS")
        
        if success:
            self.log_box.append(f">>> SUCCESS: {message}")
            QMessageBox.information(self, "Calibration Done", message)
        else:
            self.log_box.append(f">>> ERROR: {message}")
            QMessageBox.critical(self, "Calibration Failed", f"Error: {message}")

    def _execute_intrinsics_thread(self, func, session_name):
        try:
            # This calls the logic in generate_intrinsics.py
            func(session_name)
            self.root.after(0, lambda: messagebox.showinfo("Done", "Intrinsics Calibration Complete!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Calibration failed: {e}"))
        finally:
            self.root.after(0, lambda: self.btn_intrinsics.config(state=tk.NORMAL, text="A. GENERATE INTRINSICS"))
    
    def run_pipeline(self, step="all") -> None: # Add step parameter back
        session = self.session_combo.currentText()
        if not session: return

        dlg = PipelineConfigDialog(self, str(self.data_dir / session))
        if dlg.exec_() != QDialog.Accepted: return
        config = dlg.get_data()
        
        self.current_pipeline_config = {
            "session": session,
            "args": config,
            "step": step # Save the step in config
        }

        # Pass the step to the process starter
        self._start_pipeline_process(session, config, step=step)

    def _start_pipeline_process(self, session, config, step="all"): # Add step parameter
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.finished.connect(self._process_finished)
        
        script_args = [
            str(self.script_path),
            "--session", session,
            "--gpu_index", config["gpu"],
            "--resolution", config["res"],
            "--step", step,
            "--pose_estimator", config.get("pose_estimator", "openpose").lower(), 
            "--trials"
        ] + config["trials"]
        
        self.log_box.clear()
        self.log_box.append(f">>> Executing {config.get('pose_estimator').upper()}: {' '.join(script_args)}")
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Pulsing busy mode
        self.progress_label.setText("Processing with stable backend...")
        
        self.process.start(sys.executable, script_args)

    def _process_finished(self, exit_code, exit_status):
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Idle")

        if exit_code == 5:
            self.log_box.append("\n>>> INTERRUPT: Multiple subjects detected.")
            self._handle_subject_selection()
            return

        if exit_code == 0:
            self.log_box.append("\n>>> SUCCESS: Pipeline Finished.")
            QMessageBox.information(self, "Success", "Processing Complete!")
        else:
            self.log_box.append(f"\n>>> ERROR: Failed with code {exit_code}")

    def _handle_subject_selection(self):
        """Reads tracking data, opens GUI, restarts pipeline."""
        session = self.current_pipeline_config['session']
        session_dir = self.data_dir / session
        
        track_file = session_dir / "temp_tracking_data.json"
        if not track_file.exists():
            QMessageBox.critical(self, "Error", "Pipeline requested review but tracking data is missing.")
            return

        with open(track_file) as f: meta = json.load(f)
        video_path = meta.get('video_path', '')
        
        if not os.path.exists(video_path):
             QMessageBox.warning(self, "Video Missing", f"Could not find video for review:\n{video_path}")
             return

        selector = SubjectSelectorDialog(self, video_path, str(track_file))
        if selector.exec_() == QDialog.Accepted and selector.selection_made:
            exclusion_file = session_dir / "exclusion_list.json"
            with open(exclusion_file, 'w') as f:
                json.dump({"exclude_ids": selector.excluded_ids}, f)
            
            self.log_box.append(f"\n>>> Resuming pipeline. Excluding IDs: {selector.excluded_ids}")
            
            # Resume WITH the correct step
            current_step = self.current_pipeline_config.get("step", "all")
            self._start_pipeline_process(
                session, 
                self.current_pipeline_config['args'], 
                resume_file=str(exclusion_file),
                step=current_step
            )
        else:
            self.log_box.append("\n>>> Pipeline Cancelled by User during selection.")
    
    # -------------------------------------------------------------------------
    # SESSION CREATION
    # -------------------------------------------------------------------------
    
    def open_new_session_dialog(self) -> None:
        dlg = CreateSessionDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        
        data = dlg.get_data()
        session_path = self.data_dir / data['name']
        
        if session_path.exists():
            QMessageBox.warning(self, "Session Exists", f"Session '{data['name']}' already exists.")
            return
        
        try:
            for folder in ["CalibrationImages", "MarkerData", "OpenSimData", "Videos", "VisualizerJsons"]:
                (session_path / folder).mkdir(parents=True, exist_ok=True)
            for i in range(data['cams']):
                (session_path / "Videos" / f"Cam{i}" / "InputMedia").mkdir(parents=True, exist_ok=True)
            
            # --- MAPPING LOGIC ---
            # Map Placement
            placement_map = {"Vertical (Wall)": "Perpendicular", "Horizontal (Ground)": "Lying"}
            mapped_placement = placement_map.get(data['placement'], "Perpendicular")
            
            # --- UPDATED MAPPING LOGIC ---
            num_cams = int(data['cams'])
            if data['cam_type'] == "Mixed":
                mixed_dlg = MixedCameraDialog(self, num_cams)
                if mixed_dlg.exec_() == QDialog.Accepted:
                    cam_model_map = mixed_dlg.get_config()
                else:
                    return # Cancel session creation if mixed config is cancelled
            else:
                # Standard single-type logic
                base_model = "Generic_Webcam"
                if data['cam_type'] == "iPhone":
                    base_model = "iPhone_Auto_Detect"
                elif data['cam_type'] == "Android":
                    base_model = "Android_Generic"
                
                cam_model_map = {f"Cam{i}": base_model for i in range(num_cams)}

            orientation_val = "landscape" if "Landscape" in data['orientation'] else "portrait"

            metadata = {
                "augmentermodel": "v0.3",
                "calibrationSettings": {"overwriteDeployedIntrinsics": False, "saveSessionIntrinsics": False},
                "checkerBoard": {
                    "black2BlackCornersHeight_n": int(data['rows']),
                    "black2BlackCornersWidth_n": int(data['cols']),
                    "placement": mapped_placement,
                    "squareSideLength_mm": float(data['size'])
                },
                "videoOrientation": orientation_val,
                "filterfrequency": "default",
                "gender_mf": data['gender'],
                "height_m": float(data['height']),
                "iphoneModel": cam_model_map,
                "markerAugmentationSettings": {"markerAugmenterModel": "LSTM"},
                "mass_kg": float(data['weight']),
                "openSimModel": "LaiUhlrich2022",
                "posemodel": "openpose",
                "scalingsetup": "upright_standing_pose",
                "subjectID": data['subject']
            }
            
            with open(session_path / "sessionMetadata.yaml", 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Created new session: {data['name']}")
            
            # 1. Rescan the hard drive so the dropdown knows the new folder exists
            self.refresh_sessions()
            
            # 2. Force the dropdown to select it (This automatically triggers the tree & camera slots to refresh!)
            self.session_combo.setCurrentText(data['name'])
            
            self._update_status(f"Created session: {data['name']}", success=True)
            
        except Exception as e:
            logger.error(f"Error creating session: {e}", exc_info=True)
            QMessageBox.critical(self, "Creation Failed", f"Error: {str(e)}")
    
    # -------------------------------------------------------------------------
    # RECENT SESSIONS
    # -------------------------------------------------------------------------
    
    def _update_recent_menu(self) -> None:
        """Update recent sessions menu"""
        self.recent_menu.clear()
        
        recent = self.settings.value('recent_sessions', [])
        if not isinstance(recent, list):
            recent = []
        
        for session_name in recent[:Config.MAX_RECENT_SESSIONS]:
            if (self.data_dir / session_name).exists():
                action = self.recent_menu.addAction(session_name)
                action.triggered.connect(
                    lambda checked, s=session_name: self._load_recent_session(s)
                )
        
        if not recent:
            action = self.recent_menu.addAction('(empty)')
            action.setEnabled(False)
    
    def _load_recent_session(self, session_name: str) -> None:
        """Load a recent session"""
        self.session_combo.setCurrentText(session_name)
    
    def _add_to_recent(self, session_name: str) -> None:
        """Add session to recent list"""
        recent = self.settings.value('recent_sessions', [])
        if not isinstance(recent, list):
            recent = []
        
        # Remove if already exists
        if session_name in recent:
            recent.remove(session_name)
        
        # Add to front
        recent.insert(0, session_name)
        
        # Keep only last N
        recent = recent[:Config.MAX_RECENT_SESSIONS]
        
        self.settings.setValue('recent_sessions', recent)
        self._update_recent_menu()
    
    # -------------------------------------------------------------------------
    # UI HELPERS
    # -------------------------------------------------------------------------
    
    def _update_status(self, message: str, success: bool = False, 
                       warning: bool = False, error: bool = False) -> None:
        """Update status bar with message"""
        self.status_bar.showMessage(message, Config.STATUS_MESSAGE_TIMEOUT)
        
        if success:
            logger.info(f"SUCCESS: {message}")
        elif warning:
            logger.warning(f"WARNING: {message}")
        elif error:
            logger.error(f"ERROR: {message}")
        else:
            logger.info(message)
    
    def _show_about(self) -> None:
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About OpenCap Portable Pro",
            "<h3>OpenCap Portable Pro</h3>"
            "<p>Version 2.0 (Optimized)</p>"
            "<p>A comprehensive motion capture processing launcher.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Session management</li>"
            "<li>Multi-camera video import</li>"
            "<li>Pipeline execution with GPU support</li>"
            "<li>3D skeletal visualization</li>"
            "<li>Synchronized video playback</li>"
            "</ul>"
            "<p>© 2024 OpenCap Team</p>"
        )
    
    # -------------------------------------------------------------------------
    # WINDOW EVENTS
    # -------------------------------------------------------------------------
    
    def _show_tree_context_menu(self, position):
        """Displays a right-click menu for tree items."""
        item = self.tree.itemAt(position)
        if not item: return
            
        data = item.data(0, Qt.UserRole)
        if not data or data.get("type") != "video": return 
            
        menu = QMenu()
        
        # --- NEW RENAME ACTION ---
        rename_action = QAction("Rename File", self)
        rename_action.triggered.connect(lambda: self._rename_tree_file(data["path"]))
        menu.addAction(rename_action)
        
        # --- EXISTING DELETE ACTION ---
        delete_action = QAction("Delete File", self)
        delete_action.triggered.connect(lambda: self._delete_tree_file(data["path"]))
        menu.addAction(delete_action)
        
        menu.exec_(self.tree.viewport().mapToGlobal(position))

    def _delete_tree_file(self, file_path):
        """Deletes the file from the hard drive and updates the GUI."""
        file_name = os.path.basename(file_path)
        
        # 1. Ask for confirmation before destroying data
        reply = QMessageBox.question(
            self, 
            "Confirm Delete", 
            f"Are you sure you want to permanently delete this file?\n\n{file_name}",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # 2. Delete from the hard drive
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.log_box.append(f">>> DELETED: {file_path}")
                    
                    # 3. Clear the video player if they were currently watching the deleted video
                    if self.pending_click_data and self.pending_click_data.get("path") == file_path:
                        self.video_player.stop()
                        self.video_player.raw_label.clear()
                        self.video_player.overlay_label.clear()
                        self.skeleton_viewer.clear_skeleton()
                    
                    # 4. Refresh the tree so the file disappears from the UI
                    session = self.session_combo.currentText()
                    self.refresh_tree(session)
                    self._update_status(f"Deleted {file_name}", success=True)
                    
            except Exception as e:
                logger.error(f"Failed to delete file: {e}")
                QMessageBox.critical(self, "Error", f"Could not delete file:\n{e}")

    def _rename_tree_file(self, file_path):
        """Renames a file directly from the right-click menu."""
        # We import QInputDialog here dynamically since it wasn't in your top imports
        from PyQt5.QtWidgets import QInputDialog, QLineEdit
        
        old_path = Path(file_path)
        old_name = old_path.name
        
        new_name, ok = QInputDialog.getText(
            self, 
            "Rename File", 
            "Enter new file name (including extension):", 
            QLineEdit.Normal, 
            old_name
        )
        
        if ok and new_name and new_name != old_name:
            new_path = old_path.parent / new_name
            
            if new_path.exists():
                QMessageBox.warning(self, "Error", "A file with that name already exists in this folder.")
                return
                
            try:
                # 1. Clear the video player just in case they are currently watching the file they are renaming
                if self.pending_click_data and self.pending_click_data.get("path") == file_path:
                    self.video_player.stop()
                    self.video_player.raw_label.clear()
                    
                # 2. Rename it on the hard drive
                old_path.rename(new_path)
                self.log_box.append(f">>> RENAMED: {old_name} -> {new_name}")
                
                # 3. Refresh the UI
                session = self.session_combo.currentText()
                self.refresh_tree(session)
                self._update_status(f"Renamed to {new_name}", success=True)
                
            except Exception as e:
                logger.error(f"Failed to rename file: {e}")
                QMessageBox.critical(self, "Error", f"Could not rename file:\n{e}")

    def closeEvent(self, event) -> None:
        """Handle window close event"""
        # Save window state
        self.settings.setValue('window/geometry', self.saveGeometry())
        self.settings.setValue('splitter/state', self.splitter.saveState())
        
        # Check for running process
        if self.process and self.process.state() == QProcess.Running:
            reply = QMessageBox.question(
                self,
                'Process Running',
                'A pipeline process is still running.\n\n'
                'Quit anyway? The process will be terminated.',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
            else:
                logger.info("Terminating running process on close")
                self.process.terminate()
                self.process.waitForFinished(3000)
        
        # Clean up video player
        self.video_player.stop()
        
        logger.info("Application closing")
        super().closeEvent(event)

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

def main():
    """Application entry point"""
    # Enable High DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("OpenCap Portable Pro")
    app.setOrganizationName("OpenCap")
    
    # Create and show main window
    window = OpenCapPro()
    window.show()
    
    logger.info("Application started")
    
    # Run event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
