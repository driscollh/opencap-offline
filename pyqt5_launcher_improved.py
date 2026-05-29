# ==============================================================================
# File: pyqt5_launcher_improved.py
# Author: Harry G. Driscoll
# Date: Jan 2026
#
# OpenCap Offline - Motion Capture Processing Launcher
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
import requests
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

# --- FORCE PYQT5 ---
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
    QSizePolicy, QProgressBar, QShortcut, QMenuBar, QMenu, QAction, QStatusBar,
    QStackedWidget
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal as Signal, QThread, QSize, QProcess, QSettings,
    QRunnable, QThreadPool, pyqtSlot, QRect
)
from PyQt5.QtGui import (
    QImage, QPixmap, QIcon, QColor, QFont, QPalette, QKeySequence, QPainter, QBrush,
    QDesktopServices
)

from PyQt5.QtCore import (
    QDateTime, QUrl
)

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

class Lang:
    """Centralized dictionary for Korean/English localization."""
    CURRENT = 'EN'
    
    STRINGS = {
        'EN': {
            # Main UI
            'new_session': '+ New Session',
            'refresh_list': 'Refresh List',
            'edit_meta': 'Edit Metadata',
            'view_menu': '&View',
            'lang_toggle': '한국어로 변경 (Switch to Korean)',
            'theme_toggle': '&Toggle Dark/Light Mode',
            'select_gpu': 'Select GPU:',
            'pose_estimator': 'Pose Estimator:',
            'resolution': 'Resolution:',
            'run_pipeline': 'RUN PIPELINE',
            'cancel': 'Cancel',
            'cancel_process': 'Cancel Processing',
            'idle': 'Idle',
            'select_trials': 'Select Trials to Process:',
            'play': 'Play',
            'pause': 'Pause',
            'confirm': 'Confirm Selection',
            'step_1': '1. Process Intrinsics',
            'step_2_full': '2. Run Full Pipeline',
            'step_2_ext': '2. Calibrate Extrinsics',
            'step_3': '3. 2D Pose Estimation',
            'step_4': '4. 3D Pose and OpenSim',
            'acknowledge_close': 'Acknowledge & Close',
            'confirm_assign': 'Confirm Assignments',
            'import_trial_down': '▼ IMPORT TRIAL',
            'import_trial_right': '▶ IMPORT TRIAL',
            'execute_import': 'EXECUTE IMPORT',
            'pipeline_exec_down': '▼ PIPELINE EXECUTION',
            'pipeline_exec_right': '▶ PIPELINE EXECUTION',
            'research_mode': 'Research Mode (Granular Controls)',
            'show_calib': 'Show Calibration Setup',
            'browse': 'Browse',
            'clear': 'Clear',
            
            # Dialogs & Forms
            'create_new_session': 'Create New Session',
            'session_name': 'Session Name:*',
            'subject_id': 'Subject ID:*',
            'gender': 'Gender:*',
            'height': 'Height (m):*',
            'weight': 'Weight (kg):*',
            'tag': 'Tag:',
            'cam_calib_settings': 'Camera & Calibration Settings',
            'cam_type': 'Camera Type:',
            'cam_orientation': 'Camera Orientation:',
            'cam_nums': 'Camera Nums:',
            'board_placement': 'Board Placement:',
            'rows': 'Rows:',
            'cols': 'Cols:',
            'square_size': 'Square Size (mm):',
            'create': 'Create',
            'reset_defaults': 'Reset to Defaults',
            'save_metadata': 'Save Metadata',
            
            # Popups & Context Menus
            'warning': 'Warning',
            'error': 'Error',
            'success': 'Success',
            'confirm_cancel': 'Confirm Cancel',
            'cancel_prompt': 'Are you sure you want to abort the current process? Partial data may be corrupted.',
            'perf_warning': 'Performance Warning',
            'perf_prompt': 'Processing with OpenPose at 1x736 is highly memory-intensive.\n\nIf the video contains vertical background clutter (tables, chair legs), the algorithm may experience combinatorial explosion, massively increasing processing time.\n\nDo you wish to proceed?',
            'proc_complete': 'Processing Complete!',
            'confirm_delete': 'Confirm Delete',
            'delete_prompt': 'Are you sure you want to permanently delete this file?',
            'rename_file': 'Rename File',
            'delete_file': 'Delete File',

            # Dialog Titles
            'adv_meta_editor': 'Advanced Metadata Editor',
            'review_subjects': 'Review Detected Subjects',
            'calib_review': 'Extrinsics Calibration Review',
            'pipeline_config': 'Pipeline Configuration',
            'per_cam_config': 'Per-Camera Configuration',
            
            # Tooltips & Placeholders
            'nested_json_tip': 'Nested data (JSON format)',
            'reset_meta_tip': 'Revert all fields to the original session creation values.',
            'dlc_unlocked': 'Blackwell DLC detected: RTMPose Unlocked.',
            'dlc_missing': 'DLC missing: RTMPose is disabled for this installation.',
            'gpu_tip': 'Choose GPU for processing (requires CUDA)',
            'pose_tip': 'Select the AI model for pose estimation',
            'res_tip': 'Select processing resolution preset',
            'cancel_pipe_tip': 'Cancel pipeline execution',
            'run_pipe_tip': 'Start processing selected trials',
            'process_trial_tip': 'Process trial:',
            'eg_session': 'e.g., Session_2024_01',
            'session_id_tip': 'Unique identifier for this session',
            'eg_subject': 'e.g., Subject_001',
            'subject_id_tip': 'Subject identifier for this capture session',
            'height_tip': 'Subject height in meters',
            'weight_tip': 'Subject weight in kilograms',
            'orient_tip': 'Portrait: Phone held vertically.\nLandscape: Phone held horizontally.',
            'cam_type_tip': 'Select the type of camera device being used',
            'placement_tip': 'Vertical: Board matches gravity (standard).\nHorizontal: Board is flat on ground (good for 360° setups).',
            'log_placeholder': 'Process output will appear here...',
            'trial_name_placeholder': 'Trial Name',
            'trial_name_tip': 'Custom name for dynamic trials.\nExamples: walking_1, running_fast, jump_test',
            
            # UI Labels
            'deselect_exclude': '<b>Deselect to Exclude:</b>',
            'raw_video': 'Raw Video',
            'overlay': 'Overlay',
            'no_video': 'No Video',
            'buffering': 'Buffering videos to RAM...',
            'error_loading_raw': 'Error loading raw video.',
            'no_calib_images': 'No calibration images found. The detection may have failed on the first frame.',
            'no_trials_avail': 'No trials available for processing',
            'req_fields': '* Required fields',
            'trial_type_label': 'TRIAL TYPE:',
            'ready': 'Ready',
            'no_files_sel': 'No files selected',
            'calibrating_btn': 'CALIBRATING...',
            'proc_stable_backend': 'Processing with stable backend...',
            'aborted_user': 'Process aborted by user.',
            
            # Menus
            'file_menu': '&File',
            'new_session_menu': '&New Session...',
            'refresh_menu': '&Refresh Sessions',
            'recent_menu': 'Recent Sessions',
            'quit_menu': '&Quit',
            'help_menu': '&Help',
            'about_menu': '&About',
            
            # Prompts & Message Boxes
            'confirm_reset': 'Confirm Reset',
            'confirm_reset_prompt': 'Are you sure you want to discard all changes and return to the original session values?',
            'format_error': 'Formatting Error',
            'vid_not_found': 'Error: Video not found\n',
            'data_missing': 'Data Missing',
            'val_error': 'Validation Error',
            'custom_tag_title': 'Custom Subject Tag',
            'custom_tag_prompt': 'Enter specific subject condition/tag:',
            'input_error': 'Input Error',
            'input_error_prompt': 'Ensure session is selected, trial is named, and files are browsed.',
            'folder_exists': 'Folder Already Exists',
            'calib_intrinsics': 'Calibrate Intrinsics',
            'sel_cam_calib': 'Select Camera to Calibrate:',
            'calib_done': 'Calibration Done',
            'calib_failed': 'Calibration Failed',
            'no_pose_data': 'No Pose Data',
            'no_pose_data_prompt': 'No processed pose data found in this session.\n\nPlease run \'3. Run Pose\' first.',
            'track_data_missing': 'Pipeline requested review but tracking data is missing.',
            'vid_missing': 'Video Missing',
            'session_exists': 'Session Exists',
            'creation_failed': 'Creation Failed',
            'about_title': 'About OpenCap Offline',
            'update_avail': 'Update Available',
            'up_to_date': 'Up to Date',
            'api_error': 'Could not check for updates. The GitHub API might be rate-limiting.',
            'conn_error': 'Could not connect to GitHub. Please check your internet connection.',
            'rename_title': 'Rename File',
            'rename_prompt': 'Enter new file name (including extension):',
            'file_exists': 'A file with that name already exists in this folder.',
            'process_running': 'Process Running',
            'process_running_prompt': 'A pipeline process is still running.\n\nQuit anyway? The process will be terminated.',
        },
        'KO': {
            # Main UI
            'new_session': '+ 새 세션',
            'refresh_list': '목록 새로고침',
            'edit_meta': '메타데이터 편집',
            'view_menu': '&보기',
            'lang_toggle': 'Switch to English',
            'theme_toggle': '&다크/라이트 모드 전환',
            'select_gpu': 'GPU 선택:',
            'pose_estimator': '포즈 추정기:',
            'resolution': '해상도:',
            'run_pipeline': '파이프라인 실행',
            'cancel': '취소',
            'cancel_process': '처리 취소',
            'idle': '대기 중',
            'select_trials': '처리할 테스트 선택:',
            'play': '재생',
            'pause': '일시정지',
            'confirm': '선택 확인',
            'step_1': '1. 내부 파라미터 처리',
            'step_2_full': '2. 전체 파이프라인 실행',
            'step_2_ext': '2. 외부 파라미터 캘리브레이션',
            'step_3': '3. 2D 포즈 추정',
            'step_4': '4. 3D 포즈 및 OpenSim',
            'acknowledge_close': '확인 및 닫기',
            'confirm_assign': '할당 확인',
            'import_trial_down': '▼ 테스트 가져오기',
            'import_trial_right': '▶ 테스트 가져오기',
            'execute_import': '가져오기 실행',
            'pipeline_exec_down': '▼ 파이프라인 실행',
            'pipeline_exec_right': '▶ 파이프라인 실행',
            'research_mode': '연구 모드 (세부 제어)',
            'show_calib': '캘리브레이션 설정 보기',
            'browse': '찾아보기',
            'clear': '지우기',
            
            # Dialogs & Forms
            'create_new_session': '새 세션 생성',
            'session_name': '세션 이름:*',
            'subject_id': '피험자 ID:*',
            'gender': '성별:*',
            'height': '키 (m):*',
            'weight': '몸무게 (kg):*',
            'tag': '태그:',
            'cam_calib_settings': '카메라 및 캘리브레이션 설정',
            'cam_type': '카메라 종류:',
            'cam_orientation': '카메라 방향:',
            'cam_nums': '카메라 수:',
            'board_placement': '보드 배치:',
            'rows': '행(Rows):',
            'cols': '열(Cols):',
            'square_size': '사각형 크기 (mm):',
            'create': '생성',
            'reset_defaults': '기본값으로 재설정',
            'save_metadata': '메타데이터 저장',
            
            # Popups & Context Menus
            'warning': '경고',
            'error': '오류',
            'success': '성공',
            'confirm_cancel': '취소 확인',
            'cancel_prompt': '현재 프로세스를 중단하시겠습니까? 일부 데이터가 손상될 수 있습니다.',
            'perf_warning': '성능 경고',
            'perf_prompt': '1x736 해상도에서 OpenPose로 처리하는 것은 메모리 소모가 매우 큽니다.\n\n비디오에 수직 배경물(테이블, 의자 다리 등)이 많을 경우 처리 시간이 기하급수적으로 늘어날 수 있습니다.\n\n계속하시겠습니까?',
            'proc_complete': '처리 완료!',
            'confirm_delete': '삭제 확인',
            'delete_prompt': '이 파일을 영구적으로 삭제하시겠습니까?',
            'rename_file': '파일 이름 변경',
            'delete_file': '파일 삭제',

            # Dialog Titles
            'adv_meta_editor': '고급 메타데이터 편집기',
            'review_subjects': '감지된 피험자 검토',
            'calib_review': '외부 파라미터 캘리브레이션 검토',
            'pipeline_config': '파이프라인 구성',
            'per_cam_config': '카메라별 구성',
            
            # Tooltips & Placeholders
            'nested_json_tip': '중첩된 데이터 (JSON 형식)',
            'reset_meta_tip': '모든 필드를 초기 세션 생성 값으로 되돌립니다.',
            'dlc_unlocked': 'Blackwell DLC 감지됨: RTMPose 잠금 해제.',
            'dlc_missing': 'DLC 누락: 이 설치에서는 RTMPose가 비활성화되었습니다.',
            'gpu_tip': '처리할 GPU 선택 (CUDA 필요)',
            'pose_tip': '포즈 추정을 위한 AI 모델 선택',
            'res_tip': '처리 해상도 사전 설정 선택',
            'cancel_pipe_tip': '파이프라인 실행 취소',
            'run_pipe_tip': '선택한 테스트 처리 시작',
            'process_trial_tip': '테스트 처리:',
            'eg_session': '예: Session_2024_01',
            'session_id_tip': '이 세션의 고유 식별자',
            'eg_subject': '예: Subject_001',
            'subject_id_tip': '이 캡처 세션의 피험자 식별자',
            'height_tip': '피험자 키 (미터)',
            'weight_tip': '피험자 몸무게 (킬로그램)',
            'orient_tip': 'Portrait: 휴대폰을 세로로 듭니다.\nLandscape: 휴대폰을 가로로 듭니다.',
            'cam_type_tip': '사용 중인 카메라 장치 유형 선택',
            'placement_tip': 'Vertical: 보드가 중력과 일치합니다(표준).\nHorizontal: 보드가 바닥에 평평합니다(360° 설정에 적합).',
            'log_placeholder': '처리 출력이 여기에 표시됩니다...',
            'trial_name_placeholder': '테스트 이름',
            'trial_name_tip': '동적 테스트를 위한 사용자 지정 이름.\n예: walking_1, running_fast, jump_test',
            
            # UI Labels
            'deselect_exclude': '<b>제외할 항목 선택 취소:</b>',
            'raw_video': '원본 비디오',
            'overlay': '오버레이',
            'no_video': '비디오 없음',
            'buffering': 'RAM에 비디오 버퍼링 중...',
            'error_loading_raw': '원본 비디오 로드 오류.',
            'no_calib_images': '캘리브레이션 이미지를 찾을 수 없습니다. 첫 프레임에서 감지에 실패했을 수 있습니다.',
            'no_trials_avail': '처리할 수 있는 테스트가 없습니다.',
            'req_fields': '* 필수 입력란',
            'trial_type_label': '테스트 유형:',
            'ready': '준비 완료',
            'no_files_sel': '선택된 파일 없음',
            'calibrating_btn': '캘리브레이션 중...',
            'proc_stable_backend': '안정적인 백엔드로 처리 중...',
            'aborted_user': '사용자가 프로세스를 중단했습니다.',
            
            # Menus
            'file_menu': '&파일',
            'new_session_menu': '&새 세션...',
            'refresh_menu': '&세션 새로고침',
            'recent_menu': '최근 세션',
            'quit_menu': '&종료',
            'help_menu': '&도움말',
            'about_menu': '&정보',
            
            # Prompts & Message Boxes
            'confirm_reset': '초기화 확인',
            'confirm_reset_prompt': '모든 변경 사항을 취소하고 원래 세션 값으로 돌아가시겠습니까?',
            'format_error': '형식 오류',
            'vid_not_found': '오류: 비디오를 찾을 수 없음\n',
            'data_missing': '데이터 누락',
            'val_error': '유효성 검사 오류',
            'custom_tag_title': '사용자 지정 피험자 태그',
            'custom_tag_prompt': '특정 피험자 조건/태그 입력:',
            'input_error': '입력 오류',
            'input_error_prompt': '세션이 선택되었는지, 테스트 이름이 지정되었는지, 파일을 찾아보았는지 확인하세요.',
            'folder_exists': '폴더가 이미 존재함',
            'calib_intrinsics': '내부 파라미터 캘리브레이션',
            'sel_cam_calib': '캘리브레이션할 카메라 선택:',
            'calib_done': '캘리브레이션 완료',
            'calib_failed': '캘리브레이션 실패',
            'no_pose_data': '포즈 데이터 없음',
            'no_pose_data_prompt': '이 세션에서 처리된 포즈 데이터를 찾을 수 없습니다.\n\n먼저 \'3. 2D 포즈 추정\'을 실행하세요.',
            'track_data_missing': '파이프라인이 검토를 요청했지만 추적 데이터가 누락되었습니다.',
            'vid_missing': '비디오 누락',
            'session_exists': '세션 존재',
            'creation_failed': '생성 실패',
            'about_title': 'OpenCap Offline 정보',
            'update_avail': '업데이트 가능',
            'up_to_date': '최신 버전',
            'api_error': '업데이트를 확인할 수 없습니다. GitHub API 요청 제한일 수 있습니다.',
            'conn_error': 'GitHub에 연결할 수 없습니다. 인터넷 연결을 확인하세요.',
            'rename_title': '파일 이름 변경',
            'rename_prompt': '새 파일 이름 입력 (확장자 포함):',
            'file_exists': '해당 이름의 파일이 이 폴더에 이미 존재합니다.',
            'process_running': '프로세스 실행 중',
            'process_running_prompt': '파이프라인 프로세스가 아직 실행 중입니다.\n\n그래도 종료하시겠습니까? 프로세스가 종료됩니다.',
        }
    }

    @classmethod
    def get(cls, key):
        return cls.STRINGS[cls.CURRENT].get(key, key)

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

class UpdateCheckerThread(QThread):
    """Checks GitHub for updates in the background so the GUI doesn't freeze."""
    finished_check = Signal(str, str) 

    def run(self):
        api_url = "https://api.github.com/repos/driscollh/opencap-offline/releases/latest"
        try:
            # Quick 3-second timeout
            response = requests.get(api_url, timeout=3)
            if response.status_code == 200:
                latest = response.json().get("tag_name", "")
                self.finished_check.emit(latest, "")
            else:
                self.finished_check.emit("", f"API Error {response.status_code}")
        except Exception as e:
            self.finished_check.emit("", "Offline or Timeout")

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
        """Execute calibration logic in background thread."""
        try:
            import generate_intrinsics
            if self.target_cam:
                # Targeted calibration for single camera
                generate_intrinsics.calibrate_camera(self.session_name, self.target_cam)
                msg = f"Intrinsics for {self.target_cam} complete."
            else:
                # Batch calibration for all cameras
                generate_intrinsics.calibrate_session(self.session_name)
                msg = "Full session intrinsics calibration complete."
            
            self.finished.emit(True, msg)
        except Exception as e:
            self.finished.emit(False, str(e))
            
        except Exception as e:
            logger.error(f"IntrinsicsWorker error: {str(e)}", exc_info=True)
            self.finished.emit(False, f"Calibration Error: {str(e)}")

class MetadataEditorDialog(QDialog):
    """Popup to edit ALL metadata with a safety reset to the session's initial state."""
    def __init__(self, parent, session_path):
        super().__init__(parent)
        self.setWindowTitle(Lang.get('adv_meta_editor'))
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
        btn_reset = QPushButton(Lang.get('reset_defaults'))
        btn_reset.setToolTip(Lang.get('reset_meta_tip'))
        btn_reset.clicked.connect(self.reset_to_original)
        
        btn_save = QPushButton(Lang.get('save_metadata'))
        btn_save.setObjectName("AccentButton")
        btn_save.clicked.connect(self.save_metadata)
        
        btn_cancel = QPushButton(Lang.get('cancel'))
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
                edit.setToolTip(Lang.get('nested_json_tip'))
            else:
                edit = QLineEdit(str(value))
            
            self.form_layout.addRow(f"<b>{key}:</b>", edit)
            self.inputs[key] = edit

    def reset_to_original(self):
        """Restores the UI fields to the snapshot taken when the dialog opened."""
        reply = QMessageBox.question(
            self, Lang.get('confirm_reset'), 
            Lang.get('confirm_reset_prompt'),
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
                QMessageBox.critical(self, Lang.get('format_error'), f"Invalid input for {key}: {e}")
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
        self.setWindowTitle(Lang.get('review_subjects'))
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
        layout.addWidget(QLabel(Lang.get('select_gpu')))
        self.gpu_combo = QComboBox()
        self.gpu_map = self._get_gpu_info()
        self.gpu_combo.addItems(list(self.gpu_map.keys()))
        layout.addWidget(self.gpu_combo)

        # --- DYNAMIC DLC CHECK FOR POSE ESTIMATOR ---
        layout.addWidget(QLabel(Lang.get('pose_estimator')))
        self.pose_combo = QComboBox()
        
        # Identify the DLC path relative to the launcher
        base_path = os.path.dirname(os.path.abspath(__file__))
        dlc_path = os.path.join(base_path, "Blackwell_RTMPose")
        
        # Build the choices list
        pose_choices = ["OpenPose"]
        if os.path.exists(dlc_path):
            pose_choices.append("RTMPose")
            self.pose_combo.setToolTip(Lang.get('dlc_unlocked'))
        else:
            self.pose_combo.setToolTip(Lang.get('dlc_missing'))

        self.pose_combo.addItems(pose_choices)
        layout.addWidget(self.pose_combo)
        # --------------------------------------------
        
        # Resolution Selection
        layout.addWidget(QLabel(Lang.get('resolution')))
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
        
        self.btn_play = QPushButton(Lang.get('play'))
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.btn_play)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.sliderMoved.connect(self.seek_frame)
        ctrl_layout.addWidget(self.slider)
        
        layout.addLayout(ctrl_layout)

        # 4. Checkboxes for Exclusion (Scrollable Grid Layout)
        chk_frame = QFrame()
        chk_main_layout = QVBoxLayout(chk_frame)
        chk_main_layout.addWidget(QLabel(Lang.get('deselect_exclude')))
        
        # Constrain layout boundaries using a scroll viewport
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(120) 
        
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        
        self.checkboxes = {}
        columns_per_row = 6 
        
        for idx, uid in enumerate(sorted(list(unique_ids))):
            chk = QCheckBox(f"Subject {uid}")
            chk.setChecked(True) 
            c = TRACK_COLORS[uid % len(TRACK_COLORS)]
            chk.setStyleSheet(f"color: rgb({c[0]},{c[1]},{c[2]}); font-weight: bold;")
            
            # Map elements into grid rows and columns
            row = idx // columns_per_row
            col = idx % columns_per_row
            scroll_layout.addWidget(chk, row, col)
            self.checkboxes[uid] = chk
            
        scroll_area.setWidget(scroll_widget)
        chk_main_layout.addWidget(scroll_area)
            
        btn_confirm = QPushButton(Lang.get('confirm'))
        btn_confirm.setObjectName("AccentButton")
        btn_confirm.clicked.connect(self.confirm_selection)
        chk_main_layout.addWidget(btn_confirm)
        
        layout.addWidget(chk_frame)

        # Internal State
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.current_frame = 0
        self.playing = False
        self.cap = None

    def _load_video(self):
        if not os.path.exists(self.video_path):
            self.video_label.setText(f"{Lang.get('vid_not_found')}{self.video_path}")
            return
        self.cap = cv2.VideoCapture(self.video_path)
        self.show_frame(0)

    def toggle_play(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.btn_play.setText(Lang.get('play'))
        else:
            self.timer.start(33) # ~30fps
            self.playing = True
            self.btn_play.setText(Lang.get('pause'))

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
        
        # Reference parent window boundaries to prevent layout feedback loops
        target_width = max(400, self.width() - 40)
        target_height = max(300, self.height() - 320) # Reserve vertical space for timeline controls
        
        self.video_label.setPixmap(pix.scaled(
            target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
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

    def closeEvent(self, event):
        if self.cap: self.cap.release()
        super().closeEvent(event)

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

    def _update_res_options(self, estimator: str) -> None:
        """
        Updates resolution options or model complexity based on selected estimator.
        """
        self.res_combo.clear()
        if estimator.lower() == "openpose":
            self.res_combo.addItems(["1x368", "1x736", "1x736_2scales", "736x1 (Landscape)"])
            self.res_combo.setEnabled(True)
        else:
            self.res_combo.addItems(["RTMPose-m (Fast/Balanced)", "RTMPose-l (High Accuracy)"])
            self.res_combo.setEnabled(True)

# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class VideoLoader(QThread):
    finished = Signal(list, str) 
    
    def __init__(self, path, max_resolution=800): 
        super().__init__()
        self.path = str(path) 
        self.max_resolution = max_resolution 
        self.frames = []
        self._is_cancelled = False
        
    def cancel(self):
        self._is_cancelled = True

    def run(self):
        if not self.path or not os.path.exists(self.path):
            self.finished.emit([], "error")
            return

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.finished.emit([], "error")
            return
            
        # --- NEW: DYNAMIC ASPECT RATIO SCALING ---
        # Read the actual native dimensions of the video
        orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if orig_w == 0 or orig_h == 0:
            self.finished.emit([], "error")
            return
            
        # Scale the video down to fit into RAM, but KEEP the native aspect ratio
        scale = self.max_resolution / max(orig_w, orig_h)
        target_size = (int(orig_w * scale), int(orig_h * scale))
        # -----------------------------------------
            
        while not self._is_cancelled: 
            ret, frame = cap.read()
            if not ret:
                break
                
            resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            self.frames.append(qimg.copy())
            
        cap.release()
        
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
        video_layout = QHBoxLayout()
        
        self.raw_label = QLabel(Lang.get('raw_video'))
        self.raw_label.setAlignment(Qt.AlignCenter)
        self.raw_label.setStyleSheet("background-color: black; border: 1px solid #333;")
        self.raw_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.raw_label.setMinimumSize(100, 100) 
        
        self.overlay_label = QLabel(Lang.get('overlay'))
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("background-color: black; border: 1px solid #333;")
        self.overlay_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.overlay_label.setMinimumSize(100, 100)
        
        video_layout.addWidget(self.raw_label)
        video_layout.addWidget(self.overlay_label)
        layout.addLayout(video_layout)
        
        # Controls
        controls = QHBoxLayout()
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(40)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.seek)
        self.slider.setEnabled(False)
        
        controls.addWidget(self.play_button)
        controls.addWidget(self.slider)
        layout.addLayout(controls)
        
        # Status Label (Only created ONCE now)
        self.status_lbl = QLabel(Lang.get('no_video'), self)
        self.status_lbl.setStyleSheet("color: white; background: rgba(0,0,0,0.5); padding: 5px; border-radius: 3px;")
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
        self.status_lbl.setText(Lang.get('buffering'))
        self.status_lbl.show()
        
        if raw_path and os.path.exists(raw_path):
            loader1 = VideoLoader(raw_path, max_resolution=800)
            loader1.finished.connect(self._on_raw_loaded)
            self.loaders.append(loader1)
            loader1.start()
            
        if overlay_path and os.path.exists(overlay_path):
            loader2 = VideoLoader(overlay_path, max_resolution=800)
            loader2.finished.connect(self._on_overlay_loaded)
            self.loaders.append(loader2)
            loader2.start()

    def _on_raw_loaded(self, frames, path):
        if path == "error" or not frames:
            self.status_lbl.setText(Lang.get('error_loading_raw'))
            return
        # THREAD SAFETY: Convert QImage to QPixmap on the Main Thread
        self.raw_cache = [QPixmap.fromImage(img) for img in frames]
        self._check_loading_complete()

    def _on_overlay_loaded(self, frames, path):
        if path != "error" and frames:
            self.overlay_cache = [QPixmap.fromImage(img) for img in frames]
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

    def resizeEvent(self, event):
        """Dynamic scaling logic to maintain 9:16 aspect ratio on resize."""
        super().resizeEvent(event)
        self._update_display()

    def _update_display(self):
        """Re-renders the current frame to fit the new widget size."""
        if self.current_frame < len(self.raw_cache):
            self.show_frame(self.current_frame)

    def show_frame(self, idx: int) -> None:
        if self.total_frames == 0: return
        idx = max(0, min(idx, self.total_frames - 1))
        self.current_frame = idx
        self.frame_changed.emit(idx)
        
        # DYNAMIC SCALING: Smoothly scale the buffered frame to fit the current window size
        if idx < len(self.raw_cache):
            pix = self.raw_cache[idx].scaled(self.raw_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.raw_label.setPixmap(pix)
            
        if idx < len(self.overlay_cache):
            pix_ov = self.overlay_cache[idx].scaled(self.overlay_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.overlay_label.setPixmap(pix_ov)

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.total_frames > 0:
            self.show_frame(self.current_frame)

class SkeletonViewer3D(QtInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)
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

    def load_trc(self, path: str):
        self.clear_skeleton()
        if not path or not Path(path).exists(): return
        
        try:
            with open(path, 'r') as f: lines = f.readlines()
            if len(lines) < 6: return
            
            # Header
            header_meta = lines[2].strip().split('\t')
            num_markers = int(header_meta[3])
            
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
            
            logger.info(f"Marker Data Loaded: {num_markers} markers.")

            # Draw Joints (Markers)
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

    def toggle_calibration(self, session_path, show):
        # 1. Strict Name-Based Cleanup
        # Gathers a static list of all calibration actors (including hidden point meshes) and purges them
        keys_to_remove = [k for k in self.actors.keys() if str(k).startswith("calib_")]
        for key in keys_to_remove:
            self.remove_actor(key)
        
        # Ensure legacy array does not cause reference errors
        if hasattr(self, 'calib_actors'):
            self.calib_actors.clear()
        else:
            self.calib_actors = []
            
        if not show or not session_path:
            self.render()
            return
            
        import pickle, glob, os, yaml
        import numpy as np
        
        # Load board dimensions from metadata
        meta_path = os.path.join(session_path, "sessionMetadata.yaml")
        cols, rows, size = 5, 4, 35.0 # Defaults
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = yaml.safe_load(f)
                cols = meta.get('checkerBoard', {}).get('black2BlackCornersWidth_n', 5)
                rows = meta.get('checkerBoard', {}).get('black2BlackCornersHeight_n', 4)
                size = meta.get('checkerBoard', {}).get('squareSideLength_mm', 35.0)
                
        # Calculate real-world size in meters
        width_m = (cols - 1) * size * 0.001
        height_m = (rows - 1) * size * 0.001

        # --- SMART WORLD ORIENTATION ---
        placement = meta.get('checkerBoard', {}).get('placement', 'Vertical')
        is_horizontal = placement in ['ground', 'Lying', 'Horizontal']

        if is_horizontal:
            theta_x = np.radians(90)
            cx, sx = np.cos(theta_x), np.sin(theta_x)
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            
            theta_y = np.radians(90)
            cy, sy = np.cos(theta_y), np.sin(theta_y)
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            
            R_world = Ry @ Rx
        else:
            yaw_deg = -90 
            theta = np.radians(yaw_deg)
            c, s = np.cos(theta), np.sin(theta)
            R_world = np.array([
                [ c, 0, s],
                [ 0, 1, 0],
                [-s, 0, c]
            ])
            
        # 1. Transform and Draw Realistic Checkerboard
        num_squares_x = cols + 1
        num_squares_y = rows + 1
        
        phys_width_m = num_squares_x * size * 0.001
        phys_height_m = num_squares_y * size * 0.001
        
        center_x = width_m / 2.0
        center_y = height_m / 2.0
        
        raw_center = np.array([center_x, center_y, 0.0])
        rot_center = R_world @ raw_center
        center_swizzled = self._swizzle(rot_center)
        
        board_plane = pv.Plane(
            center=(center_x, center_y, 0.0),
            i_size=phys_width_m,
            j_size=phys_height_m,
            i_resolution=num_squares_x,
            j_resolution=num_squares_y,
            direction=(0, 0, 1)
        )
        
        rotated_points = (R_world @ board_plane.points.T).T
        board_plane.points = np.array([self._swizzle(p) for p in rotated_points])
        
        grid_x, grid_y = np.mgrid[0:num_squares_x, 0:num_squares_y]
        checker_pattern = (grid_x + grid_y) % 2
        board_plane.cell_data['Colors'] = checker_pattern.flatten(order='F')
        
        self.add_mesh(
            board_plane, 
            scalars='Colors', 
            cmap=['white', 'black'], 
            show_scalar_bar=False,
            lighting=True,
            name="calib_board"
        )
        
        # 2. Transform and Draw Origin (First Corner)
        raw_origin = np.array([0.0, 0.0, 0.0])
        rot_origin = R_world @ raw_origin
        origin_pos = self._swizzle(rot_origin)
        
        origin_sphere = pv.Sphere(radius=0.02, center=origin_pos)
        self.add_mesh(origin_sphere, color="#6184D8", name="calib_origin")

        # --- Draw Board Axes to Mimic Rainbow Grid Orientation ---
        raw_x_end = np.array([width_m, 0.0, 0.0])
        rot_x_end = R_world @ raw_x_end
        x_end_pos = self._swizzle(rot_x_end)
        
        x_line = pv.Line(origin_pos, x_end_pos)
        self.add_mesh(x_line, color="red", line_width=5, name="calib_x_axis")
        
        raw_y_end = np.array([0.0, height_m, 0.0])
        rot_y_end = R_world @ raw_y_end
        y_end_pos = self._swizzle(rot_y_end)
        
        y_line = pv.Line(origin_pos, y_end_pos)
        self.add_mesh(y_line, color="blue", line_width=5, name="calib_y_axis")
        
        # 3. Transform and Draw Cameras
        cam_folders = sorted(glob.glob(os.path.join(session_path, "Videos", "Cam*")))
        for cf in cam_folders:
            pkl_path = os.path.join(cf, "cameraIntrinsicsExtrinsics.pickle")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                
                R = data['rotation']
                t = data['translation']
                
                C = -np.matrix(R).T @ np.matrix(t)
                C_m = np.array(C).flatten() * 0.001 
                
                rot_C = R_world @ C_m
                cam_pos = self._swizzle(rot_C)
                
                look_vec = np.array(center_swizzled) - np.array(cam_pos)
                cone = pv.Cone(center=cam_pos, direction=look_vec, height=0.15, radius=0.08)
                
                cam_name = os.path.basename(cf)
                
                self.add_mesh(cone, color="#E57373", name=f"calib_cam_cone_{cam_name}")
                
                line = pv.Line(cam_pos, origin_pos)
                self.add_mesh(line, color="#E57373", line_width=1, opacity=0.5, name=f"calib_cam_ray_{cam_name}")
                
                self.add_point_labels(
                    [cam_pos], 
                    [cam_name], 
                    point_size=0, 
                    font_size=14, 
                    text_color="white", 
                    shape_opacity=0.3,
                    always_visible=True,
                    name=f"calib_cam_lbl_{cam_name}"
                )
                
        self.reset_camera()
        self.render()

    def update_frame(self, idx: int):
        if self.markers is None: return
        try:
            frame = self.markers[idx % len(self.markers)]
            for i, pos in enumerate(frame):
                key = f'joint_{i}'
                if key in self.skel_actors:
                    self.skel_actors[key].SetPosition(self._swizzle(pos))
            
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

class CalibrationReviewDialog(QDialog):
    """Displays the generated checkerboard calibration images."""
    def __init__(self, parent, session_path):
        super().__init__(parent)
        self.setWindowTitle(Lang.get('calib_review'))
        self.resize(1000, 700)
        self.setStyleSheet(parent.styleSheet())
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)

        calib_dir = Path(session_path) / "CalibrationImages"
        images = list(calib_dir.glob("*_calib.jpg"))

        if not images:
            layout.addWidget(QLabel(Lang.get('no_calib_images')))
        else:
            row, col = 0, 0
            for img_path in images:
                # Create Image Label
                lbl = QLabel()
                pix = QPixmap(str(img_path))
                lbl.setPixmap(pix.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                lbl.setStyleSheet("border: 1px solid #555;")
                
                # Create Title Label
                text_lbl = QLabel(f"<b>{img_path.name}</b>")
                text_lbl.setAlignment(Qt.AlignCenter)

                # Bundle together
                vbox = QVBoxLayout()
                vbox.addWidget(text_lbl)
                vbox.addWidget(lbl)
                
                container = QWidget()
                container.setLayout(vbox)
                grid.addWidget(container, row, col)

                col += 1
                if col > 1: # 2 images per row
                    col = 0
                    row += 1

        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        btn = QPushButton(Lang.get('acknowledge_close'))
        btn.setObjectName("AccentButton")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

class PipelineConfigDialog(QDialog):
    """Configuration dialog for pipeline execution."""
    
    # Updated signature to accept step and valid_tags
    def __init__(self, parent, session_path: str, step: str = "all", valid_tags: list = None):
        super().__init__(parent)
        self.session_path = Path(session_path)
        self.step = step
        self.valid_tags = valid_tags or []
        
        self.setWindowTitle(Lang.get('pipeline_config'))
        self.resize(400, 500)
        self.setStyleSheet(parent.styleSheet())
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # GPU Selection
        layout.addWidget(QLabel(Lang.get('select_gpu')))
        self.gpu_combo = QComboBox()
        self.gpu_combo.setToolTip(Lang.get('gpu_tip'))
        self.gpu_map = self._get_gpu_info()
        self.gpu_combo.addItems(list(self.gpu_map.keys()))
        layout.addWidget(self.gpu_combo)

        # --- NEW: Pose Estimator Selection ---
        layout.addWidget(QLabel(Lang.get('pose_estimator')))
        self.pose_combo = QComboBox()
        self.pose_combo.setToolTip(Lang.get('pose_tip'))
        self.pose_combo.addItems(["OpenPose", "RTMPose"])
        layout.addWidget(self.pose_combo)
        # -------------------------------------
        
        # Resolution Selection
        layout.addWidget(QLabel(Lang.get('resolution')))
        self.res_combo = QComboBox()
        self.res_combo.setToolTip(Lang.get('res_tip'))
        self.res_combo.addItems(["1x368", "1x736", "1x736_2scales","736x1 (Landscape)"])
        layout.addWidget(self.res_combo)

        # Inside _init_ui, after creating pose_combo:
        self.pose_combo.currentTextChanged.connect(self._update_res_options)
        self._update_res_options(self.pose_combo.currentText())
        
        # Trial Selection
        layout.addWidget(QLabel(Lang.get('select_trials')))
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
        btn_cancel = QPushButton(Lang.get('cancel'))
        btn_cancel.clicked.connect(self.reject)
        btn_cancel.setToolTip(Lang.get('cancel_pipe_tip'))
        
        btn_run = QPushButton(Lang.get('run_pipeline'))
        btn_run.setObjectName("AccentButton")
        # Route to our new validation checker before accepting
        btn_run.clicked.connect(self._validate_and_accept_pipeline)
        btn_run.setToolTip(Lang.get('run_pipe_tip'))
        
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_run)
        layout.addLayout(btn_box)

    def _validate_and_accept_pipeline(self):
        """Validates conditions before allowing pipeline execution."""
        est = self.pose_combo.currentText().lower()
        res_text = self.res_combo.currentText()
        
        # 1. High-Resolution Warning
        if est == "openpose" and "736" in res_text:
            reply = QMessageBox.question(
                self,
                Lang.get('perf_warning'),
                Lang.get('perf_prompt'),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # 2. Kinematics Prerequisites
        if self.step == "kinematics":
            if est == "openpose":
                folder_tag = f"OpenPose_{res_text}"
            else:
                cres = "l" if "-l" in res_text.lower() else "m"
                folder_tag = f"RTMPose_{cres}"
                
            if folder_tag not in self.valid_tags:
                QMessageBox.warning(
                    self, 
                    Lang.get('data_missing'), 
                    Lang.get('no_pose_data_prompt')
                )
                return
                
        self.accept()

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
            checkbox.setToolTip(f"{Lang.get('process_trial_tip')} {trial_name}")
            
            # Make the neutral trial bold so it stands out as the anchor trial
            if trial_name.lower() == 'neutral':
                checkbox.setStyleSheet("font-weight: bold;")
                
            self.trial_layout.addWidget(checkbox)
            self.checks[trial_name] = checkbox
        
        if not self.checks:
            label = QLabel(Lang.get('no_trials_avail'))
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
        self.setWindowTitle(Lang.get('create_new_session'))
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
        self.name_edit.setPlaceholderText(Lang.get('eg_session'))
        self.name_edit.setToolTip(Lang.get('session_id_tip'))
        
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText(Lang.get('eg_subject'))
        self.subject_edit.setToolTip(Lang.get('subject_id_tip'))

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["m", "f"]) 
        
        self.height_edit = QLineEdit(str(Config.DEFAULT_HEIGHT))
        self.height_edit.setToolTip(f"{Lang.get('height_tip')} ({Config.MIN_HEIGHT}-{Config.MAX_HEIGHT}m)")
        
        self.weight_edit = QLineEdit(str(Config.DEFAULT_WEIGHT))
        self.weight_edit.setToolTip(f"{Lang.get('weight_tip')} ({Config.MIN_WEIGHT}-{Config.MAX_WEIGHT}kg)")
        
        self.tag_combo = QComboBox()
        self.tag_combo.addItems([
            "Healthy", "Unimpaired", "Impaired", "Exo-assisted", "Other"
        ])
        self.tag_combo.currentTextChanged.connect(self._handle_tag_change)

        self.placement_combo = QComboBox()
        self.placement_combo.addItems(["Vertical (Wall)", "Horizontal (Ground)"])
        
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Portrait (Default)", "Landscape", "Mixed"])
        self.orientation_combo.setToolTip(Lang.get('orient_tip'))
        
        form.addRow(Lang.get('session_name'), self.name_edit)
        form.addRow(Lang.get('subject_id'), self.subject_edit)
        form.addRow(Lang.get('gender'), self.gender_combo)
        form.addRow(Lang.get('height'), self.height_edit)
        form.addRow(Lang.get('weight'), self.weight_edit)
        form.addRow(Lang.get('tag'), self.tag_combo)
        
        layout.addLayout(form)
        
        # --- Camera & Calibration Settings ---
        layout.addWidget(QLabel(f"<b>{Lang.get('cam_calib_settings')}</b>"))
        form_calib = QFormLayout()

        # 1. Camera Type (Priority 2)
        self.cam_type_combo = QComboBox()
        self.cam_type_combo.addItems(["iPhone", "Android", "Other", "Mixed"])
        self.cam_type_combo.setToolTip(Lang.get('cam_type_tip'))

        # 2. Checkerboard Placement (Priority 1)
        self.placement_combo = QComboBox()
        self.placement_combo.addItems(["Vertical (Wall)", "Horizontal (Ground)"])
        self.placement_combo.setToolTip(Lang.get('placement_tip'))

        self.rows_edit = QLineEdit(str(Config.DEFAULT_CHECKERBOARD_ROWS))
        self.cols_edit = QLineEdit(str(Config.DEFAULT_CHECKERBOARD_COLS))
        self.size_edit = QLineEdit(str(Config.DEFAULT_SQUARE_SIZE))
        self.cams_edit = QLineEdit(str(Config.DEFAULT_NUM_CAMERAS))
        
        form_calib.addRow(Lang.get('cam_type'), self.cam_type_combo)
        form_calib.addRow(Lang.get('cam_orientation'), self.orientation_combo)
        form_calib.addRow(Lang.get('cam_nums'), self.cams_edit)
        form_calib.addRow(Lang.get('board_placement'), self.placement_combo)
        form_calib.addRow(Lang.get('rows'), self.rows_edit)
        form_calib.addRow(Lang.get('cols'), self.cols_edit)
        form_calib.addRow(Lang.get('square_size'), self.size_edit)

        layout.addLayout(form_calib)
        
        # Required fields note
        note_label = QLabel(Lang.get('req_fields'))
        note_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(note_label)
        
        # Buttons
        btn_box = QHBoxLayout()
        ok_btn = QPushButton(Lang.get('create'))
        ok_btn.setObjectName("AccentButton")
        ok_btn.clicked.connect(self._validate_and_accept)
        cancel_btn = QPushButton(Lang.get('cancel'))
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
                Lang.get('val_error'),
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
        if text == "Other":
            from PyQt5.QtWidgets import QInputDialog
            custom_tag, ok = QInputDialog.getText(
                self, 
                Lang.get('custom_tag_title'), 
                Lang.get('custom_tag_prompt'),
                text=""
            )
            
            if ok and custom_tag.strip():
                # Temporarily add the custom tag to the combo and select it
                self.tag_combo.addItem(custom_tag.strip())
                self.tag_combo.setCurrentText(custom_tag.strip())
            else:
                # If user cancelled, revert to a default like 'healthy'
                self.tag_combo.setCurrentIndex(0)

class PerCameraConfigDialog(QDialog):
    """Dialogue to assign device types and orientations to individual cameras."""
    def __init__(self, parent, num_cameras, default_type, default_orient):
        super().__init__(parent)
        self.setWindowTitle(Lang.get('per_cam_config'))
        layout = QVBoxLayout(self)
        self.configs = {}

        form = QFormLayout()
        for i in range(num_cameras):
            cam_name = f"Cam{i}"
            
            # Device Type Dropdown
            type_cb = QComboBox()
            type_cb.addItems(["iPhone", "Android", "Other"])
            if default_type in ["iPhone", "Android", "Other"]:
                type_cb.setCurrentText(default_type)
                
            # Orientation Dropdown
            orient_cb = QComboBox()
            orient_cb.addItems(["Portrait", "Landscape"])
            if "Landscape" in default_orient:
                orient_cb.setCurrentText("Landscape")
            elif "Portrait" in default_orient:
                orient_cb.setCurrentText("Portrait")
                
            row_layout = QHBoxLayout()
            row_layout.addWidget(type_cb)
            row_layout.addWidget(orient_cb)
            
            form.addRow(f"{cam_name}:", row_layout)
            self.configs[cam_name] = {'type': type_cb, 'orient': orient_cb}
        
        layout.addLayout(form)
        
        btn = QPushButton(Lang.get('confirm_assign'))
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def get_config(self):
        # Maps user-friendly names to the internal strings
        type_mapping = {"iPhone": "iPhone_Auto_Detect", "Android": "Android_Generic", "Other": "Generic_Webcam"}
        orient_mapping = {"Portrait": "portrait", "Landscape": "landscape"}
        
        final_config = {}
        for cam, widgets in self.configs.items():
            final_config[cam] = {
                'model': type_mapping[widgets['type'].currentText()],
                'orientation': orient_mapping[widgets['orient'].currentText()]
            }
        return final_config

# =============================================================================
# MAIN WINDOW
# =============================================================================

class OpenCapPro(QMainWindow):
    """
    Main application window for OpenCap Offline.
    
    Provides complete interface for session management, video import,
    pipeline execution, and results visualization.
    """
    
    def __init__(self):
        super().__init__()

        self.current_version = "v2.2.0"
        self.github_releases_url = "https://github.com/driscollh/opencap-offline/releases/latest"
        
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
        self.setWindowTitle("OpenCap Offline")

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
        
        logger.info("OpenCap Offline initialized")

        self.update_thread = UpdateCheckerThread()
        self.update_thread.finished_check.connect(self._log_version_status)
        self.update_thread.start()

    def cancel_pipeline(self):
        if self.process and self.process.state() == QProcess.Running:
            reply = QMessageBox.question(
                self, Lang.get('confirm_cancel'), 
                Lang.get('cancel_prompt'),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.log_box.append("\n<span style='color: #ffaa00;'><b>>>> ABORTING PROCESS...</b></span>")
                self.process.kill() # kill() is safer than terminate() for Python subprocesses
                self.process.waitForFinished(2000)
                
                self.progress_bar.setVisible(False)
                self.btn_cancel_process.setVisible(False)
                self.progress_label.setText(Lang.get('aborted_user'))

    def _toggle_calibration_view(self, checked):
        """Passes the current session path down to the 3D viewer to draw the cameras."""
        session = self.session_combo.currentText()
        if session:
            session_path = str(self.data_dir / session)
            self.skeleton_viewer.toggle_calibration(session_path, checked)

    def _log_version_status(self, latest_version, error):
        """Appends the version check results directly to the log box on startup."""
        self.log_box.append("\n--- Version Status ---")
        self.log_box.append(f"Current Version: <b>{self.current_version}</b>")
        
        if latest_version:
            if latest_version != self.current_version:
                # Orange warning text, but the URL is just plain text
                self.log_box.append(f"<span style='color: #ffaa00;'><b>New version available ({latest_version})!</b></span>")
                self.log_box.append(f"Please download from: {self.github_releases_url}")
            else:
                # Green success text
                self.log_box.append("<span style='color: #00aa00;'>You are running the current version.</span>")
        else:
            self.log_box.append(f"<span style='color: #888;'><i>Unable to check for updates ({error})</i></span>")
            
        # Auto-scroll to ensure the user sees the message
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

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

    def _process_finished(self, exit_code, exit_status):
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Idle")

        if exit_code == 5:
            self.log_box.append("\n>>> INTERRUPT: Multiple subjects detected.")
            self._handle_subject_selection()
            return

        if exit_code == 0:
            self.log_box.append("\n>>> SUCCESS: Pipeline Finished.")
            
            # --- NEW: Show Calibration Images ---
            current_step = self.current_pipeline_config.get("step", "all")
            if current_step in ["calibrate", "all"]:
                self.show_calib_cb.setEnabled(True)
                session = self.current_pipeline_config['session']
                calib_dir = self.data_dir / session / "CalibrationImages"
                
                # Only show dialog if images were actually generated
                if calib_dir.exists() and any(calib_dir.iterdir()):
                    dlg = CalibrationReviewDialog(self, self.data_dir / session)
                    dlg.exec_()
                else:
                    QMessageBox.information(self, Lang.get('success'), Lang.get('proc_complete'))
            else:
                QMessageBox.information(self, Lang.get('success'), Lang.get('proc_complete'))
        else:
            self.log_box.append(f"\n>>> ERROR: Failed with code {exit_code}")

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
        
        self.progress_label = QLabel(Lang.get('idle'))
        self.progress_label.setStyleSheet("color: #888; font-size: 11px; border: none;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(15)

        # --- NEW: Cancel Button ---
        self.btn_cancel_process = QPushButton(Lang.get('cancel_process'))
        self.btn_cancel_process.setStyleSheet("background-color: #8b0000; color: white; font-weight: bold;")
        self.btn_cancel_process.setVisible(False)
        self.btn_cancel_process.clicked.connect(self.cancel_pipeline)
        
        # Group the progress bar and button horizontally
        prog_bar_layout = QHBoxLayout()
        prog_bar_layout.addWidget(self.progress_bar)
        prog_bar_layout.addWidget(self.btn_cancel_process)
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addLayout(prog_bar_layout) # Replaced the direct progress_bar add
        log_layout.addWidget(progress_container)
        
        # Log output - Removed fixed height so it respects the splitter
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Process output will appear here...")
        log_layout.addWidget(self.log_box)
    
    def _create_menu_bar(self):
        menubar = self.menuBar()
        self.file_menu = menubar.addMenu(Lang.get('file_menu'))
        
        self.new_action = QAction(Lang.get('new_session_menu'), self)
        self.new_action.setShortcut('Ctrl+N')
        self.new_action.triggered.connect(self.open_new_session_dialog)
        self.file_menu.addAction(self.new_action)
        
        self.refresh_action = QAction(Lang.get('refresh_menu'), self)
        self.refresh_action.setShortcut('Ctrl+R')
        self.refresh_action.triggered.connect(self.refresh_sessions)
        self.file_menu.addAction(self.refresh_action)
        
        self.file_menu.addSeparator()
        self.recent_menu = self.file_menu.addMenu(Lang.get('recent_menu'))
        self._update_recent_menu()
        
        self.file_menu.addSeparator()
        self.quit_action = QAction(Lang.get('quit_menu'), self)
        self.quit_action.setShortcut('Ctrl+Q')
        self.quit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.quit_action)

        self.view_menu = menubar.addMenu(Lang.get('view_menu'))
        
        self.theme_action = QAction(Lang.get('theme_toggle'), self)
        self.theme_action.setShortcut('Ctrl+T')
        self.theme_action.triggered.connect(self.toggle_theme)
        self.view_menu.addAction(self.theme_action)
        
        self.lang_action = QAction(Lang.get('lang_toggle'), self)
        self.lang_action.triggered.connect(self.toggle_language)
        self.view_menu.addAction(self.lang_action)
        
        self.help_menu = menubar.addMenu(Lang.get('help_menu'))
        self.about_action = QAction(Lang.get('about_menu'), self)
        self.about_action.triggered.connect(self._show_about)
        self.help_menu.addAction(self.about_action)

    def toggle_language(self):
        """Swaps the global language state and forces a UI refresh."""
        Lang.CURRENT = 'KO' if Lang.CURRENT == 'EN' else 'EN'
        
        # 1. Update Menus (Use setTitle for menus, setText for actions)
        self.file_menu.setTitle(Lang.get('file_menu'))
        self.view_menu.setTitle(Lang.get('view_menu'))
        self.help_menu.setTitle(Lang.get('help_menu'))
        self.recent_menu.setTitle(Lang.get('recent_menu'))
        
        self.new_action.setText(Lang.get('new_session_menu'))
        self.refresh_action.setText(Lang.get('refresh_menu'))
        self.quit_action.setText(Lang.get('quit_menu'))
        self.theme_action.setText(Lang.get('theme_toggle'))
        self.lang_action.setText(Lang.get('lang_toggle'))
        self.about_action.setText(Lang.get('about_menu'))
        
        # 2. Update Header Buttons
        self.new_session_btn.setText(Lang.get('new_session'))
        self.edit_meta_btn.setText(Lang.get('edit_meta'))
        self.refresh_list_btn.setText(Lang.get('refresh_list'))
        
        # 3. Update Import Panel
        self.trial_type_label.setText(Lang.get('trial_type_label'))
        self.trial_name_edit.setPlaceholderText(Lang.get('trial_name_placeholder'))
        self.trial_name_edit.setToolTip(Lang.get('trial_name_tip'))
        self.execute_import_btn.setText(Lang.get('execute_import'))
        
        # Update Import Toggle Button dynamically
        is_import_visible = self.toggle_import_btn.isChecked()
        self.toggle_import_btn.setText(Lang.get('import_trial_down') if is_import_visible else Lang.get('import_trial_right'))
        
        # 4. Update Pipeline Panel & Checkboxes
        self.research_mode_cb.setText(Lang.get('research_mode'))
        self.show_calib_cb.setText(Lang.get('show_calib'))
        
        is_pipe_visible = self.toggle_pipeline_btn.isChecked()
        self.toggle_pipeline_btn.setText(Lang.get('pipeline_exec_down') if is_pipe_visible else Lang.get('pipeline_exec_right'))
        
        self.btn_calib_clin.setText(Lang.get('step_1'))
        self.btn_pipe_full.setText(Lang.get('step_2_full'))
        self.btn_calib_res.setText(Lang.get('step_1'))
        self.btn_extrinsics.setText(Lang.get('step_2_ext'))
        self.btn_pose.setText(Lang.get('step_3'))
        self.btn_kinematics.setText(Lang.get('step_4'))
        
        # 5. Update Log Area
        self.btn_cancel_process.setText(Lang.get('cancel_process'))
        self.log_box.setPlaceholderText(Lang.get('log_placeholder'))
        if self.progress_label.text() in ['Idle', '대기 중']:
            self.progress_label.setText(Lang.get('idle'))
            
        # 6. Re-render Camera Slots (Updates Browse & Clear buttons)
        current_session = self.session_combo.currentText()
        if current_session:
            self.refresh_cam_slots(current_session)
        
        msg = "언어가 변경되었습니다. UI를 새로고침하려면 창을 다시 열어야 할 수 있습니다." if Lang.CURRENT == 'KO' else "Language changed. Re-open dialogs to see updates."
        QMessageBox.information(self, "Language / 언어", msg)

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
        self.new_session_btn = QPushButton(Lang.get('new_session'))
        self.new_session_btn.setObjectName("AccentButton")
        self.new_session_btn.clicked.connect(self.open_new_session_dialog)
        self.new_session_btn.setToolTip("Create new session (Ctrl+N)")
        header_layout.addWidget(self.new_session_btn)

        header_layout.addStretch()

        # --- RIGHT SIDE: Management Controls ---
        self.edit_meta_btn = QPushButton(Lang.get('edit_meta'))
        self.edit_meta_btn.clicked.connect(self.open_metadata_editor)
        header_layout.addWidget(self.edit_meta_btn)
        
        self.refresh_list_btn = QPushButton(Lang.get('refresh_list'))
        self.refresh_list_btn.clicked.connect(self.refresh_sessions)
        header_layout.addWidget(self.refresh_list_btn)
        
        # --- RESTORED: Session Dropdown Menu ---
        self.session_combo = QComboBox()
        self.session_combo.setMinimumWidth(200)
        self.session_combo.currentTextChanged.connect(self.on_session_change)
        header_layout.addWidget(self.session_combo)
        
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
        self.import_container = QFrame()
        self.import_container.setObjectName("ImportPanel")
        layout = QVBoxLayout(self.import_container)

        # Header with Toggle Button
        header_row = QHBoxLayout()
        self.toggle_import_btn = QPushButton(Lang.get('import_trial_down'))
        self.toggle_import_btn.setCheckable(True)
        self.toggle_import_btn.setChecked(True)
        self.toggle_import_btn.setStyleSheet("text-align: left; font-weight: bold; border: none; background: none;")
        self.toggle_import_btn.clicked.connect(self._toggle_import_visibility)
        header_row.addWidget(self.toggle_import_btn)
        layout.addLayout(header_row)

        # Content Wrapper (Hidden when toggled)
        self.import_content = QWidget()
        content_layout = QVBoxLayout(self.import_content)
        content_layout.setContentsMargins(10, 0, 10, 10)

        type_row = QHBoxLayout()
        self.trial_type_label = QLabel(Lang.get('trial_type_label'), styleSheet="font-weight:bold; color:#888;")
        type_row.addWidget(self.trial_type_label)
        
        # Radio buttons for trial types
        self.type_group = QButtonGroup(self)
        self.type_buttons: Dict[str, QRadioButton] = {}
        
        # Define the trial types and add their radio buttons
        for text, t_type in [
            ("Intrinsics", TrialType.INTRINSICS),
            ("Calibration", TrialType.CALIBRATION),
            ("Neutral", TrialType.NEUTRAL),
            ("Dynamic", TrialType.DYNAMIC)
        ]:
            radio = QRadioButton(text)
            if t_type == TrialType.CALIBRATION:
                radio.setChecked(True)
            
            self.type_group.addButton(radio)
            type_row.addWidget(radio)
            self.type_buttons[t_type.value] = radio
            radio.toggled.connect(self._update_trial_name_input)
        
        # Trial name input
        self.trial_name_edit = QLineEdit()
        self.trial_name_edit.setPlaceholderText(Lang.get('trial_name_placeholder'))
        self.trial_name_edit.setToolTip(Lang.get('trial_name_tip'))
        type_row.addWidget(self.trial_name_edit)
        content_layout.addLayout(type_row)
        
        self.cam_slot_layout = QGridLayout()
        content_layout.addLayout(self.cam_slot_layout)
        
        self.execute_import_btn = QPushButton(Lang.get('execute_import'))
        self.execute_import_btn.clicked.connect(self.run_import)
        content_layout.addWidget(self.execute_import_btn)
        
        layout.addWidget(self.import_content)
        self.main_layout.addWidget(self.import_container)

    def _toggle_import_visibility(self):
        """Minimizes the import section to save space."""
        is_visible = self.toggle_import_btn.isChecked()
        self.import_content.setVisible(is_visible)
        self.toggle_import_btn.setText(Lang.get('import_trial_down') if is_visible else Lang.get('import_trial_right'))

    def _toggle_pipeline_visibility(self):
        """Minimizes the pipeline section to save space."""
        is_visible = self.toggle_pipeline_btn.isChecked()
        self.pipeline_content.setVisible(is_visible)
        self.toggle_pipeline_btn.setText(Lang.get('pipeline_exec_down') if is_visible else Lang.get('pipeline_exec_right'))
    
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
        """Creates the pipeline execution buttons inside a collapsible container."""
        self.pipeline_container = QFrame()
        self.pipeline_container.setObjectName("ImportPanel") 
        container_layout = QVBoxLayout(self.pipeline_container)

        # --- Header Row (Toggle Button ONLY) ---
        header_row = QHBoxLayout()
        self.toggle_pipeline_btn = QPushButton(Lang.get('pipeline_exec_down'))
        self.toggle_pipeline_btn.setCheckable(True)
        self.toggle_pipeline_btn.setChecked(True)
        self.toggle_pipeline_btn.setStyleSheet("text-align: left; font-weight: bold; border: none; background: none;")
        self.toggle_pipeline_btn.clicked.connect(self._toggle_pipeline_visibility)
        
        header_row.addWidget(self.toggle_pipeline_btn)
        container_layout.addLayout(header_row)

        # --- CONTENT WRAPPER (Hidden when toggled) ---
        self.pipeline_content = QWidget()
        strip_layout = QVBoxLayout(self.pipeline_content)
        strip_layout.setContentsMargins(10, 0, 10, 10)
        
        # --- RESEARCH MODE & CALIBRATION VIEW TOGGLES ---
        toggle_layout = QHBoxLayout()
        self.research_mode_cb = QCheckBox(Lang.get('research_mode'))
        self.research_mode_cb.setStyleSheet("font-weight: bold; color: #888;")
        self.research_mode_cb.toggled.connect(self._toggle_research_mode)
        
        self.show_calib_cb = QCheckBox(Lang.get('show_calib'))
        self.show_calib_cb.setStyleSheet("font-weight: bold; color: #888;")
        self.show_calib_cb.setEnabled(False) # Grey out by default
        self.show_calib_cb.toggled.connect(self._toggle_calibration_view)
        
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.show_calib_cb)
        toggle_layout.addWidget(self.research_mode_cb)
        strip_layout.addLayout(toggle_layout)
        
        # --- STACKED WIDGET ---
        self.pipeline_stack = QStackedWidget()
        
        # 1. CLINICAL PAGE
        self.clinical_page = QWidget()
        clin_layout = QHBoxLayout(self.clinical_page)
        clin_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_calib_clin = QPushButton(Lang.get('step_1'))
        self.btn_calib_clin.setFixedHeight(Config.BUTTON_HEIGHT)
        self.btn_calib_clin.clicked.connect(self.run_intrinsics)
        
        self.btn_pipe_full = QPushButton(Lang.get('step_2_full'))
        self.btn_pipe_full.setObjectName("AccentButton")
        self.btn_pipe_full.setFixedHeight(Config.BUTTON_HEIGHT)
        self.btn_pipe_full.clicked.connect(lambda: self.run_pipeline(step="all"))
        
        clin_layout.addWidget(self.btn_calib_clin, 1) 
        clin_layout.addWidget(self.btn_pipe_full, 3)         
        
        # 2. RESEARCH PAGE
        self.research_page = QWidget()
        res_layout = QHBoxLayout(self.research_page)
        res_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_calib_res = QPushButton(Lang.get('step_1'))
        self.btn_calib_res.setFixedHeight(Config.BUTTON_HEIGHT)
        self.btn_calib_res.clicked.connect(self.run_intrinsics)
        
        self.btn_extrinsics = QPushButton(Lang.get('step_2_ext'))
        self.btn_extrinsics.setFixedHeight(Config.BUTTON_HEIGHT)
        self.btn_extrinsics.clicked.connect(lambda: self.run_pipeline(step="calibrate"))
        
        self.btn_pose = QPushButton(Lang.get('step_3'))
        self.btn_pose.setFixedHeight(Config.BUTTON_HEIGHT)
        self.btn_pose.clicked.connect(lambda: self.run_pipeline(step="pose"))
        
        self.btn_kinematics = QPushButton(Lang.get('step_4'))
        self.btn_kinematics.setFixedHeight(Config.BUTTON_HEIGHT)
        self.btn_kinematics.clicked.connect(lambda: self.run_pipeline(step="kinematics"))
        
        res_layout.addWidget(self.btn_calib_res, 1)
        res_layout.addWidget(self.btn_extrinsics, 1)
        res_layout.addWidget(self.btn_pose, 1)
        res_layout.addWidget(self.btn_kinematics, 1)
        
        # Add pages to stack
        self.pipeline_stack.addWidget(self.clinical_page)
        self.pipeline_stack.addWidget(self.research_page)
        
        strip_layout.addWidget(self.pipeline_stack)
        container_layout.addWidget(self.pipeline_content)
        
        self.main_layout.addWidget(self.pipeline_container)

    def _toggle_research_mode(self, checked):
        """Swaps the visible button panel"""
        if checked:
            self.pipeline_stack.setCurrentIndex(1)
            self._update_status("Research Mode Enabled: Granular execution unlocked.")
        else:
            self.pipeline_stack.setCurrentIndex(0)
            self._update_status("Clinical Mode Enabled: Automated pipeline locked in.")
    
    def _create_dashboard(self):
        """Initialize visualization dashboard with dynamic scaling."""
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setContentsMargins(0, 5, 0, 0) 
        
        # Left: Trial tree
        self.tree = QTreeWidget()
        
        # --- FIX: Swap fixed width for minimum width ---
        self.tree.setMinimumWidth(100) 
        # -----------------------------------------------
        
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self.on_tree_click)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_tree_context_menu)
        self.splitter.addWidget(self.tree)
        
        # Center: 3D viewer
        self.skeleton_viewer = SkeletonViewer3D()
        self.splitter.addWidget(self.skeleton_viewer)
        
        # Right: Video player 
        self.video_container = QWidget()
        self.video_container.setMinimumWidth(300) 
        video_layout = QVBoxLayout(self.video_container)
        
        self.video_player = DualVideoPlayer()
        self.video_player.frame_changed.connect(self.skeleton_viewer.update_frame)
        video_layout.addWidget(self.video_player)
        self.splitter.addWidget(self.video_container)
        
        # The splitter will still use these as the default starting sizes
        self.splitter.setSizes([300, 600, 600])
    
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
        
        self.progress_label = QLabel(Lang.get('idle'))
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
        self.log_box.setPlaceholderText(Lang.get('log_placeholder'))
        log_layout.addWidget(self.log_box)
        
        self.main_layout.addWidget(log_container)
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Session label (permanent)
        self.session_status_label = QLabel()
        self.status_bar.addPermanentWidget(self.session_status_label)
        
        self.status_bar.showMessage(Lang.get('ready'))
    
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
            path_label = QLabel(Lang.get('no_files_sel'))
            path_label.setStyleSheet("color: gray;")
            
            # Browse Button
            browse_btn = QPushButton(Lang.get('browse'))
            browse_btn.clicked.connect(lambda checked, c=cam, l=path_label: self._browse_file(c, l))
            
            # Clear Button (NEW)
            clear_btn = QPushButton(Lang.get('clear'))
            clear_btn.setFixedWidth(60)
            clear_btn.clicked.connect(lambda checked, c=cam, l=path_label: self._clear_selection(c, l))
            
            self.cam_slot_layout.addWidget(label, i, 0)
            self.cam_slot_layout.addWidget(path_label, i, 1)
            self.cam_slot_layout.addWidget(browse_btn, i, 2)
            self.cam_slot_layout.addWidget(clear_btn, i, 3) # Add to column 3
            
            self.cam_buttons[cam] = {"paths": [], "label": path_label}

        
        logger.info(f"Created {len(cameras)} camera slot(s)")
    
    def _browse_file(self, camera: str, label: QLabel) -> None:
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
        label.setText(Lang.get('no_files_sel'))
        label.setStyleSheet("color: gray; font-weight: normal;")
        logger.info(f"Cleared selection for {camera}")
    
    def refresh_tree(self, session_name: str):
        # --- 1. SAVE EXPANSION STATE ---
        expanded_paths = set()
        
        def save_state(item, current_path):
            path = f"{current_path}/{item.text(0)}" if current_path else item.text(0)
            if item.isExpanded():
                expanded_paths.add(path)
            for i in range(item.childCount()):
                save_state(item.child(i), path)
                
        was_populated = self.tree.topLevelItemCount() > 0
        if was_populated:
            for i in range(self.tree.topLevelItemCount()):
                save_state(self.tree.topLevelItem(i), "")

        self.tree.clear()
        
        video_root = self.data_dir / session_name / "Videos"
        if not video_root.exists(): return
        
        self.log_box.append(f"\n--- Scanning Session: {session_name} ---")
        cameras = sorted([d.name for d in video_root.iterdir() if d.name.startswith("Cam")])
        
        for cam in cameras:
            cam_item = QTreeWidgetItem([cam])
            cam_item.setIcon(0, self.style().standardIcon(QStyle.SP_DriveCDIcon))
            self.tree.addTopLevelItem(cam_item)
            
            cam_root = video_root / cam
            media_path = cam_root / "InputMedia"
            
            # ---> DEFINING output_candidates HERE (Prevents the NameError) <---
            output_candidates = sorted([
                d.name for d in cam_root.iterdir()
                if d.is_dir() and (
                    d.name.startswith("OutputVideos") or 
                    d.name.startswith("OutputMedia") or 
                    d.name.startswith("OutputJsons")
                )
            ], reverse=True)
            
            self.log_box.append(f"[{cam}] Found Output Folders: {output_candidates}")
            
            if media_path.exists():
                trials = sorted([t.name for t in media_path.iterdir() if t.is_dir()], key=get_trial_sort_key)
                
                for trial_name in trials:
                    trial_path = media_path / trial_name
                    
                    # Create Trial Folder Item
                    trial_item = QTreeWidgetItem([trial_name])
                    trial_item.setIcon(0, self.style().standardIcon(QStyle.SP_DirIcon))
                    cam_item.addChild(trial_item)
                    
                    # Get the raw video paths
                    videos = sorted([f for f in trial_path.iterdir() if f.suffix.lower() in Config.VIDEO_EXTENSIONS])
                    if not videos:
                        continue
                        
                    raw_vid_path = str(videos[0])
                    
                    # Add a "Raw Only" clickable item for EVERY video found
                    for vid in videos:
                        raw_item = QTreeWidgetItem([f"Raw Video: {vid.name}"])
                        raw_item.setIcon(0, self.style().standardIcon(QStyle.SP_FileIcon))
                        # Use absolute paths!
                        raw_item.setData(0, Qt.UserRole, {
                            "type": "video",
                            "path": str(vid.absolute()),
                            "overlay": None,
                            "trc": None
                        })
                        trial_item.addChild(raw_item)
                    
                    # Add a clickable item for EVERY resolution/overlay it finds!
                    for out_folder in output_candidates:
                        res_string = out_folder.replace("OutputVideos_", "").replace("OutputMedia_", "").replace("OutputJsons_", "")
                        trial_out_dir = cam_root / out_folder / trial_name
                        
                        overlay_file = None
                        if trial_out_dir.exists():
                            for f in trial_out_dir.iterdir():
                                if f.suffix.lower() in ['.avi', '.mp4']:
                                    name_lower = f.name.lower()
                                    # RTMPose specific file catch
                                    if (name_lower.endswith('_overlay.avi') or 
                                        name_lower.endswith('_diagnostic.avi') or 
                                        '_rtmpose_diagnostic' in name_lower):
                                        overlay_file = f
                                        break
                        
                        potential_trc = self.data_dir / session_name / "MarkerData" / res_string / "PreAugmentation" / f"{trial_name}.trc"
                        
                        if overlay_file and overlay_file.exists():
                            file_type_label = "Diagnostic" if "diagnostic" in overlay_file.name.lower() else "Overlay"
                            
                            proc_item = QTreeWidgetItem([f"{file_type_label} ({res_string})"])
                            proc_item.setIcon(0, self.style().standardIcon(QStyle.SP_MediaPlay))
                            
                            # CRITICAL: Convert all Path objects to absolute strings for VideoLoader
                            proc_item.setData(0, Qt.UserRole, {
                                "type": "video",
                                "path": str(Path(raw_vid_path).absolute()),
                                "overlay": str(overlay_file.absolute()),
                                "trc": str(potential_trc.absolute()) if potential_trc.exists() else None
                            })
                            trial_item.addChild(proc_item)
                        
            # Set default expansion for first-time loads
            cam_item.setExpanded(True)
            
        # --- 2. RESTORE EXPANSION STATE ---
        if was_populated:
            def restore_state(item, current_path):
                path = f"{current_path}/{item.text(0)}" if current_path else item.text(0)
                item.setExpanded(path in expanded_paths)
                for i in range(item.childCount()):
                    restore_state(item.child(i), path)
                    
            for i in range(self.tree.topLevelItemCount()):
                restore_state(self.tree.topLevelItem(i), "")

        # --- NEW: Check for Extrinsics to Enable 3D View ---
        has_extrinsics = False
        cam_folders = sorted([d for d in video_root.iterdir() if d.is_dir() and d.name.startswith("Cam")])
        for cf in cam_folders:
            if (cf / "cameraIntrinsicsExtrinsics.pickle").exists():
                has_extrinsics = True
                break
                
        self.show_calib_cb.setEnabled(has_extrinsics)
        if not has_extrinsics and self.show_calib_cb.isChecked():
            self.show_calib_cb.setChecked(False) 
        elif has_extrinsics and self.show_calib_cb.isChecked():
            # Trigger a redraw if session changed but box was left checked
            self._toggle_calibration_view(True)
            
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())
    
    def on_tree_click(self, item, col):
        """Intercepts the click and starts a tiny delay to prevent spam-crashing."""
        data = item.data(0, Qt.UserRole)
        # Verify that the item clicked actually contains video data
        if not data or data.get("type") != "video": 
            return
        
        self.pending_click_data = data
        # Restart the 200ms timer to debounce rapid clicks
        self.tree_click_timer.start(200) 
        self._update_status("Loading media...", warning=True)

    def _execute_tree_click(self):
        data = self.pending_click_data
        if not data: return
    
        # Ensure absolute string paths for the backend
        raw_path = str(data.get("path", ""))
        overlay_path = str(data.get("overlay", "")) if data.get("overlay") else None
        trc_path = str(data.get("trc", "")) if data.get("trc") else None

        # Load Skeleton
        if trc_path and os.path.exists(trc_path):
            self.skeleton_viewer.load_trc(trc_path)
        else:
            self.skeleton_viewer.clear_skeleton()
    
        # Load Video
        self.video_player.blockSignals(True)
        self.video_player.load(raw_path, overlay_path)
        self.video_player.blockSignals(False)
        
        # 3. Synchronize first frame
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
            QMessageBox.warning(self, Lang.get('input_error'), Lang.get('input_error_prompt'))
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
                Lang.get('folder_exists'), 
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
            
            QMessageBox.information(self, Lang.get('success'), f"Imported {files_copied} files into '{final_name}'")
            self.refresh_tree(session)
            self._update_status(f"Imported trial: {final_name}", success=True)

        except Exception as e:
            logger.error(f"Import failed: {e}", exc_info=True)
            QMessageBox.critical(self, Lang.get('error'), f"Import failed: {str(e)}")
            self.refresh_tree(session)
    
    # -------------------------------------------------------------------------
    # PIPELINE EXECUTION
    # -------------------------------------------------------------------------
    
    def run_intrinsics(self):
        """Triggers the calibration and manages button state."""
        session = self.session_combo.currentText()
        if not session:
            QMessageBox.warning(self, Lang.get('warning'), "Please select a session first.")
            return

        from PyQt5.QtWidgets import QInputDialog
        cams = list(self.cam_buttons.keys())
        choices = ["All Cameras"] + cams
        
        cam_choice, ok = QInputDialog.getItem(
            self, Lang.get('calib_intrinsics'), Lang.get('sel_cam_calib'), choices, 0, False
        )
        
        # Only execute if the user actually clicks 'OK' (doesn't cancel)
        if ok:
            target = None if cam_choice == "All Cameras" else cam_choice
            
            # 1. Disable the button to prevent multiple simultaneous runs
            self.intrinsics_btn = self.sender()
            self.intrinsics_btn.setEnabled(False)
            
            # Save original text to restore later, then change it
            self.original_btn_text = self.intrinsics_btn.text()
            self.intrinsics_btn.setText(Lang.get('calibrating_btn'))
            
            target_str = target if target else 'All Cameras'
            self.log_box.append(f"\n>>> Starting Intrinsics Calibration for: {target_str}")
            
            # 2. Start background worker EXACTLY ONCE
            self.worker = IntrinsicsWorker(session, target_cam=target)
            self.worker.finished.connect(self._on_intrinsics_finished)
            self.worker.start()

    def _on_intrinsics_finished(self, success, message):
        """Rectifies the 'greyed-out' issue by re-enabling the button."""
        # Reset Button State using the dynamically saved text
        if hasattr(self, 'intrinsics_btn'):
            self.intrinsics_btn.setEnabled(True)
            if hasattr(self, 'original_btn_text'):
                self.intrinsics_btn.setText(self.original_btn_text)
        
        if success:
            self.log_box.append(f">>> SUCCESS: {message}")
            QMessageBox.information(self, Lang.get('calib_done'), message)
        else:
            self.log_box.append(f">>> ERROR: {message}")
            QMessageBox.critical(self, Lang.get('calib_failed'), f"Error: {message}")

    def _execute_intrinsics_thread(self, func, session_name):
        try:
            # This calls the logic in generate_intrinsics.py
            func(session_name)
            self.root.after(0, lambda: messagebox.showinfo("Done", "Intrinsics Calibration Complete!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Calibration failed: {e}"))
        finally:
            self.root.after(0, lambda: self.btn_intrinsics.config(state=tk.NORMAL, text="A. GENERATE INTRINSICS"))
    
    def run_pipeline(self, step="all") -> None:
        session = self.session_combo.currentText()
        if not session: 
            return

        # --- Bypass Dialog for CPU-bound Extrinsics ---
        if step == "calibrate":
            config = {
                "gpu": "0",                
                "res": "default",          
                "pose_estimator": "openpose",  
                "trials": ["calibration"]  
            }
            
            self.current_pipeline_config = {
                "session": session,
                "args": config,
                "step": step 
            }
            
            self.log_box.append("\n>>> Starting Extrinsics Calibration (CPU Mode)...")
            self._start_pipeline_process(session, config, step=step)
            return

        # --- NEW: Pre-scan for processed Pose Data ---
        valid_pose_tags = []
        cam0_dir = self.data_dir / session / "Videos" / "Cam0"
        
        if cam0_dir.exists():
            for d in cam0_dir.iterdir():
                if d.is_dir() and d.name.startswith("OutputPkl_"):
                    # Extract the exact identifier (e.g., "RTMPose_m" or "OpenPose_1x736")
                    valid_pose_tags.append(d.name.replace("OutputPkl_", ""))

        # Hard stop if they click Kinematics but haven't run any Pose estimation
        if step == "kinematics" and not valid_pose_tags:
            QMessageBox.warning(self, Lang.get('no_pose_data'), Lang.get('no_pose_data_prompt'))
            return
        # ---------------------------------------------

        # Pass the step and valid_tags to the dialog
        dlg = PipelineConfigDialog(self, str(self.data_dir / session), step=step, valid_tags=valid_pose_tags)
        if dlg.exec_() != QDialog.Accepted: 
            return
            
        config = dlg.get_data()
        
        # --- NEW: High-Resolution Warning ---
        if config.get("pose_estimator") == "openpose" and config.get("res") == "1x736":
            reply = QMessageBox.question(
                self,
                "Performance Warning",
                "Processing with OpenPose at 1x736 resolution is highly memory-intensive.\n\n"
                "If the video contains a large amount of vertical background clutter (tables, chairs, tripods), "
                "the algorithm may experience combinatorial explosion, massively increasing processing time.\n\n"
                "Do you wish to proceed?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                self.log_box.append(">>> Pipeline cancelled by user (Resolution Warning).")
                return
        # ------------------------------------
        
        self.current_pipeline_config = {
            "session": session,
            "args": config,
            "step": step 
        }

        self._start_pipeline_process(session, config, step=step)

    def _start_pipeline_process(self, session, config, step="all", resume_file=None):
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
        
        # --- NEW: Append the exclusion file flag if resuming ---
        if resume_file:
            script_args.extend(["--exclusion_file", resume_file])
            
        self.log_box.clear()
        self.log_box.append(f">>> Executing {config.get('pose_estimator').upper()}: {' '.join(script_args)}")
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText(Lang.get('proc_stable_backend'))
        
        self.process.start(sys.executable, script_args)
        self.btn_cancel_process.setVisible(True) # Show the kill switch

    def _process_finished(self, exit_code, exit_status):
        self.progress_bar.setVisible(False)
        self.progress_label.setText(Lang.get('idle'))
        self.btn_cancel_process.setVisible(False) # Hide the kill switch

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
            QMessageBox.critical(self, Lang.get('error'), Lang.get('track_data_missing'))
            return

        with open(track_file) as f: meta = json.load(f)
        video_path = meta.get('video_path', '')
        
        if not os.path.exists(video_path):
             QMessageBox.warning(self, Lang.get('vid_missing'), f"Could not find video for review:\n{video_path}")
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
            QMessageBox.warning(self, Lang.get('session_exists'), f"Session '{data['name']}' already exists.")
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
            
            # --- PER-CAMERA MAPPING LOGIC ---
            num_cams = int(data['cams'])
            cam_model_map = {}
            cam_orient_map = {}
            
            # If they chose Mixed for either setting, open the advanced dialog
            if data['cam_type'] == "Mixed" or data['orientation'] == "Mixed":
                config_dlg = PerCameraConfigDialog(self, num_cams, data['cam_type'], data['orientation'])
                if config_dlg.exec_() == QDialog.Accepted:
                    configs = config_dlg.get_config()
                    cam_model_map = {cam: conf['model'] for cam, conf in configs.items()}
                    cam_orient_map = {cam: conf['orientation'] for cam, conf in configs.items()}
                else:
                    return # User cancelled
            else:
                # Standard unified setup
                base_model = "Generic_Webcam"
                if data['cam_type'] == "iPhone":
                    base_model = "iPhone_Auto_Detect"
                elif data['cam_type'] == "Android":
                    base_model = "Android_Generic"
                cam_model_map = {f"Cam{i}": base_model for i in range(num_cams)}

                base_orient = "landscape" if "Landscape" in data['orientation'] else "portrait"
                cam_orient_map = {f"Cam{i}": base_orient for i in range(num_cams)}
            
            # Use 'mixed' for the legacy key so we don't break old OpenCap scripts, 
            # and store our new explicit map in 'cameraOrientations'
            orientation_val = "mixed" if data['orientation'] == "Mixed" else ("landscape" if "Landscape" in data['orientation'] else "portrait")

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
                "cameraOrientations": cam_orient_map,
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
            QMessageBox.critical(self, Lang.get('creation_failed'), f"Error: {str(e)}")
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
            Lang.get('about_title'),
            "<h3>OpenCap Offline</h3>"
            f"<p><b>Version:</b> {self.current_version}</p>"
            "<p>A comprehensive motion capture processing launcher.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Session management</li>"
            "<li>Multi-camera video import</li>"
            "<li>Pipeline execution with GPU support</li>"
            "<li>3D skeletal visualization</li>"
            "<li>Synchronized video playback</li>"
            "</ul>"
        )

    def manual_update_check(self):
        """Manually ping GitHub for the latest release version."""
        api_url = "https://api.github.com/repos/driscollh/opencap-offline/releases/latest"
        
        try:
            # Ping GitHub with a 5-second timeout
            response = requests.get(api_url, timeout=5)
            
            if response.status_code == 200:
                latest_version = response.json().get("tag_name")
                
                if latest_version and latest_version != self.current_version:
                    msg = QMessageBox(self)
                    msg.setWindowTitle(Lang.get('update_avail'))
                    msg.setIcon(QMessageBox.Information)
                    msg.setText(f"A new version of OpenCap Offline (<b>{latest_version}</b>) is available!")
                    msg.setInformativeText("Would you like to download it now?")
                    
                    # Set up Yes/No buttons
                    download_btn = msg.addButton("Download Update", QMessageBox.AcceptRole)
                    cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
                    
                    msg.exec_()
                    
                    if msg.clickedButton() == download_btn:
                        # Opens the user's default web browser to your GitHub
                        QDesktopServices.openUrl(QUrl(self.github_releases_url))
                else:
                    QMessageBox.information(
                        self, 
                        Lang.get('up_to_date'), 
                        f"You are currently running the latest version ({self.current_version})."
                    )
            else:
                QMessageBox.warning(self, Lang.get('error'), Lang.get('api_error'))
                
        except requests.exceptions.RequestException:
            QMessageBox.warning(
                self, 
                Lang.get('error'), 
                Lang.get('conn_error')
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
        
        rename_action = QAction(Lang.get('rename_file'), self)
        rename_action.triggered.connect(lambda: self._rename_tree_file(data["path"]))
        menu.addAction(rename_action)
        
        delete_action = QAction(Lang.get('delete_file'), self)
        delete_action.triggered.connect(lambda: self._delete_tree_file(data["path"]))
        menu.addAction(delete_action)
        
        menu.exec_(self.tree.viewport().mapToGlobal(position))

    def _delete_tree_file(self, file_path):
        file_name = os.path.basename(file_path)
        reply = QMessageBox.question(
            self, 
            Lang.get('confirm_delete'), 
            f"{Lang.get('delete_prompt')}\n\n{file_name}",
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
            Lang.get('rename_title'), 
            Lang.get('rename_prompt'), 
            QLineEdit.Normal, 
            old_name
        )
        
        if ok and new_name and new_name != old_name:
            new_path = old_path.parent / new_name
            
            if new_path.exists():
                QMessageBox.warning(self, Lang.get('error'), Lang.get('file_exists'))
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
                Lang.get('process_running'),
                Lang.get('process_running_prompt'),
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
    app.setApplicationName("OpenCap Offline")
    app.setOrganizationName("OpenCap")
    
    # Create and show main window
    window = OpenCapPro()
    window.show()
    
    logger.info("Application started")
    
    # Run event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
