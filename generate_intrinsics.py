# ==============================================================================
# File: generate_intrinsics.py
# Author: Harry G. Driscoll
# Date: Jan 2026
#
# Modified from code originally in main.py
# authors: Scott Uhlrich, Antoine Falisse, Łukasz Kidziński
#
# License: Distributed under the Apache 2.0 License
# ==============================================================================

import os
import sys
import glob
import cv2
import numpy as np
import pickle
import json
import subprocess
import yaml

# --- SETUP PATHS GLOBAL ---
if getattr(sys, 'frozen', False):
    APP_PATH = os.path.dirname(sys.executable)
else:
    APP_PATH = os.path.dirname(os.path.abspath(__file__))

# Ensure FFmpeg is found for ffprobe calls
ffmpeg_bin = os.path.join(APP_PATH, "dependencies", "ffmpeg", "bin")
os.environ["PATH"] += os.pathsep + ffmpeg_bin

def get_iphone_model(video_path):
    """Probes video metadata to identify the phone model."""
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        data = json.loads(result.stdout)
        tags = data.get('format', {}).get('tags', {})
        model = tags.get('com.apple.quicktime.model') or tags.get('model')
        return model if model else "Unknown_iPhone"
    except Exception as e:
        print(f"  [WARNING] Could not read metadata for {video_path}: {e}")
        return "Unknown_iPhone"

def calibrate_camera(session_name, cam_name):
    """
    Performs intrinsics calibration for a single specific camera.
    This allows researchers to fix one camera without re-processing the entire session.
    """
    print(f"%%STATUS: Calibrating {cam_name} in {session_name}...")
    
    session_path = os.path.join(APP_PATH, "Data", session_name)
    meta_path = os.path.join(session_path, "sessionMetadata.yaml")
    library_path = os.path.join(APP_PATH, "CustomCameraIntrinsics")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}")

    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)
    
    # Extract board parameters from YAML
    rows = meta['checkerBoard']['black2BlackCornersHeight_n']
    cols = meta['checkerBoard']['black2BlackCornersWidth_n']
    square_size = meta['checkerBoard']['squareSideLength_mm']
    board_dims = (cols, rows) 

    # Prepare 3D object points based on dynamic square size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    cam_folder = os.path.join(session_path, "Videos", cam_name)
    input_dir = os.path.join(cam_folder, "InputMedia", "Intrinsics")
    video_files = [f for f in glob.glob(os.path.join(input_dir, "*.*")) 
                   if f.lower().endswith(('.mov', '.mp4', '.avi'))]
    
    if not video_files:
        print(f"  [SKIP] No videos found in {input_dir}")
        return False

    raw_model_name = get_iphone_model(video_files[0])
    unique_model_name = f"{raw_model_name}_{cam_name}"
    
    # Store camera-specific model in metadata
    if 'iphoneModel' not in meta: meta['iphoneModel'] = {}
    meta['iphoneModel'][cam_name] = unique_model_name

    imgpoints, objpoints = [], []
    final_size = None
    total_valid_frames = 0
    
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, frames_total-1, 30, dtype=int)
        
        for f_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            if final_size is None: final_size = (frame.shape[1], frame.shape[0])
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            
            # Detect chessboard corners using the SB (Sector Based) algorithm
            ret, corners = cv2.findChessboardCornersSB(gray, board_dims, flags=flags)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                total_valid_frames += 1
        cap.release()

    if total_valid_frames < 10:
        print(f"  [FAIL] Only {total_valid_frames} valid frames found for {cam_name}.")
        return False
        
    # Solve for Intrinsic Matrix and Distortion Coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, final_size, None, None)
    
    data = {
        'intrinsicMat': mtx, 
        'distortion': dist, 
        'imageSize': np.array(final_size).reshape(2,1), 
        'reprojection_error': ret
    }

    # Save to local session folder and global library
    for path in [os.path.join(cam_folder, "cameraIntrinsics.pickle"),
                 os.path.join(library_path, unique_model_name, "cameraIntrinsics.pickle")]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
    # Save the updated YAML with the new model tag
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f)
        
    print(f"  [SUCCESS] {cam_name} calibrated. Error: {ret:.4f}")
    return True

def calibrate_session(session_name):
    """Wrapper to calibrate all cameras found in the session."""
    session_path = os.path.join(APP_PATH, "Data", session_name)
    cams = sorted(glob.glob(os.path.join(session_path, "Videos", "Cam*")))
    
    for i, cf in enumerate(cams):
        cam_name = os.path.basename(cf)
        calibrate_camera(session_name, cam_name)
        print(f"%%PROGRESS: {((i + 1) / len(cams)) * 100}")
        
    print("%%STATUS: Full Session Intrinsics Complete.")