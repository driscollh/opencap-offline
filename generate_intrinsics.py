# ==============================================================================
# File: check_imports.py
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
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        tags = data.get('format', {}).get('tags', {})
        model = tags.get('com.apple.quicktime.model') or tags.get('model')
        return model if model else "Unknown_iPhone"
    except:
        return "Unknown_iPhone"

def calibrate_session(session_name):
    print(f"%%STATUS: Starting Intrinsics Calibration for {session_name}...")
    
    # Setup paths
    session_path = os.path.join(APP_PATH, "Data", session_name)
    meta_path = os.path.join(session_path, "sessionMetadata.yaml")
    library_path = os.path.join(APP_PATH, "CustomCameraIntrinsics")
    
    # 1. LOAD THE YAML FILE
    if not os.path.exists(meta_path):
        print(f"  [ERROR] Metadata not found at {meta_path}")
        return

    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)
    
    # 2. EXTRACT DYNAMIC PARAMETERS
    try:
        rows = meta['checkerBoard']['black2BlackCornersHeight_n']
        cols = meta['checkerBoard']['black2BlackCornersWidth_n']
        square_size = meta['checkerBoard']['squareSideLength_mm']
    except KeyError as e:
        print(f"  [ERROR] Missing key in YAML: {e}")
        return
    
    board_dims = (cols, rows) # OpenCV expects (width_corners, height_corners)
    print(f"  [INFO] Using board: {cols}x{rows} corners, {square_size}mm squares.")

    # 3. PREPARE OBJECT POINTS (Now using dynamic square_size)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    cams = sorted(glob.glob(os.path.join(session_path, "Videos", "Cam*")))
    total_cams = len(cams)
    
    for i, cf in enumerate(cams):
        cam_name = os.path.basename(cf)
        print(f"\nProcessing {cam_name}...")
        
        input_dir = os.path.join(cf, "InputMedia", "Intrinsics")
        video_files = [f for f in glob.glob(os.path.join(input_dir, "*.*")) 
                       if f.lower().endswith(('.mov', '.mp4', '.avi'))]
        
        if not video_files:
            print(f"  [SKIP] No videos found in {input_dir}")
            continue

        model_name = get_iphone_model(video_files[0])
        print(f"  [INFO] Detected Model: {model_name}")
        
        # Store model in metadata
        if 'iphoneModel' not in meta: meta['iphoneModel'] = {}
        meta['iphoneModel'][cam_name] = model_name

        imgpoints, objpoints = [], []
        final_size = None
        total_valid_frames = 0
        
        for vid_idx, video_path in enumerate(video_files):
            print(f"  [READ] Reading video {vid_idx+1}/{len(video_files)}: {os.path.basename(video_path)}")
            cap = cv2.VideoCapture(video_path)
            frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract up to 30 frames spread across the video
            indices = np.linspace(0, frames_total-1, 30, dtype=int)
            
            for f_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret: continue
                
                if final_size is None: final_size = (frame.shape[1], frame.shape[0])
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
                
                # Using dynamic board_dims from YAML
                ret, corners = cv2.findChessboardCornersSB(gray, board_dims, flags=flags)
                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    total_valid_frames += 1
            cap.release()

        if total_valid_frames < 10:
            print(f"  [FAIL] Not enough valid frames ({total_valid_frames}) for {cam_name}.")
            continue
            
        print(f"  [CALC] Solving with {total_valid_frames} frames...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, final_size, None, None)
        
        data = {
            'intrinsicMat': mtx, 
            'distortion': dist, 
            'imageSize': np.array(final_size).reshape(2,1), 
            'reprojection_error': ret
        }

        # Save locally to Cam folder
        local_save = os.path.join(cf, "cameraIntrinsics.pickle")
        with open(local_save, 'wb') as f:
            pickle.dump(data, f)
        
        # Save to global library
        lib_folder = os.path.join(library_path, model_name)
        os.makedirs(lib_folder, exist_ok=True)
        lib_save = os.path.join(lib_folder, "cameraIntrinsics.pickle")
        with open(lib_save, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"  [SUCCESS] Saved intrinsics for {cam_name}.")
        
        progress = ((i + 1) / total_cams) * 100
        print(f"%%PROGRESS: {progress}")

    # Write updated metadata back to YAML
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f)
        
    print("%%STATUS: Intrinsics Complete.")