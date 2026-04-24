# ==============================================================================
# File: run_pipeline.py
# Author: Harry G. Driscoll
# Date: Jan 2026
#
# License: Distributed under the Apache 2.0 License
# ==============================================================================

import os
import sys
import glob
import pickle
import shutil
import subprocess
import numpy as np
import cv2 
import yaml
import traceback
import logging
import argparse
import re
import generate_intrinsics

# --- 1. SETUP PATHS ---
APP_PATH = os.path.dirname(os.path.abspath(__file__))
OPENPOSE_ROOT = os.path.join(APP_PATH, "dependencies", "openpose")
FFMPEG_DIR = os.path.join(APP_PATH, "dependencies", "ffmpeg", "bin")
LOG_PATH = os.path.join(APP_PATH, "pipeline_debug.log")

# Ensure FFmpeg is in path
if FFMPEG_DIR not in os.environ["PATH"]:
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ["PATH"]

# --- 2. LOGGING SETUP ---
# Redirect logging to both file and GUI console
logging.basicConfig(level=logging.INFO, format='%(message)s', filename=LOG_PATH, filemode='w')
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# --- 3. IMPORTS ---
# Override utils.getDataDirectory to point to APP_PATH before importing main
import utils
utils.getDataDirectory = lambda isDocker=False: APP_PATH

import utilsSync  
import utilsChecker 
import utilsAugmenter 

# --- 4. APPLY LOGIC PATCHES (Safety Nets) ---
# We keep these patches because the GUI environment is fragile.
# They won't interfere if the data is good (as proved by reprocessOffline).

_original_findOverlap = utilsSync.findOverlap
_original_cross_corr = utilsSync.cross_corr
_original_cross_corr_multi = utilsSync.cross_corr_multiple_timeseries

CURRENT_TRIAL_IS_STATIC = False

def smart_findOverlap(confidenceList, markers4VertVel):
    # Try normal detection first (matches your personal code)
    overlapInds, minLength = _original_findOverlap(confidenceList, markers4VertVel)
    
    # If successful, return result
    if np.any(overlapInds): 
        return overlapInds, minLength
    
    # If failed AND it's a static trial, FORCE success (Safety Net)
    if CURRENT_TRIAL_IS_STATIC:
        print("   > [SmartSync] Static Trial: Forcing overlap detection.")
        min_frames = min([c.shape[1] for c in confidenceList])
        return np.arange(min_frames), min_frames
        
    return overlapInds, minLength

def smart_cross_corr(*args, **kwargs):
    if CURRENT_TRIAL_IS_STATIC: return 1.0, 0
    return _original_cross_corr(*args, **kwargs)

def smart_cross_corr_multi(*args, **kwargs):
    if CURRENT_TRIAL_IS_STATIC: return 1.0, 0
    return _original_cross_corr_multi(*args, **kwargs)

utilsSync.findOverlap = smart_findOverlap
utilsSync.cross_corr = smart_cross_corr
utilsSync.cross_corr_multiple_timeseries = smart_cross_corr_multi

# [Patch B] Augmenter Model Fix
def apply_augmenter_hotfix():
    target_file = os.path.join(APP_PATH, "utilsAugmenter.py")
    if not os.path.exists(target_file): return
    try:
        with open(target_file, 'r') as f: lines = f.readlines()
        new_lines = []
        modified = False
        for line in lines:
            if "outputs = model.predict(inputs, verbose=2)" in line and "np.expand_dims" not in line:
                indent = line[:line.find("outputs")]
                new_lines.append(f"{indent}if inputs.ndim == 2: inputs = np.expand_dims(inputs, axis=0)\n")
                new_lines.append(line)
                new_lines.append(f"{indent}if outputs.ndim == 3: outputs = np.squeeze(outputs, axis=0)\n")
                modified = True
            else:
                new_lines.append(line)
        if modified:
            with open(target_file, 'w') as f: f.writelines(new_lines)
    except: pass

apply_augmenter_hotfix()

# [Patch C] Dynamic Model Locator
def find_real_model_dir():
    search_root = os.path.join(APP_PATH, "MarkerAugmenter")
    for root, dirs, files in os.walk(search_root):
        if "metadata.json" in files: return root
    return None

REAL_MODEL_DIR = find_real_model_dir()
_original_join = utilsAugmenter.os.path.join

def smart_path_join(*args):
    if REAL_MODEL_DIR and len(args) > 0:
        if args[-1] in ["metadata.json", "mean.npy", "std.npy"] or args[-1].endswith(".h5"):
            return _original_join(REAL_MODEL_DIR, args[-1])
    return _original_join(*args)

if REAL_MODEL_DIR: utilsAugmenter.os.path.join = smart_path_join

# --- 5. IMPORT MAIN ---
from main import main 

# --- 6. GUI-SAFE RUN COMMAND ---
if sys.platform == 'win32':
    _startupinfo = subprocess.STARTUPINFO()
    _startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    _startupinfo.wShowWindow = subprocess.SW_HIDE
else:
    _startupinfo = None

def run_command(cmd, cwd=None):
    """
    Executes a subprocess safely. 
    NOTE: Unlike reprocessOffline (which prints to real console), 
    we pipe output here so it appears in the GUI window.
    """
    try:
        process = subprocess.Popen(
            cmd, 
            cwd=cwd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            stdin=subprocess.DEVNULL,
            text=True, 
            startupinfo=_startupinfo,
            encoding='utf-8', 
            errors='replace'
        )
        
        for line in process.stdout:
            print(line, end='') 
        
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
    except Exception as e:
        print(f"   > [System] Subprocess Error: {e}")
        raise e

# --- 7. WORKFLOW FUNCTIONS (Matches reprocessOffline.py exactly) ---

def check_and_generate_intrinsics(session_path, session_name):
    meta_path = os.path.join(session_path, "sessionMetadata.yaml")
    if not os.path.exists(meta_path): return

    with open(meta_path, 'r') as f: meta = yaml.safe_load(f)
    
    iphone_models = meta.get('iphoneModel', {})
    needs_generation = False
    
    # Check flags
    for model in iphone_models.values():
        if model in ["Android_Generic", "GoPro_Generic", "iPhone_Auto_Detect", "Generic_Webcam"]:
            needs_generation = True
            break
            
    # Check missing files
    if not needs_generation:
        for cf in glob.glob(os.path.join(session_path, 'Videos', 'Cam*')):
            if not os.path.exists(os.path.join(cf, "cameraIntrinsics.pickle")):
                needs_generation = True; break

    if needs_generation:
        print(f"%%STATUS: Intrinsics missing or auto-detect requested. Running generation...")
        try:
            generate_intrinsics.calibrate_session(session_name)
        except Exception as e:
            print(f"!! INTRINSICS GENERATION FAILED: {e}")

def load_custom_intrinsics(cam_folder):
    custom_path = os.path.join(cam_folder, "cameraIntrinsics.pickle")
    if os.path.exists(custom_path):
        with open(custom_path, 'rb') as f: return pickle.load(f)
    return None

def visualize_calibration(img, corners, rvec, tvec, intrinsics, save_path, board_dims, square_size_mm):
    if corners is not None:
        for c in corners: cv2.circle(img, (int(c[0,0]), int(c[0,1])), 3, (0, 0, 255), -1)
    objp = np.zeros((board_dims[0]*board_dims[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1,2) * square_size_mm
    mtx, dist = intrinsics['intrinsicMat'], intrinsics['distortion']
    imgpts, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    for p in imgpts: cv2.circle(img, (int(p[0,0]), int(p[0,1])), 5, (255, 255, 0), 1)
    cv2.imwrite(save_path, img)

def run_auto_calibration(session_path):
    print("%%STATUS: Running Auto-Calibration...")
    vid_dir = os.path.join(session_path, 'Videos')
    calib_out = os.path.join(session_path, 'CalibrationImages')
    os.makedirs(calib_out, exist_ok=True)
    
    meta_path = os.path.join(session_path, "sessionMetadata.yaml")
    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)
        
    cols = meta['checkerBoard']['black2BlackCornersWidth_n']
    rows = meta['checkerBoard']['black2BlackCornersHeight_n']
    SQUARE_SIZE_MM = meta['checkerBoard']['squareSideLength_mm']
    BOARD_DIMS = (cols, rows)

    for cf in sorted(glob.glob(os.path.join(vid_dir, 'Cam*'))):
        input_dir = os.path.join(cf, 'InputMedia', 'calibration')
        vpath = None
        for ext in ['.mp4', '.mov', '.avi']:
            cand = os.path.join(input_dir, f"calibration{ext}")
            if os.path.exists(cand):
                vpath = cand
                break
        
        if not vpath: continue
        intrinsics = load_custom_intrinsics(cf)
        if not intrinsics: continue

        cap = cv2.VideoCapture(vpath)
        ret, frame = cap.read()
        cap.release()
        if not ret: continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ret, corners = cv2.findChessboardCornersSB(gray, BOARD_DIMS, flags=flags)
        
        if not ret: continue
            
        objp = np.zeros((BOARD_DIMS[0]*BOARD_DIMS[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:BOARD_DIMS[0], 0:BOARD_DIMS[1]].T.reshape(-1,2) * SQUARE_SIZE_MM
        
        mtx, dist = intrinsics['intrinsicMat'], intrinsics['distortion']
        ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

        visualize_calibration(frame.copy(), corners, rvec, tvec, intrinsics, os.path.join(calib_out, f'{os.path.basename(cf)}_calib.jpg'))

        R_matrix, _ = cv2.Rodrigues(rvec)
        t_col = np.array(tvec).reshape(3, 1) 
        img_size = np.array(intrinsics.get('imageSize', (frame.shape[1], frame.shape[0]))).reshape(2, 1)
        
        with open(os.path.join(cf, 'cameraIntrinsicsExtrinsics.pickle'), 'wb') as f:
            pickle.dump({
                'intrinsicMat': mtx, 'distortion': dist, 'imageSize': img_size,
                'rotation': R_matrix, 'translation': t_col, 
                'rotation_EulerAngles': np.zeros((3,1)), 'R': R_matrix, 't': t_col
            }, f)
    print("%%PROGRESS: 100")

def sanitize_video_file(session_path, trial_name):
    # Matches reprocessOffline.py exactly
    for cf in glob.glob(os.path.join(session_path, 'Videos', 'Cam*')):
        input_dir = os.path.join(cf, 'InputMedia', trial_name)
        backup_dir = os.path.join(input_dir, "_Backup")
        os.makedirs(backup_dir, exist_ok=True)
        mp4_path = os.path.join(input_dir, f"{trial_name}.mp4")
        
        raw_vid = None
        for ext in ['.mov', '.avi', '.mp4']:
            cand = os.path.join(input_dir, f"{trial_name}{ext}")
            if os.path.exists(cand) and "_Backup" not in cand:
                if cand == mp4_path: continue 
                raw_vid = cand
                break
        
        if raw_vid:
            backup_path = os.path.join(backup_dir, os.path.basename(raw_vid))
            shutil.move(raw_vid, backup_path)
            
            cmd = [
                'ffmpeg', '-y', '-i', backup_path, 
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                '-r', '60', 
                '-fflags', '+genpts', 
                '-reset_timestamps', '1',
                '-video_track_timescale', '60000', 
                mp4_path
            ]
            run_command(cmd)

def generate_neutral_thumbnails(session_path, trial_name, res_override):
    if trial_name != 'neutral': return
    thumb_dir = os.path.join(session_path, 'NeutralPoseImages', f'OpenPose_{res_override}')
    os.makedirs(thumb_dir, exist_ok=True)
    os.makedirs(thumb_dir, exist_ok=True)
    for cf in sorted(glob.glob(os.path.join(session_path, 'Videos', 'Cam*'))):
        input_vid = os.path.join(cf, 'InputMedia', trial_name, f"{trial_name}.mp4")
        if not os.path.exists(input_vid): continue
        cap = cv2.VideoCapture(input_vid)
        ret, frame = cap.read()
        cap.release()
        if ret: 
            cv2.imwrite(os.path.join(thumb_dir, f"{os.path.basename(cf)}_0.png"), frame)

def run_openpose_direct(session_path, trial_name, res_override, gpu_start_idx):
    bin_path = os.path.join(OPENPOSE_ROOT, 'bin', 'OpenPoseDemo.exe')
    for cf in sorted(glob.glob(os.path.join(session_path, 'Videos', 'Cam*'))):
        input_vid = os.path.join(cf, 'InputMedia', trial_name, f"{trial_name}.mp4")
        if not os.path.exists(input_vid): continue
        
        json_out = os.path.join(cf, f'OutputJsons_{res_override}', trial_name)
        os.makedirs(json_out, exist_ok=True)
        # Skip if already processed (check file count)
        if len(glob.glob(os.path.join(json_out, "*.json"))) > 10: continue
        
        video_out_path = os.path.join(cf, f'OutputVideos_{res_override}', trial_name, f"{trial_name}_overlay.avi")
        os.makedirs(os.path.dirname(video_out_path), exist_ok=True)
        
        cmd = [
            bin_path, '--video', input_vid, '--write_json', json_out, 
            '--display', '0', '--render_pose', '1', '--write_video', video_out_path, 
            '--num_gpu', '1', '--num_gpu_start', str(gpu_start_idx)
        ]
        if res_override != 'default':
            base_res = res_override.split('_')[0]
            cmd.extend(['--net_resolution', f"-{base_res}"])
            if '_2scales' in res_override: cmd.extend(['--scale_number', '2', '--scale_gap', '0.75'])
            
        print(f"%%STATUS: Running OpenPose for {trial_name}...")
        try: 
            run_command(cmd, cwd=OPENPOSE_ROOT)
        except subprocess.CalledProcessError as e: 
            print(f"!! OPENPOSE ERROR: {e}"); raise e

def generate_missing_pickles(session_path, trial_name, res_override, is_static=False, pose_estimator="openpose"):
    import re
    # Match the old script's printout
    print(f"\n--- [Step 3/3] Helper Files (Static Freeze: {is_static}) ---")
    for cf in sorted(glob.glob(os.path.join(session_path, 'Videos', 'Cam*'))):
        # The folder MUST match what you pass to main()
        suffix = res_override if pose_estimator == "openpose" else "RTMPose"
        json_dir = os.path.join(cf, f'OutputJsons_{suffix}', trial_name)
        pkl_out = os.path.join(cf, f'OutputPkl_{suffix}', trial_name)
        os.makedirs(pkl_out, exist_ok=True)
        
        files = sorted(glob.glob(os.path.join(json_dir, "*.json")), 
                       key=lambda f: int(re.findall(r'(\d+)_keypoints', f)[-1]) if re.findall(r'(\d+)_keypoints', f) else 0)
        
        if not files:
            print(f"    [WARNING] No JSONs found in {json_dir}")
            continue

        frames_list = []
        for jf in files:
            try:
                with open(jf) as f: d = json.load(f)
                frames_list.append(d['people'] if 'people' in d and d['people'] else [{'pose_keypoints_2d': [0]*75}])
            except: frames_list.append([{'pose_keypoints_2d': [0]*75}])
        
        if is_static and frames_list:
            kp_array = []
            for frame in frames_list: kp_array.append(frame[0].get('pose_keypoints_2d', [0]*75))
            median_kp = np.median(kp_array, axis=0).tolist()
            frames_list = [[{'pose_keypoints_2d': median_kp}]] * len(frames_list)

        if len(frames_list) == 1: frames_list.append(frames_list[0]) 
        
        # This filename is hardcoded in Stanford's utilsSync.py
        pkl_path = os.path.join(pkl_out, f"{trial_name}_rotated_pp.pkl")
        with open(pkl_path, 'wb') as f: 
            pickle.dump(frames_list, f)

# --- 8. EXECUTION ---
def run_session_pipeline(session_name, gpu_id="0", resolution="1x736"):
    global CURRENT_TRIAL_IS_STATIC
    print(f"%%STATUS: Pipeline Config: GPU {gpu_id}, Res {resolution}")
    
    try:
        session_path = os.path.join(APP_PATH, "Data", session_name)
        check_and_generate_intrinsics(session_path, session_name)
        cam0 = os.path.join(session_path, "Videos", "Cam0", "InputMedia")
        trials = []
        if os.path.exists(os.path.join(cam0, "calibration")): trials.append({'name':'calibration','type':'calibration','id':'calibration'})
        if os.path.exists(os.path.join(cam0, "neutral")): trials.append({'name':'neutral','type':'static','id':'neutral'})
        ignored = ['calibration', 'neutral', 'Intrinsics', '_Backup']
        for f in os.listdir(cam0):
            if f not in ignored and os.path.isdir(os.path.join(cam0, f)):
                trials.append({'name': f, 'type': 'dynamic', 'id': f})

        # 1. Extrinsics
        run_auto_calibration(session_path) 
        
        # 2. Process Trials
        for trial in trials:
            if trial['type'] == 'calibration': continue
            
            # --- UPDATE GLOBAL STATE ---
            CURRENT_TRIAL_IS_STATIC = (trial['type'] == 'static' or trial['name'] == 'neutral')
            
            print(f"%%STATUS: Processing trial: {trial['name']} (StaticMode={CURRENT_TRIAL_IS_STATIC})...")
            print("%%PROGRESS: 5")
            
            # Helper steps matching reprocessOffline.py
            sanitize_video_file(session_path, trial['name'])
            generate_neutral_thumbnails(session_path, trial['name'])
            run_openpose_direct(session_path, trial['name'], resolution, gpu_start_idx=gpu_id) 
            print("%%PROGRESS: 60")
            generate_missing_pickles(session_path, trial['name'], resolution, is_static=CURRENT_TRIAL_IS_STATIC)
            print("%%PROGRESS: 75")
            
            print(f"%%STATUS: Triangulating {trial['name']}...")
            
            # Main Pipeline Call (Matches reprocessOffline.py args)
            main(session_name, trial['name'], trial['id'], 
                 isDocker=False, 
                 extrinsicsTrial=False, 
                 poseDetector='OpenPose', 
                 imageUpsampleFactor=1, 
                 scaleModel=CURRENT_TRIAL_IS_STATIC, 
                 resolutionPoseDetection=resolution, 
                 genericFolderNames=False, 
                 bbox_thr=0.8,
                 calibrationOptions=None) # Start fresh
                 
            print("%%PROGRESS: 100")
            
        print("%%STATUS: Pipeline Complete.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--resolution", required=True)
    args = parser.parse_args()
    run_session_pipeline(args.session, args.gpu, args.resolution)