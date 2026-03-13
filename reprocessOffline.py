import os
import sys
import json
import glob
import utils
import pickle
import re
import shutil
import subprocess
import numpy as np
import cv2 
import yaml
import argparse
import traceback

# Get the absolute path to your portable folder
base_path = os.path.dirname(os.path.abspath(__file__))

# Define the hidden path where your DLLs actually live
dll_path = os.path.join(base_path, 'python_env', 'Library', 'bin')

if os.path.exists(dll_path):
    print(f"[GPU CONFIG] Found CUDA libraries at: {dll_path}")
    # This is the critical line for Windows 10/11
    os.add_dll_directory(dll_path)
else:
    print(f"[WARNING] Could not find DLL folder at {dll_path}")

# Force TensorFlow to see the local path for any legacy subprocesses
os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']

# --- DYNAMIC CONFIGURATION (Argparse for GUI Inputs) -----------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Session folder name in Data/")
    parser.add_argument("--gpu_index", default="0", help="CUDA device index")
    parser.add_argument("--resolution", default="1x736", help="Net resolution for OpenPose")
    parser.add_argument("--trials", nargs='+', required=True, help="List of trials from GUI")
    return parser.parse_args()

args = get_args()

# --- REPLICATED CONSTANTS & OVERRIDES -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
OPENPOSE_GPU_START = 0 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = CURRENT_DIR # The script appends 'Data' later
OPENPOSE_ROOT_DIR = os.path.join(CURRENT_DIR, "dependencies", "openpose")
INTRINSICS_LIBRARY_ROOT = os.path.join(CURRENT_DIR, "CameraIntrinsics") 

# 2. Tell Windows/Python to trust and search this folder for DLLs
if sys.platform == 'win32':
    # This is the modern way to load local DLLs (cudart, cublas, etc.)
    os.add_dll_directory(CURRENT_DIR)
    
    # Also check the python_env bin if they are hidden in there
    env_bin = os.path.join(CURRENT_DIR, 'python_env', 'Scripts')
    if os.path.exists(env_bin):
        os.add_dll_directory(env_bin)

# 3. Optional: Prevent TensorFlow from hogging all VRAM immediately
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Replicated Path Overrides
utils.getDataDirectory = lambda isDocker=False: CURRENT_DIR
from main import main 

SESSION_NAME = args.session
RESOLUTION = args.resolution
CALIB_SOURCE_TRIAL = 'calibration' 

# Define paths
SESSION_PATH = os.path.join(base_path, 'Data', SESSION_NAME)
metadata_path = os.path.join(SESSION_PATH, 'sessionMetadata.yaml')

# 1. Load the metadata once at the start
metadata = {}
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)

# 2. Build TRIALS based ONLY on the selected args.trials
TRIALS = []
for t_name in args.trials:
    # Default values
    t_type = 'dynamic'
    t_id = t_name
    
    # Check if this specific trial name exists in metadata to get its real type/id
    if metadata and 'trials' in metadata:
        for m_id, m_info in metadata['trials'].items():
            if m_info['name'] == t_name:
                t_type = m_info['type']
                t_id = m_id
                break
    
    # Force 'neutral' to be static regardless of metadata typos
    if t_name.lower() == 'neutral':
        t_type = 'static'
        
    TRIALS.append({'name': t_name, 'type': t_type, 'id': t_id})

# Sort TRIALS so that 'neutral' always comes first.
# This works by checking if the name is 'neutral' (True becomes 0, False becomes 1)
TRIALS.sort(key=lambda x: 0 if x['name'].lower() == 'neutral' else 1)

# 3. Print Name and Type for confirmation
print(f"[CONFIG] Trials to process: {[{'name': t['name'], 'type': t['type']} for t in TRIALS]}")

# -----------------------------------------------------------------------------
# DYNAMIC METADATA LOADING
# -----------------------------------------------------------------------------
session_path = os.path.join(CURRENT_DIR, 'Data', SESSION_NAME)
metadata_path = os.path.join(session_path, "sessionMetadata.yaml")

if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        meta = yaml.safe_load(f)
    # Pulling Board Dims dynamically from the GUI-generated metadata
    SQUARE_SIZE_MM = meta['checkerBoard']['squareSideLength_mm']
    BOARD_DIMS = (meta['checkerBoard']['black2BlackCornersWidth_n'], 
                  meta['checkerBoard']['black2BlackCornersHeight_n'])
    print(f" [METADATA] Loaded Board: {BOARD_DIMS}, Size: {SQUARE_SIZE_MM}mm")
else:
    # Fallback to manual defaults if file is missing
    SQUARE_SIZE_MM = 35       
    BOARD_DIMS = (5, 4)       

# -----------------------------------------------------------------------------
# REPLICATED HELPER FUNCTIONS (Exact Parity)
# -----------------------------------------------------------------------------

def load_custom_intrinsics(cam_folder, model_tag=None):
    custom_path = os.path.join(cam_folder, "cameraIntrinsics.pickle")
    if os.path.exists(custom_path):
        print(f"    [OVERRIDE] Using CUSTOM intrinsics found in {os.path.basename(cam_folder)}")
        with open(custom_path, 'rb') as f: return pickle.load(f)

    if not model_tag: return None
    target_folder = os.path.join(INTRINSICS_LIBRARY_ROOT, model_tag, "Deployed_720_60fps")
    pickle_path = os.path.join(target_folder, "cameraIntrinsics.pickle")
    
    if not os.path.exists(pickle_path):
        target_folder = os.path.join(INTRINSICS_LIBRARY_ROOT, model_tag, "Deployed")
        pickle_path = os.path.join(target_folder, "cameraIntrinsics.pickle")
        
    if not os.path.exists(pickle_path): return None
    
    with open(pickle_path, 'rb') as f: return pickle.load(f)

def visualize_calibration(img, corners, rvec, tvec, intrinsics, save_path):
    if corners is not None:
        for c in corners: cv2.circle(img, (int(c[0,0]), int(c[0,1])), 3, (0, 0, 255), -1)

    objp = np.zeros((BOARD_DIMS[0]*BOARD_DIMS[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:BOARD_DIMS[0], 0:BOARD_DIMS[1]].T.reshape(-1,2) * SQUARE_SIZE_MM
    mtx, dist = intrinsics['intrinsicMat'], intrinsics['distortion']
    imgpts, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    for p in imgpts: cv2.circle(img, (int(p[0,0]), int(p[0,1])), 5, (255, 255, 0), 1)
    
    cv2.imwrite(save_path, img)

def run_auto_calibration(session_path):
    print(f"\n--- [Step 0/3] Auto-Calibration (Extrinsics) ---")
    vid_dir = os.path.join(session_path, 'Videos')
    calib_out = os.path.join(session_path, 'CalibrationImages')
    os.makedirs(calib_out, exist_ok=True)

    for cf in sorted(glob.glob(os.path.join(vid_dir, 'Cam*'))):
        print(f"  > Calibrating {os.path.basename(cf)}...")
        input_dir = os.path.join(cf, 'InputMedia', CALIB_SOURCE_TRIAL)
        
        # Look for existing sanitized MP4 first
        vpath = None
        for ext in ['.mp4', '.mov', '.avi']:
            cand = os.path.join(input_dir, f"{CALIB_SOURCE_TRIAL}{ext}")
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
        
        if not ret:
            print("    [FAIL] Checkerboard NOT detected in first frame.")
            continue
            
        objp = np.zeros((BOARD_DIMS[0]*BOARD_DIMS[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:BOARD_DIMS[0], 0:BOARD_DIMS[1]].T.reshape(-1,2) * SQUARE_SIZE_MM
        
        mtx, dist = intrinsics['intrinsicMat'], intrinsics['distortion']
        ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

        visualize_calibration(frame.copy(), corners, rvec, tvec, intrinsics, os.path.join(calib_out, f'{os.path.basename(cf)}_calib.jpg'))

        R_matrix, _ = cv2.Rodrigues(rvec)
        t_col = np.array(tvec).reshape(3, 1) 
        
        img_size = np.array(intrinsics.get('imageSize', (frame.shape[1], frame.shape[0]))).reshape(2, 1)
        euler = np.zeros((3,1)) 

        with open(os.path.join(cf, 'cameraIntrinsicsExtrinsics.pickle'), 'wb') as f:
            pickle.dump({
                'intrinsicMat': mtx, 'distortion': dist, 'imageSize': img_size,
                'rotation': R_matrix, 'translation': t_col, 
                'rotation_EulerAngles': euler, 'R': R_matrix, 't': t_col
            }, f)
        print(f"    [SUCCESS] Saved calibration.")

def sanitize_video_file(session_path, trial_name):
    """
    1. Moves RAW videos to a '_Backup' folder so OpenCap can't see them.
    2. Generates a strict 60fps MP4 with 0.00 start timestamp.
    """
    print(f"\n--- [Step 1/3] Video Sanitization ({trial_name}) ---")
    for cf in glob.glob(os.path.join(session_path, 'Videos', 'Cam*')):
        input_dir = os.path.join(cf, 'InputMedia', trial_name)
        backup_dir = os.path.join(input_dir, "_Backup")
        os.makedirs(backup_dir, exist_ok=True)
        
        mp4_path = os.path.join(input_dir, f"{trial_name}.mp4")

        # 1. Identify Input Candidate (MOV, AVI, or existing MP4)
        raw_vid = None
        for ext in ['.mov', '.avi', '.mp4']:
            cand = os.path.join(input_dir, f"{trial_name}{ext}")
            if os.path.exists(cand) and "_Backup" not in cand:
                # Avoid re-processing the clean MP4 if it's the only one left
                if cand == mp4_path:
                    # Check if we should re-process (e.g. if it's the only file)
                    continue 
                raw_vid = cand
                break
        
        # If we have a raw video that isn't the final MP4, process it
        if raw_vid:
            print(f"    [PROCESSING] Found raw file: {os.path.basename(raw_vid)}")
            
            # Move raw file to backup immediately
            backup_path = os.path.join(backup_dir, os.path.basename(raw_vid))
            try:
                shutil.move(raw_vid, backup_path)
                print(f"    [MOVE] Moved raw file to {backup_dir}")
            except Exception as e:
                print(f"    [ERROR] Could not move file: {e}")
                continue

            # Generate Clean MP4 from the BACKUP file
            # -fflags +genpts -reset_timestamps 1: Forces timestamp reset to 0
            cmd = [
                'ffmpeg', '-y', '-i', backup_path, 
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                '-r', '60', 
                '-fflags', '+genpts', 
                '-reset_timestamps', '1',
                '-video_track_timescale', '60000', 
                mp4_path
            ]
            
            try: 
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"    [CLEAN] Generated clean MP4: {os.path.basename(mp4_path)}")
            except: 
                print(f"    [ERROR] FFmpeg failed. Restoring backup.")
                shutil.move(backup_path, raw_vid)

def generate_neutral_thumbnails(session_path, trial_name):
    if trial_name != 'neutral': return
    print(f"\n--- [Helper] Generating Neutral Pose Thumbnails ---")
    thumb_dir = os.path.join(session_path, 'NeutralPoseImages')
    os.makedirs(thumb_dir, exist_ok=True)
    
    vid_dir = os.path.join(session_path, 'Videos')
    for i, cf in enumerate(sorted(glob.glob(os.path.join(vid_dir, 'Cam*')))):
        input_vid = os.path.join(cf, 'InputMedia', trial_name, f"{trial_name}.mp4")
        if not os.path.exists(input_vid): continue
        
        cap = cv2.VideoCapture(input_vid)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            cam_name = os.path.basename(cf) 
            out_path = os.path.join(thumb_dir, f"{cam_name}_0.png")
            cv2.imwrite(out_path, frame)
            print(f"    [OK] Saved {out_path}")

def run_openpose_direct(session_path, trial_name, res_override):
    print(f"\n--- [Step 2/3] Running OpenPose ---")
    bin_path = os.path.join(OPENPOSE_ROOT_DIR, 'bin', 'OpenPoseDemo.exe')
    for cf in sorted(glob.glob(os.path.join(session_path, 'Videos', 'Cam*'))):
        input_vid = os.path.join(cf, 'InputMedia', trial_name, f"{trial_name}.mp4")
        
        if not os.path.exists(input_vid): 
            print(f"    [SKIP] No video found for {os.path.basename(cf)}")
            continue
            
        json_out = os.path.join(cf, f'OutputJsons_{res_override}', trial_name)
        os.makedirs(json_out, exist_ok=True)
        
        # Output overlay video folder
        video_out_dir = os.path.join(cf, f'OutputVideos_{res_override}', trial_name)
        os.makedirs(video_out_dir, exist_ok=True)
        video_out_path = os.path.join(video_out_dir, f"{trial_name}_overlay.avi")
        
        cmd = [
            bin_path, 
            '--video', input_vid, 
            '--write_json', json_out, 
            '--display', '0', 
            '--render_pose', '1', 
            '--write_video', video_out_path,
            '--num_gpu_start', str(OPENPOSE_GPU_START)
        ]

        if res_override != 'default':
            # 1. Parse base resolution (e.g. "1x736")
            base_res = res_override.split('_')[0]
            cmd.extend(['--net_resolution', f"-{base_res}"])
            
            # 2. Check for multi-scale request (e.g. "_2scales")
            if '_2scales' in res_override:
                print(f"    [CONFIG] Enabling 2-Scale Processing (High Accuracy)")
                cmd.extend(['--scale_number', '2', '--scale_gap', '0.75'])
            elif '_4scales' in res_override: # If you ever want extreme accuracy
                cmd.extend(['--scale_number', '4', '--scale_gap', '0.25'])
            
        try: 
            print(f"    [RUNNING] OpenPose for {os.path.basename(cf)}...")
            subprocess.run(cmd, cwd=OPENPOSE_ROOT_DIR, check=True)
            print(f"    [SAVED] Video: {video_out_path}")
        except: 
            print(f"   [ERROR] OpenPose Crashed on {os.path.basename(cf)}")

def generate_missing_pickles(session_path, trial_name, res_override, is_static=False):
    print(f"\n--- [Step 3/3] Helper Files (Static Freeze: {is_static}) ---")
    for cf in sorted(glob.glob(os.path.join(session_path, 'Videos', 'Cam*'))):
        json_dir = os.path.join(cf, f'OutputJsons_{res_override}', trial_name)
        pkl_out = os.path.join(cf, f'OutputPkl_{res_override}', trial_name)
        os.makedirs(pkl_out, exist_ok=True)
        
        files = sorted(glob.glob(os.path.join(json_dir, "*.json")), key=lambda f: int(re.findall(r'(\d+)_keypoints', f)[-1]) if re.findall(r'(\d+)_keypoints', f) else 0)
        frames_list = []
        for jf in files:
            try:
                with open(jf) as f: d = json.load(f)
                frames_list.append(d['people'] if 'people' in d and d['people'] else [{'pose_keypoints_2d': [0]*75}])
            except: frames_list.append([{'pose_keypoints_2d': [0]*75}])
        
        if is_static and frames_list:
            kp_array = []
            for frame in frames_list: kp_array.append(frame[0].get('pose_keypoints_2d', [0]*75))
            median_kp = np.median(np.array(kp_array), axis=0).tolist()
            frames_list = [[{'pose_keypoints_2d': median_kp}]] * len(frames_list)

        if len(frames_list) == 1: frames_list.append(frames_list[0]) 
        
        # Restore filename pattern
        with open(os.path.join(pkl_out, f"{trial_name}_rotated_pp.pkl"), 'wb') as f: pickle.dump(frames_list, f)

def ensure_session_resources(session_path, trials, cres):
    """
    Checks for required folders and dynamically detects camera names 
    from the directory structure.
    """
    # 1. Detect Camera Folders dynamically
    video_root = os.path.join(session_path, 'Videos')
    if not os.path.exists(video_root):
        os.makedirs(video_root, exist_ok=True)
        print(f"[FAILSAFE] Created root video directory: {video_root}")
        return # Cannot detect cameras if the folder was just created and is empty
        
    camera_names = [d for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d)) and d.startswith('Cam')]
    print(f"[CHECK] Detected cameras: {camera_names}")

    # 2. Define Core directories required by main.py and utils
    required_dirs = [
        os.path.join(session_path, 'MarkerData', 'PostAugmentation'),
        os.path.join(session_path, 'MarkerData', 'PreAugmentation'),
        os.path.join(session_path, 'OpenSimData', 'Model'),
        os.path.join(session_path, 'OpenSimData', 'Kinematics'),
        os.path.join(session_path, 'VisualizerJsons')
    ]

    # 3. Add dynamic trial-specific subdirectories for the Visualizer
    for trial in trials:
        required_dirs.append(os.path.join(session_path, 'VisualizerJsons', trial['name']))

    # 4. Add camera-specific processing subdirectories
    for cam in camera_names:
        cam_path = os.path.join(video_root, cam)
        required_dirs.append(os.path.join(cam_path, 'InputMedia'))
        required_dirs.append(os.path.join(cam_path, f'OutputPkl_{cres}'))
        required_dirs.append(os.path.join(cam_path, f'OutputJsons_{cres}'))

    # 5. Execute creation
    for folder in required_dirs:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"[FAILSAFE] Created missing directory: {folder}")

# -----------------------------------------------------------------------------
# MAIN OFFLINE PIPELINE EXECUTION
# -----------------------------------------------------------------------------

def run_offline_pipeline():
    session_path = os.path.join(BASE_DATA_DIR, 'Data', SESSION_NAME)
    
    if not os.path.exists(session_path):
        print(f"ERROR: Session not found: {session_path}")
        return

    cres = RESOLUTION
    ensure_session_resources(session_path, TRIALS, cres)
     
    # 1. Extrinsics
    run_auto_calibration(session_path) 
    
    # 2. Main Processing
    for trial in TRIALS:
         sanitize_video_file(session_path, trial['name'])
         generate_neutral_thumbnails(session_path, trial['name'])
         run_openpose_direct(session_path, trial['name'], cres)
         
         # FIX: Dynamically check if this is the neutral/static trial
         is_static_trial = (trial['type'] == 'static' or trial['name'].lower() == 'neutral')
         generate_missing_pickles(session_path, trial['name'], cres, is_static=is_static_trial)

    print(f"\n================ STARTING OPENCAP PIPELINE ================\n")

    # --- ADD THE OVERRIDE HERE ---
    import utilsAugmenter
    # This forces the pipeline to look in your LSTM folder instead of v0.3_lower
    lstm_path = os.path.join(CURRENT_DIR, 'MarkerAugmenter', 'LSTM')
    
    # We must ensure the folder exists before point the tool to it
    if os.path.exists(lstm_path):
        print(f"[CONFIG] Pointing MarkerAugmenter to: {lstm_path}")
        # This global variable in utilsAugmenter is what main.py uses
        utilsAugmenter.EXTERNAL_AUGMENTER_MODEL_DIR = lstm_path 
    else:
        print(f"[WARNING] LSTM folder not found at {lstm_path}!")
    # -----------------------------

    for trial in TRIALS:
        try:
            print(f"\n--- Processing Trial: {trial['name']} ({trial['type']}) ---")
            
            is_static_trial = (trial['type'] == 'static')
            
            main(SESSION_NAME, trial['name'], trial['id'], 
                 isDocker=False, 
                 extrinsicsTrial=False, 
                 poseDetector='OpenPose', 
                 imageUpsampleFactor=1, 
                 scaleModel=is_static_trial, 
                 resolutionPoseDetection=cres, 
                 genericFolderNames=False, 
                 bbox_thr=0.8, 
                 calibrationOptions=None) 
            
            print(f"[SUCCESS] Finished {trial['name']}")
        except Exception as e: 
            print(f"[FAILED] {trial['name']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    run_offline_pipeline()