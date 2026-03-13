# ==============================================================================
# File: create_session.py
# Author: Harry G. Driscoll
# Date: Jan 2026
#
# License: Distributed under the Apache 2.0 License
# ==============================================================================

import os
import sys
import yaml

def create_session_structure_args(session_name, n_cams, mass, height):
    print(f"--- Creating Session: {session_name} ---")
    
    # --- FIX: DETECT EXE LOCATION ---
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
    session_dir = os.path.join(base_dir, "Data", session_name)
    
    if os.path.exists(session_dir):
        print(f"[WARNING] Session '{session_name}' already exists. Skipping creation.")
        return

    # 2. Create Folder Structure
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(os.path.join(session_dir, "CalibrationImages"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "MarkerData"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "OpenSimData"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "VisualizerJsons"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "VisualizerVideos"), exist_ok=True)

    iphone_models = {}
    for i in range(n_cams):
        cam_name = f"Cam{i}"
        cam_path = os.path.join(session_dir, "Videos", cam_name)
        
        os.makedirs(os.path.join(cam_path, "InputMedia", "Intrinsics"), exist_ok=True)
        os.makedirs(os.path.join(cam_path, "InputMedia", "calibration"), exist_ok=True)
        os.makedirs(os.path.join(cam_path, "InputMedia", "neutral"), exist_ok=True)
        
        # Placeholder for dynamic trial
        os.makedirs(os.path.join(cam_path, "InputMedia", "Dynamic_1"), exist_ok=True)

        iphone_models[cam_name] = "Unknown_iPhone"

    # 3. Generate Metadata YAML
    metadata = {
        'mass_kg': mass, 'height_m': height, 'openSimModel': 'LaiUhlrich2022',
        'augmentermodel': 'v0.3', 'filterfrequency': 'default',
        'checkerBoard': {
            'black2BlackCornersWidth_n': 4, 'black2BlackCornersHeight_n': 5,
            'squareSideLength_mm': 35, 'placement': 'backWall'
        },
        'iphoneModel': iphone_models,
        'markerAugmentationSettings': {'markerAugmenterModel': 'v0.3'}
    }
    
    with open(os.path.join(session_dir, "sessionMetadata.yaml"), 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"[SUCCESS] Session created at {session_dir}")