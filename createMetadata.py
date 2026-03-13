# ==============================================================================
# File: createMetadata.py
# Author: Harry G. Driscoll
# Date: Jan 2026
#
# License: Distributed under the Apache 2.0 License
# ==============================================================================

import os
import yaml

def create_metadata(session_dir, subject_id, height, mass, tag, rows=4, cols=5, size=35.0):
    """
    Writes sessionMetadata.yaml following the user's specific logic.
    """
    # Find created camera folders
    cam_folders = [d for d in os.listdir(session_dir) if d.startswith('Cam')]
    cam_folders.sort()
    iphone_models = {cam: "iPhone_Unknown" for cam in cam_folders}

    metadata = {
        'augmentermodel': 'v0.3',
        'calibrationSettings': {
            'overwriteDeployedIntrinsics': False,
            'saveSessionIntrinsics': False
        },
        'checkerBoard': {
            'black2BlackCornersHeight_n': rows,
            'black2BlackCornersWidth_n': cols,
            'placement': 'Perpendicular',
            'squareSideLength_mm': size
        },
        'filterfrequency': 'default',
        'gender_mf': None,
        'height_m': height,
        'iphoneModel': iphone_models,
        'markerAugmentationSettings': {
            'markerAugmenterModel': 'LSTM'
        },
        'mass_kg': mass,
        'subject_tags': tag,
        'openSimModel': 'LaiUhlrich2022',
        'posemodel': 'hrnet', 
        'scalingsetup': 'upright_standing_pose',
        'subjectID': subject_id
    }

    output_path = os.path.join(session_dir, 'sessionMetadata.yaml')
    with open(output_path, 'w') as file:
        yaml.dump(metadata, file, default_flow_style=False, sort_keys=False)