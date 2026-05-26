import os
import json
import glob
import numpy as np
import re

def generate_tracking_data(json_dir, video_path, output_json_path):
    """
    Scans a folder of pose JSONs, tracks subjects across frames using spatial centroids,
    and generates a tracking JSON for the GUI to consume.
    Incorporates a track retention window to prevent ID splitting from frame drops.
    
    Returns:
        bool: True if multiple subjects were detected at any point, False otherwise.
    """
    json_files = sorted(
        glob.glob(os.path.join(json_dir, "*.json")),
        key=lambda f: int(re.findall(r'(\d+)', os.path.basename(f))[-1]) if re.findall(r'(\d+)', os.path.basename(f)) else 0
    )

    if not json_files:
        print(f"[TRACKER] Error: No JSON files found in {json_dir}")
        return False

    total_frames = len(json_files)
    tracks_out = {}
    
    # Track structure: id -> {"centroid": [x, y], "lost_frames": int}
    active_tracks = {} 
    next_id = 0
    max_dist_threshold = 200.0 
    max_lost_frames = 15 # Retain track IDs across up to 15 frames of missing detections

    for frame_idx, jf in enumerate(json_files):
        with open(jf, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            people = data.get('people', [])
        elif isinstance(data, list):
            people = data
        else:
            people = []
            
        current_frame_tracks = []
        detections = []
        
        # --- A. Extract Data for all people in this frame ---
        for p in people:
            if isinstance(p, dict):
                kp = p.get('pose_keypoints_2d', [])
            elif isinstance(p, list):
                kp = p
            else:
                kp = []
                
            if not kp or len(kp) == 0: continue
            
            kp_array = np.array(kp).reshape(-1, 3)
            # Filter for valid keypoints (confidence > 0.1)
            valid_kp = kp_array[kp_array[:, 2] > 0.1]
            if len(valid_kp) < 20: continue # Enforce minimum keypoint count to reject artifacts
            
            min_x, min_y = np.min(valid_kp[:, 0]), np.min(valid_kp[:, 1])
            max_x, max_y = np.max(valid_kp[:, 0]), np.max(valid_kp[:, 1])
            w, h = max_x - min_x, max_y - min_y
            centroid = np.mean(valid_kp[:, :2], axis=0)
            
            detections.append({
                'centroid': centroid,
                'bbox': [float(min_x), float(min_y), float(w), float(h)]
            })
            
        # --- B. Match Detections to Existing Tracks ---
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(active_tracks.keys())
        new_active_tracks = {}
        
        if active_tracks and detections:
            for det_idx in list(unmatched_detections):
                det = detections[det_idx]
                best_id = None
                best_dist = max_dist_threshold
                
                for track_id in unmatched_tracks:
                    dist = np.linalg.norm(det['centroid'] - active_tracks[track_id]["centroid"])
                    if dist < best_dist:
                        best_dist = dist
                        best_id = track_id
                        
                if best_id is not None:
                    current_frame_tracks.append({'id': best_id, 'bbox': det['bbox']})
                    new_active_tracks[best_id] = {"centroid": det['centroid'], "lost_frames": 0}
                    unmatched_detections.remove(det_idx)
                    unmatched_tracks.remove(best_id)
                    
        # --- C. Keep Track of Lost Targets Within Grace Period ---
        for track_id in unmatched_tracks:
            lost_count = active_tracks[track_id]["lost_frames"] + 1
            if lost_count <= max_lost_frames:
                new_active_tracks[track_id] = {
                    "centroid": active_tracks[track_id]["centroid"], 
                    "lost_frames": lost_count
                }
                    
        # --- D. Assign New IDs to Confirmed New Detections ---
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            current_id = next_id
            next_id += 1
            
            current_frame_tracks.append({'id': current_id, 'bbox': det['bbox']})
            new_active_tracks[current_id] = {"centroid": det['centroid'], "lost_frames": 0}
            
        active_tracks = new_active_tracks
        tracks_out[frame_idx] = current_frame_tracks

    output_data = {
        "total_frames": total_frames,
        "video_path": video_path,
        "tracks": tracks_out
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"    [TRACKER] Scanned {total_frames} frames. Unique subjects detected: {next_id}")
    return next_id > 1


def purge_excluded_subjects(session_path, pose_folder_name, trial_name, exclusion_list_path):
    """
    Reads the exclusion list and permanently deletes the specified subjects 
    from the pose estimation JSON files across all cameras.
    """
    if not os.path.exists(exclusion_list_path):
        return

    with open(exclusion_list_path, 'r') as f:
        excluded_ids = json.load(f).get("exclude_ids", [])

    if not excluded_ids:
        return

    cam_folders = glob.glob(os.path.join(session_path, 'Videos', 'Cam*'))
    
    for cf in cam_folders:
        json_dir = os.path.join(cf, f"OutputJsons_{pose_folder_name}", trial_name)
        if not os.path.exists(json_dir):
            continue
            
        json_files = sorted(
            glob.glob(os.path.join(json_dir, "*.json")),
            key=lambda f: int(re.findall(r'(\d+)', os.path.basename(f))[-1]) if re.findall(r'(\d+)', os.path.basename(f)) else 0
        )

        # Re-run spatial tracking per camera to accurately identify IDs in this specific viewing angle
        active_tracks = {}
        next_id = 0
        max_dist_threshold = 200.0

        for jf in json_files:
            with open(jf, 'r') as f:
                data = json.load(f)

            if isinstance(data, dict):
                people = data.get('people', [])
            elif isinstance(data, list):
                people = data
            else:
                people = []
                
            if not people:
                continue

            detections = []
            for idx, p in enumerate(people):
                if isinstance(p, dict):
                    kp = p.get('pose_keypoints_2d', [])
                elif isinstance(p, list):
                    kp = p
                else:
                    kp = []
                    
                if not kp or len(kp) == 0: continue
                kp_array = np.array(kp).reshape(-1, 3)
                valid_kp = kp_array[kp_array[:, 2] > 0.1]
                if len(valid_kp) < 20: continue # Enforce minimum keypoint count to reject artifacts
                centroid = np.mean(valid_kp[:, :2], axis=0)
                detections.append({'original_index': idx, 'centroid': centroid})

            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(active_tracks.keys())
            new_active_tracks = {}
            indices_to_keep = []

            if active_tracks and detections:
                for det_idx in list(unmatched_detections):
                    det = detections[det_idx]
                    best_id = None
                    best_dist = max_dist_threshold

                    for track_id in unmatched_tracks:
                        dist = np.linalg.norm(det['centroid'] - active_tracks[track_id])
                        if dist < best_dist:
                            best_dist = dist
                            best_id = track_id

                    if best_id is not None:
                        if best_id not in excluded_ids:
                            indices_to_keep.append(det['original_index'])
                        new_active_tracks[best_id] = det['centroid']
                        unmatched_detections.remove(det_idx)
                        unmatched_tracks.remove(best_id)

            for det_idx in unmatched_detections:
                det = detections[det_idx]
                current_id = next_id
                next_id += 1
                if current_id not in excluded_ids:
                    indices_to_keep.append(det['original_index'])
                new_active_tracks[current_id] = det['centroid']

            active_tracks = new_active_tracks

            # Filter the JSON and overwrite the file
            filtered_people = [people[i] for i in indices_to_keep]
            
            if isinstance(data, dict):
                data['people'] = filtered_people
            else:
                data = filtered_people

            with open(jf, 'w') as f:
                json.dump(data, f)
                
    print(f"    [FILTER] Successfully purged excluded subjects from {trial_name}.")