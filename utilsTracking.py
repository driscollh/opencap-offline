import cv2
import os
import json
import glob
import numpy as np
import re

def _track_subjects(json_dir):
    """
    The single, shared subject tracker.

    Scans one camera's folder of pose JSON files and assigns a stable "id"
    to each person across the whole clip (so person 1 stays person 1 even if
    the detector briefly loses them for a few frames). Both the picker
    (generate_tracking_data) and the deleter (purge_excluded_subjects) call
    THIS one function, so they can never disagree about who is who.

    Returns three things:
      json_files    - the list of frame files, in order
      frames_tracks - for each frame, the people found, each recorded as
                      {'id': which subject, 'idx': their slot in the raw file,
                       'bbox': their box on screen}
      n_ids         - how many different people were seen in total
    """
    json_files = sorted(
        glob.glob(os.path.join(json_dir, "*.json")),
        key=lambda f: int(re.findall(r'(\d+)', os.path.basename(f))[-1]) if re.findall(r'(\d+)', os.path.basename(f)) else 0
    )

    active_tracks = {}
    next_id = 0
    max_dist_threshold = 200.0
    max_lost_frames = 15
    frames_tracks = []

    for jf in json_files:
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

        # --- A. Find the real people in this frame (and remember their slot) ---
        for idx, p in enumerate(people):
            if isinstance(p, dict):
                kp = p.get('pose_keypoints_2d', [])
            elif isinstance(p, list):
                kp = p
            else:
                kp = []

            if not kp or len(kp) == 0:
                continue

            kp_array = np.array(kp).reshape(-1, 3)
            valid_kp = kp_array[kp_array[:, 2] > 0.1]
            # Ignore weak/false detections (e.g. a chair): too few or too faint joints
            if len(valid_kp) < 5 or np.sum(valid_kp[:, 2]) < 4.0:
                continue

            min_x, min_y = np.min(valid_kp[:, 0]), np.min(valid_kp[:, 1])
            max_x, max_y = np.max(valid_kp[:, 0]), np.max(valid_kp[:, 1])
            centroid = np.mean(valid_kp[:, :2], axis=0)
            detections.append({
                'idx': idx,
                'centroid': centroid,
                'bbox': [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]
            })

        # --- B. Match this frame's people to people we were already following ---
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
                    current_frame_tracks.append({'id': best_id, 'idx': det['idx'], 'bbox': det['bbox']})
                    new_active_tracks[best_id] = {"centroid": det['centroid'], "lost_frames": 0}
                    unmatched_detections.remove(det_idx)
                    unmatched_tracks.remove(best_id)

        # --- C. Someone we were following wasn't seen this frame: keep them briefly ---
        for track_id in unmatched_tracks:
            lost_count = active_tracks[track_id]["lost_frames"] + 1
            if lost_count <= max_lost_frames:
                new_active_tracks[track_id] = {
                    "centroid": active_tracks[track_id]["centroid"],
                    "lost_frames": lost_count
                }

        # --- D. A brand-new person appeared: give them a fresh id ---
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            current_id = next_id
            next_id += 1
            current_frame_tracks.append({'id': current_id, 'idx': det['idx'], 'bbox': det['bbox']})
            new_active_tracks[current_id] = {"centroid": det['centroid'], "lost_frames": 0}

        active_tracks = new_active_tracks
        frames_tracks.append(current_frame_tracks)

    return json_files, frames_tracks, next_id

def generate_tracking_data(json_dir, video_path, output_json_path, cam_name, trial_name):
    """
    Scans a folder of pose JSONs, tracks subjects across frames using spatial centroids,
    and generates a tracking JSON for the GUI to consume.
    Incorporates a track retention window to prevent ID splitting from frame drops.
    
    Returns:
        bool: True if multiple subjects were detected at any point, False otherwise.
    """
    json_files, frames_tracks, n_ids = _track_subjects(json_dir)

    if not json_files:
        print(f"[TRACKER] Error: No JSON files found in {json_dir}")
        return False

    total_frames = len(json_files)

    # The GUI picker only needs each person's id and their on-screen box.
    # (_track_subjects also hands us 'idx', but the picker doesn't use it.)
    tracks_out = {
        i: [{'id': t['id'], 'bbox': t['bbox']} for t in frame]
        for i, frame in enumerate(frames_tracks)
    }

    trial_name = os.path.basename(os.path.dirname(video_path))
    output_data = {
        "trial_name": trial_name,
        "cam_name": cam_name,
        "total_frames": total_frames,
        "video_path": video_path,
        "tracks": tracks_out
    }

    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"    [TRACKER] Scanned {total_frames} frames. Unique subjects detected: {n_ids}")
    return n_ids > 1


def purge_excluded_subjects(cam_dir, pose_folder_name, trial_name, exclusion_list_path):
    """Permanently deletes specified subjects from a SINGLE camera's JSON files."""
    if not os.path.exists(exclusion_list_path):
        return

    with open(exclusion_list_path, 'r') as f:
        excluded_ids = json.load(f).get("exclude_ids", [])

    if not excluded_ids:
        return

    json_dir = os.path.join(cam_dir, f"OutputJsons_{pose_folder_name}", trial_name)
    if not os.path.exists(json_dir):
        return

    # Run the SAME tracker the picker used, so the ids line up exactly.
    json_files, frames_tracks, _ = _track_subjects(json_dir)

    # Walk each frame alongside the tracker's verdict for that frame.
    for jf, frame_tracks in zip(json_files, frames_tracks):
        with open(jf, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            people = data.get('people', [])
        elif isinstance(data, list):
            people = data
        else:
            people = []

        # Keep only the people whose id was NOT ticked for exclusion.
        keep_idx = [t['idx'] for t in frame_tracks if t['id'] not in excluded_ids]
        filtered_people = [people[i] for i in keep_idx]

        if isinstance(data, dict):
            data['people'] = filtered_people
        else:
            data = filtered_people

        with open(jf, 'w') as f:
            json.dump(data, f)

    print(f"    [FILTER] Successfully purged excluded subjects from {trial_name}.")

def render_purged_video(cam_dir, pose_folder_name, trial_name):
    """Redraws the overlay video for a SINGLE camera."""
    
    print(f"    [RENDERER] Overwriting overlay videos for {trial_name}...")
    
    pairs = [
        (1,8), (1,2), (1,5), (2,3), (3,4), (5,6), (6,7), (8,9), 
        (9,10), (10,11), (8,12), (12,13), (13,14), (1,0), (0,15), 
        (15,17), (0,16), (16,18), (14,19), (19,20), (14,21), 
        (11,22), (22,23), (11,24)
    ]
    
    rgb_colors = [
        (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0),
        (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
        (255, 0, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
        (0, 170, 255), (0, 85, 255), (0, 0, 255), (255, 0, 170),
        (170, 0, 255), (255, 0, 255), (85, 0, 255), (0, 0, 255),
        (0, 0, 255), (0, 0, 255), (0, 255, 255), (0, 255, 255),
        (0, 255, 255)
    ]
    
    colors = [(int(c[2]), int(c[1]), int(c[0])) for c in rgb_colors]

    raw_vid = os.path.join(cam_dir, 'InputMedia', trial_name, f"{trial_name}.mp4")
    json_dir = os.path.join(cam_dir, f"OutputJsons_{pose_folder_name}", trial_name)
    out_dir = os.path.join(cam_dir, f"OutputMedia_{pose_folder_name}", trial_name)
    out_vid = os.path.join(out_dir, f"{trial_name}_overlay.avi")
    
    if not os.path.exists(raw_vid) or not os.path.exists(json_dir):
        return
        
    json_files = sorted(
        glob.glob(os.path.join(json_dir, "*.json")),
        key=lambda f: int(re.findall(r'(\d+)', os.path.basename(f))[-1]) if re.findall(r'(\d+)', os.path.basename(f)) else 0
    )
                        
    cap = cv2.VideoCapture(raw_vid)
    if not cap.isOpened():
        return
            
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_out_vid = out_vid.replace(".avi", "_temp.avi")
    writer = cv2.VideoWriter(temp_out_vid, fourcc, fps, (w, h))
    
    frame_idx = 0
    total_people_drawn = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx < len(json_files):
            with open(json_files[frame_idx], 'r') as jf:
                data = json.load(jf)
            
            people = data.get('people', []) if isinstance(data, dict) else data
            total_people_drawn += len(people)
            
            for p in people:
                kp = p.get('pose_keypoints_2d', []) if isinstance(p, dict) else p
                if not kp: continue
                
                kp_array = np.array(kp).reshape(-1, 3)
                
                for i, pair in enumerate(pairs):
                    if pair[0] < len(kp_array) and pair[1] < len(kp_array):
                        p1, p2 = kp_array[pair[0]], kp_array[pair[1]]
                        if p1[2] > 0.1 and p2[2] > 0.1:
                            color = colors[pair[1]] 
                            pt1 = (int(p1[0]), int(p1[1]))
                            pt2 = (int(p2[0]), int(p2[1]))
                            cv2.line(frame, pt1, pt2, color, 6, cv2.LINE_AA)
                
                for i, pt in enumerate(kp_array):
                    if pt[2] > 0.1:
                        color = colors[i] 
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, color, -1, cv2.LINE_AA)
        
        writer.write(frame)
        frame_idx += 1
        
    cap.release()
    writer.release()
    
    cam_name = os.path.basename(cam_dir)
    print(f"    [RENDERER] {cam_name}: Drew skeletons for {total_people_drawn} total detections across {frame_idx} frames.")
    
    try:
        if os.path.exists(out_vid):
            os.remove(out_vid)
        os.rename(temp_out_vid, out_vid)
    except Exception as e:
        print(f"    [RENDERER] Error replacing video file: {e}")