# ==============================================================================
# File: run_rtmpose.py
# Author: Harry G. Driscoll
# Date: Mar 2026
#
# License: Distributed under the Apache 2.0 License
# ==============================================================================

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# OpenMMLab imports
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.structures import merge_data_samples
from mmpose.visualization import PoseLocalVisualizer
from mmdet.apis import inference_detector, init_detector
from mmengine.registry import DefaultScope 

def rtmpose_to_openpose_body25(rtm_keypoints):
    """Translates 133-point RTMPose to 25-point OpenPose with Dynamic Foot Fallbacks"""
    op_kp = np.zeros((25, 3)) # [x, y, confidence]
    k = rtm_keypoints
    
    # Direct mappings (Upper Body & Legs)
    op_kp[0] = k[0]; op_kp[15] = k[2]; op_kp[16] = k[1]; op_kp[17] = k[4]; op_kp[18] = k[3]
    op_kp[2] = k[6]; op_kp[3] = k[8]; op_kp[4] = k[10]; op_kp[5] = k[5]; op_kp[6] = k[7]; op_kp[7] = k[9]
    op_kp[9] = k[12]; op_kp[10] = k[14]; op_kp[11] = k[16]
    op_kp[12] = k[11]; op_kp[13] = k[13]; op_kp[14] = k[15]

    # Calculated mappings (Midpoints for Neck and MidHip)
    if k[5][2] > 0 and k[6][2] > 0:
        op_kp[1] = [(k[5][0]+k[6][0])/2, (k[5][1]+k[6][1])/2, min(k[5][2], k[6][2])]
    if k[11][2] > 0 and k[12][2] > 0:
        op_kp[8] = [(k[11][0]+k[12][0])/2, (k[11][1]+k[12][1])/2, min(k[11][2], k[12][2])]

    # Dynamic Foot Mapping (COCO-WholeBody indices: 17=LBigToe, 18=LSmallToe, 19=LHeel, 20=RBigToe, 21=RSmallToe, 22=RHeel)
    
    # Left Foot Fallback Logic
    if k[17][2] > 0.05 or k[19][2] > 0.05: # If Left Big Toe or Heel is detected
        op_kp[19] = k[17] # LBigToe
        op_kp[20] = k[18] # LSmallToe
        op_kp[21] = k[19] # LHeel
    else:
        op_kp[19] = k[15]; op_kp[20] = k[15]; op_kp[21] = k[15] # Anchor to LAnkle

    # Right Foot Fallback Logic
    if k[20][2] > 0.05 or k[22][2] > 0.05: # If Right Big Toe or Heel is detected
        op_kp[22] = k[20] # RBigToe
        op_kp[23] = k[21] # RSmallToe
        op_kp[24] = k[22] # RHeel
    else:
        op_kp[22] = k[16]; op_kp[23] = k[16]; op_kp[24] = k[16] # Anchor to RAnkle

    return op_kp.flatten().tolist()

def process_video_rtmpose(video_path, output_dir, video_out_dir, device='cuda:0', model_complexity='m'):
    # --- 1. BULLETPROOF PATH RESOLUTION ---
    # The script is in the root, so point the dependency directory INTO the DLC folder
    dep_dir = Path(__file__).resolve().parent / "Blackwell_RTMPose"
    
    # --- 2. ROBUST RECURSIVE FILE SEARCH ---
    # Pose Config
    pose_configs = list(dep_dir.rglob(f"rtmpose-{model_complexity}*.py"))
    if not pose_configs:
        raise FileNotFoundError(f"Missing Pose Config! Could not find 'rtmpose-{model_complexity}*.py' anywhere in {dep_dir}")
    pose_config = str(pose_configs[0])

    # Pose Weights
    pose_ckpts = list(dep_dir.rglob(f"rtmpose-{model_complexity}*.pth"))
    if not pose_ckpts:
        raise FileNotFoundError(f"Missing Pose Weights! Could not find 'rtmpose-{model_complexity}*.pth' anywhere in {dep_dir}")
    pose_ckpt = str(pose_ckpts[0])
    
    # Detector Config
    det_configs = list(dep_dir.rglob("rtmdet_m*.py"))
    if not det_configs:
        raise FileNotFoundError(f"Missing Detector Config! Could not find 'rtmdet_m*.py' anywhere in {dep_dir}")
    det_config = str(det_configs[0])

    # Detector Weights
    det_ckpts = list(dep_dir.rglob("rtmdet_m*.pth"))
    if not det_ckpts:
        raise FileNotFoundError(f"Missing Detector Weights! Could not find 'rtmdet_m*.pth' anywhere in {dep_dir}")
    det_ckpt = str(det_ckpts[0])
    # ----------------------------------------

    detector = init_detector(det_config, det_ckpt, device=device)
    pose_estimator = init_pose_estimator(pose_config, pose_ckpt, device=device)

    # Setup Visualizer
    visualizer = PoseLocalVisualizer(radius=4, line_width=2)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)

    # Setup Video Capture & Writer
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_out_dir, exist_ok=True)
    
    video_filename = f"{Path(video_path).stem}_rtmpose_diagnostic.avi"
    out_video_path = os.path.join(video_out_dir, video_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    print(f"Processing frames to: {output_dir}")
    
    dumped_raw = False 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # DETECT & FILTER: Ensure only the Top-1 subject is processed
        with DefaultScope.overwrite_default_scope('mmdet'):
            det_result = inference_detector(detector, frame)
            
        pred_instances = det_result.pred_instances
        valid_idx = (pred_instances.labels == 0) & (pred_instances.scores > 0.5)
        
        if valid_idx.any():
            scores = pred_instances.scores[valid_idx].cpu().numpy()
            bboxes = pred_instances.bboxes[valid_idx].cpu().numpy()
            best_idx = np.argmax(scores)
            person_bboxes = bboxes[best_idx : best_idx + 1]
        else:
            person_bboxes = np.zeros((0, 4))
        
        # ESTIMATE: Only run pose on the highest-confidence bounding box
        with DefaultScope.overwrite_default_scope('mmpose'):
            pose_results = inference_topdown(pose_estimator, frame, person_bboxes)
        
        # Format to OpenPose JSON
        people = []
        if pose_results:
            pose = pose_results[0]
            kpts = pose.pred_instances.keypoints[0]
            scores = pose.pred_instances.keypoint_scores[0][:, None]
            combined = np.concatenate((kpts, scores), axis=1)
            
            # --- START DIAGNOSTIC DUMP ---
            if not dumped_raw:
                diag_file = os.path.join(output_dir, f"RAW_133_TENSOR_{Path(video_path).stem}.json")
                with open(diag_file, 'w') as df:
                    json.dump(combined.tolist(), df, indent=4)
                dumped_raw = True 
            # --- END DIAGNOSTIC DUMP ---
            
            body_25_kpts = rtmpose_to_openpose_body25(combined)
            people.append({
                "person_id": [-1], 
                "pose_keypoints_2d": body_25_kpts
            })
            
            # 1. Convert BGR to RGB for the Visualizer
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Draw skeleton
            visualizer.add_datasample(
                'result',
                frame_rgb,
                data_sample=pose_results[0],
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=True,
                show=False,
                wait_time=0,
                out_file=None
            )
            
            # 3. Extract the drawn image and convert back to BGR for OpenCV
            vis_frame = visualizer.get_image()
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            vis_frame_bgr = cv2.resize(vis_frame_bgr, (width, height))
            video_writer.write(vis_frame_bgr)
            
        else:
            # Even if no one is found, we write an empty skeleton
            people.append({
                "person_id": [-1], 
                "pose_keypoints_2d": [0.0] * 75
            })
            frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
            
        json_data = {"version": 1.3, "people": people}
        json_file = os.path.join(output_dir, f"{Path(video_path).stem}_{frame_idx:012d}_keypoints.json")
        with open(json_file, 'w') as f: json.dump(json_data, f)
        
        frame_idx += 1
        if frame_idx % 30 == 0: print(f"Processed {frame_idx} frames...")

    cap.release()
    video_writer.release()
    print(f"RTMPose processing complete! Diagnostic video saved to: {video_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--video_out", required=True) 
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model_complexity", default="m") 
    args = parser.parse_args()
    
    process_video_rtmpose(args.video, args.output_dir, args.video_out, f"cuda:{args.gpu}", args.model_complexity)