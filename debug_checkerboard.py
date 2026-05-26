import cv2
import os
import sys
import numpy as np

# =============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# =============================================================================
# Provide the full absolute path to the exact video file that is failing
VIDEO_PATH = r"C:\Users\user\Documents\Opencap\opencap-portable_prod\Data\3cam\Videos\Cam2\InputMedia\calibration\calibration.mov"

# Set your internal corner dimensions (Width, Height).
# Note: If it fails, try swapping these (e.g., 4, 5 instead of 5, 4) 
# to immediately test if OpenCV is perceiving the board sideways.
CORNERS_WIDTH = 4  
CORNERS_HEIGHT = 5 

# Where to save the output images for visual review
OUTPUT_DIR = "checkerboard_debug_output"
# =============================================================================

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video file not found: {VIDEO_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    board_dims = (CORNERS_WIDTH, CORNERS_HEIGHT)
    
    print(f"Loading video: {VIDEO_PATH}")
    print(f"Searching for internal corner grid: {board_dims}")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Could not open video file.")
        sys.exit(1)

    # Replicating the exact logic from reprocessOffline.py
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    
    frames_checked = 0
    success_count = 0
    max_frames_to_check = 60 # Scan the first 2 seconds of a 30fps video
    
    print(f"Scanning the first {max_frames_to_check} frames...")
    
    while True:
        ret, frame = cap.read()
        # Artificially upscale the image to help the algorithm find compressed corners
        frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        if not ret or frames_checked >= max_frames_to_check:
            break

        # Convert to grayscale exactly as the pipeline does
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Switch to the legacy adaptive thresholding algorithm
        legacy_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, board_dims, flags=legacy_flags)

        if found:
            # Draw the colored lines/nodes onto the original color frame
            cv2.drawChessboardCorners(frame, board_dims, corners, found)
            out_name = f"success_frame_{frames_checked:03d}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, frame)
            
            if success_count == 0:
                print(f"  [SUCCESS] Checkerboard DETECTED on Frame {frames_checked}! (Saved as {out_name})")
            success_count += 1
            
        elif frames_checked == 0:
            # Save the very first frame even if it fails, so we can see what the algorithm saw
            out_path = os.path.join(OUTPUT_DIR, "failed_frame_000.jpg")
            cv2.imwrite(out_path, frame)
            print(f"  [FAILED] Could not detect board on Frame 0. Saved snapshot for visual review.")

        frames_checked += 1

    cap.release()
    
    print("\n--- Summary ---")
    if success_count > 0:
        print(f"Result: Found the {board_dims} checkerboard in {success_count} frames.")
        print(f"Action: Check the '{OUTPUT_DIR}' folder. Look at the RED dot on the drawn grid to verify corner #0 orientation.")
    else:
        print(f"Result: FAILED to find the {board_dims} checkerboard in all {frames_checked} frames.")
        print(f"Action: ")
        print(f"  1. Look at 'failed_frame_000.jpg' in the output folder. Is it blurry? Is someone holding it?")
        print(f"  2. Edit this script and swap the width/height (e.g., from {CORNERS_WIDTH}x{CORNERS_HEIGHT} to {CORNERS_HEIGHT}x{CORNERS_WIDTH}) and run again.")

if __name__ == "__main__":
    main()