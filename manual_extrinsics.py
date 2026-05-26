import os
import sys
import glob
import cv2
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# =============================================================================
# CONFIGURATION
# =============================================================================
SESSION_NAME = "3cam" 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_PATH = os.path.join(BASE_DIR, "Data", SESSION_NAME)

def main():
    if not os.path.exists(SESSION_PATH):
        print(f"Error: Session not found at {SESSION_PATH}")
        sys.exit(1)

    # 1. Load Metadata
    meta_path = os.path.join(SESSION_PATH, "sessionMetadata.yaml")
    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)
    
    cols = meta['checkerBoard']['black2BlackCornersWidth_n']
    rows = meta['checkerBoard']['black2BlackCornersHeight_n']
    sq_size = meta['checkerBoard']['squareSideLength_mm']
    board_dims = (cols, rows)
    
    # Calculate True Physical Size (used for math)
    true_w = (cols - 1) * sq_size
    true_h = (rows - 1) * sq_size
    
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_size

    # 2. Extract Raw OpenCV Solutions
    cams = sorted(glob.glob(os.path.join(SESSION_PATH, 'Videos', 'Cam*')))
    cam_data = {} 
    
    print("Extracting raw coordinates. Please wait...")
    for cf in cams:
        cam_name = os.path.basename(cf)
        print(f"  > Processing {cam_name}...")
        
        # Load Intrinsics
        pkl_path = os.path.join(cf, 'cameraIntrinsicsExtrinsics.pickle')
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join(cf, 'cameraIntrinsics.pickle')
        with open(pkl_path, 'rb') as f:
            intrinsics = pickle.load(f)
            
        mtx = intrinsics['intrinsicMat']
        dist = intrinsics['distortion']

        vid_dir = os.path.join(cf, 'InputMedia', 'calibration')
        vpath_list = glob.glob(os.path.join(vid_dir, "calibration.*"))
        if not vpath_list:
            continue
            
        cap = cv2.VideoCapture(vpath_list[0])
        found = False
        corners = None
        
        # Frame Hunt
        for _ in range(60):
            ret, frame = cap.read()
            if not ret: break
            
            gray_1x = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Attempt 1: SB
            ret_sb, c_found = cv2.findChessboardCornersSB(gray_1x, board_dims, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if ret_sb:
                corners = c_found
                found = True
                break
                
            # Attempt 2: Upscale Legacy
            gray_2x = cv2.cvtColor(cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
            ret_legacy, c_found = cv2.findChessboardCorners(gray_2x, board_dims, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
            if ret_legacy:
                corners = c_found / 2.0
                found = True
                break
                
        cap.release()

        if not found:
            print(f"    [FAIL] No board found in {cam_name}.")
            continue

        # Solve for 4 Pure, Unadulterated Realities using IPPE
        retA, rvecsA, tvecsA, _ = cv2.solvePnPGeneric(objp, corners, mtx, dist, flags=cv2.SOLVEPNP_IPPE)
        retB, rvecsB, tvecsB, _ = cv2.solvePnPGeneric(objp, corners[::-1], mtx, dist, flags=cv2.SOLVEPNP_IPPE)
        
        # We apply NO transforms, NO iterative polishing, NO origin alignment.
        raw_sols = [
            (rvecsA[0], tvecsA[0]),
            (rvecsA[1] if len(rvecsA) > 1 else rvecsA[0], tvecsA[1] if len(tvecsA) > 1 else tvecsA[0]),
            (rvecsB[0], tvecsB[0]),
            (rvecsB[1] if len(rvecsB) > 1 else rvecsB[0], tvecsB[1] if len(tvecsB) > 1 else tvecsB[0])
        ]
        
        refined_sols = []
        for rv, tv in raw_sols:
            R, _ = cv2.Rodrigues(rv)
            refined_sols.append((R, tv))
            
        cam_data[cam_name] = {
            'solutions': refined_sols,
            'current_idx': 0,
            'pickle_path': os.path.join(cf, 'cameraIntrinsicsExtrinsics.pickle'),
            'intrinsics_data': intrinsics
        }

    # 3. INTERACTIVE 3D UI
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)
    
    # Exaggerate Board Size to 500mm (50cm) for visual clarity ONLY
    EXAGGERATED_LONG_EDGE = 500.0
    if true_w > true_h:
        vis_w = EXAGGERATED_LONG_EDGE
        vis_h = EXAGGERATED_LONG_EDGE * (true_h / true_w)
    else:
        vis_h = EXAGGERATED_LONG_EDGE
        vis_w = EXAGGERATED_LONG_EDGE * (true_w / true_h)

    def update_plot():
        ax.cla()
        
        # 1. Draw Exaggerated Board (Notice Y is drawn negative to maintain true geometry)
        ax.plot([0, vis_w, vis_w, 0, 0], [0, 0, -vis_h, -vis_h, 0], [0, 0, 0, 0, 0], 'k-', linewidth=2)
        
        # Tiny Origin Dot
        ax.scatter([0], [0], [0], color='black', s=20, label='Origin (0,0,0)', zorder=10)
        
        # Red X-Axis (Rows) -> Positive direction
        ax.plot([0, vis_w], [0, 0], [0, 0], color='red', linewidth=4)
        x_label = "LONG EDGE (X-Axis)" if vis_w > vis_h else "SHORT EDGE (X-Axis)"
        ax.text(vis_w/2, 50, 0, x_label, color='red', weight='bold')

        # Blue Y-Axis (Cols) -> Negative direction
        ax.plot([0, 0], [0, -vis_h], [0, 0], color='blue', linewidth=4)
        y_label = "LONG EDGE (Y-Axis)" if vis_h > vis_w else "SHORT EDGE (Y-Axis)"
        ax.text(-50, -vis_h/2, 0, y_label, color='blue', weight='bold')

        # 2. Draw Cameras
        colors = ['red', 'green', 'magenta', 'orange', 'cyan']
        max_z = 1000 # Floor baseline tracking
        
        for i, (cam_name, data) in enumerate(cam_data.items()):
            idx = data['current_idx']
            R, t = data['solutions'][idx]
            
            # Raw Math: Center = -R^T * t
            C = -np.matrix(R).T @ np.matrix(t)
            C = np.array(C).flatten()
            
            # --- THE FIX ---
            # We take the absolute value of Z, and invert Y to counter the mirror effect.
            plot_x = C[0]
            plot_y = -C[1]
            plot_z = abs(C[2])
            
            if plot_z > max_z: max_z = plot_z
            
            color = colors[i % len(colors)]
            
            # Tiny Camera Dot
            ax.scatter(plot_x, plot_y, plot_z, color=color, s=20, label=f"{cam_name} (Pose {idx+1})")
            
            # Line connecting Camera to Origin
            ax.plot([plot_x, 0], [plot_y, 0], [plot_z, 0], color=color, linestyle='--', alpha=0.5)

        # Enforce Ground Plane
        ax.set_zlim(0, max_z * 1.1)
        ax.set_xlabel('X-Axis (mm)')
        ax.set_ylabel('Y-Axis (mm) [Inverted for GUI parity]')
        ax.set_zlabel('Z-Axis (mm) [Absolute Magnitude]')
        ax.set_title('RAW EXTRINSICS VISUALIZER (Matched to PyVista GUI)')
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        
        # Equal Aspect Ratio
        extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            if dim == 'z':
                ax.set_zlim(0, maxsize) # Keep floor at 0
            else:
                getattr(ax, f'set_{dim}lim')(ctr - r, ctr + r)
            
        fig.canvas.draw_idle()

    # Buttons
    btn_axes = []
    buttons = []
    
    def make_cycler(cam):
        def cycle(event):
            cam_data[cam]['current_idx'] = (cam_data[cam]['current_idx'] + 1) % 4
            update_plot()
        return cycle

    for i, cam_name in enumerate(cam_data.keys()):
        ax_btn = plt.axes([0.1 + i*0.18, 0.05, 0.12, 0.075])
        btn = Button(ax_btn, f'Cycle {cam_name}')
        btn.on_clicked(make_cycler(cam_name))
        btn_axes.append(ax_btn)
        buttons.append(btn)

    def save_data(event):
        for cam_name, data in cam_data.items():
            idx = data['current_idx']
            R, t = data['solutions'][idx]
            out_data = data['intrinsics_data']
            out_data['rotation'] = R
            out_data['translation'] = t
            out_data['R'] = R
            out_data['t'] = t
            out_data['rotation_EulerAngles'], _ = cv2.Rodrigues(R)
            
            with open(data['pickle_path'], 'wb') as f:
                pickle.dump(out_data, f)
            print(f"Saved {cam_name} (Pose {idx+1})")

    ax_save = plt.axes([0.8, 0.05, 0.15, 0.075])
    btn_save = Button(ax_save, 'SAVE EXTRINSICS', color='lightgreen')
    btn_save.on_clicked(save_data)
    buttons.append(btn_save)

    update_plot()
    plt.show()

if __name__ == "__main__":
    main()