# ==============================================================================
# File: simple_launcher.py
# Author: Harry G. Driscoll
# Date: Jan 2026
#
# License: Distributed under the Apache 2.0 License
# ==============================================================================

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import subprocess
import threading
import os
import sys
import shutil
import createMetadata 

class OpenCapGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCap Portable - Session Manager")
        self.root.geometry("950x850")
        self.root.configure(bg="#f0f0f0")

        self.app_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.app_path, "Data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.python_exe = sys.executable 
        self.script_path = os.path.join(self.app_path, "reprocessOffline.py")

        # --- 1. SESSION MANAGEMENT ---
        top_frame = tk.LabelFrame(root, text="Session Management", padx=10, pady=10)
        top_frame.pack(pady=10, padx=20, fill="x")

        tk.Label(top_frame, text="Select Session:").grid(row=0, column=0, sticky="w")
        self.cb_sessions = ttk.Combobox(top_frame, state="readonly", width=30)
        self.cb_sessions.grid(row=0, column=1, padx=5, pady=5)
        self.cb_sessions.bind("<<ComboboxSelected>>", self.refresh_camera_slots)
        
        tk.Button(top_frame, text="Refresh", command=self.refresh_sessions).grid(row=0, column=2, padx=5)
        tk.Button(top_frame, text="New Session", command=self.open_create_window, bg="#3498db", fg="white").grid(row=0, column=3, padx=20)

        # --- 2. VIDEO IMPORT SECTION ---
        import_frame = tk.LabelFrame(root, text="Import Videos to Session", padx=10, pady=10)
        import_frame.pack(pady=10, padx=20, fill="x")

        # --- Trial Name Selection with Quick Buttons ---
        name_frame = tk.Frame(import_frame)
        name_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=5)

        tk.Label(name_frame, text="Trial Name:", font=("Arial", 9, "bold")).pack(side="left")

        # --- Trial Type Selection (Radio Buttons) ---
        type_frame = tk.LabelFrame(import_frame, text="Step 1: Select Trial Type", padx=5, pady=5)
        type_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=5)

        self.trial_choice = tk.StringVar(value="calibration")
        
        # Radio Options
        opts = [("Intrinsics", "intrinsics"), ("Calibration", "calibration"), ("Neutral", "neutral"), ("Dynamic", "dynamic")]
        for i, (text, val) in enumerate(opts):
            rb = tk.Radiobutton(type_frame, text=text, variable=self.trial_choice, value=val, 
                                command=self.toggle_trial_name_entry)
            rb.pack(side="left", padx=10)

        tk.Label(type_frame, text=" |  Custom Name:").pack(side="left", padx=(10, 0))
        self.ent_trial_name = tk.Entry(type_frame, width=20)
        self.ent_trial_name.insert(0, "calibration")
        self.ent_trial_name.config(state="disabled", disabledbackground="#e0e0e0")
        self.ent_trial_name.pack(side="left", padx=5)

        self.cam_slots_frame = tk.Frame(import_frame)
        self.cam_slots_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky="w")
        self.cam_file_paths = {} # To store paths for each cam

        self.btn_import = tk.Button(import_frame, text="EXECUTE IMPORT", command=self.execute_import, bg="#f39c12", fg="white", font=("Arial", 10, "bold"))
        self.btn_import.grid(row=2, column=0, columnspan=4, pady=5, sticky="ew")

        # --- 3. PROCESSING SECTION ---
        proc_frame = tk.LabelFrame(root, text="Processing", padx=10, pady=10)
        proc_frame.pack(pady=10, padx=20, fill="x")

        # New Intrinsics Button
        self.btn_intrinsics = tk.Button(proc_frame, text="1. RUN INTRINSICS CALIBRATION", 
                                        command=self.run_intrinsics, 
                                        bg="#3498db", fg="white", font=("Arial", 10, "bold"))
        self.btn_intrinsics.pack(side="left", padx=10, expand=True, fill="x")

        # Existing Pipeline Button
        self.run_btn = tk.Button(proc_frame, text="2. RUN FULL PIPELINE", command=self.start_thread, 
                                 bg="#2ecc71", fg="white", font=("Arial", 10, "bold"))
        self.run_btn.pack(side="left", padx=10, expand=True, fill="x")

        # --- 4. SESSION OVERVIEW PANEL ---
        view_frame = tk.LabelFrame(root, text="Session Content Overview", padx=10, pady=10)
        view_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # Tree and Scrollbar container
        tree_container = tk.Frame(view_frame)
        tree_container.pack(side="left", fill="both", expand=True)

        self.tree = ttk.Treeview(tree_container, columns=("Info"), height=10)
        self.tree.heading("#0", text="Camera / Trial / File")
        self.tree.heading("Info", text="Status")
        self.tree.column("Info", width=120, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)

        tree_scroll = ttk.Scrollbar(tree_container, orient="vertical", command=self.tree.yview)
        tree_scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=tree_scroll.set)

        # Action Buttons for the Tree
        tree_btn_frame = tk.Frame(view_frame)
        tree_btn_frame.pack(side="right", fill="y", padx=5)

        tk.Button(tree_btn_frame, text="REFRESH\nLIST", command=self.refresh_session_overview, 
                  bg="#95a5a6", fg="white", width=10).pack(pady=5)
        
        tk.Button(tree_btn_frame, text="DELETE\nSELECTED", command=self.delete_selected_item, 
                  bg="#e74c3c", fg="white", width=10, font=("Arial", 8, "bold")).pack(pady=5)

        # --- 5. OUTPUT ---
        self.output_area = scrolledtext.ScrolledText(root, bg="white", fg="black", font=("Consolas", 10))
        self.output_area.pack(pady=10, padx=20, fill="both", expand=True)

        self.refresh_sessions()

    def refresh_session_overview(self):
        """Clears and repopulates the visual tree of trial folders and videos."""
        self.tree.delete(*self.tree.get_children())
        session = self.cb_sessions.get()
        if not session: return

        video_root = os.path.join(self.data_dir, session, "Videos")
        if not os.path.exists(video_root): return

        cams = sorted([d for d in os.listdir(video_root) if d.startswith("Cam")])
        
        for cam in cams:
            cam_node = self.tree.insert("", "end", text=cam, open=True, values=("Camera Folder",))
            media_path = os.path.join(video_root, cam, "InputMedia")
            
            if os.path.exists(media_path):
                trials = sorted([d for d in os.listdir(media_path) if os.path.isdir(os.path.join(media_path, d))])
                for trial in trials:
                    t_path = os.path.join(media_path, trial)
                    vids = [f for f in os.listdir(t_path) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
                    # Simply show the trial in the tree; do not create Checkbuttons here
                    trial_node = self.tree.insert(cam_node, "end", text=trial, values=(f"{len(vids)} files",))

    def delete_selected_item(self):
        selected_item = self.tree.selection()
        if not selected_item:
            return messagebox.showwarning("Delete", "Please select a file or folder in the list above.")

        item_text = self.tree.item(selected_item)['text']
        item_values = self.tree.item(selected_item)['values']
        parent_item = self.tree.parent(selected_item)
        
        session = self.cb_sessions.get()
        video_root = os.path.join(self.data_dir, session, "Videos")

        path_to_delete = None
        
        # Determine what was selected based on parentage
        if parent_item == "":  # Selected a Camera folder (Cam0)
            return messagebox.showinfo("Delete", "You cannot delete root camera folders.")

        grandparent = self.tree.parent(parent_item)
        
        if grandparent == "": # Selected a Trial folder (e.g. calibration)
            cam = self.tree.item(parent_item)['text']
            path_to_delete = os.path.join(video_root, cam, "InputMedia", item_text)
        else: # Selected an individual Video file
            trial = self.tree.item(parent_item)['text']
            cam = self.tree.item(grandparent)['text']
            path_to_delete = os.path.join(video_root, cam, "InputMedia", trial, item_text)

        if path_to_delete and os.path.exists(path_to_delete):
            confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete:\n{item_text}?")
            if confirm:
                try:
                    if os.path.isdir(path_to_delete):
                        shutil.rmtree(path_to_delete)
                    else:
                        os.remove(path_to_delete)
                    
                    self.log(f"[DELETE] Removed {item_text}\n")
                    self.refresh_session_overview()
                except Exception as e:
                    messagebox.showerror("Error", f"Could not delete: {e}")

    def toggle_trial_name_entry(self):
        choice = self.trial_choice.get()
        
        # Unlock to change text
        self.ent_trial_name.config(state="normal")
        self.ent_trial_name.delete(0, tk.END)
        
        if choice == "dynamic":
            self.ent_trial_name.insert(0, "walking_1")
            self.ent_trial_name.focus()
            # Keep it normal so user can type
        else:
            self.ent_trial_name.insert(0, choice)
            self.ent_trial_name.config(state="disabled")

    def refresh_sessions(self):
        sessions = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        self.cb_sessions['values'] = sessions
        if sessions: 
            self.cb_sessions.current(0)
            self.refresh_camera_slots()

        self.refresh_session_overview()

    def refresh_camera_slots(self, event=None):
        # Clear existing slots
        for widget in self.cam_slots_frame.winfo_children():
            widget.destroy()
        
        session = self.cb_sessions.get()
        if not session: return

        # UPDATE: Point to the Videos subfolder where CamX folders live
        sess_videos_path = os.path.join(self.data_dir, session, "Videos")
        
        if not os.path.exists(sess_videos_path):
            return

        cams = sorted([d for d in os.listdir(sess_videos_path) if d.startswith("Cam")])
        
        self.cam_file_paths = {}
        for i, cam in enumerate(cams):
            tk.Label(self.cam_slots_frame, text=f"{cam}:").grid(row=i, column=0, padx=5, sticky="w")
            lbl_path = tk.Label(self.cam_slots_frame, text="No file selected", fg="gray", width=60, anchor="w")
            lbl_path.grid(row=i, column=1, padx=5)
            
            # The browse button that opens the pop-up
            btn_browse = tk.Button(self.cam_slots_frame, text="Browse", 
                                   command=lambda c=cam, l=lbl_path: self.browse_video(c, l))
            btn_browse.grid(row=i, column=2, padx=5)

        self.refresh_session_overview()

    def browse_video(self, cam_id, label_widget):
        # Changed to askopenfilenames to allow multi-select
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.MOV")]
        )
        if file_paths:
            # Store as a list in our dictionary
            self.cam_file_paths[cam_id] = list(file_paths)
            
            # Show count if multiple, else show filename
            if len(file_paths) > 1:
                label_widget.config(text=f"{len(file_paths)} files selected", fg="blue")
            else:
                label_widget.config(text=os.path.basename(file_paths[0]), fg="black")

    def execute_import(self):
        session = self.cb_sessions.get()
        trial = self.ent_trial_name.get().strip()
        
        if not session or not trial:
            return messagebox.showerror("Error", "Select a session and trial type.")

        try:
            for cam, paths in self.cam_file_paths.items():
                # Target: Data/Session/Videos/CamX/InputMedia/TrialName
                dest_dir = os.path.join(self.data_dir, session, "Videos", cam, "InputMedia", trial)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Clean out the folder first to ensure only the NEW renamed video exists
                for old_file in os.listdir(dest_dir):
                    try: os.remove(os.path.join(dest_dir, old_file))
                    except: pass
                
                for src_path in paths:
                    # Get the file extension (e.g., .mp4 or .mov)
                    ext = os.path.splitext(src_path)[1]
                    
                    # CHANGE: Instead of original basename, use the trial name
                    dest_path = os.path.join(dest_dir, f"{trial}{ext}")
                    
                    shutil.copy2(src_path, dest_path)
                    
                self.log(f"[IMPORT] Renamed and Copied to {cam} -> {trial}{ext}\n")
            
            messagebox.showinfo("Success", f"Imported videos as '{trial}'.")
            self.cam_file_paths = {}
            self.refresh_camera_slots()
        except Exception as e:
            messagebox.showerror("Import Failed", str(e))

        self.refresh_session_overview()

    def run_intrinsics(self):
        session = self.cb_sessions.get()
        if not session:
            return messagebox.showwarning("Warning", "Please select a session first.")
        
        self.btn_intrinsics.config(state="disabled", text="CALIBRATING...")
        self.output_area.insert(tk.END, f"\n>>> Calibrating Intrinsics for: {session}\n")
        
        import generate_intrinsics
        
        # Run the updated function in a background thread
        def task():
            try:
                generate_intrinsics.calibrate_session(session)
                self.root.after(0, lambda: messagebox.showinfo("Done", "Intrinsics saved to CustomCameraIntrinsics and Session folders."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.btn_intrinsics.config(state="normal", text="1. RUN INTRINSICS CALIBRATION"))

        threading.Thread(target=task, daemon=True).start()

    def _execute_intrinsics_thread(self, func, session_name):
        try:
            # This calls the logic in generate_intrinsics.py
            func(session_name)
            self.root.after(0, lambda: messagebox.showinfo("Done", "Intrinsics Calibration Complete!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Calibration failed: {e}"))
        finally:
            self.root.after(0, lambda: self.btn_intrinsics.config(state=tk.NORMAL, text="A. GENERATE INTRINSICS"))

    def log(self, text):
        self.output_area.insert(tk.END, text)
        self.output_area.see(tk.END)

    def open_create_window(self):
        create_win = tk.Toplevel(self.root)
        create_win.title("Create New Session")
        create_win.geometry("450x650")
        
        frame = tk.Frame(create_win, padx=20, pady=20)
        frame.pack(fill="both", expand=True)

        # --- Basic Info ---
        tk.Label(frame, text="Session Name (Folder):", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky="w", pady=5)
        ent_sess_id = tk.Entry(frame, width=25)
        ent_sess_id.grid(row=0, column=1, pady=5)

        tk.Label(frame, text="Subject Name/ID:").grid(row=1, column=0, sticky="w", pady=5)
        ent_sub_id = tk.Entry(frame, width=25)
        ent_sub_id.grid(row=1, column=1, pady=5)

        tk.Label(frame, text="Height (m):").grid(row=2, column=0, sticky="w", pady=5)
        ent_height = tk.Entry(frame, width=10); ent_height.insert(0, "1.75")
        ent_height.grid(row=2, column=1, sticky="w", pady=5)

        tk.Label(frame, text="Weight (kg):").grid(row=3, column=0, sticky="w", pady=5)
        ent_weight = tk.Entry(frame, width=10); ent_weight.insert(0, "70.0")
        ent_weight.grid(row=3, column=1, sticky="w", pady=5)

        tk.Label(frame, text="Subject Tag:").grid(row=4, column=0, sticky="w", pady=5)
        tags = ["healthy", "unimpaired", "impaired", "exo assisted", "exo unassisted", "Other"]
        cb_tags = ttk.Combobox(frame, values=tags, state="readonly", width=22); cb_tags.current(0)
        cb_tags.grid(row=4, column=1, pady=5)

        # Divider
        tk.Frame(frame, height=2, bd=1, relief="sunken").grid(row=5, columnspan=2, sticky="ew", pady=10)
        tk.Label(frame, text="Checkerboard Settings", font=('Arial', 9, 'bold')).grid(row=6, column=0, sticky="w", pady=5)

        tk.Label(frame, text="Rows (Inner Corners):").grid(row=7, column=0, sticky="w", pady=2)
        ent_rows = tk.Spinbox(frame, from_=1, to=20, width=5); ent_rows.delete(0, "end"); ent_rows.insert(0, "4")
        ent_rows.grid(row=7, column=1, sticky="w")

        tk.Label(frame, text="Cols (Inner Corners):").grid(row=8, column=0, sticky="w", pady=2)
        ent_cols = tk.Spinbox(frame, from_=1, to=20, width=5); ent_cols.delete(0, "end"); ent_cols.insert(0, "5")
        ent_cols.grid(row=8, column=1, sticky="w")

        tk.Label(frame, text="Square Size (mm):").grid(row=9, column=0, sticky="w", pady=2)
        ent_size = tk.Entry(frame, width=10); ent_size.insert(0, "35.0")
        ent_size.grid(row=9, column=1, sticky="w")

        tk.Label(frame, text="Number of Cameras:").grid(row=10, column=0, sticky="w", pady=10)
        ent_cams = tk.Spinbox(frame, from_=1, to=10, width=5); ent_cams.grid(row=10, column=1, sticky="w")

        def submit():
            sess_name = ent_sess_id.get().strip()
            sub_id = ent_sub_id.get().strip()
            tag = cb_tags.get()
            try:
                h = float(ent_height.get())
                w = float(ent_weight.get())
                c_rows = int(ent_rows.get())
                c_cols = int(ent_cols.get())
                c_size = float(ent_size.get())
                n_cams = int(ent_cams.get())
            except ValueError:
                return messagebox.showerror("Error", "Please enter valid numbers.")

            sess_path = os.path.join(self.data_dir, sess_name)
            if not os.path.exists(sess_path):
                # 1. Create Base Structure from your create_session logic
                os.makedirs(sess_path, exist_ok=True)
                for folder in ["CalibrationImages", "MarkerData", "OpenSimData", "Videos", "VisualizerJsons", "VisualizerVideos"]:
                    os.makedirs(os.path.join(sess_path, folder), exist_ok=True)
                
                # 2. Create Camera Folders
                for i in range(n_cams):
                    cam_path = os.path.join(sess_path, "Videos",f"Cam{i}")
                    os.makedirs(cam_path, exist_ok=True)
                
                # 3. Create Metadata
                createMetadata.create_metadata(sess_path, sub_id, h, w, tag, rows=c_rows, cols=c_cols, size=c_size)
                
                messagebox.showinfo("Success", f"Session {sess_name} created.")
                self.refresh_sessions()
                create_win.destroy()
            else:
                messagebox.showerror("Error", "Session folder already exists.")

        tk.Button(frame, text="Create Session", command=submit, bg="#2ecc71", fg="white", font=("Arial", 10, "bold")).grid(row=11, columnspan=2, pady=20)

    def start_thread(self):
        session = self.cb_sessions.get()
        if not session: return messagebox.showwarning("Warning", "Select a session first.")
        self.run_btn.config(state=tk.DISABLED, text="RUNNING...")
        self.output_area.delete('1.0', tk.END)
        threading.Thread(target=self.run_pipeline, args=(session,), daemon=True).start()

    def run_pipeline(self, session_name):
        session_path = os.path.join(self.data_dir, session_name)
        popup = PipelineConfigPopup(self.root, session_path)
        self.root.wait_window(popup)
        
        if not popup.result: return # User closed window

        # Construct Command
        cmd = [
            self.python_exe, "-u", self.script_path,
            "--session", session_name,
            "--gpu_index", popup.result["gpu_index"], 
            "--resolution", popup.result["res"],
            "--trials"
        ] + popup.result["trials"]

        try:
            # log to the GUI that the process has started elsewhere
            self.log(f"\n>>> Launching Pipeline for {session_name}...\n")
            self.log(">>> CHECK BACKGROUND TERMINAL FOR PROGRESS\n")

            # REMOVED: stdout=subprocess.PIPE and stderr=subprocess.STDOUT
            # This allows the process to print directly to your open terminal
            process = subprocess.Popen(
                cmd, 
                cwd=self.app_path, 
                env=os.environ.copy()
            )
            
            # Wait for the process to finish in this background thread
            rc = process.wait()
            
            self.root.after(0, self.log, f"\n--- FINISHED: Code {rc} ---")
        except Exception as e:
            self.root.after(0, self.log, f"\n[ERROR] {str(e)}")
        
        self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL, text="RUN PIPELINE"))

class PipelineConfigPopup(tk.Toplevel):
    def __init__(self, parent, session_path):
        super().__init__(parent)
        self.title("Pipeline Configuration")
        self.geometry("450x600")
        self.result = None
        self.grab_set()

        # Fetch GPU Mapping
        self.gpu_map = self.get_gpu_info()
        
        # 1. GPU Selection (Showing Names)
        tk.Label(self, text="Select Graphics Card (GPU):", font=("Arial", 10, "bold")).pack(pady=5)
        self.gpu_cb = ttk.Combobox(self, values=list(self.gpu_map.keys()), state="readonly", width=40)
        self.gpu_cb.current(0)
        self.gpu_cb.pack(pady=5)

        # 2. Resolution
        tk.Label(self, text="Processing Resolution:", font=("Arial", 10, "bold")).pack(pady=5)
        self.res_cb = ttk.Combobox(self, values=["default", "1x736", "1x736_2scales"], state="readonly", width=15)
        self.res_cb.current(0)
        self.res_cb.pack(pady=5)

        # 3. Trial Selection (Filtered)
        tk.Label(self, text="Select Trials to Process:", font=("Arial", 10, "bold")).pack(pady=10)
        self.trial_vars = {}
        
        media_path = os.path.join(session_path, "Videos", "Cam0", "InputMedia")
        if os.path.exists(media_path):
            # 1. Get all folder names
            trials = [d for d in os.listdir(media_path) if os.path.isdir(os.path.join(media_path, d))]
            
            for t in sorted(trials):
                t_lower = t.lower()
                
                # STRICT FILTER: Do not show intrinsics OR calibration
                # Calibration is now handled automatically by reprocessOffline.py
                if t_lower in ["intrinsics", "calibration"]: 
                    continue
                
                var = tk.BooleanVar(value=True)
                self.trial_vars[t] = var
                tk.Checkbutton(self, text=t, variable=var).pack(anchor="w", padx=80)
        
        tk.Button(self, text="RUN PIPELINE", command=self.on_start, 
                  bg="#2ecc71", fg="white", font=("Arial", 11, "bold"), height=2).pack(pady=20, fill="x", padx=50)

    def get_gpu_info(self):
        try:
            cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
            res = subprocess.check_output(cmd).decode().strip().split('\n')
            return {name.strip(): str(i) for i, name in enumerate(res)}
        except:
            return {"CPU / Default": "0"}

    def on_start(self):
        selected = [t for t, v in self.trial_vars.items() if v.get()]
        if not selected:
            return messagebox.showwarning("Warning", "Select at least one trial.")
        
        # Ensure this uses the map to get the index string
        self.result = {
            "gpu_index": self.gpu_map[self.gpu_cb.get()], 
            "res": self.res_cb.get(),
            "trials": selected
        }
        self.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = OpenCapGUI(root)
    root.mainloop()