import yaml
import json
import os
import socket
import pickle
import glob
import subprocess
import datetime
import shutil

import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial.transform import Rotation as R
from numpy.lib.recfunctions import append_fields
import utilsDataman

# %% Directory and Path Utilities

def getDataDirectory(isDocker=False):
    computername = socket.gethostname()
    # Define paths based on hostname or Docker status
    if isDocker:
        dataDir = os.getcwd()
    else:
        # Default to current working directory if no specific override found
        dataDir = os.getcwd()
    return dataDir

def getOpenPoseDirectory(isDocker=False):
    if isDocker:
        openPoseDirectory = "docker"
    else:
        # Default local path - user should ensure this matches their config
        openPoseDirectory = r"C:\openpose" 
    return openPoseDirectory

def getMMposeDirectory(isDocker=False):
    # MMpose placeholder
    return ''

def loadCameraParameters(filename):
    open_file = open(filename, "rb")
    cameraParams = pickle.load(open_file)
    open_file.close()
    return cameraParams

def importMetadata(filePath):
    with open(filePath) as myYamlFile:
        parsedYamlFile = yaml.load(myYamlFile, Loader=yaml.FullLoader)
    return parsedYamlFile

def deleteCalibrationFiles(session_path):
    calImagePath = os.path.join(session_path, 'CalibrationImages')
    if os.path.exists(calImagePath):
        shutil.rmtree(calImagePath)
    
    # Delete camera directories
    camDirs = glob.glob(os.path.join(session_path, 'Videos', 'Cam*'))
    
    # Find extrinsic Filename
    extrinsicFileFound = False
    extrinsicTrialName = None
    if len(camDirs) > 1 and os.path.exists(os.path.join(camDirs[0], 'InputMedia')):
        inputDir = os.path.join(camDirs[0], 'InputMedia')
        dirContents = os.listdir(inputDir)
        trialNames = [tName for tName in dirContents if os.path.isdir(os.path.join(inputDir, tName))]
        for tName in trialNames:
            if os.path.exists(os.path.join(inputDir, tName, 'extrinsicImage0.png')):
                extrinsicTrialName = tName
                extrinsicFileFound = True
    
    for camDir in camDirs:
        extPath = os.path.join(camDir, 'cameraIntrinsicsExtrinsics.pickle')
        if os.path.exists(extPath):
            os.remove(extPath)
        # Delete extrinsic folder
        if extrinsicFileFound:
            extFolder = os.path.join(camDir, 'InputMedia', extrinsicTrialName)
            if os.path.isdir(extFolder):
                shutil.rmtree(extFolder)

def deleteStaticFiles(session_path, staticTrialName='neutral'):
    vidDir = os.path.join(session_path, 'Videos')
    camDirs = glob.glob(os.path.join(vidDir, 'Cam*'))
    markerDirs = glob.glob(os.path.join(session_path, 'MarkerData'))
    openSimDir = os.path.join(session_path, 'OpenSimData')
    
    # Delete media folders
    for camDir in camDirs:
        mediaDirs = glob.glob(os.path.join(camDir, '*'))
        for medDir in mediaDirs:
            try:
                shutil.rmtree(os.path.join(camDir, medDir, staticTrialName))
                print(f'Deleting {medDir}/{staticTrialName}')
            except:
                pass
            
    # Delete marker data
    for mkrDir in markerDirs:
        mkrFiles = glob.glob(os.path.join(mkrDir, '*'))
        for mkrFile in mkrFiles:
            if staticTrialName in mkrFile:
                os.remove(mkrFile)
                print(f'Deleting {os.path.basename(mkrFile)}')
           
    if os.path.exists(openSimDir):
        shutil.rmtree(openSimDir)
        print('Deleting openSimDir')

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def getVideoExtension(pathFileWithoutExtension):
    pathVideoDir = os.path.split(pathFileWithoutExtension)[0]
    videoName = os.path.split(pathFileWithoutExtension)[1]
    extension = ''
    if os.path.exists(pathVideoDir):
        for file in os.listdir(pathVideoDir):
            if videoName == file.rsplit('.', 1)[0]:
                extension = '.' + file.rsplit('.', 1)[1]
                break
    return extension

def checkCudaTF():
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Found {len(gpus)} GPU(s).")
        for gpu in gpus:
            print(f"GPU: {gpu.name}")
    else:
        print("No GPU detected by TensorFlow.")

def writeToJsonLog(path, new_dict, max_entries=1000, indent=2):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(new_dict)

    while len(data) > max_entries:
        data.pop(0)

    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

def writeToErrorLog(path, session_id, trial_id, error, stack, max_entries=1000):
    error_entry = {
        'session_id': session_id,
        'trial_id': trial_id,
        'datetime': datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        'error': str(error),
        'stack': stack
    }
    writeToJsonLog(path, error_entry, max_entries)

def getCommitHash():
    return os.getenv('GIT_COMMIT_HASH')

def getHostname():
    return socket.gethostname()

def getGendersDict():
    genders_dict = {
          "woman": "Woman",
          "man": "Man",
          "transgender": "Transgender",
          "non-binary": "Non-Binary/Non-Conforming",
          "prefer-not-respond": "Prefer not to respond",
        }
    return genders_dict

# %% TRC and Data Conversion Utilities

def numpy2TRC(f, data, headers, fc=50.0, t_start=0.0, units="m"):
    
    header_mapping = {}
    for count, header in enumerate(headers):
        header_mapping[count+1] = header 
        
    # Line 1.
    f.write('PathFileType  4\t(X/Y/Z) %s\n' % os.getcwd())
    
    # Line 2.
    f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\t'
                'Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    
    num_frames=data.shape[0]
    num_markers=len(header_mapping.keys())
    
    # Line 3.
    f.write('%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n' % (
            fc, fc, num_frames,
            num_markers, units, fc,
            1, num_frames))
    
    # Line 4.
    f.write("Frame#\tTime\t")
    for key in sorted(header_mapping.keys()):
        f.write("%s\t\t\t" % format(header_mapping[key]))

    # Line 5.
    f.write("\n\t\t")
    for imark in np.arange(num_markers) + 1:
        f.write('X%i\tY%s\tZ%s\t' % (imark, imark, imark))
    f.write('\n')
    
    # Line 6.
    f.write('\n')

    for frame in range(data.shape[0]):
        f.write("{}\t{:.8f}\t".format(frame+1,(frame)/fc+t_start)) # opensim frame labeling is 1 indexed

        for key in sorted(header_mapping.keys()):
            f.write("{:.5f}\t{:.5f}\t{:.5f}\t".format(data[frame,0+(key-1)*3], data[frame,1+(key-1)*3], data[frame,2+(key-1)*3]))
        f.write("\n")

def numpy2storage(labels, data, storage_file):
    
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"
    
    f = open(storage_file, 'w')
    f.write('name %s\n' %storage_file)
    f.write('datacolumns %d\n' %data.shape[1])
    f.write('datarows %d\n' %data.shape[0])
    f.write('range %f %f\n' %(np.min(data[:, 0]), np.max(data[:, 0])))
    f.write('endheader \n')
    
    for i in range(len(labels)):
        f.write('%s\t' %labels[i])
    f.write('\n')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' %data[i, j])
        f.write('\n')
        
    f.close() 

def storage2numpy(storage_file, excess_header_entries=0):
    f = open(storage_file, 'r')
    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names, skip_header=skip_header)

    return data

def storage2df(storage_file, headers):
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    return out

def TRC2numpy(pathFile, markers, rotation=None):
    trc_file = utilsDataman.TRCFile(pathFile)
    time = trc_file.time
    num_frames = time.shape[0]
    data = np.zeros((num_frames, len(markers)*3))
    
    if rotation is not None:
        for axis, angle in rotation.items():
            trc_file.rotate(axis, angle)
    for count, marker in enumerate(markers):
        data[:,3*count:3*count+3] = trc_file.marker(marker)    
    this_dat = np.empty((num_frames, 1))
    this_dat[:, 0] = time
    data_out = np.concatenate((this_dat, data), axis=1)
    
    return data_out

# %% Filtering and Kinematics

def lowpassFilter(inputData, filtFreq, order=4):
    time = inputData[:,0]
    fs = 1/np.mean(np.diff(time))
    wn = filtFreq/(fs/2)
    sos = signal.butter(order/2, wn, btype='low', output='sos')
    inputDataFilt = signal.sosfiltfilt(sos, inputData[:,1:], axis=0)    
    data = np.concatenate((np.expand_dims(time, 1), inputDataFilt), axis=1)
    return data

def getIK(storage_file, joints, degrees=False):
    data = storage2numpy(storage_file)
    Qs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, joint in enumerate(joints):  
        if joint in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
            Qs.insert(count + 1, joint, data[joint])         
        else:
            if degrees:
                Qs.insert(count + 1, joint, data[joint])                  
            else:
                Qs.insert(count + 1, joint, data[joint] * np.pi / 180)              
            
    fs = 1/np.mean(np.diff(Qs['time']))    
    fc = 6
    order = 4
    w = fc / (fs / 2)
    b, a = signal.butter(order/2, w, 'low')  
    output = signal.filtfilt(b, a, Qs.loc[:, Qs.columns != 'time'], axis=0, 
                             padtype='odd', padlen=3*(max(len(b), len(a))-1))    
    output = pd.DataFrame(data=output, columns=joints)
    QsFilt = pd.concat([pd.DataFrame(data=data['time'], columns=['time']), 
                        output], axis=1)    
    
    return Qs, QsFilt

# %% Video Utilities

def rewriteVideos(inputPath, startFrame, nFrames, frameRate, outputDir=None,
                  imageScaleFactor=.5, outputFileName=None):
        
    inputDir, vidName = os.path.split(inputPath)
    vidName, vidExt = os.path.splitext(vidName)

    if outputFileName is None:
        outputFileName = vidName + '_sync' + vidExt
    if outputDir is not None:
        outputFullPath = os.path.join(outputDir, outputFileName)
    else:
        outputFullPath = os.path.join(inputDir, outputFileName)
      
    imageScaleArg = '' 
    maintainQualityArg = '-acodec copy -vcodec copy'
    if imageScaleFactor is not None:
        imageScaleArg = '-vf scale=iw/{:.0f}:-1'.format(1/imageScaleFactor)
        maintainQualityArg = ''

    startTime = startFrame/frameRate

    ffmpegCmd = "ffmpeg -loglevel error -y -ss {:.3f} -i {} {} -vframes {:.0f} {} {}".format(
                startTime, inputPath, maintainQualityArg, 
                nFrames, imageScaleArg, outputFullPath).rstrip().replace("  ", " ")

    subprocess.run(ffmpegCmd.split(" "))
    return

# %% Marker Definitions

def getOpenPoseMarkerNames():
    markerNames = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                   "LShoulder", "LElbow", "LWrist", "midHip", "RHip",
                   "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                   "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                   "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    return markerNames

def getOpenPoseFaceMarkers():
    faceMarkerNames = ['Nose', 'REye', 'LEye', 'REar', 'LEar']
    markerNames = getOpenPoseMarkerNames()
    idxFaceMarkers = [markerNames.index(i) for i in faceMarkerNames]
    return faceMarkerNames, idxFaceMarkers

def getMMposeMarkerNames():
    markerNames = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", 
                   "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
                   "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
                   "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe",
                   "RHeel"]        
    return markerNames

def getOpenPoseMarkers_fullBody():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]
    return feature_markers, response_markers

def getMMposeMarkers_fullBody():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe", 
        "RElbow", "LElbow", "RWrist", "LWrist"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]
    return feature_markers, response_markers        

def getOpenPoseMarkers_lowerExtremity():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]
    return feature_markers, response_markers

def getOpenPoseMarkers_lowerExtremity2():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]

    response_markers = [
        'r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study',
        'L.PSIS_study', 'r_knee_study', 'r_mknee_study', 
        'r_ankle_study', 'r_mankle_study', 'r_toe_study', 
        'r_5meta_study', 'r_calc_study', 'L_knee_study', 
        'L_mknee_study', 'L_ankle_study', 'L_mankle_study',
        'L_toe_study', 'L_calc_study', 'L_5meta_study', 
        'r_shoulder_study', 'L_shoulder_study', 'C7_study', 
        'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
        'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
        'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 'L_sh1_study',
        'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']
    return feature_markers, response_markers

def getMMposeMarkers_lowerExtremity():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe"]

    response_markers = ["C7_study", "r_shoulder_study", "L_shoulder_study",
                        "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", 
                        "L.PSIS_study", "r_knee_study", "L_knee_study",
                        "r_mknee_study", "L_mknee_study", "r_ankle_study", 
                        "L_ankle_study", "r_mankle_study", "L_mankle_study",
                        "r_calc_study", "L_calc_study", "r_toe_study", 
                        "L_toe_study", "r_5meta_study", "L_5meta_study",
                        "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
                        "L_thigh1_study", "L_thigh2_study", "L_thigh3_study", 
                        "r_sh1_study", "r_sh2_study", "r_sh3_study", 
                        "L_sh1_study", "L_sh2_study", "L_sh3_study",
                        "RHJC_study", "LHJC_study"]
    return feature_markers, response_markers

def getMarkers_upperExtremity_pelvis():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RElbow", "LElbow",
        "RWrist", "LWrist"]

    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]
    return feature_markers, response_markers

def getMarkers_upperExtremity_noPelvis():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist",
        "LWrist"]

    response_markers = ["r_lelbow_study", "L_lelbow_study", "r_melbow_study",
                        "L_melbow_study", "r_lwrist_study", "L_lwrist_study",
                        "r_mwrist_study", "L_mwrist_study"]
    return feature_markers, response_markers

def getMarkers_upperExtremity_noPelvis2():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist",
        "LWrist"]

    response_markers = ["r_lelbow_study", "r_melbow_study", "r_lwrist_study",
                        "r_mwrist_study", "L_lelbow_study", "L_melbow_study",
                        "L_lwrist_study", "L_mwrist_study"]
    return feature_markers, response_markers