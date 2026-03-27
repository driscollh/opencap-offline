"""
    @authors: Scott Uhlrich, Antoine Falisse, Łukasz Kidziński

    Modified by: Harry G. Driscoll
    Modified date: Dec 2025

    Modfied for offline (no cloud data link) use
    
    This function calibrates the cameras, runs the pose detection algorithm, 
    reconstructs the 3D marker positions, augments the marker set,
    and runs the OpenSim pipeline.

"""

import os 
import glob
import numpy as np
import yaml
import traceback
import ffmpeg
from utilsChecker import calcIntrinsics, video2Images

import logging
logging.basicConfig(level=logging.INFO)

from utils import importMetadata, loadCameraParameters, getVideoExtension
from utils import getDataDirectory, getOpenPoseDirectory, getMMposeDirectory
from utilsChecker import saveCameraParameters
from utilsChecker import calcExtrinsicsFromVideo
from utilsChecker import isCheckerboardUpsideDown
from utilsChecker import autoSelectExtrinsicSolution
from utilsChecker import triangulateMultiviewVideo
from utilsChecker import writeTRCfrom3DKeypoints
from utilsChecker import popNeutralPoseImages
from utilsChecker import rotateIntrinsics as legacyRotateIntrinsics
from utilsChecker import calcIntrinsics, video2Images, getVideoRotation
from utilsSync import synchronizeVideos
from utilsDetector  import runPoseDetector
from utilsAugmenter import augmentTRC
from utilsOpenSim import runScaleTool, getScaleTimeRange, runIKTool, generateVisualizerJson
from defaults import DEFAULT_SYNC_VER

def get_iphone_model_from_metadata(video_path):
    try:
        meta = ffmpeg.probe(video_path)
        tags = meta.get('format', {}).get('tags', {})
        model_raw = tags.get('com.apple.quicktime.model') or tags.get('model')
        if not model_raw: return "iPhone"
        return model_raw.replace(" ", "")
    except Exception:
        return "iPhone"

def smart_rotate_intrinsics(CamParams, videoPath):
    """
    Intelligently rotates intrinsics only if the aspect ratio mismatch 
    between the video rotation and the intrinsic parameters suggests it is needed.
    """
    try:
        # 1. Get Video Rotation from Metadata
        # 90 = Portrait (Standard iPhone), 0 = Landscape
        rotation = getVideoRotation(videoPath)
        
        # 2. Get Intrinsic Dimensions
        # imageSize is [rows, cols] (Height, Width)
        # e.g., Portrait = [1920, 1080], Landscape = [1080, 1920]
        h_int, w_int = CamParams['imageSize'][0], CamParams['imageSize'][1]
        
        is_intrinsics_portrait = h_int > w_int
        
        # 3. Determine Video Orientation based on Rotation Tag
        # If rotation is 90 or 270, the video content is displayed as Portrait
        # If rotation is 0 or 180, the video content is displayed as Landscape
        is_video_portrait = (rotation == 90 or rotation == 270)
        
        logging.info(f"SmartRotation: VideoRot={rotation} (Portrait={is_video_portrait}), Intrinsics={h_int}x{w_int} (Portrait={is_intrinsics_portrait})")

        # 4. Compare and Rotate if Mismatch
        if is_video_portrait and not is_intrinsics_portrait:
            logging.info(">> Mismatch detected: Video is Portrait, Intrinsics are Landscape. ROTATING.")
            return legacyRotateIntrinsics(CamParams, videoPath)
            
        elif not is_video_portrait and is_intrinsics_portrait:
            logging.info(">> Mismatch detected: Video is Landscape, Intrinsics are Portrait. ROTATING.")
            # We force the rotation by simulating a condition legacyRotateIntrinsics expects
            # However, legacyRotateIntrinsics uses the video path to get rotation again.
            # If video tag says 0 (Landscape), legacyRotateIntrinsics WILL swap dims.
            # So calling it here is safe.
            return legacyRotateIntrinsics(CamParams, videoPath)
            
        else:
            logging.info(">> Orientation matches. No rotation needed.")
            return CamParams
            
    except Exception as e:
        logging.warning(f"SmartRotation failed: {e}. Falling back to legacy.")
        return legacyRotateIntrinsics(CamParams, videoPath)

# ------------------------

def main(sessionName, trialName, trial_id, cameras_to_use=['all'],
         intrinsicsFinalFolder='Deployed', isDocker=False,
         extrinsicsTrial=False, alternateExtrinsics=None, 
         calibrationOptions=None,
         markerDataFolderNameSuffix=None, imageUpsampleFactor=4,
         poseDetector='OpenPose', resolutionPoseDetection='default', 
         scaleModel=False, bbox_thr=0.8, augmenter_model='v0.3',
         genericFolderNames=False, offset=True, benchmark=False,
         dataDir=None, overwriteAugmenterModel=False,
         filter_frequency='default', overwriteFilterFrequency=False,
         scaling_setup='upright_standing_pose', overwriteScalingSetup=False,
         overwriteCamerasToUse=False, syncVer=None,):

    # %% High-level settings.
    runCameraCalibration = True
    runPoseDetection = True
    runSynchronization = True
    runTriangulation = True
    runMarkerAugmentation = True
    runOpenSimPipeline = True    
    
    if poseDetector == 'hrnet':
        poseDetector = 'mmpose'        
    elif poseDetector == 'openpose':
        poseDetector = 'OpenPose'
    elif poseDetector == 'rtmpose':
        poseDetector = 'RTMPose'
    
    # Unified output folder
    outputMediaFolder = f'OutputMedia_{resolutionPoseDetection}'
    
    if extrinsicsTrial:
        runCameraCalibration = True
        runPoseDetection = False
        runSynchronization = False
        runTriangulation =  False
        runMarkerAugmentation = False
        runOpenSimPipeline = False
        
    # %% Paths and metadata.
    baseDir = os.path.dirname(os.path.abspath(__file__))
    if dataDir is None:
        dataDir = getDataDirectory(isDocker)
    
    sessionDir = os.path.join(dataDir, 'Data', sessionName)
    sessionMetadata = importMetadata(os.path.join(sessionDir, 'sessionMetadata.yaml'))
    
    if 'augmentermodel' in sessionMetadata and not overwriteAugmenterModel:
        augmenterModel = sessionMetadata['augmentermodel']
    else:
        augmenterModel = augmenter_model
        
    if 'filterfrequency' in sessionMetadata and not overwriteFilterFrequency:
        filterfrequency = sessionMetadata['filterfrequency']
    else:
        filterfrequency = filter_frequency
    if filterfrequency == 'default':
        filtFreqs = {'gait':12, 'default':500}
    else:
        filtFreqs = {'gait':filterfrequency, 'default':filterfrequency}

    if 'scalingsetup' in sessionMetadata and not overwriteScalingSetup:
        scalingSetup = sessionMetadata['scalingsetup']
    else:
        scalingSetup = scaling_setup

    if 'camerastouse' in sessionMetadata and not overwriteCamerasToUse:
        camerasToUse = sessionMetadata['camerastouse']
    else:
        camerasToUse = cameras_to_use

    syncVer = syncVer or sessionMetadata.get('sync_ver', DEFAULT_SYNC_VER)

    if poseDetector == 'OpenPose':
        poseDetectorDirectory = getOpenPoseDirectory(isDocker)
    elif poseDetector == 'mmpose':
        poseDetectorDirectory = getMMposeDirectory(isDocker)    
    elif poseDetector == 'RTMPose':
        poseDetectorDirectory = os.path.join(baseDir, 'Blackwell_RTMPose')   
        
    # %% Create marker folders
    if genericFolderNames:
        markerDataFolderName = os.path.join('MarkerData') 
    else:
        markerDataFolderName = os.path.join('MarkerData', resolutionPoseDetection)

    preAugmentationDir = os.path.join(sessionDir, markerDataFolderName, 'PreAugmentation')
    os.makedirs(preAugmentationDir, exist_ok=True)
    
    postAugmentationDir = os.path.join(sessionDir, markerDataFolderName, 
                                       'PostAugmentation' if genericFolderNames else 'PostAugmentation_{}'.format(augmenterModel))
    os.makedirs(postAugmentationDir, exist_ok=True)
        
    # %% Camera calibration.
    if runCameraCalibration:    
        CheckerBoardParams = {
            'dimensions': (
                sessionMetadata['checkerBoard']['black2BlackCornersWidth_n'],
                sessionMetadata['checkerBoard']['black2BlackCornersHeight_n']),
            'squareSize': sessionMetadata['checkerBoard']['squareSideLength_mm']}       
        
        cameraDirectories = {}
        cameraModels = {}
        for pathCam in glob.glob(os.path.join(sessionDir, 'Videos', 'Cam*')):
            camName = os.path.basename(pathCam)
            cameraDirectories[camName] = pathCam
            cameraModels[camName] = sessionMetadata['iphoneModel'][camName]        
        
        CamParamDict = {}
        loadedCamParams = {}
        for camName in cameraDirectories:
            camDir = cameraDirectories[camName]
            
            # 1. Check if combined calibration already exists
            if os.path.exists(os.path.join(camDir,"cameraIntrinsicsExtrinsics.pickle")):
                logging.info(f"Load extrinsics for {camName} - already existing")
                CamParams = loadCameraParameters(os.path.join(camDir, "cameraIntrinsicsExtrinsics.pickle"))
                loadedCamParams[camName] = True
            else:
                logging.info(f"Compute extrinsics for {camName} - not yet existing")
                phoneName = cameraModels[camName]
                
                # --- AUTO-DETECT IPHONE MODEL ---
                if phoneName == "iPhone_Auto_Detect":
                    pathVideoTrial = os.path.join(camDir, 'InputMedia', trialName, trial_id)
                    full_video_path = pathVideoTrial + getVideoExtension(pathVideoTrial)
                    
                    if not os.path.exists(full_video_path):
                         calib_vid = os.path.join(camDir, 'InputMedia', 'calibration', 'calibration.mp4')
                         if os.path.exists(calib_vid): full_video_path = calib_vid
                    
                    if os.path.exists(full_video_path):
                        raw_phone = get_iphone_model_from_metadata(full_video_path)
                        # FIX: Apply the same _CamX suffix so it matches the custom library folder
                        phoneName = f"{raw_phone}_{camName}" 
                        cameraModels[camName] = phoneName 
                    else:
                        phoneName = f"iPhone12_{camName}" # Fallback

                localIntrinsicFile = os.path.join(camDir, 'cameraIntrinsics.pickle')
                customIntrinsicPath = os.path.join(baseDir, 'CustomCameraIntrinsics', phoneName, 'cameraIntrinsics.pickle')

                # Priority 1: Local file
                if genericFolderNames and os.path.exists(localIntrinsicFile):
                    CamParams = loadCameraParameters(localIntrinsicFile)
                # Priority 2: Custom library
                elif os.path.exists(customIntrinsicPath):
                    CamParams = loadCameraParameters(customIntrinsicPath)
                # Priority 3: Default library
                elif os.path.exists(os.path.join(baseDir, 'CameraIntrinsics', phoneName, intrinsicsFinalFolder, 'cameraIntrinsics.pickle')):
                    CamParams = loadCameraParameters(os.path.join(baseDir, 'CameraIntrinsics', phoneName, intrinsicsFinalFolder, 'cameraIntrinsics.pickle'))
                # Priority 4: Auto-Calculate (Android/Other)
                else:
                    logging.info(f"Priority 4: Intrinsics missing for {phoneName}. Calculating from video...")
                    calibVidPath = os.path.join(camDir, 'InputMedia', 'calibration', 'calibration.mp4')
                    if not os.path.exists(calibVidPath): # Try other extensions
                         for ext in ['.mov', '.avi']:
                             if os.path.exists(calibVidPath.replace('.mp4', ext)): 
                                 calibVidPath = calibVidPath.replace('.mp4', ext); break
                    
                    if os.path.exists(calibVidPath):
                        imageFolder = os.path.join(camDir, 'InputMedia', 'calibration', 'IntrinsicImages')
                        os.makedirs(imageFolder, exist_ok=True)
                        video2Images(calibVidPath, nImages=20, outputFolder=imageFolder, filePrefix='int')
                        CamParams = calcIntrinsics(imageFolder, CheckerBoardParams, visualize=False, saveFileName=localIntrinsicFile)
                        if CamParams is None: raise Exception(f"Intrinsics calculation failed for {camName}")
                    else:
                        raise Exception(f"Intrinsics not found and no calibration video at {calibVidPath}")
                        
                # Extrinsics Calculation
                useSecondExtrinsicsSolution = (alternateExtrinsics is not None and camName in alternateExtrinsics)
                pathVideoWithoutExtension = os.path.join(camDir, 'InputMedia', trialName, trial_id)
                extension = getVideoExtension(pathVideoWithoutExtension)
                extrinsicPath = pathVideoWithoutExtension + extension
                                              
                # Ensure FFmpeg is available for the rotation check
                ffmpeg_path = os.path.join(baseDir, "dependencies", "ffmpeg", "bin")
                if ffmpeg_path not in os.environ["PATH"]:
                    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

                # --- SMART ROTATION (PRIORITY 3) ---
                try:
                    CamParams = smart_rotate_intrinsics(CamParams, extrinsicPath)
                except Exception as ffmpeg_err:
                    logging.warning(f"Smart Rotation failed: {ffmpeg_err}. Proceeding without rotation check.")
                # -----------------------------------
                
                try:
                    CamParams = calcExtrinsicsFromVideo(
                        extrinsicPath, CamParams, CheckerBoardParams, 
                        visualize=False, imageUpsampleFactor=imageUpsampleFactor,
                        useSecondExtrinsicsSolution=useSecondExtrinsicsSolution)
                except Exception as e:
                    exception = "Camera calibration failed. Verify checkerboard visibility."
                    raise Exception(exception, traceback.format_exc())
                loadedCamParams[camName] = False
                
            if CamParams is not None:
                CamParamDict[camName] = CamParams.copy()

        if not all([loadedCamParams[i] for i in loadedCamParams]):
            for camName in CamParamDict:
                saveCameraParameters(os.path.join(cameraDirectories[camName], "cameraIntrinsicsExtrinsics.pickle"), CamParamDict[camName])
            
    # %% 3D reconstruction & Remainder of Pipeline
    pathOutputFiles = {trialName: os.path.join(preAugmentationDir, (trialName if benchmark else trial_id) + ".trc")}
    
    if runPoseDetection:
        checkerBoardMount = sessionMetadata['checkerBoard']['placement']
        if checkerBoardMount in ['backWall', 'Perpendicular', 'Vertical']:
            rotationAngles = {'y': -90} if isCheckerboardUpsideDown(CamParamDict) else {'y': 90, 'z': 180}
        elif checkerBoardMount in ['ground', 'Lying', 'Horizontal']:
            rotationAngles = {'x': 90, 'y': 90}
             
        cameras_available = []
        for camName in cameraDirectories:
            pathVid = os.path.join(cameraDirectories[camName], 'InputMedia', trialName, trial_id)
            if len(glob.glob(pathVid + '*')) > 0:
                cameras_available.append(camName)

        if camerasToUse[0] == 'all':
            camerasToUse_c = list(cameraDirectories.keys())
        elif camerasToUse[0] == 'all_available':
            camerasToUse_c = cameras_available
        else:
            camerasToUse_c = camerasToUse

        if len(camerasToUse_c) < 2:
            raise Exception("At least two videos are required for triangulation.", "Video upload failure.")

    if runSynchronization:
        videoExtension = getVideoExtension(os.path.join(cameraDirectories[camerasToUse_c[0]], 'InputMedia', trialName, trial_id))
        trialRelativePath = os.path.join('InputMedia', trialName, trial_id + videoExtension)
        
        keypoints2D, confidence, keypointNames, frameRate, nansInOut, startEndFrames, cameras2Use = synchronizeVideos( 
            cameraDirectories, trialRelativePath, poseDetectorDirectory,
            undistortPoints=True, CamParamDict=CamParamDict, filtFreqs=filtFreqs, 
            confidenceThreshold=0.4, cams2Use=camerasToUse_c, poseDetector=poseDetector, 
            trialName=trialName, resolutionPoseDetection=resolutionPoseDetection, syncVer=syncVer)
                
    if runTriangulation:
        keypoints3D, confidence3D = triangulateMultiviewVideo(
            CamParamDict, keypoints2D, ignoreMissingMarkers=False, cams2Use=cameras2Use, 
            confidenceDict=confidence, spline3dZeros=True, splineMaxFrames=int(frameRate/5), 
            nansInOut=nansInOut, CameraDirectories=cameraDirectories, trialName=trialName, 
            startEndFrames=startEndFrames, trialID=trial_id, outputMediaFolder=outputMediaFolder)
        
        writeTRCfrom3DKeypoints(keypoints3D, pathOutputFiles[trialName], keypointNames, frameRate=frameRate, rotationAngles=rotationAngles)

    # %% Augmentation & OpenSim
    augmenterModelName = sessionMetadata['markerAugmentationSettings']['markerAugmenterModel']
    pathAugmentedOutputFiles = {trialName: os.path.join(postAugmentationDir, trial_id + (".trc" if genericFolderNames else "_" + augmenterModelName + ".trc"))}
    
    if runMarkerAugmentation:
        vertical_offset = augmentTRC(pathOutputFiles[trialName], sessionMetadata['mass_kg'], sessionMetadata['height_m'], 
                                     pathAugmentedOutputFiles[trialName], os.path.join(baseDir, "MarkerAugmenter"), 
                                     augmenterModelName=augmenterModelName, augmenter_model=augmenterModel, offset=offset)
        if offset:
            vertical_offset_settings = float(np.copy(vertical_offset)-0.01)
            vertical_offset = 0.01   
        
    if runOpenSimPipeline:
        openSimPipelineDir = os.path.join(baseDir, "opensimPipeline")
        openSimDir = os.path.join(sessionDir, 'OpenSimData' if genericFolderNames else os.path.join('OpenSimData', resolutionPoseDetection))        
        outputScaledModelDir = os.path.join(openSimDir, 'Model')
        suffix_model = '_shoulder' if 'shoulder' in sessionMetadata['openSimModel'] else ''
        
        if scaleModel:
            os.makedirs(outputScaledModelDir, exist_ok=True)
            pathGenericModel4Scaling = os.path.join(openSimPipelineDir, 'Models', sessionMetadata['openSimModel'] + '.osim')            
            timeRange4Scaling = getScaleTimeRange(pathAugmentedOutputFiles[trialName])
            pathScaledModel = runScaleTool(os.path.join(openSimPipelineDir, 'Scaling', 'Setup_scaling_LaiUhlrich2022.xml'), 
                                           pathGenericModel4Scaling, sessionMetadata['mass_kg'], pathAugmentedOutputFiles[trialName], 
                                           timeRange4Scaling, outputScaledModelDir, subjectHeight=sessionMetadata['height_m'], suffix_model=suffix_model)
            neutral_img_dir = os.path.join(sessionDir, 'NeutralPoseImages') if genericFolderNames else os.path.join(sessionDir, 'NeutralPoseImages', resolutionPoseDetection)            
            popNeutralPoseImages(cameraDirectories, cameras2Use, timeRange4Scaling[0], neutral_img_dir, trial_id, writeVideo=True)   
            pathOutputIK, pathModelIK = pathScaledModel[:-5] + '.mot', pathScaledModel
        else:
            pathScaledModel = os.path.join(outputScaledModelDir, sessionMetadata['openSimModel'] + "_scaled.osim")
            pathOutputIK, pathModelIK = runIKTool(os.path.join(openSimPipelineDir, 'IK', f'Setup_IK{suffix_model}.xml'), 
                                                  pathScaledModel, pathAugmentedOutputFiles[trialName], os.path.join(openSimDir, 'Kinematics'))
        
        vis_json_dir = os.path.join(sessionDir, 'VisualizerJsons') if genericFolderNames else os.path.join(sessionDir, 'VisualizerJsons', resolutionPoseDetection)
        os.makedirs(os.path.join(vis_json_dir, trialName), exist_ok=True)
        generateVisualizerJson(pathModelIK, pathOutputIK, os.path.join(vis_json_dir, trialName, trialName + '.json'), 
                               vertical_offset=vertical_offset)
        
    # %% Rewrite settings, adding offset  
    if not extrinsicsTrial:
        # Check if settings exists before trying to write to it
        if 'settings' in locals() and offset:
            settings['verticalOffset'] = vertical_offset_settings 
            with open(pathSettings, 'w') as file:
                yaml.dump(settings, file)