import os
import cv2
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from fastprogress import progress_bar
from concurrent.futures import ThreadPoolExecutor, as_completed

def intrinsic_calibration(image_paths, run_name, camera, checkerboard_size=(8,6), world_scaling=1, sample_rate=None):

    os.makedirs('matrix/internal/', exist_ok=True)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = world_scaling* objp

    objpoints = []  
    imgpoints_A = [] 
    images_A = image_paths
    images_A = sorted(images_A, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for img_path_A in progress_bar(images_A, total=len(images_A)):
        img_A = cv2.imread(img_path_A)
        gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        ret_A, corners_A = cv2.findChessboardCorners(gray_A, checkerboard_size, None)
        if ret_A:
            objpoints.append(objp)
            corners2_A = cv2.cornerSubPix(gray_A, corners_A, (11,11), (-1,-1), criteria=criteria)
            imgpoints_A.append(corners2_A)
    if sample_rate is not None:
        objpoints = objpoints[::sample_rate]
        imgpoints_A = imgpoints_A[::sample_rate]

    ret_A, mtx_A, dist_A, rvecs_A, tvecs_A = cv2.calibrateCamera(objpoints, imgpoints_A, gray_A.shape[::-1], None, None)
    np.savez(f'matrix/internal/{run_name}_C{camera}_internal_calibration.npz', mint=mtx_A, dist=dist_A)
    print(f'Camera A calibration done. Intrinsic matrix saved as matrix/internal/{run_name}_internal_calibration.npz')

def stereo_cal(run_name, objpoints, imgpoints_A, imgpoints_B, valid_frames_A, valid_frames_B, file_num, gray_A_shape, pair, main, repeat):

    if len(imgpoints_A) >= file_num:
        random_indices = random.sample(range(len(imgpoints_A)), file_num)
    else:
        random_indices = random.sample(range(len(imgpoints_A)), len(imgpoints_A))

    objpoints_sample = [objpoints[i] for i in random_indices]
    imgpoints_A_sample = [imgpoints_A[i] for i in random_indices]
    imgpoints_B_sample = [imgpoints_B[i] for i in random_indices]
    valid_frames_A_sample = [valid_frames_A[i] for i in random_indices]
    valid_frames_B_sample = [valid_frames_B[i] for i in random_indices]

    data_internal1 = np.load(f'matrix/internal/{run_name}_C{main}_internal_calibration.npz')
    mtx_A = data_internal1['mint']
    dist_A = data_internal1['dist']
    data_internal2 = np.load(f'matrix/internal/{run_name}_C{pair}_internal_calibration.npz')
    mtx_B = data_internal2['mint']
    dist_B = data_internal2['dist']

    # Stereo calibration
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        # objpoints_sample, imgpoints_A_sample, imgpoints_B_sample, mtx_A, dist_A, mtx_B, dist_B, gray_A_shape
        objpoints_sample, imgpoints_A_sample, imgpoints_B_sample, mtx_A, dist_A, mtx_B, dist_B, gray_A_shape, flags=stereocalibration_flags, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    )

    valid_frames_A_sample = sorted(valid_frames_A_sample, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    valid_frames_B_sample = sorted(valid_frames_B_sample, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    save_path = f'matrix/external/main{main}_pair{pair}_external_calibration_repeat_{repeat}_{file_num}_frames.npz'
    np.savez(save_path, R=R, T=T, min1=cameraMatrix1, dist1=distCoeffs1, mint2=cameraMatrix2, dist2=distCoeffs2, mtx_A=mtx_A, dist_A=dist_A, mtx_B=mtx_B, dist_B=dist_B)
    
    return True, {'pair': pair, 'repeat': repeat, 'num_frames': file_num, 'valid_A': len(imgpoints_A_sample), 'valid_B': len(imgpoints_B_sample), 'retval': retval, 'valid_frames_A': valid_frames_A_sample, 'valid_frames_B': valid_frames_B_sample, 'path': save_path}


def optimize_stereo_cal(main_cam_paths, pair_cam_paths, run_name, main_cam, pair_cam, checkerboard_size=(8,6), world_scaling=1, rotate=False, min_frames=5, max_frames=10, repititions=1000):

    os.makedirs('matrix/external/', exist_ok=True)
    os.makedirs('csvs/', exist_ok=True)

    data = []
    main = main_cam
    pair = pair_cam
    valid_frames_A = []
    valid_frames_B = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = world_scaling* objp

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints_A = []  # 2d points in image plane for Camera A.
    imgpoints_B = []  # 2d points in image plane for Camera B.

    # Load the images for Camera A and Camera B
    images_A = main_cam_paths
    images_A = sorted(images_A, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    images_B = pair_cam_paths
    images_B = sorted(images_B, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Assuming you have the same number of images for both cameras
    for img_path_A, img_path_B in zip(images_A, images_B):
        img_A = cv2.imread(img_path_A)
        img_B = cv2.imread(img_path_B)

        if rotate:
            img_A = cv2.rotate(img_A, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_B = cv2.rotate(img_B, cv2.ROTATE_90_COUNTERCLOCKWISE)

        gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        gray_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_A, corners_A = cv2.findChessboardCorners(gray_A, checkerboard_size, None)
        ret_B, corners_B = cv2.findChessboardCorners(gray_B, checkerboard_size, None)

        # If found, add object points, image points (after refining them)
        if ret_A and ret_B:
            valid_frames_A.append(img_path_A)
            valid_frames_B.append(img_path_B)
            objpoints.append(objp)

            corners2_A = cv2.cornerSubPix(gray_A, corners_A, (11,11), (-1,-1), criteria=criteria)
            imgpoints_A.append(corners2_A)

            corners2_B = cv2.cornerSubPix(gray_B, corners_B, (11,11), (-1,-1), criteria=criteria)
            imgpoints_B.append(corners2_B)

    pbar = tqdm(total=((max_frames-min_frames)*repititions), desc='External Calibration: ')

    for file_num in range(min_frames, max_frames+1):
        futures = []
        with ThreadPoolExecutor() as executor:
            for repeat in range(repititions):
                # Submit the task to the executor
                future = executor.submit(stereo_cal, run_name, objpoints, imgpoints_A, imgpoints_B,
                                        valid_frames_A, valid_frames_B, file_num, gray_A.shape[::-1],
                                        gray_B.shape[::-1], pair, main, repeat)
                futures.append(future)

            # As tasks are completed, update the progress bar and collect results
            for future in as_completed(futures):
                flag, res = future.result()
                data.append(res)
                pbar.update(1)

    df = pd.DataFrame(data)
    df.to_csv(f'csvs/{run_name}_external_calibrations_repeats_full_main{main}_pair{pair}.csv', index=False)
    print(f'External calibration done. Results saved as csvs/{run_name}_external_calibrations_repeats_full_main{main}_pair{pair}.csv')



