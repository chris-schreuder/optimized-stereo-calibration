import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def intrinsic_calibration(image_paths, run_name, camera, checkerboard_size=(8,6), world_scaling=1, sample_rate=None):
    """
    Perform intrinsic calibration for a camera using chessboard pattern images and save calibration results.

    Parameters:
    image_paths (list): List of paths to calibration images.
    run_name (str): Name of the calibration run.
    camera (str): Identifier for the camera (e.g., 'A', 'B').
    checkerboard_size (tuple, optional): Size of the checkerboard pattern (cols, rows). Defaults to (8, 6).
    world_scaling (float, optional): Scaling factor for world coordinates. Defaults to 1.
    sample_rate (int, optional): Rate of subsampling for calibration images. Defaults to None (no subsampling).

    Returns:
    None
    """
    os.makedirs('matrix/internal/', exist_ok=True)

    # Criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # Initialize object points based on checkerboard pattern
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= world_scaling

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Sort images by their numeric suffix
    images = sorted(image_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for img_path in tqdm(images, desc=f'Calibrating Camera {camera}'):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if ret:
            objpoints.append(objp)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=criteria)
            imgpoints.append(corners_refined)

    # Optionally subsample points
    if sample_rate is not None:
        objpoints = objpoints[::sample_rate]
        imgpoints = imgpoints[::sample_rate]

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save calibration results
    np.savez(f'matrix/internal/{run_name}_C{camera}_internal_calibration.npz', mint=mtx, dist=dist)
    print(f'Camera {camera} calibration completed. Intrinsic matrix saved as matrix/internal/{run_name}_C{camera}_internal_calibration.npz')

def stereo_cal(run_name, objpoints, imgpoints_A, imgpoints_B, valid_frames_A, valid_frames_B, file_num, gray_A_shape, pair, main, repeat):
    """
    Perform stereo calibration between two cameras using selected image points and save calibration results.

    Parameters:
    run_name (str): Name of the calibration run.
    objpoints (list): List of object points in real world coordinates.
    imgpoints_A (list): List of image points from Camera A.
    imgpoints_B (list): List of image points from Camera B.
    valid_frames_A (list): List of valid image paths for Camera A.
    valid_frames_B (list): List of valid image paths for Camera B.
    file_num (int): Number of frames used in calibration.
    gray_A_shape (tuple): Shape of grayscale image from Camera A.
    pair (int): Identifier for the paired camera.
    main (int): Identifier for the main camera.
    repeat (int): Repetition index for calibration.

    Returns:
    tuple: Tuple containing success flag and dictionary of calibration results.
        Success flag (bool): Indicates if calibration was successful.
        Dictionary: Contains calibration results including paths, valid frames, and calibration parameters.
    """
    # Randomly sample points for calibration
    if len(imgpoints_A) >= file_num:
        random_indices = random.sample(range(len(imgpoints_A)), file_num)
    else:
        random_indices = random.sample(range(len(imgpoints_A)), len(imgpoints_A))

    objpoints_sample = [objpoints[i] for i in random_indices]
    imgpoints_A_sample = [imgpoints_A[i] for i in random_indices]
    imgpoints_B_sample = [imgpoints_B[i] for i in random_indices]
    valid_frames_A_sample = [valid_frames_A[i] for i in random_indices]
    valid_frames_B_sample = [valid_frames_B[i] for i in random_indices]

    # Load internal calibration data for both cameras
    data_internal1 = np.load(f'matrix/internal/{run_name}_C{main}_internal_calibration.npz')
    mtx_A = data_internal1['mint']
    dist_A = data_internal1['dist']
    data_internal2 = np.load(f'matrix/internal/{run_name}_C{pair}_internal_calibration.npz')
    mtx_B = data_internal2['mint']
    dist_B = data_internal2['dist']

    # Set stereo calibration flags
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT

    # Perform stereo calibration
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints_sample, imgpoints_A_sample, imgpoints_B_sample, mtx_A, dist_A, mtx_B, dist_B, gray_A_shape,
        flags=stereocalibration_flags, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    )

    # Sort valid frames by numeric suffix
    valid_frames_A_sample = sorted(valid_frames_A_sample, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    valid_frames_B_sample = sorted(valid_frames_B_sample, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Save stereo calibration results
    save_path = f'matrix/external/main{main}_pair{pair}_external_calibration_repeat_{repeat}_{file_num}_frames.npz'
    np.savez(save_path, R=R, T=T, min1=cameraMatrix1, dist1=distCoeffs1, mint2=cameraMatrix2, dist2=distCoeffs2,
             mtx_A=mtx_A, dist_A=dist_A, mtx_B=mtx_B, dist_B=dist_B)

    # Return success flag and calibration results
    return True, {
        'pair': pair,
        'repeat': repeat,
        'num_frames': file_num,
        'valid_A': len(imgpoints_A_sample),
        'valid_B': len(imgpoints_B_sample),
        'retval': retval,
        'valid_frames_A': valid_frames_A_sample,
        'valid_frames_B': valid_frames_B_sample,
        'path': save_path
    }


def optimize_stereo_cal(main_cam_paths, pair_cam_paths, run_name, main_cam, pair_cam, checkerboard_size=(8,6), 
                        world_scaling=1, rotate=False, min_frames=5, max_frames=10, repetitions=1000):
    """
    Optimize stereo calibration between two cameras using a range of frames and repetitions, saving results to CSV.

    Parameters:
    main_cam_paths (list): List of paths to images from the main camera.
    pair_cam_paths (list): List of paths to images from the paired camera.
    run_name (str): Name of the calibration run.
    main_cam (int): Identifier for the main camera.
    pair_cam (int): Identifier for the paired camera.
    checkerboard_size (tuple, optional): Size of the checkerboard pattern (default is (8, 6)).
    world_scaling (float, optional): Scaling factor for real-world coordinates (default is 1).
    rotate (bool, optional): Flag indicating whether images should be rotated (default is False).
    min_frames (int, optional): Minimum number of frames to use for calibration (default is 5).
    max_frames (int, optional): Maximum number of frames to use for calibration (default is 10).
    repetitions (int, optional): Number of repetitions for each frame count (default is 1000).

    Returns:
    None
    """
    # Create directories if they don't exist
    os.makedirs('matrix/external/', exist_ok=True)
    os.makedirs('csvs/', exist_ok=True)

    data = []  # List to store calibration results
    main = main_cam  # Alias for main camera identifier
    pair = pair_cam  # Alias for paired camera identifier
    valid_frames_A = []  # List to store valid image paths for Camera A
    valid_frames_B = []  # List to store valid image paths for Camera B

    # Criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # Generate object points for the checkerboard pattern
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= world_scaling

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real-world space
    imgpoints_A = []  # 2D points in image plane for Camera A
    imgpoints_B = []  # 2D points in image plane for Camera B

    # Load images for Camera A and Camera B
    images_A = main_cam_paths
    images_A = sorted(images_A, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    images_B = pair_cam_paths
    images_B = sorted(images_B, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Assuming the same number of images for both cameras, iterate through each pair
    for img_path_A, img_path_B in zip(images_A, images_B):
        img_A = cv2.imread(img_path_A)
        img_B = cv2.imread(img_path_B)

        # Rotate images if specified
        if rotate:
            img_A = cv2.rotate(img_A, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_B = cv2.rotate(img_B, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Convert images to grayscale
        gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        gray_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners in both images
        ret_A, corners_A = cv2.findChessboardCorners(gray_A, checkerboard_size, None)
        ret_B, corners_B = cv2.findChessboardCorners(gray_B, checkerboard_size, None)

        # If corners are found in both images, add points to lists after refinement
        if ret_A and ret_B:
            valid_frames_A.append(img_path_A)
            valid_frames_B.append(img_path_B)
            objpoints.append(objp)

            corners2_A = cv2.cornerSubPix(gray_A, corners_A, (11, 11), (-1, -1), criteria=criteria)
            imgpoints_A.append(corners2_A)

            corners2_B = cv2.cornerSubPix(gray_B, corners_B, (11, 11), (-1, -1), criteria=criteria)
            imgpoints_B.append(corners2_B)

    # Initialize progress bar
    pbar = tqdm(total=((max_frames - min_frames) * repetitions), desc='External Calibration: ')

    # Iterate over range of frame counts for calibration
    for file_num in range(min_frames, max_frames + 1):
        futures = []
        with ThreadPoolExecutor() as executor:
            for repeat in range(repetitions):
                # Submit calibration task to executor
                future = executor.submit(stereo_cal, run_name, objpoints, imgpoints_A, imgpoints_B,
                                         valid_frames_A, valid_frames_B, file_num, gray_A.shape[::-1],
                                         gray_B.shape[::-1], pair, main, repeat)
                futures.append(future)

            # As tasks complete, update progress bar and collect results
            for future in as_completed(futures):
                flag, res = future.result()
                data.append(res)
                pbar.update(1)

    # Convert collected data to DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(f'csvs/{run_name}_external_calibrations_repeats_full_main{main}_pair{pair}.csv', index=False)
    print(f'External calibration done. Results saved as csvs/{run_name}_external_calibrations_repeats_full_main{main}_pair{pair}.csv')

