import numpy as np
from scipy import linalg
import pandas as pd
from fastprogress import progress_bar
import shutil

def triangulate_multiple_points(imgpoints_Origin, imgpoints_Pair1, projMatr_Origin, projMatr_Pair1):
    """
    Triangulates 3D points from multiple views.

    Parameters:
    imgpoints_Origin (np.array): Image points from the origin camera.
    imgpoints_Pair1 (np.array): Image points from the pair camera.
    projMatr_Origin (np.array): Projection matrix of the origin camera.
    projMatr_Pair1 (np.array): Projection matrix of the pair camera.

    Returns:
    list: A list of triangulated 3D points.
    """
    points_3d_list = []
    for points_origin, points_pair1 in zip(imgpoints_Origin, imgpoints_Pair1):
        point_origin = np.array(points_origin)
        point_pair1 = np.array(points_pair1)

        # Form the linear system for triangulation
        A = [point_origin[1]*projMatr_Origin[2,:] - projMatr_Origin[1,:],
                projMatr_Origin[0,:] - point_origin[0]*projMatr_Origin[2,:],
                point_pair1[1]*projMatr_Pair1[2,:] - projMatr_Pair1[1,:],
                projMatr_Pair1[0,:] - point_pair1[0]*projMatr_Pair1[2,:]]

        A = np.array(A)
        B = A.T @ A
        U, s, Vh = linalg.svd(B, full_matrices=False)

        # Compute the 3D point
        points4D_homogeneous = Vh[-1, 0:3] / Vh[-1, 3]
        points_3d_list.append(points4D_homogeneous[:3].T)

    return points_3d_list

def projectPixelMain(M, xyz):
    """
    Projects a 3D point into 2D image coordinates using a given projection matrix.

    Parameters:
    M (np.array): The projection matrix of size (3, 4).
    xyz (list or np.array): The 3D point coordinates [x, y, z].

    Returns:
    np.array: The 2D image coordinates [x, y].
    """
    # Convert xyz to homogeneous coordinates
    xyz = np.array([[xyz[0], xyz[1], xyz[2], 1]], dtype=object).T
    # Project the 3D point to 2D
    image_xy = M @ xyz
    # Normalize by the third coordinate
    image_xy /= image_xy[2]
    # Flatten and return the 2D coordinates
    image_xy = np.squeeze(image_xy)
    return image_xy[0:2]

def projectInternal(cam_params, xyz):
    """
    Projects a 3D point to internal coordinates using camera parameters.

    Parameters:
    cam_params (dict): The camera parameters containing rotation matrix 'R' and translation vector 'T'.
    xyz (list or np.array): The 3D point coordinates [x, y, z].

    Returns:
    np.array: The internal coordinates [x, y, z].
    """
    R = cam_params['R']
    t = cam_params['T']
    # Combine rotation and translation into a transformation matrix
    a = np.array([[0, 0, 0]])
    b = np.array([[1]])
    xyz = np.array([[xyz[0], xyz[1], xyz[2], 1]], dtype=object).T
    M = np.vstack((np.hstack((R, t)), np.hstack((a, b))))
    internal_xyz = M @ xyz
    # Flatten and return the internal coordinates
    internal_xyz = np.squeeze(internal_xyz)
    return internal_xyz[:3]

def projectPixel(cam_params, xyz):
    """
    Projects a 3D point into 2D image coordinates using intrinsic camera parameters.

    Parameters:
    cam_params (dict): The camera parameters containing the intrinsic matrix 'mtx'.
    xyz (list or np.array): The 3D point coordinates [x, y, z].

    Returns:
    np.array: The 2D image coordinates [x, y].
    """
    K = cam_params['mtx']
    # Convert xyz to homogeneous coordinates
    xyz = np.array([[xyz[0], xyz[1], xyz[2], 1]], dtype=object).T
    # Combine the intrinsic matrix with a zero vector to form the projection matrix
    a = np.array([[0, 0, 0]]).T
    M = np.hstack((K, a))
    # Project the 3D point to 2D
    image_xy = M @ xyz
    # Normalize by the third coordinate
    image_xy /= image_xy[2]
    # Flatten and return the 2D coordinates
    image_xy = np.squeeze(image_xy)
    return image_xy[0:2]

def projectPixel(cam_params, xyz):
    """
    Projects a 3D point into 2D image coordinates using intrinsic camera parameters.

    Parameters:
    cam_params (dict): The camera parameters containing the intrinsic matrix 'mtx'.
    xyz (list or np.array): The 3D point coordinates [x, y, z].

    Returns:
    np.array: The 2D image coordinates [x, y].
    """
    K = cam_params['mtx']
    # Convert xyz to homogeneous coordinates
    xyz = np.array([[xyz[0], xyz[1], xyz[2], 1]], dtype=object).T
    # Combine the intrinsic matrix with a zero vector to form the projection matrix
    a = np.array([[0, 0, 0]]).T
    M = np.hstack((K, a))
    # Project the 3D point to 2D
    image_xy = M @ xyz
    # Normalize by the third coordinate
    image_xy /= image_xy[2]
    # Flatten and return the 2D coordinates
    image_xy = np.squeeze(image_xy)
    return image_xy[0:2]


def projectEval(run_name, main, pair, all_cam_points, image_size=(1600, 1200)):
    """
    Evaluates the projection errors of 3D points using external calibration data and saves the results.

    Parameters:
    run_name (str): The name of the run.
    main (int): The main identifier.
    pair (int): The pair identifier.
    all_cam_points (list): A list containing the image points from two camera views.
    image_size (tuple, optional): Size of the image (width, height). Defaults to (1600, 1200).

    Returns:
    None
    """
    # Load the calibration results from CSV
    df = pd.read_csv(f'csvs/{run_name}_external_calibrations_repeats_full_main{main}_pair{pair}.csv')

    # Initialize lists to store errors
    errors = []
    errors_pair = []
    perc_errors = []
    perc_errors_pair = []

    # Iterate through each row in the DataFrame
    for idx, (index, row) in progress_bar(enumerate(df.iterrows()), total=len(df)):
        pair = row['pair']
        repeat = row['repeat']
        num_frames = row['num_frames']

        # Load the external calibration data
        data = np.load(f'matrix/external/main{main}_pair{pair}_external_calibration_repeat_{repeat}_{num_frames}_frames.npz')

        # Extract necessary calibration parameters
        R2 = data['R']
        T2 = data['T'].reshape((3, 1))
        cameraMatrix1 = data['mtx_A']
        cameraMatrix2 = data['mtx_B']

        # Construct projection matrices for both cameras
        RT1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = cameraMatrix1 @ RT1
        RT2 = np.hstack([R2, T2])
        P2 = cameraMatrix2 @ RT2

        # Triangulate 3D points using multiple views
        all_results = triangulate_multiple_points(all_cam_points[0], all_cam_points[1], P1, P2)
        points_initial = np.array(all_results)

        # Compute the World Coordinate System (WCS) transformation
        x_direction = points_initial[0] - points_initial[1]
        z_direction = points_initial[3] - points_initial[1]
        x_direction /= np.linalg.norm(x_direction)
        z_direction /= np.linalg.norm(z_direction)
        y_direction = np.cross(x_direction, z_direction)
        R_wcs = np.vstack((x_direction, y_direction, z_direction))
        M = np.vstack((R_wcs, points_initial[1]))
        M = np.hstack((M, np.array([[0], [0], [0], [1]])))
        M = M.T

        # Project 3D points back to 2D image plane of camera 1
        points_2d = []
        for point in points_initial:
            points_2d.append(projectPixelMain(cameraMatrix1, point))
        points_c1_2d = np.array(points_2d)

        # Compute errors in camera 1's image plane
        squared_errors = np.sum((points_c1_2d - np.array(all_cam_points[0]))**2, axis=1)
        mean_error = np.sqrt(np.mean(squared_errors))
        percentage_error = (mean_error / np.sqrt(image_size[0] * image_size[1])) * 100

        # Store errors for camera 1
        errors.append(mean_error)
        perc_errors.append(percentage_error)

        # Prepare camera 2's internal parameters
        cam_params = {
            'C2': {
                'R': R2,
                'T': T2,
                'mtx': cameraMatrix2
            }
        }

        # Project 3D points to camera 2's image plane
        points_internal_c2 = np.array([projectInternal(cam_params['C2'], p) for p in points_initial])
        points_pixel_c2 = np.array([projectPixel(cam_params['C2'], p) for p in points_internal_c2])

        # Compute errors in camera 2's image plane
        squared_errors_pair = np.sum((points_pixel_c2 - np.array(all_cam_points[1]))**2, axis=1)
        mean_error_pair = np.sqrt(np.mean(squared_errors_pair))
        percentage_error_pair = (mean_error_pair / np.sqrt(image_size[0] * image_size[1])) * 100

        # Store errors for camera 2
        errors_pair.append(mean_error_pair)
        perc_errors_pair.append(percentage_error_pair)

    # Update DataFrame with errors
    df['error'] = errors
    df['perc_error'] = perc_errors
    df['error_pair'] = errors_pair
    df['perc_error_pair'] = perc_errors_pair

    # Save results to CSV
    df.to_csv(f'csvs/{run_name}_external_calibrations_repeats_main{main}_pair{pair}_w_errors_w_pair_error.csv', index=False)
    print(f'CSV saved as csvs/{run_name}_external_calibrations_repeats_main{main}_pair{pair}_w_errors_w_pair_error.csv')

    # Find and save the best calibration
    idx = df['error'].idxmin()
    filtered_df = df.loc[idx]
    path_best = filtered_df['path'].tolist()[0]
    copy_path = f'matrix/external/best_{run_name}_external_calibration_main{main}_pair{pair}.npz'
    shutil.copyfile(path_best, copy_path)
    print(f'Best calibration saved as {copy_path}')
