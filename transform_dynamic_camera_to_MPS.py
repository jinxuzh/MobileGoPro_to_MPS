import os
import cv2 as cv2
import numpy as np
import pandas as pd
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rerun as rr
from tqdm import tqdm

from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

from config import get_parameters
from utils import load_metashape_cam_pose_walkaround_aria, load_metashape_cam_pose, cam_pose_transformation
from aria_alignment_helper import RansacEstimator, Solver, transform_from_rotm_tr

def transformation_MPS_Metashape(args, visualize=False, visualize_cam_pose=False):
    capture_name = '_'.join(args.take.split('_')[:-1])
    ################# Fetch walkaround aria results from MPS #################
    df_aria = pd.DataFrame()  # initialize dataframe for storing everything
    aria_vrs_pth = os.path.join(args.vrs_folder, 'aria01.vrs')
    assert os.path.exists(aria_vrs_pth)

    # get information from vrs provider
    provider = data_provider.create_vrs_data_provider(aria_vrs_pth)
    time_domain = TimeDomain.DEVICE_TIME  # query data based on host time
    option = TimeQueryOptions.BEFORE  # get data whose time [in TimeDomain] is BEFORE to query time
    stream_id = StreamId('214-1')  # rgb
    t_first = provider.get_first_time_ns(stream_id, TimeDomain.DEVICE_TIME)
    print("Aria Device Time starting at", t_first, 'ns.')

    # Fetch timestamps for each extracted image of aria walkaround and, optionally, images from VRS file
    dev_time = []
    abs_time = []
    abs_frame = []
    save_visualization_image = False
    walking_aria_img_dir = os.path.join(args.work_dir, capture_name, 'aria_images', 'aria_walkaround')
    walking_aria_img_list = sorted(os.listdir(walking_aria_img_dir))
    for image_file in walking_aria_img_list:
        frame_id = int(image_file.split('-')[2]) - 1
        # get timestamp for each image
        time = int((frame_id / 30 * 10 ** 3) * 10 ** 6 + t_first)
        dev_time.append((frame_id / 30 * 10 ** 3) * 10 ** 6 + t_first)
        abs_time.append(frame_id / 30 * 10 ** 3)
        abs_frame.append(frame_id)
        if save_visualization_image:
            image = provider.get_image_data_by_time_ns(stream_id, time, time_domain, option)
            plt.imshow(image[0].to_numpy_array())
            cv2.imwrite(os.path.join(walking_aria_img_dir + '_check', image_file),
                        cv2.cvtColor(image[0].to_numpy_array(), cv2.COLOR_RGB2BGR))
    df_aria['Aria Cropped Frame Device Time (ns)'] = dev_time
    df_aria['Aria Cropped Frame Absolute Time (ms)'] = abs_time
    df_aria['Aria Cropped Frame Number'] = abs_frame
    df_aria.head(5)

    # Fetch aria poses from MPS trajectory file
    aria_traj_pth = os.path.join(args.vrs_folder, "../trajectory/closed_loop_trajectory.csv")
    assert os.path.exists(aria_traj_pth)

    df_traj = pd.read_csv(aria_traj_pth, sep=',')
    timestamps_traj = df_traj.values[:, 1]  # device time ns
    transforms = df_traj.values[:, 3:10]  # [x,y,z,qx,qy,qz,qw]
    get_transform_idx = lambda x: np.argmin(np.abs(timestamps_traj - x))

    dev_time = df_aria['Aria Cropped Frame Device Time (ns)']
    transform_idx = [get_transform_idx(i / 1000) for i in dev_time]
    df_aria['Aria Pose in Aria World'] = [transforms[transform_idx][i, :] for i in range(len(transform_idx))]
    df_aria.head(5)

    ################# Fetch walkaround aria results from Metashape #################
    metashape_output_pth = os.path.join(args.work_dir, capture_name, TAKE, 'outputs', 'Metashape')
    assert os.path.exists(metashape_output_pth)

    tgt_file = 'walkaround_aria'  # aria only
    imgs, poses = load_metashape_cam_pose_walkaround_aria(metashape_output_pth, tgt_file)

    cropped_times = []
    frame_nums = []
    metashape_poses = []
    for filename in sorted(os.listdir(walking_aria_img_dir)):
        frame_num = int(filename.split("/")[-1].split("-")[2]) - 1
        frame_nums.append(frame_num)
        # cropped_times.append(cropped_aria[file_num] + frame_num * 1 / FPS * 1000)  # cropped times in ms
        cropped_times.append((frame_num / 30 * 10 ** 3) * 10 ** 6 + t_first)  # cropped times in ms

        if frame_num in imgs:
            file_id = imgs[frame_num]
            metashape_poses.append(poses[file_id])
        else:
            metashape_poses.append(np.nan)

    # print("Fetched", len(frame_nums), "for alignment from", len(set(file_nums)), 'video sequences.')
    df_aria['Aria Cropped Frame Number'] = frame_nums
    df_aria['Aria Pose in Metashape World'] = metashape_poses
    df_aria.head(5)

    ################# Fit a transformation between MPS coordinate system and Metashape coordinate system #################
    if visualize_cam_pose:
        rr.init("MPS camera pose mps metashape {}".format(args.take))
        rr.spawn()
        K = np.array([[1.5, 0, 1.5], [0, 1.5, 1.5], [0, 0, 1]])

    # Setup pair points for alignment
    points_metashape = []
    points_aria = []
    camera_name = "camera-rgb"
    T_device_RGB = provider.get_device_calibration().get_transform_device_sensor(camera_name)
    for i in range(len(df_aria)):
        T_metashape_world_cam = df_aria.loc[i, 'Aria Pose in Metashape World']
        point = (T_metashape_world_cam @ np.array([0, 0, 0, 1]))[:3]
        points_metashape.append(point.reshape((1, 3)))

        # device pose
        T_mps_world_device = df_aria.loc[i, 'Aria Pose in Aria World']
        R_mps_world_device = Rotation.from_quat(T_mps_world_device[3:]).as_matrix()
        T_mps_world_device_matrix = transform_from_rotm_tr(R_mps_world_device, T_mps_world_device[:3])
        # transform from world_device to world_rgb
        T_mps_world_rgb = T_mps_world_device_matrix @ T_device_RGB.matrix()
        points_aria.append(T_mps_world_rgb[:3, -1].reshape((1, 3)))

        if visualize_cam_pose:
            t_metashape = T_metashape_world_cam[:3, -1].tolist()
            rot_metashape = Rotation.from_matrix(T_metashape_world_cam[:3, :3])
            rot_metashape = rot_metashape.as_quat().tolist()
            rr.log_rigid3("hand/camera/metashape_{}".format(i), parent_from_child=(t_metashape, rot_metashape))
            # rr.log_pinhole("hand/camera/metashape_{}/image".format(i), child_from_parent=np.array(K), width=0, height=0)

            t_mps = T_mps_world_rgb[:3, -1].tolist()
            rot_mps = Rotation.from_matrix(T_mps_world_rgb[:3, :3])
            rot_mps = rot_mps.as_quat().tolist()
            rr.log_rigid3("hand/camera/mps_{}".format(i), parent_from_child=(t_mps, rot_mps))
            # rr.log_pinhole("hand/camera/mps_{}/image".format(i), child_from_parent=np.array(K), width=0, height=0)

            if i > 0:
                rr.log_line_segments('hand/metashape_link/{}'.format(i), [t_metashape, points_metashape[-2].tolist()[0]], color=[255, 0, 0])
                rr.log_line_segments('hand/mps_link/{}'.format(i), [t_mps, points_aria[-2].tolist()[0]], color=[0, 0, 255])

    points_metashape = np.concatenate(points_metashape)
    points_aria = np.concatenate(points_aria)
    dst_pc = points_aria.astype(np.float64)
    src_pc = points_metashape.astype(np.float64)

    # estimate with RANSAC
    ransac = RansacEstimator(
        min_samples=16,
        residual_threshold=0.01,  # (0.001)**2,
        max_trials=10000,
    )
    ret = ransac.fit(Solver(), [src_pc, dst_pc])
    transform_ransac = ret["best_params"]  # Scale included, aria2meta (changed to metashape2mps)
    inliers_ransac = ret["best_inliers"]
    scale_ransac = ret['best_scale']
    print('Percentage of Inliers:', np.count_nonzero(inliers_ransac) / len(inliers_ransac))
    mse_ransac = np.sqrt(Solver(transform_ransac).residuals(src_pc, dst_pc).mean())
    print("MSE ransac all: {}".format(mse_ransac))
    mse_ransac_inliers = np.sqrt(
        Solver(transform_ransac).residuals(src_pc[inliers_ransac], dst_pc[inliers_ransac]).mean())
    print("MSE ransac inliers: {}".format(mse_ransac_inliers))
    print("Scale from Aria to Metashape is", scale_ransac)

    T_mps_metashape = transform_ransac

    # transform playing aria and playing gopro results from Metashape, project to exo gopro camera in MPS, and check
    if visualize:
        # read metashape cam pose
        cam_pose_metashape = {}
        for tgt_name in ['playing_aria', 'playing_gopro']:
            cam_pose_metashape[tgt_name] = load_metashape_cam_pose(metashape_output_pth, tgt_name)

        # read exo camera pose in MPS from gopro_calibs.csv
        gopro_calib_path = os.path.join(args.vrs_folder, '../trajectory/gopro_calibs.csv')
        assert os.path.exists(gopro_calib_path), f"{gopro_calib_path} doesn't exist. Please check if trajectory data is downloaded."
        calib_df = pd.read_csv(gopro_calib_path)
        exo_gp_calib = calib_df.loc[calib_df['cam_uid'] == args.exo_cam]
        # Intrinsics
        intri_index = [f"intrinsics_{idx}" for idx in range(8)]
        gp_camera_params = exo_gp_calib[intri_index].values.flatten().tolist()
        # Extrinsics
        extri_index = ['tx_world_cam','ty_world_cam','tz_world_cam','qx_world_cam','qy_world_cam','qz_world_cam','qw_world_cam']
        T_gp_world_cam = exo_gp_calib[extri_index].values.flatten().tolist()

        R_gp_world_cam = Rotation.from_quat(T_gp_world_cam[3:]).as_matrix()
        T_gp_world_cam_matrix = transform_from_rotm_tr(R_gp_world_cam, T_gp_world_cam[:3])
        gp_camera_intrinsics_projection = calibration.CameraProjection(calibration.CameraModelType.KANNALA_BRANDT_K3,
                                                                       gp_camera_params)

        # project camera center
        gp01_image_dir = os.path.join(args.work_dir, capture_name, TAKE, f"vis_exo-playing-{args.exo_cam}/original_img")
        vis_folder = os.path.join(os.path.dirname(gp01_image_dir), 'gp01_vis')
        os.makedirs(vis_folder, exist_ok=True)
        print("Saving visualization of aria+gp position in MPS projected onto GP01")
        for file in tqdm(sorted(os.listdir(gp01_image_dir))):
            im = cv2.imread(os.path.join(gp01_image_dir, file))

            # project camera locations location
            for camera_name, c in zip(['playing_aria', 'playing_gopro'], [(0,0,255), (255,0,0)]):
                if file in cam_pose_metashape[camera_name]['cam_pose'].keys():
                    point_metashape = (cam_pose_metashape[camera_name]['cam_pose'][file] @ np.array([0, 0, 0, 1])) # T_world_camera * [0,0,0,1]
                    point_mps_world = T_mps_metashape @ point_metashape
                    point_mps_gp01 = np.linalg.inv(T_gp_world_cam_matrix) @ point_mps_world
                    point_2D_gp01 = gp_camera_intrinsics_projection.project(point_mps_gp01[:3])
                    im = cv2.circle(im, (int(point_2D_gp01[0]), int(point_2D_gp01[1])), 10, c, -1)

            cv2.imwrite(os.path.join(vis_folder, file), im)

    return transform_ransac

def transformation_MPS_gp_aria(args, T_mps_metashape, visualize_cam_pose=False):

    metashape_output_pth = os.path.join(args.work_dir, capture_name, TAKE, 'outputs', 'Metashape')
    cam_pose_metashape = {}
    for tgt_name in ['playing_aria', 'playing_gopro']:
        cam_pose_metashape[tgt_name] = load_metashape_cam_pose(metashape_output_pth, tgt_name)

    # form pairs of gopro, aria pose in metashape, then transform it to MPS frame
    # fit a transformation between aria and gopro for each frame
    if visualize_cam_pose:
        rr.init("MPS logging {}".format(args.take))
        rr.spawn()
        K = np.array([[1.5, 0, 1.5], [0, 1.5, 1.5], [0, 0, 1]])

    T_mps_gp_aria = []
    for file in cam_pose_metashape['playing_aria']['cam_pose'].keys():
        if file in cam_pose_metashape['playing_gopro']['cam_pose'].keys():
            T_metashape_world_aria = cam_pose_metashape['playing_aria']['cam_pose'][file]
            T_metashape_world_gp = cam_pose_metashape['playing_gopro']['cam_pose'][file]

            T_mps_world_aria = cam_pose_transformation(T_metashape_world_aria, T_mps_metashape)
            T_mps_world_gp = cam_pose_transformation(T_metashape_world_gp, T_mps_metashape)

            if visualize_cam_pose:
                t_aria = T_mps_world_aria[:3, -1].tolist()
                rot_aria = Rotation.from_matrix(T_mps_world_aria[:3, :3])
                rot_aria = rot_aria.as_quat().tolist()
                rr.log_rigid3("hand/camera/aria_{}".format(file), parent_from_child=(t_aria, rot_aria))
                rr.log_pinhole("hand/camera/aria_{}/image".format(file), child_from_parent=np.array(K), width=0, height=0)

                t_gp = T_mps_world_gp[:3, -1].tolist()
                rot_gp = Rotation.from_matrix(T_mps_world_gp[:3, :3])
                rot_gp = rot_gp.as_quat().tolist()
                rr.log_rigid3("hand/camera/gp_{}".format(file), parent_from_child=(t_gp, rot_gp))
                rr.log_pinhole("hand/camera/gp_{}/image".format(file), child_from_parent=np.array(K), width=0, height=0)

                rr.log_line_segments('hand/link/{}'.format(file), [t_gp, t_aria], color=[255,255,255])

            T_mps_gp_aria.append(np.linalg.inv(T_mps_world_gp) @ T_mps_world_aria)

    # TODO: a better way of choosing
    return T_mps_gp_aria[0]


if __name__ == '__main__':
    ######################################
    TAKE = None # MODIFY e.g. "upenn_0718_Violin_2_5"
    ######################################

    args = get_parameters(TAKE)
    capture_name = '_'.join(TAKE.split('_')[:-1])

    # fit a transformation between MPS coordinate system and Metashape coordinate system
    T_mps_metashape = transformation_MPS_Metashape(args, visualize=False)
    save_dir = os.path.join(args.work_dir, capture_name, TAKE, 'outputs', 'Metashape')
    save_name = 'transformation_MPS_Metashape.json'
    with open(os.path.join(save_dir, save_name), 'w') as f:
        json.dump(T_mps_metashape.tolist(), f, indent=4)

    # fit a transformation between dynamic gopro and aria rgb in mps coordinate system
    save_dir = os.path.join(args.work_dir, capture_name, TAKE, 'outputs', 'Metashape')
    save_name = 'transformation_MPS_Metashape.json'
    with open(os.path.join(save_dir, save_name), 'r') as f:
        T_mps_metashape = np.array(json.load(f))

    T_mps_gp_aria = transformation_MPS_gp_aria(args, T_mps_metashape)
    save_dir = os.path.join(args.work_dir, capture_name, TAKE, 'outputs', 'Metashape')
    save_name = 'transformation_MPS_gp_aria.json'
    with open(os.path.join(save_dir, save_name), 'w') as f:
        json.dump(T_mps_gp_aria.tolist(), f, indent=4)
