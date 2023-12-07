import argparse
import os

def get_parameters(take_name):

    parser = argparse.ArgumentParser('Register mobile gopro to aria MPS coordinate system')

    # path
    parser.add_argument('--take', type=str, default=take_name)
    parser.add_argument('--base_folder', type=str, default='/mnt/volume2/Data/Ego4D')
    parser.add_argument('--annotation_folder', type=str, default='/mnt/volume2/Data/Ego4D/annotations/ego_pose/hand')
    parser.add_argument('--annotation_im_folder', type=str, default='/mnt/volume2/Data/Ego4D/aria_undistorted_images')
    parser.add_argument('--cam_pose_folder', type=str, default='/mnt/volume2/Data/Ego4D/annotations/ego_pose/hand/camera_pose')
    parser.add_argument('--calib_dir', type=str, default='/mnt/volume2/Data/Ego4D/register_mobileGoPro_Ego4D/calib_files',
                        help="folder for calibration files")
    parser.add_argument('--work_dir', type=str, default='/mnt/volume2/Data/Ego4D/register_mobileGoPro_Ego4D')

    # image resolution
    parser.add_argument('--mobile_width', type=int, default=1920,
                        help="width for decoded exo images")
    parser.add_argument('--mobile_height', type=int, default=1080,
                        help="width for decoded exo images")
    parser.add_argument('--exo_width', type=int, default=3840,
                        help="width for decoded exo images")
    parser.add_argument('--exo_height', type=int, default=2160,
                        help="width for decoded exo images")
    parser.add_argument('--aria_width', type=int, default=1408,
                        help="width for decoded exo images")
    parser.add_argument('--aria_height', type=int, default=1408,
                        help="width for decoded exo images")

    # frame rate
    parser.add_argument('--mobile_rate', type=int, default=5,
                        help="fps for decoding images for 3D environment reconstruction")
    parser.add_argument('--playing_rate', type=int, default=1,
                        help="fps for decoding images of playing for aria and head mounted gopro")
    parser.add_argument('--annotation_rate', type=int, default=30,
                        help="fps for decoding images of playing for aria and head mounted gopro")

    # camera setting
    parser.add_argument('--mobile_cam', type=str, default='gp05',
                        help="define the camera used for room scan and mounted on the head")
    parser.add_argument('--exo_cam', type=str, default='gp01',
                        help="define the camera used for exo camera visualization")

    # crop aria
    # need human work
    parser.add_argument('--aria_walkaround_start_frame', type=int, default=1300,
                        help="start frame of aria walkaround")
    parser.add_argument('--aria_walkaround_end_frame', type=int, default=4000,
                        help="start frame of aria walkaround")

    # Pretrained hand bbox detector and pose estimator
    parser.add_argument(
        "--det_config", 
        default="/home/jinxu/code/Ego4d/ego4d/internal/human_pose/external/mmlab/mmpose/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py", 
        help="Path of hand detector config file"
    )
    parser.add_argument(
        "--det_ckpt", 
        default="https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth", 
        help="Path of hand detector checkpoint"
    )
    parser.add_argument(
        "--pose_config", 
        default="/home/jinxu/code/Ego4d/ego4d/internal/human_pose/external/mmlab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256_dark.py", 
        help="Path of hand pose2d estimator config file"
    )
    parser.add_argument(
        "--pose_ckpt", 
        default="https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth", 
        help="Path of hand pose2d estimator checkpoint"
    )

    args = parser.parse_args()

    args.video_folder = os.path.join(args.base_folder, 'takes', take_name, 'frame_aligned_videos')
    capture_name = '_'.join(take_name.split('_')[:-1])
    args.vrs_folder = os.path.join(args.base_folder, 'captures', capture_name, 'videos')

    return args