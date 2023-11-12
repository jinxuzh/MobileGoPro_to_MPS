import argparse
import os

def get_parameters(take_name):

    parser = argparse.ArgumentParser('Register mobile gopro to aria MPS coordinate system')

    # path
    parser.add_argument('--base_folder', type=str, default='/mnt/volume2/Data/Ego4D')
    parser.add_argument('--annotation_folder', type=str, default='/mnt/volume2/Data/Ego4D/annotations/ego_pose/hand')
    parser.add_argument('--annotation_im_folder', type=str, default='/mnt/volume2/Data/Ego4D/aria_undistorted_images/annotation')
    parser.add_argument('--calib_dir', type=str, default='/mnt/8tbvol11/register_mobileGoPro_Ego4D/calib_files',
                        help="folder for calibration files")
    parser.add_argument('--work_dir', type=str, default='/mnt/8tbvol11/register_mobileGoPro_Ego4D')

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
    args = parser.parse_args()

    args.video_folder = os.path.join(args.base_folder, 'takes', take_name, 'frame_aligned_videos')
    args.vrs_folder = os.path.join(args.base_folder, 'captures', take_name[:-2], 'videos')
    args.take = take_name

    return args