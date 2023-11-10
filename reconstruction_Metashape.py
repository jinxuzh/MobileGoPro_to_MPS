import os
import argparse

import Metashape

# copy from config.py for self contain
def get_parameters(take_name):

    parser = argparse.ArgumentParser('Register mobile gopro to aria MPS coordinate system')

    # path
    parser.add_argument('--base_folder', type=str, default='/media/shan/Volume2/Ego4D-egoexo-10-18/')
    parser.add_argument('--calib_dir', type=str, default='/media/shan/Volume1/Data/Music/calib_files',
                        help="folder for calibration files")

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

    return args

def reconstruction(doc, image_dir, calib_file, save_dir, save_name, sensor_label, image_prefix, fix_calib, save_calib, save_campose):

    chunk = doc.chunk

    print('\t Building environment')
    # prepare file list
    filelist = os.listdir(image_dir)
    absolute_filelist = [os.path.join(image_dir, filepath) for filepath in filelist]
    absolute_filelist.sort()
    # add photo
    chunk.addPhotos(absolute_filelist)
    # add camera and assign the photos to it
    sensor = chunk.addSensor()
    calibration = Metashape.Calibration()
    calibration.load(calib_file)
    sensor.label = sensor_label
    sensor.type = Metashape.Sensor.Type.Fisheye
    sensor.width = calibration.width
    sensor.height = calibration.height
    sensor.user_calib = calibration.copy()
    if fix_calib:
        sensor.fixed = True
    for camera in chunk.cameras:
        if image_prefix is not None:
            if camera.label[:len(image_prefix)] == image_prefix:
                camera.sensor = sensor
        else:
            camera.sensor = sensor
    # match points and align camera
    chunk.matchPhotos(downscale=0, reference_preselection=False, keep_keypoints=True)
    chunk.alignCameras(reset_alignment=False)

    if save_calib:
        sensor.calibration.save(os.path.join(save_dir, '{}.xml'.format(sensor_label)))

    if save_campose:
        save_cam_pose(sensor_label, image_prefix, chunk, save_dir)

    doc.save(os.path.join(save_dir, save_name))

def save_cam_pose(camera_name, label_flag, chunk, save_dir):
    # save camera pose
    cam_pose_filename = os.path.join(save_dir, '{}_cam_pose.txt'.format(camera_name))
    file = open(cam_pose_filename, "wt")
    for camera in chunk.cameras:

        if camera.label[:len(label_flag)] != label_flag:
            continue

        if not camera.transform:
            continue
        Twc = camera.transform
        calib = camera.sensor.calibration
        file.write(
            "{:6d}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t\n".format(
                camera.key, Twc[0, 0], Twc[1, 0], Twc[2, 0], Twc[0, 1], Twc[1, 1], Twc[2, 1], Twc[0, 2],
                Twc[1, 2],
                Twc[2, 2], Twc[0, 3], Twc[1, 3], Twc[2, 3]))
    file.flush()
    file.close()

    # save image name
    im_name_filename = os.path.join(save_dir, '{}_im_name.txt'.format(camera_name))
    file = open(im_name_filename, "wt")
    for camera in chunk.cameras:

        if camera.label[:len(label_flag)] != label_flag:
            continue

        if not camera.transform:
            continue
        path = camera.photo.path
        file.write("{:6d}\t{:s}\n".format(camera.key, path))
    file.flush()
    file.close()


if __name__ == '__main__':

    args = get_parameters('upenn_0718_Violin_2_5')

    save_dir = os.path.join(args.video_folder, '../', 'outputs', 'Metashape')
    os.makedirs(save_dir, exist_ok=True)
    save_name = '{}.psx'.format('upenn_0718_Violin_2_5')
    save_file = os.path.join(save_dir, save_name)

    ################# Build 3D world from gopro walkaround #################
    image_dir = os.path.join(args.video_folder, 'mobile')
    assert os.path.exists(image_dir), image_dir
    calib_file = os.path.join(args.calib_dir, '{}.xml'.format(args.mobile_cam))
    assert os.path.exists(calib_file), calib_file

    doc = Metashape.app.document
    reconstruction(doc, image_dir, calib_file, save_dir, save_name,
                   sensor_label='mobile_gp', image_prefix=None, fix_calib=False, save_calib=True, save_campose=False)

    ################# Register walkarond aria frames and calibrate aria camera #################
    video_dir = os.path.join(args.video_folder, 'walkaround_aria')
    assert os.path.exists(video_dir), video_dir
    calib_file = os.path.join(args.calib_dir, 'aria01_new.xml')
    assert os.path.exists(calib_file), calib_file

    doc = Metashape.app.document
    doc.open(save_file)
    reconstruction(doc, video_dir, calib_file, save_dir, save_name,
                   sensor_label='walkaround_aria', image_prefix='214-1', fix_calib=False, save_calib=True, save_campose=True)

    ################# Register aria playing frames #################
    video_dir = os.path.join(args.video_folder, 'playing_aria')
    assert os.path.exists(video_dir), video_dir
    calib_file = os.path.join(save_dir, 'walkaround_aria.xml')
    assert os.path.exists(calib_file), calib_file

    doc = Metashape.app.document
    doc.open(save_file)

    reconstruction(doc, video_dir, calib_file, save_dir, save_name,
                   sensor_label='playing_aria', image_prefix='playing_aria', fix_calib=True, save_calib=False, save_campose=True)

    ################# Register aria playing frames #################
    video_dir = os.path.join(args.video_folder, 'playing_gopro')
    assert os.path.exists(video_dir), video_dir
    calib_file = os.path.join(save_dir, 'mobile_gp.xml')
    assert os.path.exists(calib_file), calib_file

    doc = Metashape.app.document
    doc.open(save_file)

    reconstruction(doc, video_dir, calib_file, save_dir, save_name,
                   sensor_label='playing_gopro', image_prefix='playing_gopro', fix_calib=True, save_calib=False, save_campose=True)