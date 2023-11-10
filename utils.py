import os
import numpy as np
import cv2 as cv2

def decode_video_to_images(video_path, image_dir, im_width, im_height, framerate, prefix=None):

    os.makedirs(image_dir, exist_ok=True)
    if prefix is not None:
        cmd = 'ffmpeg -i {} -vf scale={}:{} -r {} {}/{}_%06d.png'.format(
            video_path, im_width, im_height, framerate, image_dir, prefix)
    else:
        cmd = 'ffmpeg -i {} -vf scale={}:{} -r {} {}/%06d.png'.format(
            video_path, im_width, im_height, framerate, image_dir)
    os.system(cmd)

def load_metashape_cam_pose(metashape_output_pth, tgt_name):

    cam_pose_metashape = {}
    cam_pose_metashape['im_name'] = {}
    cam_pose_metashape['cam_pose'] = {}

    im_name_file = open(os.path.join(metashape_output_pth, tgt_name + '_im_name.txt'), 'r')
    im_name = im_name_file.readlines()
    im_name_file.close()

    for image_name in im_name:
        temp_content = image_name.split('\t')
        key_val = temp_content[0].strip()
        file_val = temp_content[1].strip()
        cam_pose_metashape['im_name'][key_val] = file_val

    cam_pose_file = open(os.path.join(metashape_output_pth, tgt_name + '_cam_pose.txt'), 'r')
    cam_poses = cam_pose_file.readlines()
    cam_pose_file.close()

    for cam_pose in cam_poses:
        temp_content = cam_pose.split('\t')
        key_val = temp_content[0].strip()
        param = np.array(temp_content[1:-1]).astype(float)
        param = np.reshape(param, (4, 3)).T
        param = np.concatenate([param, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
        if key_val in cam_pose_metashape['im_name'].keys():
            name_key = cam_pose_metashape['im_name'][key_val][-10:]
            cam_pose_metashape['cam_pose'][name_key] = param

    return cam_pose_metashape


# TODO: merge with load_metashape_cam_pose
def load_metashape_cam_pose_walkaround_aria(metashape_output_pth, tgt_name):

    imgs = {}
    poses = {}

    for filename in sorted(os.listdir(metashape_output_pth)):
        if tgt_name in filename:
            if 'im_name' in filename:
                f = open(os.path.join(metashape_output_pth, filename), 'r')
                image_names = f.readlines()
                f.close()
                image_info = {}
                for image_name in image_names:
                    temp_content = image_name.split('\t')
                    key_val = temp_content[0].strip()
                    file_val = temp_content[1].strip()
                    frame = int(file_val.split("/")[-1].split("-")[2]) - 1
                    image_info[frame] = key_val
                imgs = image_info
            if 'cam_pose' in filename:
                f = open(os.path.join(metashape_output_pth, filename), 'r')
                cam_poses = f.readlines()
                f.close()
                camera2world = {}
                for cam_pose in cam_poses:
                    temp_content = cam_pose.split('\t')
                    key_val = temp_content[0].strip()
                    param = np.array(temp_content[1:-1]).astype(float)
                    param = np.reshape(param, (4, 3)).T
                    param = np.concatenate([param, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
                    camera2world[key_val] = param
                poses = camera2world

    return imgs, poses

def cam_pose_transformation(T_sys1_world_cam, T_sys2_sys1):

    c_sys1 = T_sys1_world_cam @ np.array([0, 0, 0, 1])
    c_sys2 = T_sys2_sys1 @ c_sys1

    R_sys2_sys1 = T_sys2_sys1[:3, :3].copy()
    for i in range(3):
        R_sys2_sys1[:, i] = R_sys2_sys1[:, i] / np.linalg.norm(R_sys2_sys1[:, i])
    R_sys2 = R_sys2_sys1 @ T_sys1_world_cam[:3, :3]

    T_sys2_world_cam = np.eye(4)
    T_sys2_world_cam[:3, :3] = R_sys2
    T_sys2_world_cam[:3, -1] = c_sys2[:3]

    return T_sys2_world_cam

def visualize_hand_crop(im, hand2D, scale=1, pad=200):

    hand_link = [[0, 1], [1, 2], [2, 3], [3, 4],
                 [0, 5], [5, 6], [6, 7], [7, 8],
                 [0, 9], [9, 10], [10, 11], [11, 12],
                 [0, 13], [13, 14], [14, 15], [15, 16],
                 [0, 17], [17, 18], [18, 19], [19, 20]]

    im = cv2.resize(im, (im.shape[1]*scale, im.shape[0]*scale))
    im = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0])

    v = [hand2D[i, 0]>0 for i in range(21)]

    hand_x = [hand2D[i, 0] * scale for i in range(21) if (hand2D[i, 0]>0 and v[i])]
    hand_y = [hand2D[i, 1] * scale for i in range(21) if (hand2D[i, 0]>0 and v[i])]
    mean_x = int(np.mean(hand_x)) + pad
    mean_y = int(np.mean(hand_y)) + pad
    hand_crop = im[mean_y - pad:mean_y + pad, mean_x - pad:mean_x + pad, :].copy()
    shift_x = mean_x - 2 * pad
    shift_y = mean_y - 2 * pad

    if hand_crop.shape[0] == 0 or hand_crop.shape[1] == 0:
        print("out of bound")

    for link in hand_link:
        joint_0 = hand2D[link[0]]
        joint_1 = hand2D[link[1]]
        if v[link[0]]:
            hand_crop = cv2.circle(hand_crop, (int(joint_0[0]*scale-shift_x), int(joint_0[1]*scale-shift_y)), 3, [0,0,255], -1)
        if v[link[1]]:
            hand_crop = cv2.circle(hand_crop, (int(joint_1[0]*scale-shift_x), int(joint_1[1]*scale-shift_y)), 3, [0,0,255], -1)
        if v[link[0]] and v[link[1]]:
            hand_crop = cv2.line(hand_crop,
                          (int(joint_0[0]*scale-shift_x), int(joint_0[1]*scale-shift_y)),
                          (int(joint_1[0]*scale-shift_x), int(joint_1[1]*scale-shift_y)),
                          [0, 0, 255], 1)

    return hand_crop


def visualize_hand(im, hand2D):
    hand_link = [[0, 1], [1, 2], [2, 3], [3, 4],
                 [0, 5], [5, 6], [6, 7], [7, 8],
                 [0, 9], [9, 10], [10, 11], [11, 12],
                 [0, 13], [13, 14], [14, 15], [15, 16],
                 [0, 17], [17, 18], [18, 19], [19, 20]]

    v = [hand2D[i, 0] > 0 for i in range(21)]

    for link in hand_link:
        joint_0 = hand2D[link[0]]
        joint_1 = hand2D[link[1]]
        if v[link[0]]:
            im = cv2.circle(im, (int(joint_0[0]), int(joint_0[1])), 3,
                                   [0, 0, 255], -1)
        if v[link[1]]:
            im = cv2.circle(im, (int(joint_1[0]), int(joint_1[1])), 3,
                                   [0, 0, 255], -1)
        if v[link[0]] and v[link[1]]:
            im = cv2.line(im,
                          (int(joint_0[0]), int(joint_0[1])),
                          (int(joint_1[0]), int(joint_1[1])),
                          [0, 0, 255], 1)

    return im