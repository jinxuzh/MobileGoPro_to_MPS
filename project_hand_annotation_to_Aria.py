import os
import numpy as np
import json
import cv2 as cv2
from tqdm import tqdm
from config import get_parameters
from utils.utils import visualize_hand_crop, visualize_hand
import xml.etree.ElementTree as ET

left_hand = ['left_wrist',
             'left_thumb_1', 'left_thumb_2', 'left_thumb_3', 'left_thumb_4',
             'left_index_1', 'left_index_2', 'left_index_3', 'left_index_4',
             'left_middle_1', 'left_middle_2', 'left_middle_3', 'left_middle_4',
             'left_ring_1', 'left_ring_2', 'left_ring_3', 'left_ring_4',
             'left_pinky_1', 'left_pinky_2', 'left_pinky_3', 'left_pinky_4']
right_hand = ['right_wrist',
             'right_thumb_1', 'right_thumb_2', 'right_thumb_3', 'right_thumb_4',
             'right_index_1', 'right_index_2', 'right_index_3', 'right_index_4',
             'right_middle_1', 'right_middle_2', 'right_middle_3', 'right_middle_4',
             'right_ring_1', 'right_ring_2', 'right_ring_3', 'right_ring_4',
             'right_pinky_1', 'right_pinky_2', 'right_pinky_3', 'right_pinky_4']
both_hand = left_hand + right_hand

hand_link = [[0,1], [1,2], [2,3], [3,4],
             [0,5], [5,6], [6,7], [7,8],
             [0,9], [9,10], [10,11], [11,12],
             [0,13], [13,14], [14,15], [15,16],
             [0,17], [17,18], [18,19], [19,20]]
hand_link_color = [[255,125,0], [255,125,0], [255,125,0], [255,125,0],
                   [255,155,255], [255,155,255], [255,155,255], [255,155,255],
                   [105,175,255], [105,175,255], [105,175,255], [105,175,255],
                   [210,55,60], [210,55,60], [210,55,60], [210,55,60],
                   [0,255,0], [0,255,0], [0,255,0], [0,255,0]]

if __name__ == '__main__':
    # Config
    take = '' # Modify if not using args
    args = get_parameters(take)
    capture_name = '_'.join(args.take.split('_')[:-1])

    annotation_folder = os.path.join(args.annotation_folder, 'annotation')
    cam_pose_folder = os.path.join(args.annotation_folder, 'camera_pose')
    annotation_image_folder = args.annotation_im_folder
    annotation_files = os.listdir(annotation_folder)

    # Load transformation from aria to gp
    transform_save_dir = os.path.join(args.work_dir, capture_name, args.take)
    T_mps_gp_aria_save_name = 'transformation_MPS_gp_aria_allIdx.json'
    with open(os.path.join(transform_save_dir, T_mps_gp_aria_save_name), 'r') as f:
        T_mps_gp_aria = np.array(json.load(f))
    selected_gp_aria_T_idx = 0 # TODO: Choose based on criterion e.g. transformation with least projection error.
    T_mps_gp_aria = T_mps_gp_aria[selected_gp_aria_T_idx]
        
    # Create directory to store projected hand annotation on Aria and GoPro
    aria_playing_vis_dir = os.path.join(transform_save_dir, 'hand_anno_check/aria')
    os.makedirs(aria_playing_vis_dir, exist_ok=True)
    gp_playing_vis_dir = os.path.join(transform_save_dir, f'hand_anno_check/mobile_{args.mobile_cam}')
    os.makedirs(gp_playing_vis_dir, exist_ok=True)

    # Load camera intrinsics & distortion parameter from metashape calibration file
    metashape_gp_calib = os.path.join(args.work_dir, capture_name, 'Metashape/mobile_gp.xml')
    tree = ET.parse(metashape_gp_calib)
    root = tree.getroot()
    gp_calib = {}
    for i in range(1,len(root)-1):
        gp_calib[root[i].tag] = float(root[i].text)
    K_gp05 = [[gp_calib['f'], 0, gp_calib['width']/2 + gp_calib['cx']],
              [0, gp_calib['f'], gp_calib['height']/2 + gp_calib['cy']],
              [0, 0, 1]]
    D_gp05 = [gp_calib['k1'], gp_calib['k2'], gp_calib['k3'], 0]

    # Project hand annotation onto aria and dynamic gopro camera
    for file in annotation_files:
        annotation_file = os.path.join(annotation_folder, file)
        with open(annotation_file) as f:
            annot = json.load(f)

        take_uid = annot[list(annot.keys())[0]][0]['metadata']['take_uid']
        take_name = annot[list(annot.keys())[0]][0]['metadata']['take_name']

        if not take_name == args.take:
            continue

        cam_pose_file = os.path.join(cam_pose_folder, file)
        if not os.path.exists(cam_pose_file):
            print('camera pose not exist for {}, {}'.format(take_uid, take_name))
            continue
        if os.path.exists(cam_pose_file):
            with open(cam_pose_file) as f:
                cam_pose = json.load(f)

        for item in tqdm(annot.keys()):
            hand_3D = np.zeros((42, 5))  # x, y, z, 1, valid
            for idx, joint in enumerate(both_hand):
                if joint in annot[item][0]['annotation3D'].keys():
                    hand_3D[idx, 0] = annot[item][0]['annotation3D'][joint]['x']
                    hand_3D[idx, 1] = annot[item][0]['annotation3D'][joint]['y']
                    hand_3D[idx, 2] = annot[item][0]['annotation3D'][joint]['z']
                    hand_3D[idx, 3] = 1
                    hand_3D[idx, 4] = 1

            K_aria = np.array(cam_pose[item]['aria01']['camera_intrinsics'])
            T_mps_aria_world = np.array(cam_pose[item]['aria01']['camera_extrinsics'])
            hand_3D_repro = K_aria @ T_mps_aria_world @ hand_3D[:, :4].T
            hand_3D_repro = hand_3D_repro[:2, :].T / hand_3D_repro[2:, :].T

            # Project onto aria image
            hand_3D_repro = np.concatenate([hand_3D_repro, hand_3D[:, 4:]], axis=-1)
            hand_3D_repro = np.nan_to_num(hand_3D_repro)
            im_name = os.path.join(annotation_image_folder, take_name, '{:06d}.jpg'.format(int(item)))
            im = cv2.imread(im_name)
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            hand_vis = visualize_hand(im, hand_3D_repro[:21, :2])
            hand_vis = visualize_hand(hand_vis, hand_3D_repro[21:, :2])
            hand_vis = cv2.rotate(hand_vis, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(aria_playing_vis_dir, '{:06d}.jpg'.format(int(item))), hand_vis)

            # Project onto head-mounted GoPro image
            T_mps_aria_world = np.concatenate([T_mps_aria_world, np.array([[0, 0, 0, 1]])])
            hand_3D_repro = T_mps_gp_aria[:3, :] @ T_mps_aria_world @ hand_3D[:, :4].T
            hand_3D_repro = hand_3D_repro[:2, :].T / hand_3D_repro[2:, :].T
            hand_3D_repro_distort = cv2.fisheye.distortPoints(np.expand_dims(hand_3D_repro, axis=1), np.array(K_gp05),
                                                                np.array(D_gp05))
            hand_3D_repro_distort = np.nan_to_num(hand_3D_repro_distort)

            im_name = os.path.join(args.work_dir, capture_name, args.take, f"vis_gp-playing-{args.mobile_cam}", 'playing_gopro_{:06d}.jpg'.format(int(item)))
            im = cv2.imread(im_name)
            hand_vis = visualize_hand(im, hand_3D_repro_distort[:21, 0, :2])
            hand_vis = visualize_hand(hand_vis, hand_3D_repro_distort[21:, 0, :2])
            cv2.imwrite(os.path.join(gp_playing_vis_dir, '{:06d}.jpg'.format(int(item))), hand_vis)