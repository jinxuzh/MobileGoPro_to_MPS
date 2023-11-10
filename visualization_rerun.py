import rerun as rr
import numpy as np
import os
import json
import cv2 as cv2
import pandas as pd
from scipy.spatial.transform import Rotation

from config import get_parameters

hand_link = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]
hand_color = [[0,255,255], [0,255,255], [0,255,255], [0,255,255],
              [200,120,200], [200,120,200], [200,120,200], [200,120,200],
              [255,255,60], [255,255,60], [255,255,60], [255,255,60],
              [0,0,255], [0,0,255], [0,0,255], [0,0,255],
              [0,255,0], [0,255,0], [0,255,0], [0,255,0]]

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

if __name__ == '__main__':

    args = get_parameters('upenn_0718_Violin_2_5')

    # pts_file = os.path.join(args.vrs_folder, '../trajectory/semidense_observations.csv')
    # df_pts = pd.read_csv(pts_file, sep=',')

    rr.init("MPS logging {}".format('upenn_0718_Violin_2_5'))
    rr.spawn()
    rr.set_time_seconds("stable_time", 0)

    annotation_folder = os.path.join(args.annotation_folder, 'annotation')
    cam_pose_folder = os.path.join(args.annotation_folder, 'camera_pose')
    annotation_image_folder = args.annotation_im_folder

    annotation_files = os.listdir(annotation_folder)

    save_dir = os.path.join(args.video_folder, '../', 'outputs', 'Metashape')
    save_name = 'transformation_MPS_gp_aria.json'
    with open(os.path.join(save_dir, save_name), 'r') as f:
        T_mps_gp_aria = np.array(json.load(f))

    for file in annotation_files:
        annotation_file = os.path.join(annotation_folder, file)
        with open(annotation_file) as f:
            annot = json.load(f)

        take_uid = annot[list(annot.keys())[0]][0]['metadata']['take_uid']
        take_name = annot[list(annot.keys())[0]][0]['metadata']['take_name']

        if not take_name == 'upenn_0718_Violin_2_5':
            continue

        cam_pose_file = os.path.join(cam_pose_folder, file)
        if not os.path.exists(cam_pose_file):
            print('camera pose not exist for {}, {}'.format(take_uid, take_name))
            continue
        if os.path.exists(cam_pose_file):
            with open(cam_pose_file) as f:
                cam_pose = json.load(f)

        for item_idx, item in enumerate(annot.keys()):

            time = item_idx / 1
            rr.set_time_seconds("stable_time", time)

            # draw 3d hand keypoints
            hand_3D = np.zeros((42, 5))  # x, y, z, 1, valid
            for idx, joint in enumerate(both_hand):
                if joint in annot[item][0]['annotation3D'].keys():
                    hand_3D[idx, 0] = annot[item][0]['annotation3D'][joint]['x']
                    hand_3D[idx, 1] = annot[item][0]['annotation3D'][joint]['y']
                    hand_3D[idx, 2] = annot[item][0]['annotation3D'][joint]['z']
                    hand_3D[idx, 3] = 1
                    hand_3D[idx, 4] = 1

            kp_lr = [hand_3D[:21], hand_3D[21:]]
            for lr_idx, kp in enumerate(kp_lr):
                for idx in range(21):
                    c = np.array([0.8, 0.8, 0.8])
                    kp_single = kp[idx, :3]
                    if kp[idx, -1] > 0:
                        rr.log_points('hand/joint/{}/{}'.format(lr_idx, idx), kp_single[None, ...],
                                      colors=c[None, ...],
                                      radii=0.005)
                    else:
                        rr.log_points('hand/joint/{}/{}'.format(lr_idx, idx), kp_single[None, ...],
                                      colors=c[None, ...],
                                      radii=0.00)

                for l_idx, l in enumerate(hand_link):
                    p1 = kp[l[0], :3]
                    p2 = kp[l[1], :3]
                    c = hand_color[l_idx]
                    if kp[l[0], -1] > 0 and kp[l[1], -1] > 0:
                        rr.log_line_segments('hand/limb/{}/{}'.format(lr_idx, l_idx), [p1[:3], p2[:3]], color=c)
                    else:
                        rr.log_line_segments('hand/limb/{}/{}'.format(lr_idx, l_idx), [p1[:3], p1[:3]], color=[0, 0, 0, 1])

            # draw camera
            K_aria = np.array(cam_pose[item]['aria01']['camera_intrinsics'])
            K_gp = np.array([[879.44310492816487, 0, 1920 / 2 + 6.2943846689491405],
                             [0, 879.44310492816487, 1080 / 2 - 1.2995632779780959],
                             [0, 0, 1]])
            K = np.array([[0.1, 0, 0.1], [0, 0.1, 0.1], [0, 0, 1]])

            T_mps_aria_world = np.array(cam_pose[item]['aria01']['camera_extrinsics'])
            T_mps_aria_world = np.concatenate([T_mps_aria_world, np.array([[0,0,0,1]])])
            T_mps_world_aria = np.linalg.inv(T_mps_aria_world)
            t_aria = T_mps_world_aria[:3, -1].tolist()
            rot_aria = Rotation.from_matrix(T_mps_world_aria[:3, :3])
            rot_aria = rot_aria.as_quat().tolist()
            rr.log_rigid3("hand/camera/aria", parent_from_child=(t_aria, rot_aria))
            rr.log_pinhole("hand/camera/aria/image", child_from_parent=np.array(K_aria), width=0, height=0)
            # rr.log_image("hand/camera/{}/image", image=plt.imread())

            T_mps_world_gp = T_mps_world_aria @ np.linalg.inv(T_mps_gp_aria)
            t_gp = T_mps_world_gp[:3, -1].tolist()
            rot_gp = Rotation.from_matrix(T_mps_world_gp[:3, :3])
            rot_gp = rot_gp.as_quat().tolist()
            rr.log_rigid3("hand/camera/gp", parent_from_child=(t_gp, rot_gp))
            rr.log_pinhole("hand/camera/gp/image", child_from_parent=np.array(K_gp), width=0, height=0)
            # rr.log_image("hand/camera/{}/image", image=plt.imread())



