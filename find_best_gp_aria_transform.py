import json
import pandas as pd
import os
import numpy as np
import copy
import glob
import cv2
from PIL import Image
from tqdm import tqdm
from config import get_parameters
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from utils.utils import (
    load_metashape_calib, 
    load_hand_pose_anno, 
    get_bbox_from_kpts,
    visualize_hand
)

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
    vis_pose_result,
)
from mmpose.datasets import DatasetInfo


def hand_pose2d_inference(gt_anno, gt_cam_pose, vis_gp_playing_img_dir, T_mps_gp_aria, K_gp05, D_gp05, save_result=False):
    """
    Run pretrained hand pose detector to get hand pose2d keypoints, using bbox 
    proposed by projected hand kpts from using index=0 transformation. Please check 
    if the bbox makes sense.
    """
    if save_result:
        save_vis_dir = os.path.join(os.path.dirname(vis_gp_playing_img_dir), 'hand_anno_check/detector_infer_vis')
        os.makedirs(save_vis_dir, exist_ok=True)
    
    pred_hand_pose = {}
    for curr_anno_frame in tqdm(gt_anno.keys(), total=len(gt_anno)):
        curr_frame_pred = {}
        # Load mobile GoPro image at current frame
        img_path = os.path.join(vis_gp_playing_img_dir, f"playing_gopro_{int(curr_anno_frame):06d}.jpg")
        img = np.array(Image.open(img_path))
        
        # Project GT hand pose3d onto distorted gp image
        T_mps_aria_world = np.array(gt_cam_pose[curr_anno_frame]['aria01']['camera_extrinsics'])
        hand_3D_repro_distort = load_hand_pose_anno(gt_anno[curr_anno_frame], T_mps_aria_world, T_mps_gp_aria, K_gp05, D_gp05)
        
        # Get hand bbox based on GT projected pose3d kpts
        curr_right_bbox, curr_right_flag = get_bbox_from_kpts(hand_3D_repro_distort[:21, :], img.shape)
        curr_left_bbox, curr_left_flag = get_bbox_from_kpts(hand_3D_repro_distort[21:, :], img.shape)
        curr_bbox = [{'bbox': np.append(curr_right_bbox, 1)},
                     {'bbox': np.append(curr_left_bbox, 1)}]

        # Hand pose2d prediction with pretrained detector
        dataset = pose_model.cfg.data["test"]["type"]
        dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
        dataset_info = DatasetInfo(dataset_info)
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            img_path,
            curr_bbox,
            bbox_thr=0.3,
            format="xyxy",
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None,
        )

        # Append prediction result
        curr_frame_pred['kpts'] = [res['keypoints'][:,:2] for res in pose_results]
        curr_frame_pred['valid_flag'] = [curr_right_flag.tolist() if curr_right_flag is not None else curr_right_flag, 
                                        curr_left_flag.tolist() if curr_left_flag is not None else curr_left_flag]
        pred_hand_pose[curr_anno_frame] = curr_frame_pred

        # Save visualization if required
        if save_result and not os.path.exists(os.path.join(save_vis_dir, f"{int(curr_anno_frame):06d}.jpg")):
            hand_vis_pred = visualize_hand(img, pose_results[0]['keypoints'])
            hand_vis_pred = visualize_hand(hand_vis_pred, pose_results[1]['keypoints'])
            cv2.rectangle(hand_vis_pred, curr_left_bbox[:2], curr_left_bbox[2:], color=(0,0,255), thickness=2);
            cv2.rectangle(hand_vis_pred, curr_right_bbox[:2], curr_right_bbox[2:], color=(255,0,0), thickness=2);
            cv2.imwrite(os.path.join(save_vis_dir, f"{int(curr_anno_frame):06d}.jpg"), hand_vis_pred[:,:,::-1])

    return pred_hand_pose


def find_best_transformation(gt_anno, gt_cam_pose, T_mps_gp_aria_all, pred_hand_pose_2d, K_gp05, D_gp05, save_dir):
    mpjpe = []
    for selected_gp_aria_T_idx in tqdm(range(len(T_mps_gp_aria_all)), total=len(T_mps_gp_aria_all)):
        curr_mpjpe = []
        # Transformation from aria to gp with current index
        T_mps_gp_aria = T_mps_gp_aria_all[selected_gp_aria_T_idx]

        # Project GT hand pose3d onto distorted gp image for all frames
        for curr_anno_frame in gt_anno.keys():
            T_mps_aria_world = np.array(gt_cam_pose[curr_anno_frame]['aria01']['camera_extrinsics'])
            hand_3D_repro_distort = load_hand_pose_anno(gt_anno[curr_anno_frame], T_mps_aria_world, T_mps_gp_aria, K_gp05, D_gp05)
            # Compute mpjpe for each hand in current frame
            pred_pose2d = np.array(pred_hand_pose_2d[curr_anno_frame]['kpts'])
            valid_flag = pred_hand_pose_2d[curr_anno_frame]['valid_flag']
            for right_left_idx in range(2):
                curr_hand_valid_flag = valid_flag[right_left_idx]
                if curr_hand_valid_flag is not None:
                    curr_hand_pose2d_diff = pred_pose2d[right_left_idx][curr_hand_valid_flag] \
                        - hand_3D_repro_distort[21*right_left_idx:21*(right_left_idx+1)][curr_hand_valid_flag]
                    curr_mpjpe.append(np.mean(np.linalg.norm(curr_hand_pose2d_diff, axis=1)))
        mpjpe.append(np.mean(curr_mpjpe))

    # Sort mpjpe to find best index and save transformation
    best_index = np.argsort(mpjpe)[0]
    print(f"Find best transformation with idx={best_index}\t MPJPE (pixels): {mpjpe[best_index]:.2f}")
    with open(os.path.join(save_dir, 'transformation_MPS_gp_aria_best.json'), 'w') as f:
            json.dump(T_mps_gp_aria_all[best_index].tolist(), f, indent=4)



if __name__ == '__main__':
    # Config and take info
    take = '' # Modify if not using args
    save_pretrained_infer_vis = True # Save pretrained detector hand pose visualization or not
    
    args = get_parameters(take)
    capture_name = '_'.join(args.take.split('_')[:-1])
    takes = json.load(open(os.path.join(args.base_folder, "takes.json")))
    take_uid = [t['take_uid'] for t in takes if t["root_dir"] == args.take]
    assert len(take_uid) == 1, f"{args.take} doesn't have valid take metadata."
    take_uid = take_uid[0]

    # Load GT annotation and cam pose
    gt_anno_path = os.path.join(args.annotation_folder, f"annotation/{take_uid}.json")
    gt_anno = json.load(open(gt_anno_path))
    gt_cam_pose_dir = os.path.join(args.cam_pose_folder, f"{take_uid}.json")
    gt_cam_pose = json.load(open(gt_cam_pose_dir))
    vis_gp_playing_img_dir = os.path.join(args.work_dir, capture_name, args.take, f"vis_gp-playing-{args.mobile_cam}")
    # Load all transformation between aria and mobile GoPro camera in MPS system
    transform_save_dir = os.path.join(args.work_dir, capture_name, args.take)
    with open(os.path.join(transform_save_dir, 'transformation_MPS_gp_aria_allIdx.json'), 'r') as f:
        T_mps_gp_aria_all = np.array(json.load(f))
    # Load camera intrinsics & distortion parameter from metashape calibration file
    metashape_gp_calib = os.path.join(args.work_dir, capture_name, 'Metashape/mobile_gp.xml')
    K_gp05, D_gp05 = load_metashape_calib(metashape_gp_calib)

    # Load hand bbox detector and pose estimator
    det_model = init_detector(args.det_config, args.det_ckpt)
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt)

    # Run pretrained hand pose detector to get pseudo 2D pose
    pretrained_hand_pose_pred = hand_pose2d_inference(gt_anno, 
                                                      gt_cam_pose, 
                                                      vis_gp_playing_img_dir, 
                                                      T_mps_gp_aria_all[0], # Modify as needed
                                                      K_gp05, D_gp05,
                                                      save_result=save_pretrained_infer_vis)

    # Check for best transformation index
    find_best_transformation(gt_anno, 
                             gt_cam_pose, 
                             T_mps_gp_aria_all, 
                             pretrained_hand_pose_pred, 
                             K_gp05, D_gp05,
                             transform_save_dir)
