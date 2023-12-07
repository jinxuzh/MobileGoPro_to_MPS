import os
import numpy as np
import cv2 as cv2
import av
import torch
import math
from tqdm import tqdm
from torchaudio.io import StreamReader
from torchvision.transforms import Resize
from typing import Optional
import xml.etree.ElementTree as ET
from .visualization_rerun import right_hand, left_hand


def decode_video_to_images(video_path, image_dir, im_width, im_height, framerate, prefix=None, subclip_frame_num=None):

    os.makedirs(image_dir, exist_ok=True)
    if subclip_frame_num:
        # Read in video
        reader = PyAvReader(
            path=video_path,
            resize=(im_height,im_width),
            mean=None,
            frame_window_size=1,
            stride=1,
            gpu_idx=-1,
            )
        for idx in tqdm(subclip_frame_num):
            out_path = os.path.join(image_dir, f"{prefix}_{idx:06d}.jpg") if prefix else os.path.join(image_dir, f"{idx:06d}.jpg")
            if not os.path.exists(out_path):
                frame = reader[idx][0].cpu().numpy()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                assert cv2.imwrite(out_path, frame), out_path
    else:
        if prefix is not None:
            cmd = 'ffmpeg -i {} -vf scale={}:{} -r {} {}/{}_%06d.jpg'.format(
                video_path, im_width, im_height, framerate, image_dir, prefix)
        else:
            cmd = 'ffmpeg -i {} -vf scale={}:{} -r {} {}/%06d.jpg'.format(
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


def get_video_meta(path):
    with av.open(path) as cont:
        n_frames = cont.streams[0].frames
        codec = cont.streams[0].codec.name
        tb = cont.streams[0].time_base

        all_pts = []
        for x in cont.demux(video=0):
            if x.pts is None:
                continue
            all_pts.append(x.pts)

        assert len(all_pts) == n_frames
        return {
            "all_pts": sorted(all_pts),
            "codec": codec,
            "tb": tb,
        }
    

def load_metashape_calib(calib_path):
    """
    Load camera calibration (intrinsic + distortion parameter) 
    from Metahape output.
    """
    tree = ET.parse(calib_path)
    root = tree.getroot()
    gp_calib = {}
    for i in range(1,len(root)-1):
        gp_calib[root[i].tag] = float(root[i].text)
    K_gp05 = [[gp_calib['f'], 0, gp_calib['width']/2 + gp_calib['cx']],
                [0, gp_calib['f'], gp_calib['height']/2 + gp_calib['cy']],
                [0, 0, 1]]
    D_gp05 = [gp_calib['k1'], gp_calib['k2'], gp_calib['k3'], 0]
    return K_gp05, D_gp05


def load_hand_pose_anno(curr_frame_anno, T_mps_aria_world, T_mps_gp_aria, K, D):
    hand_3D = np.zeros((42, 5))  # x, y, z, 1, valid
    BOTH_HAND = right_hand + left_hand # right first, then left
    for idx, joint in enumerate(BOTH_HAND):
        if joint in curr_frame_anno[0]['annotation3D'].keys():
            hand_3D[idx, 0] = curr_frame_anno[0]['annotation3D'][joint]['x']
            hand_3D[idx, 1] = curr_frame_anno[0]['annotation3D'][joint]['y']
            hand_3D[idx, 2] = curr_frame_anno[0]['annotation3D'][joint]['z']
            hand_3D[idx, 3] = 1
            hand_3D[idx, 4] = 1

    T_mps_aria_world = np.concatenate([T_mps_aria_world, np.array([[0, 0, 0, 1]])])
    hand_3D_repro = T_mps_gp_aria[:3, :] @ T_mps_aria_world @ hand_3D[:, :4].T
    hand_3D_repro = hand_3D_repro[:2, :].T / hand_3D_repro[2:, :].T
    hand_3D_repro_distort = cv2.fisheye.distortPoints(np.expand_dims(hand_3D_repro, axis=1), 
                                                      np.array(K),
                                                      np.array(D))
    hand_3D_repro_distort = np.nan_to_num(hand_3D_repro_distort)
    return hand_3D_repro_distort.squeeze()


def get_bbox_from_kpts(kpts, img_shape, padding=30):
    img_H, img_W = img_shape[:2]
    # Filter invalid kpts and return None if less than 5 valid kpts exists
    valid_flag = (kpts[:,0] > 0) & (kpts[:,0] < img_W) & (kpts[:,1] > 0) & (kpts[:,1] < img_H)
    if np.sum(valid_flag) < 10:
        return [0,0,0,0], None
    
    kpts = kpts[valid_flag]
    # Get proposed hand bounding box from hand keypoints
    x1, y1, x2, y2 = (
        kpts[:, 0].min(),
        kpts[:, 1].min(),
        kpts[:, 0].max(),
        kpts[:, 1].max(),
    )
    # Proposed hand bounding box with padding
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
        np.clip(x1 - padding, 0, img_W - 1),
        np.clip(y1 - padding, 0, img_H - 1),
        np.clip(x2 + padding, 0, img_W - 1),
        np.clip(y2 + padding, 0, img_H - 1),
    )
    # Return bbox result
    return np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2]).astype(np.int64), valid_flag


class StridedReader:
    def __init__(self, path, stride, frame_window_size):
        self.path = path
        self.meta = get_video_meta(path)
        self.all_pts = self.meta["all_pts"]
        self.stride = stride
        self.frame_window_size = frame_window_size
        if self.stride == 0:
            self.stride = self.frame_window_size

    def __getitem__(self, _: int) -> torch.Tensor:
        raise AssertionError("Not implemented")

    def __len__(self):
        return int(
            math.ceil((len(self.all_pts) - self.frame_window_size) / self.stride)
        )
    

class PyAvReader(StridedReader):
    def __init__(
        self,
        path: str,
        resize: tuple[int, int],
        mean: Optional[torch.Tensor],
        frame_window_size: int,
        stride: int,
        gpu_idx: int,
    ):
        super().__init__(path, stride, frame_window_size)

        if gpu_idx >= 0:
            raise AssertionError("GPU decoding not support for pyav")

        self.mean = mean
        self.resize = Resize(resize) if resize is not None else None
        self.path = path
        self.create_underlying_cont(gpu_idx)

    def create_underlying_cont(self, gpu_id):
        self.cont = av.open(self.path)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame_i = self.stride * idx
        frame_j = frame_i + self.frame_window_size
        assert frame_i >= 0 and frame_j < len(self.all_pts)

        frame_i_pts = self.all_pts[frame_i]
        frame_j_pts = self.all_pts[frame_j]
        # self.cont.streams.video[0].seek(frame_i_pts)
        self.cont.seek(frame_i_pts, stream=self.cont.streams.video[0])
        fs = []
        for f in self.cont.decode(video=0):
            # print(f.pts)
            if f.pts < frame_i_pts:
                continue
            if f.pts >= frame_j_pts:
                break
            fs.append((f.pts, torch.tensor(f.to_ndarray(format="rgb24"))))
        fs = sorted(fs, key=lambda x: x[0])
        ret = torch.stack([x[1] for x in fs]).permute(0,3,1,2) # (B, C, H, W)
        if self.resize is not None:
            ret = self.resize(ret)
        if self.mean is not None:
            ret -= self.mean
        ret = ret.permute(0,2,3,1)
        return ret