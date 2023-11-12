import os
import time
import cv2 as cv2
import json
from tqdm import tqdm
from config import get_parameters
from utils import decode_video_to_images



if __name__ == '__main__':
    ######################################
    TAKE = None # MODIFY e.g. "upenn_0718_Violin_2_5"
    ######################################

    args = get_parameters(TAKE)
    capture_name = '_'.join(args.take.split('_')[:-1])

    ################# Extract images from gopro walkaround #################
    print(f'{"="*20} Extracting gp-walkaround images {"="*20}')
    gp_mobile_image_dir = os.path.join(args.work_dir, capture_name, 'gp_walkaround')
    gp_mobile_video_path = None # "/mnt/8tbvol2/MusicData/07182023_Violin/0718_Violin_2/exo/mobile/GX010351.MP4"
    decode_video_to_images(gp_mobile_video_path, gp_mobile_image_dir, args.mobile_width, args.mobile_height, args.mobile_rate)

    ################# Extract images from aria walkaround #################
    # TODO: find a way to extract just part of the vrs
    print(f'{"="*20} Extracting raw aria images {"="*20}')
    vrs_file = os.path.join(args.vrs_folder, 'aria01.vrs')
    vrs_image_folder = os.path.join(args.work_dir, capture_name, 'aria_images', 'raw_aria_images')
    os.makedirs(vrs_image_folder, exist_ok=True)
    cmd = 'vrs extract-images {} --to {} + 214-1'.format(vrs_file, vrs_image_folder)
    os.system(cmd)
    
    # move a subset to take folder
    print(f'{"="*20} Moving raw aria images {"="*20}')
    start_frame = args.aria_walkaround_start_frame
    end_frame = args.aria_walkaround_end_frame
    sub_img_list = sorted(os.listdir(vrs_image_folder))[start_frame:end_frame][::10]
    aria_mobile_image_dir = os.path.join(args.work_dir, capture_name, 'aria_images', 'aria_walkaround')
    os.makedirs(aria_mobile_image_dir, exist_ok=True)
    for img_file in tqdm(sub_img_list):
        cmd = 'cp {} {}'.format(os.path.join(vrs_image_folder, img_file), os.path.join(aria_mobile_image_dir, img_file))
        os.system(cmd)

    ################# Extract images from aria playing #################
    print(f'{"="*20} Extracting aria-playing images {"="*20}')
    aria_playing_image_dir = os.path.join(args.work_dir, capture_name, args.take, 'playing_aria')
    aria_playing_video_path = os.path.join(args.video_folder, 'aria01_214-1.mp4')
    decode_video_to_images(aria_playing_video_path, aria_playing_image_dir, args.aria_width, args.aria_height, args.playing_rate, 'playing_aria')

    # rotate image 90 degree counterclockwise
    # TODO: find a better way
    print(f'{"="*20} Rotating aria-playing images {"="*20}')
    aria_playing_image_dir_ = sorted(os.listdir(aria_playing_image_dir))
    for img_file in tqdm(aria_playing_image_dir_):
        im = cv2.imread(os.path.join(aria_playing_image_dir, img_file))
        im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(aria_playing_image_dir, img_file), im)

    ################# Extract images from gopro playing #################
    # playing gopro frames
    print(f'{"="*20} Extracting gp-playing images {"="*20}')
    gp_playing_image_dir = os.path.join(args.work_dir, capture_name, args.take, 'playing_gopro')
    gp_playing_video_path = os.path.join(args.video_folder, '{}.mp4'.format(args.mobile_cam))
    decode_video_to_images(gp_playing_video_path, gp_playing_image_dir, args.mobile_width, args.mobile_height, args.playing_rate, 'playing_gopro')

    ################# Extract images from exo gopro for visualization #################
    print(f'{"="*20} Extracting sample exo-playing images for annotation visualization {"="*20}')
    vis_exo_image_dir = os.path.join(args.work_dir, capture_name, args.take, f"vis_exo-playing-{args.exo_cam}/original_img")
    vis_exo_video_path = os.path.join(args.video_folder, '{}.mp4'.format(args.exo_cam))
    decode_video_to_images(vis_exo_video_path, vis_exo_image_dir, args.exo_width, args.exo_height, args.playing_rate)

    ################# Extract images from mobile gopro for hand annotations #################
    print(f'{"="*20} Extracting gp-playing images for annotation visualization {"="*20}')
    gp_anno_image_dir = os.path.join(args.work_dir, capture_name, args.take, f"vis_gp-playing-{args.mobile_cam}")
    gp_video_path = os.path.join(args.video_folder, '{}.mp4'.format(args.mobile_cam))
    subclip_frame_path = os.path.join(args.annotation_folder, 'selected_frames_info_annotation', f"{args.take}.json")
    subclip_frame = json.load(open(subclip_frame_path))
    decode_video_to_images(gp_video_path, gp_anno_image_dir, args.mobile_width, args.mobile_height, args.annotation_rate, 'playing_gopro', subclip_frame)
