import os
import time
import cv2 as cv2

from config import get_parameters
from utils import decode_video_to_images



if __name__ == '__main__':

    args = get_parameters('upenn_0718_Violin_2_5')

    # ################# Extract images from gopro walkaround #################
    # image_dir = os.path.join(args.video_folder, 'mobile')
    # video_path = os.path.join(args.video_folder, 'mobile.MP4')
    # decode_video_to_images(video_path, image_dir, args.mobile_width, args.mobile_height, args.mobile_rate)

    ################# Extract images from aria playing #################
    image_dir = os.path.join(args.video_folder, 'playing_aria')
    video_path = os.path.join(args.video_folder, 'aria01_214-1.mp4')
    decode_video_to_images(video_path, image_dir, args.aria_width, args.aria_height, args.playing_rate, 'playing_aria')

    # rotate image 90 degree counterclockwise
    # TODO: find a better way
    img_files = sorted(os.listdir(image_dir))
    for img_file in img_files:
        im = cv2.imread(os.path.join(image_dir, img_file))
        im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(image_dir, img_file), im)

    ################# Extract images from gopro playing #################
    # playing gopro frames
    image_dir = os.path.join(args.video_folder, 'playing_gopro')
    video_path = os.path.join(args.video_folder, '{}.mp4'.format(args.mobile_cam))
    decode_video_to_images(video_path, image_dir, args.mobile_width, args.mobile_height, args.playing_rate, 'playing_gopro')

    # ################# Extract images from aria walkaround #################
    # # TODO: find a way to extract just part of the vrs
    # vrs_file = os.path.join(args.vrs_folder, 'aria01.vrs')
    # vrs_image_folder = os.path.join(os.path.dirname(vrs_file), 'images')
    # os.makedirs(vrs_image_folder, exist_ok=True)
    # cmd = 'vrs extract-images {} --to {} + 214-1'.format(vrs_file, vrs_image_folder)
    # os.system(cmd)
    #
    # # move a subset to take folder
    # start_frame = args.aria_walkaround_start_frame
    # end_frame = args.aria_walkaround_end_frame
    # sub_img_list = sorted(os.listdir(vrs_image_folder))[start_frame:end_frame][::10]
    # image_dir = os.path.join(args.video_folder, 'walkaround_aria')
    # os.makedirs(image_dir, exist_ok=True)
    # for img_file in sub_img_list:
    #     cmd = 'cp {} {}'.format(os.path.join(vrs_image_folder, img_file), os.path.join(image_dir, img_file))
    #     os.system(cmd)

    # ################# Extract images from exo gopro for visualization #################
    # image_dir = os.path.join(args.video_folder, args.exo_cam)
    # video_path = os.path.join(args.video_folder, '{}.mp4'.format(args.exo_cam))
    # decode_video_to_images(video_path, image_dir, args.exo_width, args.exo_height, args.playing_rate)

    ################# Extract images from mobile gopro for hand annotations (fps=30) #################
    image_dir = os.path.join(args.video_folder, args.mobile_cam)
    video_path = os.path.join(args.video_folder, '{}.mp4'.format(args.mobile_cam))
    decode_video_to_images(video_path, image_dir, args.mobile_width, args.mobile_height, args.annotation_rate, 'playing_gopro')

    ################# Extract images from aria for hand annotations (fps=30) #################
    image_dir = os.path.join(args.video_folder, 'aria01')
    video_path = os.path.join(args.video_folder, 'aria01_214-1.mp4')
    decode_video_to_images(video_path, image_dir, args.aria_width, args.aria_height, args.annotation_rate, 'playing_aria')