# first_order_model/demo.py
import matplotlib
matplotlib.use('Agg')
import os
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import argparse
from tqdm import tqdm

from modules.generator import Generator
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from utils.animate import normalize_kp
from utils.reconstruction import make_animation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint")
    parser.add_argument("--source_image", required=True, help="path to source image")
    parser.add_argument("--driving_video", required=True, help="path to driving video")
    parser.add_argument("--result_video", required=True, help="path to output")
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale")

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generator = Generator(**config['model_params']['generator_params'])
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'])

    checkpoint = torch.load(opt.checkpoint, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator.eval()
    kp_detector.eval()
    
    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    predictions = make_animation(source_image, driving_video, generator, kp_detector, 
                               relative=opt.relative, adapt_movement_scale=opt.adapt_scale)
    
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
