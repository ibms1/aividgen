import subprocess
import sys

# تثبيت المكتبات المطلوبة
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    "matplotlib",
    "numpy",
    "torch",
    "opencv-python",
    "Pillow",
    "scikit-learn",
    "tqdm",
    "ffmpeg-python",
    "gdown"
]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"جاري تثبيت {package}...")
        install_package(package)

# استيراد المكتبات المطلوبة
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image
from skimage import img_as_ubyte
from tqdm import tqdm
import yaml
from argparse import ArgumentParser
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp

# تحميل التكوين
def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

# تحميل النموذج
def load_checkpoint(config_path, checkpoint_path, device):
    config = load_config(config_path)
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                       **config['model_params']['common_params'])
    generator.to(device)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

# تحويل الصورة إلى Tensor
def image_to_tensor(image_path, device):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
    return image.to(device)

# تحويل الفيديو إلى Tensor
def video_to_tensor(video_path, device):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
        frames.append(frame)
    cap.release()
    return torch.cat(frames, dim=0).to(device)

# إنشاء الفيديو المتحرك
def animate(source_image, driving_video, generator, kp_detector, device, output_path):
    with torch.no_grad():
        source = image_to_tensor(source_image, device)
        driving = video_to_tensor(driving_video, device)

        kp_source = kp_detector(source)
        kp_driving = kp_detector(driving)

        predictions = []
        for frame in tqdm(driving):
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving, kp_driving_current=frame)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            predictions.append(out['prediction'].squeeze().cpu().numpy().transpose(1, 2, 0))

        # حفظ الفيديو
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (predictions[0].shape[1], predictions[0].shape[0]))
        for frame in predictions:
            out.write(img_as_ubyte(frame))
        out.release()

# الدالة الرئيسية
def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the config file")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file")
    parser.add_argument("--source_image", required=True, help="Path to the source image")
    parser.add_argument("--driving_video", required=True, help="Path to the driving video")
    parser.add_argument("--result_video", required=True, help="Path to the output video")
    parser.add_argument("--relative", action="store_true", help="Use relative keypoint displacement")
    parser.add_argument("--adapt_scale", action="store_true", help="Adapt the scale of the keypoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator, kp_detector = load_checkpoint(args.config, args.checkpoint, device)
    animate(args.source_image, args.driving_video, generator, kp_detector, device, args.result_video)

if __name__ == "__main__":
    main()