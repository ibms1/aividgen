# first_order_model/utils/video.py
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch

def load_checkpoints(config_path, checkpoint_path, device='cpu'):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = Generator(**config['model_params']['generator_params'])
    generator.to(device)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'])
    kp_detector.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector

def make_animation(source_image, driving_video, generator, kp_detector,
                  relative=True, adapt_movement_scale=True, device='cpu'):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in range(driving.shape[2]):
            driving_frame = driving[:, :, frame_idx]
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                 kp_driving_initial=kp_driving_initial, 
                                 adapt_movement_scale=adapt_movement_scale,
                                 use_relative_movement=relative)
            out = generator(source, kp_driving=kp_norm, kp_source=kp_source)
            predictions.append(np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])[0])

    return predictions