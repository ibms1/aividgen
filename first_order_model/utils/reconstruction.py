# first_order_model/utils/reconstruction.py
import torch
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize
from .animate import normalize_kp

def make_animation(source_image, driving_video, generator, kp_detector, relative=True,
                  adapt_movement_scale=True, device='cpu'):
    """
    Creates animation from source image and driving video
    """
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in range(driving.shape[2]):
            driving_frame = driving[:, :, frame_idx].to(device)
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                 kp_driving_initial=kp_driving_initial, 
                                 adapt_movement_scale=adapt_movement_scale,
                                 use_relative_movement=relative)
            out = generator(source, kp_driving=kp_norm, kp_source=kp_source)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

    return predictions