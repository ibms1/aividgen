# first_order_model/modules/dense_motion.py
import torch
from torch import nn
import torch.nn.functional as F
from .util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid

class DenseMotionNetwork(nn.Module):
    """
    Module that predicts dense motion field from sparse motion representation
    """
    def __init__(self, block_expansion, num_blocks, max_features, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_channels * (num_channels + 1)),
                                 max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_channels + 1, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_channels = num_channels
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        self.kp_variance = kp_variance

    def forward(self, source_image, kp_driving, kp_source):
        # Source image processing
        bs, _, h, w = source_image.shape
        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        input_ = torch.cat([heatmap_representation, source_image], dim=1)
        input_ = F.interpolate(input_, size=(h // 2, w // 2), mode='bilinear')

        prediction = self.hourglass(input_)
        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask

        # Compute deformed feature maps
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source

        # Compute occlusion map if required
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict