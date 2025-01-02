# first_order_model/modules/generator.py
import torch
from torch import nn
import torch.nn.functional as F
from .util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d

class Generator(nn.Module):
    def __init__(self, num_channels=3, block_expansion=64, max_features=512,
                 num_down_blocks=2, num_bottleneck_blocks=6, estimate_occlusion_map=True,
                 dense_motion_params=None, estimate_jacobian=False):
        super(Generator, self).__init__()
        
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def forward(self, source_image, kp_driving, kp_source):
        # Implementation details...
        out = self.first(source_image)
        for down_block in self.down_blocks:
            out = down_block(out)
        
        # Bottleneck
        for up_block in self.up_blocks:
            out = up_block(out)
            
        out = self.final(out)
        return out