import sys

sys.path.append('/data3/dinov2_3d-cleanup')
from pathlib import Path
from typing import Union

import torch.nn as nn
from dinov2.configs import load_and_merge_config_3d
from dinov2.eval.setup import build_model_for_eval

from .common3D import LayerNorm3d


class DinoV2ImageEncoder(nn.Module):
    DINO_EMBED_DIM = 1024
    
    def __init__(
            self,
            model_cfg: Union[str, Path],
            pretrained_weights: Union[str, Path],
            img_size: int,
            out_chans: int,
        ):
        super().__init__()
        cfg = load_and_merge_config_3d(model_cfg)
        self.model = build_model_for_eval(cfg, pretrained_weights)
        self.img_size = img_size

        self.desired_out_spatial = self.img_size // 2 ** 4
        self.desired_out_channel = out_chans

        self.neck = nn.Sequential(
            nn.Conv3d(
                self.DINO_EMBED_DIM,
                self.desired_out_channel,
                kernel_size=1,
                bias=False,
                stride=1,
            ),
            LayerNorm3d(self.desired_out_channel),
        )

    def forward(self, x):
        x = self.model(x, is_training=True)["x_norm_patchtokens"]
        x = self.neck(x) # B HWD C
        x = x.transpose(1,2) # B C HWD
        x = x.reshape(
            x.shape[0],
            self.desired_out_channel,
            self.desired_out_spatial,
            self.desired_out_spatial,
            self.desired_out_spatial,
        )
        
        return x