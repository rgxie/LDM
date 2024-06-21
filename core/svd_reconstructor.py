import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Literal
from functools import partial

import itertools
from core.lrm.embedder import CameraEmbedder
from core.lrm.transformer import TransformerDecoder


class LDM_SVD_Net(nn.Module):
    """
    predict SVD using transformer
    """
    def __init__(self, camera_embed_dim: int, 
                 transformer_dim: int, transformer_layers: int, transformer_heads: int,
                 triplane_low_res: int, triplane_high_res: int, 
                 encoder_freeze: bool = True, encoder_type: str = 'dino',
                 encoder_model_name: str = 'facebook/dino-vitb16', encoder_feat_dim: int = 768, app_n_comp=24,
                 density_n_comp=8):
        super().__init__()
        
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.camera_embed_dim = camera_embed_dim
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.transformer_dim=transformer_dim

        # modules
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            modulation_dim=self.camera_embed_dim,  #mod camera vector 
            freeze=encoder_freeze,
        )
        self.camera_embedder = CameraEmbedder(
            raw_dim=12+4, embed_dim=camera_embed_dim,
        )

        self.n_comp=app_n_comp+density_n_comp
        self.app_n_comp=app_n_comp
        self.density_n_comp=density_n_comp
        
        self.pos_embed = nn.Parameter(torch.randn(1, 3*(triplane_low_res**2)+3*triplane_low_res, transformer_dim) * (1. / transformer_dim) ** 0.5)
        self.transformer = TransformerDecoder(
            block_type='cond',
            num_layers=transformer_layers, num_heads=transformer_heads,
            inner_dim=transformer_dim, cond_dim=encoder_feat_dim, mod_dim=None,
        )
        
        self.upsampler = nn.ConvTranspose2d(transformer_dim, self.n_comp, kernel_size=2, stride=2, padding=0)
        self.dim_map = nn.Linear(transformer_dim,self.n_comp)
        self.up_line = nn.Linear(triplane_low_res,triplane_low_res*2)


    @staticmethod
    def _encoder_fn(encoder_type: str):
        encoder_type = encoder_type.lower()
        assert encoder_type in ['dino', 'dinov2'], "Unsupported encoder type"
        if encoder_type == 'dino':
            from .encoders.dino_wrapper import DinoWrapper
            return DinoWrapper
        elif encoder_type == 'dinov2':
            from .encoders.dinov2_wrapper import Dinov2Wrapper
            return Dinov2Wrapper

    def forward_transformer(self, image_feats, camera_embeddings=None):

        N = image_feats.shape[0]
        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        x = self.transformer(
            x,
            cond=image_feats,
            mod=camera_embeddings,
        )
        return x
    
    def reshape_upsample(self, tokens):
        #B,_,3*ncomp
        N = tokens.shape[0]
        H = W = self.triplane_low_res
        P=self.n_comp
        # planes
        plane_tokens= tokens[:,:3*H*W,:].view(N,H,W,3,self.transformer_dim)
        plane_tokens = torch.einsum('nhwip->inphw', plane_tokens)  # [3, N, P, H, W]
        plane_tokens = plane_tokens.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        plane_tokens = self.upsampler(plane_tokens)  # [3*N, P, H', W']
        plane_tokens = plane_tokens.view(3, N, *plane_tokens.shape[-3:])  # [3, N, P, H', W']
        plane_tokens = torch.einsum('inphw->niphw', plane_tokens)  # [N, 3, P, H', W']
        plane_tokens = plane_tokens.reshape(N, 3*P, *plane_tokens.shape[-2:])  # # [N, 3*P, H', W']
        plane_tokens = plane_tokens.contiguous()
        
        #lines
        line_tokens= tokens[:,3*H*W:3*H*W+3*H,:].view(N,H,3,self.transformer_dim)
        line_tokens= self.dim_map(line_tokens)
        line_tokens = torch.einsum('nhip->npih', line_tokens) # [ N, P, 3, H]
        line_tokens=self.up_line(line_tokens) 
        line_tokens = torch.einsum('npih->niph', line_tokens) # [ N, 3, P, H]
        line_tokens=line_tokens.reshape(N,3*P,line_tokens.shape[-1],1)
        line_tokens = line_tokens.contiguous()
        

        return plane_tokens[:,:self.app_n_comp*3,:,:],line_tokens[:,:self.app_n_comp*3,:,:],plane_tokens[:,self.app_n_comp*3:,:,:],line_tokens[:,self.app_n_comp*3:,:,:]


    def forward_planes(self, image, camera):
        N,V,_,H,W = image.shape
        image=image.reshape(N*V,3,H,W)
        camera=camera.reshape(N*V,-1)
        
    
        camera_embeddings = self.camera_embedder(camera)
        assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
            f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"

        image_feats = self.encoder(image, camera_embeddings)
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        image_feats=image_feats.reshape(N,V*image_feats.shape[-2],image_feats.shape[-1])
    
        tokens = self.forward_transformer(image_feats)
        
        app_planes,app_lines,density_planes,density_lines = self.reshape_upsample(tokens)

        return app_planes,app_lines,density_planes,density_lines

    def forward(self, image,source_camera):

        assert image.shape[0] == source_camera.shape[0], "Batch size mismatch for image and source_camera"
        planes = self.forward_planes(image, source_camera)

        #B,3,dim,H,W
        return planes