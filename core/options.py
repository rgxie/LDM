import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    # The seed used during inference
    seed: Optional[int] = None
    
    # dataset config
    is_crop: Optional[bool] = True
    is_fix_views: bool = False
    
    # True for text prompts
    txt_or_image: Optional[bool] = False 
    text_prompt: Optional[str] = 'a cute owl'
    
    infer_render_size: int = 256
    # True for mvdream  False for zero123plus
    mvdream_or_zero123: Optional[bool] = True 
    
    rar_data: bool = True
    
    # Unet image input size
    input_size: int = 512
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 64
    # render size
    output_size: Optional[int] = 128

    # for tensorsdf
    density_n_comp: int = 8
    app_n_comp: int = 32
    shadingMode: Literal['MLP_Fea']='MLP_Fea' #'MLP_Fea'
    view_pe: int = 2
    fea_pe: int = 2
    pos_pe: int = 6
    # points number sampled per ray
    n_sample: int = 64  

    volume_mode: Literal['TRF_Mesh','TRF_SDF'] = 'TRF_SDF'


    # for LRM_Net
    camera_embed_dim: int=1024
    transformer_dim: int=1024
    transformer_layers: int=16
    transformer_heads: int=16
    triplane_low_res: int=32
    triplane_high_res: int=64
    encoder_type: str ='dinov2'
    encoder_model_name: str = 'dinov2_vitb14_reg'
    encoder_feat_dim: int = 768 #768
    encoder_freeze: bool = False
    
    # training
    over_fit: Optional[bool] = False

    ### dataset
    # data mode (only support s3 now)
    data_mode: Literal['s5','s6'] = 's5'
    data_path: str = 'train_data'
    data_debug_list: str = 'dataset_debug/gobj_merged_debug.json'
    
    # TODO Please replace with your training data list
    data_list_path: str = 'gobjs_selected.json' 
    # fovy of the dataset
    fovy: float = 39.6 
    # camera near plane
    znear: float = 0.5
    # camera far plane
    zfar: float = 2.5
    # number of all views (input + output)
    num_views: int = 12
    # number of views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8 #8

    ### training
    # workspace
    workspace: str = './workspace_test'
    # resume
    resume: Optional[str] = None
    ckpt_nerf: Optional[str] = None
    # batch size (per-GPU)
    batch_size: int = 8
    # gradient accumulation
    gradient_accumulation_steps: Optional[int] = 1
    # training epochs
    num_epochs:  Optional[int] = 50
    # lpips loss weight
    lambda_lpips: float = 1.0
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16'
    # learning rate
    lr: Optional[float] = 4e-4
    lr_scheduler: str = 'OneCycleLR'
    warmup_real_iters: int = 3000

    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5

    ### testing
    # test image path
    test_path: Optional[str] = None

    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    

# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['ldm'] = 'the default settings for LDM'
config_defaults['ldm'] = Options()


config_doc['tiny_trf_trans_mesh'] = 'tiny model for ablation'
config_defaults['tiny_trf_trans_mesh'] = Options(
    input_size=512, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    volume_mode='TRF_Mesh',
    splat_size=64,
    output_size=512,
    data_mode='s6',
    batch_size=1,  #8
    num_views=8,
    gradient_accumulation_steps=1,  #2
    mixed_precision='no',
)

config_doc['tiny_trf_trans_sdf'] = 'tiny model for ablation'
config_defaults['tiny_trf_trans_sdf'] = Options(
    input_size=512, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    volume_mode='TRF_SDF',
    splat_size=64,
    output_size=62, #crop patch
    data_mode='s5',
    batch_size=4,  #8
    num_views=8,
    gradient_accumulation_steps=1,  #2
    mixed_precision='bf16',
)

config_doc['tiny_trf_trans_sdf_123plus'] = 'tiny model for ablation'
config_defaults['tiny_trf_trans_sdf_123plus'] = Options(
    input_size=512, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    volume_mode='TRF_SDF',
    mvdream_or_zero123 = False,
    splat_size=64,
    output_size=64, #crop patch
    data_mode='s5',
    batch_size=3,  #8
    num_views=10,
    num_input_views=6,
    gradient_accumulation_steps=1,  #2
    mixed_precision='bf16',
)


config_doc['tiny_trf_trans_sdf_nocrop'] = 'tiny model for ablation'
config_defaults['tiny_trf_trans_sdf_nocrop'] = Options(
    input_size=512, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    volume_mode='TRF_SDF',
    splat_size=64,
    output_size=62, #crop patch
    data_mode='s5',
    batch_size=4,  #8
    is_crop=False,
    num_views=8,
    gradient_accumulation_steps=1,  #2
    mixed_precision='bf16',
)


AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
