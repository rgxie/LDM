
import os
import tyro
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera
from core.utils import get_rays, grid_distortion, orbit_camera_jitter
from core.options import AllConfigs, Options
from core.models import LDM_Mesh,LDM_SDF
from core.utils import save_obj, save_obj_with_mtl
from mvdream.pipeline_mvdream import MVDreamPipeline
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
import random
from PIL import Image as pil_image

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

opt = tyro.cli(AllConfigs)


# model
if opt.volume_mode == 'TRF_Mesh':
    model = LDM_Mesh(opt)
elif opt.volume_mode == 'TRF_SDF':
    model = LDM_SDF(opt)
else:
    raise NotImplementedError
    
# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else: #ckpt
        ckpt_dict = torch.load(opt.resume, map_location='cpu')
        ckpt=ckpt_dict["model"]

    state_dict = model.state_dict()
    for k, v in ckpt.items():
        k=k.replace('module.', '')
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
        else:
            print(f'[WARN] unexpected param {k}: {v.shape}')
    print(f'[INFO] load resume success!')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if opt.volume_mode == 'TRF_Mesh':
    model = model.float().to(device)
else:
    model = model.half().to(device)
model.eval()


tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

if opt.mvdream_or_zero123:
    if not opt.txt_or_image:
    # load image dream
        pipe = MVDreamPipeline.from_pretrained(
            "ashawkey/imagedream-ipmv-diffusers", # remote weights
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        )
        pipe = pipe.to(device)
    else:
        pipe = MVDreamPipeline.from_pretrained(
            'ashawkey/mvdream-sd2.1-diffusers', # remote weights
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        )
        pipe = pipe.to(device)
else:
    print('Loading 123plus model ...')
    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing='trailing'
    )

    unet_path='./pretrained/diffusion_pytorch_model.bin' 

    print('Loading custom white-background unet ...')
    if os.path.exists(unet_path):
        unet_ckpt_path = unet_path
    else:
        unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipe.unet.load_state_dict(state_dict, strict=True)
    pipe = pipe.to(device)

# load rembg
bg_remover = rembg.new_session()

# process function
def process_image(opt: Options, path):

    if opt.seed == None:
        seed = random.randint(0, 99999)
    else:
        seed = opt.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"seed:{seed}")
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)

    input_image = pil_image.open(path)
    input_image = np.array(input_image).astype(np.uint8)

    # bg removal
    carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # recenter
    image = recenter(carved_image, mask, border_ratio=0.2)
    
    # generate mv
    image = image.astype(np.float32) / 255.0

    # rgba to rgb white bg
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        
    kiui.write_image(f'{opt.workspace}/{name}/{name}_input.jpg', image)

    if opt.mvdream_or_zero123:
        mv_image = pipe('', image, guidance_scale=5.0, num_inference_steps=30, elevation=0)  
        mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    else:
        from PIL import Image
        from einops import rearrange, repeat
        
        image=image* 255
        image = Image.fromarray(image.astype('uint8'))
        mv_image = pipe(image, num_inference_steps=30).images[0]
        mv_image = np.asarray(mv_image, dtype=np.float32) / 255.0
        mv_image = torch.from_numpy(mv_image).permute(2, 0, 1).contiguous().float()     
        mv_image = rearrange(mv_image, 'c (n h) (m w) -> (n m) h w c', n=3, m=2).numpy() 
        

    predict_images=np.concatenate(mv_image,axis=1)
    kiui.write_image(f'{opt.workspace}/{name}/{name}_mv.jpg', predict_images)
    
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) 
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)

    data = {}
    
    images_input_vit = F.interpolate(input_image, size=(224, 224), mode='bilinear', align_corners=False)

    input_image = input_image.unsqueeze(0) 
    images_input_vit=images_input_vit.unsqueeze(0)
    data['input_vit']=images_input_vit
    
    elevation = 0
    cam_poses =[]
    
    if opt.mvdream_or_zero123:
        azimuth = np.arange(0, 360, 90, dtype=np.int32)
        for azi in tqdm.tqdm(azimuth):
            cam_pose = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
            cam_poses.append(cam_pose)
    else:
        azimuth = np.arange(30, 360, 60, dtype=np.int32)
        cnt = 0
        for azi in tqdm.tqdm(azimuth):
            if (cnt+1) % 2!= 0:
                elevation=-20
            else:
                elevation=30
            cam_pose = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
            cam_poses.append(cam_pose)
            cnt=cnt+1
            
    cam_poses = torch.cat(cam_poses,0)
    radius = torch.norm(cam_poses[0, :3, 3])
    cam_poses[:, :3, 3] *= opt.cam_radius / radius
    transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32).to(device) @ torch.inverse(cam_poses[0])
    cam_poses = transform.unsqueeze(0) @ cam_poses 
    
    cam_poses=cam_poses.unsqueeze(0)
    data['source_camera']=cam_poses

    
    with torch.no_grad():
        if opt.volume_mode == 'TRF_Mesh':
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                svd_volume = model.forward_svd_volume(input_image,data)
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                svd_volume = model.forward_svd_volume(input_image,data)
         
        export_texmap=False
        
        mesh_out = model.extract_mesh(svd_volume,use_texture_map=export_texmap)
        

        mesh_path=os.path.join(opt.workspace, name, name + '.obj')
        
        if export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            
            for i in range(len(tex_map)):
                mesh_path=os.path.join(opt.workspace, name, name + str(i) + '_'+ str(seed)+ '.obj')
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map[i].permute(1, 2, 0).data.cpu().numpy(),
                    mesh_path,
                )
        else:
            vertices, faces, vertex_colors = mesh_out
            
            for i in range(len(vertex_colors)):
                mesh_path=os.path.join(opt.workspace, name, name + str(i) + '_'+ str(seed)+ '.obj')
                save_obj(vertices, faces, vertex_colors[i], mesh_path)
                       


def process_text(opt: Options):

    assert opt.mvdream_or_zero123
    
    if opt.seed == None:
        seed = random.randint(0, 99999)
    else:
        seed = opt.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"seed:{seed}")

    prompt= opt.text_prompt
    name=prompt.replace(" ", "_")
    #name=prompt.split(' ')[-1]
    mv_image = pipe(prompt, negative_prompt='', num_inference_steps=30, guidance_scale=5.0, elevation=0)
    for i in range(mv_image.shape[0]):
        carved_image = rembg.remove((mv_image[i]*255).astype(np.uint8), session=bg_remover) # [H, W, 4]
        mask = carved_image[..., -1] > 0
        # recenter
        carved_image = recenter(carved_image, mask, border_ratio=0.2).astype(np.float32) / 255.0
        carved_image = carved_image[..., :3] * carved_image[..., 3:4] + (1 - carved_image[..., 3:4])
        mv_image[i] =  carved_image   
   
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32

    os.makedirs(f'{opt.workspace}/{name}', exist_ok=True)
    predict_images=np.concatenate(mv_image,axis=1)
    kiui.write_image(f'{opt.workspace}/{name}/{name}_mv.jpg', predict_images)
    
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) 
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)

    data = {}
    
    images_input_vit = F.interpolate(input_image, size=(224, 224), mode='bilinear', align_corners=False)

    input_image = input_image.unsqueeze(0) 
    images_input_vit=images_input_vit.unsqueeze(0)
    data['input_vit']=images_input_vit
    
    elevation = 0
    cam_poses =[]
    
    
    azimuth = np.arange(0, 360, 90, dtype=np.int32)
    for azi in tqdm.tqdm(azimuth):
        cam_pose = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
        cam_poses.append(cam_pose)

            
    cam_poses = torch.cat(cam_poses,0)
    radius = torch.norm(cam_poses[0, :3, 3])
    cam_poses[:, :3, 3] *= opt.cam_radius / radius
    transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32).to(device) @ torch.inverse(cam_poses[0])
    cam_poses = transform.unsqueeze(0) @ cam_poses 
    
    cam_poses=cam_poses.unsqueeze(0)
    data['source_camera']=cam_poses

    
    with torch.no_grad():
        if opt.volume_mode == 'TRF_Mesh':
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                svd_volume = model.forward_svd_volume(input_image,data)
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                svd_volume = model.forward_svd_volume(input_image,data)
         
        export_texmap=False
        
        mesh_out = model.extract_mesh(svd_volume,use_texture_map=export_texmap)
        

        mesh_path=os.path.join(opt.workspace, name, name + '.obj')
        
        if export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            
            for i in range(len(tex_map)):
                mesh_path=os.path.join(opt.workspace, name, name + str(i) + '_'+ str(seed)+ '.obj')
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map[i].permute(1, 2, 0).data.cpu().numpy(),
                    mesh_path,
                )
        else:
            vertices, faces, vertex_colors = mesh_out
            
            for i in range(len(vertex_colors)):
                mesh_path=os.path.join(opt.workspace, name, name + str(i) + '_'+ str(seed)+ '.obj')
                save_obj(vertices, faces, vertex_colors[i], mesh_path)






if opt.txt_or_image:
    # text condition
    process_text(opt)

else:
    assert opt.test_path is not None
    # image condition
    if os.path.isdir(opt.test_path):
        file_paths = glob.glob(os.path.join(opt.test_path, "*"))
    else:
        file_paths = [opt.test_path]
        
    for path in file_paths:
        process_image(opt, path)