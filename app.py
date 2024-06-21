import os
import tyro
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import gradio as gr

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

import spaces

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
GRADIO_VIDEO_PATH = 'gradio_output.mp4'
GRADIO_OBJ_PATH = 'gradio_output_rgb.obj'
GRADIO_OBJ_ALBEDO_PATH = 'gradio_output_albedo.obj'
GRADIO_OBJ_SHADING_PATH = 'gradio_output_shading.obj'

#opt = tyro.cli(AllConfigs)

ckpt_path = hf_hub_download(repo_id="rgxie/LDM", filename="LDM6v01.ckpt")

opt = Options(
    input_size=512, 
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False, False),
    volume_mode='TRF_SDF',
    splat_size=64,
    output_size=62, #crop patch
    data_mode='s5',
    num_views=8,
    gradient_accumulation_steps=1,  #2
    mixed_precision='bf16',
    resume=ckpt_path,
)


# model
if opt.volume_mode == 'TRF_Mesh':
    model = LDM_Mesh(opt)
elif opt.volume_mode == 'TRF_SDF':
    model = LDM_SDF(opt)
else:
    model = LGM(opt)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32).to(device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# load dreams
pipe_text = MVDreamPipeline.from_pretrained(
    'ashawkey/mvdream-sd2.1-diffusers', # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe_text = pipe_text.to(device)

# mvdream
pipe_image = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers", # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe_image = pipe_image.to(device)


print('Loading 123plus model ...')
pipe_image_plus = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    #local_files_only=True,
)
pipe_image_plus.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe_image_plus.scheduler.config, timestep_spacing='trailing'
)

unet_path='./pretrained/diffusion_pytorch_model.bin' 

print('Loading custom white-background unet ...')
if os.path.exists(unet_path):
    unet_ckpt_path = unet_path
else:
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipe_image_plus.unet.load_state_dict(state_dict, strict=True)
pipe_image_plus = pipe_image_plus.to(device)

# load rembg
bg_remover = rembg.new_session()


@spaces.GPU
def generate_mv(condition_input_image, prompt, prompt_neg='', input_elevation=0, input_num_steps=30, input_seed=42, mv_moedl_option=None):
    # seed
    kiui.seed_everything(input_seed)

    os.makedirs(os.path.join(opt.workspace, "gradio"), exist_ok=True)
    
    # text-conditioned
    if condition_input_image is None:
        mv_image_uint8 = pipe_text(prompt, negative_prompt=prompt_neg, num_inference_steps=input_num_steps, guidance_scale=7.5, elevation=input_elevation)
        mv_image_uint8 = (mv_image_uint8 * 255).astype(np.uint8)
        # bg removal
        mv_image = []
        for i in range(4):
            image = rembg.remove(mv_image_uint8[i], session=bg_remover) # [H, W, 4]
            # to white bg
            image = image.astype(np.float32) / 255
            image = recenter(image, image[..., 0] > 0, border_ratio=0.2)
            image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
            mv_image.append(image)
            
        mv_image_grid = np.concatenate([mv_image[1], mv_image[2],mv_image[3], mv_image[0]],axis=1)
        input_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0)
        
        processed_image=None
    # image-conditioned
    else:
        condition_input_image = np.array(condition_input_image) # uint8
        # bg removal
        carved_image = rembg.remove(condition_input_image, session=bg_remover) # [H, W, 4]
        mask = carved_image[..., -1] > 0
        image = recenter(carved_image, mask, border_ratio=0.2)
        image = image.astype(np.float32) / 255.0
        processed_image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        
        if mv_moedl_option=='mvdream':
            mv_image = pipe_image(prompt, processed_image, negative_prompt=prompt_neg, num_inference_steps=input_num_steps, guidance_scale=5.0,  elevation=input_elevation)
        
            mv_image_grid = np.concatenate([mv_image[1], mv_image[2],mv_image[3], mv_image[0]],axis=1)
            input_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0)
        else:
            from PIL import Image
            from einops import rearrange, repeat
            
            # input_image=input_image* 255
            processed_image = Image.fromarray((processed_image * 255).astype(np.uint8))
            mv_image = pipe_image_plus(processed_image, num_inference_steps=input_num_steps).images[0]
            mv_image = np.asarray(mv_image, dtype=np.float32) / 255.0
            mv_image = torch.from_numpy(mv_image).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
            mv_image_grid = rearrange(mv_image, 'c (n h) (m w) -> (m h) (n w) c', n=3, m=2).numpy()
            mv_image = rearrange(mv_image, 'c (n h) (m w) -> (n m) h w c', n=3, m=2).numpy()
            input_image = mv_image
    return mv_image_grid, processed_image, input_image 

@spaces.GPU
def generate_3d(input_image, condition_input_image, mv_moedl_option=None, input_seed=42):
    kiui.seed_everything(input_seed)
    
    output_obj_rgb_path = os.path.join(opt.workspace,"gradio", GRADIO_OBJ_PATH)
    output_obj_albedo_path = os.path.join(opt.workspace,"gradio", GRADIO_OBJ_ALBEDO_PATH)
    output_obj_shading_path = os.path.join(opt.workspace,"gradio", GRADIO_OBJ_SHADING_PATH)
    
    output_video_path = os.path.join(opt.workspace,"gradio", GRADIO_VIDEO_PATH)
    # generate gaussians
     # [4, 256, 256, 3], float32
    input_image = torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)

    images_input_vit = F.interpolate(input_image, size=(224, 224), mode='bilinear', align_corners=False)
    
    data = {}
    input_image = input_image.unsqueeze(0) # [1, 4, 9, H, W]
    images_input_vit=images_input_vit.unsqueeze(0)
    data['input_vit']=images_input_vit
    
    elevation = 0
    cam_poses =[]
    if mv_moedl_option=='mvdream' or condition_input_image is None:
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
        
        #time-consuming
        export_texmap=False
        
        mesh_out = model.extract_mesh(svd_volume,use_texture_map=export_texmap)
        
        if export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            
            for i in range(len(tex_map)):
                mesh_path=os.path.join(opt.workspace, name + str(i) + '_'+ str(seed)+ '.obj')
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

            save_obj(vertices, faces, vertex_colors[0], output_obj_rgb_path)
            save_obj(vertices, faces, vertex_colors[1], output_obj_albedo_path)
            save_obj(vertices, faces, vertex_colors[2], output_obj_shading_path)
        
        # images=[]  
        # azimuth = np.arange(0, 360, 6, dtype=np.int32)
        # for azi in tqdm.tqdm(azimuth):

        #     cam_pose = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True))

        #     if opt.volume_mode == 'TRF_Mesh': 
        #         cam_view = torch.inverse(cam_pose)
        #         cam_view=cam_view.unsqueeze(0).unsqueeze(0).to(device)
        #         data['w2c'] = cam_view
        #         with torch.autocast(device_type='cuda', dtype=torch.float32):
        #             render_images=model.render_frame(data)
        #     else:
        #         rays_o, rays_d = get_rays(cam_pose, opt.infer_render_size, opt.infer_render_size, opt.fovy) # [h, w, 3]
        #         rays_o=rays_o.unsqueeze(0).unsqueeze(0).to(device)# B,V,H,W,3
        #         rays_d=rays_d.unsqueeze(0).unsqueeze(0).to(device)
        #         data['all_rays_o']=rays_o
        #         data['all_rays_d']=rays_d
        #         with torch.autocast(device_type='cuda', dtype=torch.float16):
        #             render_images=model.render_frame(data)
        #     image=render_images['images_pred']

        #     images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        # images = np.concatenate(images, axis=0)
        # imageio.mimwrite(output_video_path, images, fps=30)
        

    return output_obj_rgb_path, output_obj_albedo_path, output_obj_shading_path #, output_video_path


# gradio UI

_TITLE = '''LDM: Large Tensorial SDF Model for Textured Mesh Generation'''

_DESCRIPTION = '''


* Input can be text prompt, image. 
* The currently supported multi-view diffusion models include the image-conditioned MVdream and Zero123plus, as well as the text-conditioned Imagedream.
* If you find the output unsatisfying, try using different multi-view diffusion models or seeds!
'''

block = gr.Blocks(title=_TITLE).queue()
with block:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('# ' + _TITLE)
    gr.Markdown(_DESCRIPTION)
    
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            with gr.Tab("Image-to-3D"):
                # input image
                with gr.Row():
                    condition_input_image = gr.Image(
                        label="Input Image", 
                        image_mode="RGBA", 
                        type="pil"
                    )
                    
                    processed_image = gr.Image(
                        label="Processed Image", 
                        image_mode="RGBA", 
                        type="pil", 
                        interactive=False
                    )
                
                
                with gr.Row():
                        mv_moedl_option = gr.Radio([
                                "zero123plus",
                                "mvdream"
                            ], value="zero123plus",
                            label="Multi-view Diffusion")
                        
                with gr.Row(variant="panel"):
                    gr.Examples(
                        examples=[
                            os.path.join("example", img_name) for img_name in sorted(os.listdir("example"))
                        ],
                        inputs=[condition_input_image],
                        fn=lambda x: process(condition_input_image=x, prompt=''),
                        cache_examples=False,
                        examples_per_page=20,
                        label='Image-to-3D Examples'
                    )
                
            with gr.Tab("Text-to-3D"):  
                # input prompt
                with gr.Row():
                    input_text = gr.Textbox(label="prompt")
                # negative prompt
                with gr.Row():
                    input_neg_text = gr.Textbox(label="negative prompt", value='ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate')

                with gr.Row(variant="panel"):
                    gr.Examples(
                        examples=[
                            "a hamburger",
                            "a furry red fox head",
                            "a teddy bear",
                            "a motorbike",
                        ],
                        inputs=[input_text],
                        fn=lambda x: process(condition_input_image=None, prompt=x),
                        cache_examples=False,
                        label='Text-to-3D Examples'
                    )
            
            # elevation
            input_elevation = gr.Slider(label="elevation", minimum=-90, maximum=90, step=1, value=0)
            # inference steps
            input_num_steps = gr.Slider(label="inference steps", minimum=1, maximum=100, step=1, value=30)
            # random seed
            input_seed = gr.Slider(label="random seed", minimum=0, maximum=100000, step=1, value=0)
            # gen button
            button_gen = gr.Button("Generate")

        
        with gr.Column(scale=1):
            with gr.Row():
                # multi-view results
                mv_image_grid = gr.Image(interactive=False, show_label=False)
            # with gr.Row():    
            #     output_video_path = gr.Video(label="video")
            with gr.Row():    
                output_obj_rgb_path = gr.Model3D(
                    label="RGB Model (OBJ Format)",
                    interactive=False,
                )
            with gr.Row():    
                output_obj_albedo_path = gr.Model3D(
                    label="Albedo Model (OBJ Format)",
                    interactive=False,
                )
            with gr.Row():
                output_obj_shading_path = gr.Model3D(
                    label="Shading Model (OBJ Format)",
                    interactive=False,
                )

            
        input_image = gr.State()
        button_gen.click(fn=generate_mv, inputs=[condition_input_image, input_text, input_neg_text, input_elevation, input_num_steps, input_seed, mv_moedl_option], 
                         outputs=[mv_image_grid, processed_image, input_image],).success(
                            fn=generate_3d,
                            inputs=[input_image, condition_input_image, mv_moedl_option, input_seed], 
                            outputs=[output_obj_rgb_path, output_obj_albedo_path, output_obj_shading_path] , #output_video_path
                         )
        
        
block.launch(server_name="0.0.0.0", share=False)