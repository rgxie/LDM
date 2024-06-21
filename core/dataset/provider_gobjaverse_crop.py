import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision.transforms import v2
import tarfile

import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class GobjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.total_epoch = 30
        self.cur_epoch = 0
        self.cur_itrs = 0
        
        self.original_scale = 0.1
        self.bata_line_scale = self.original_scale * 0.5
        self.beta_line_ites = 3000
        
        self.opt = opt
        self.training = training

        if opt.over_fit:
            data_list_path=opt.data_debug_list
        else:
            data_list_path=opt.data_list_path

        self.items = []
        with open(data_list_path, 'r') as f:
            data = json.load(f)
            for item in data:
                self.items.append(item)

        # naive split
        if not opt.over_fit:
            if self.training:
                self.items = self.items[:-self.opt.batch_size]
            else:
                self.items = self.items[-self.opt.batch_size:]
        else:
            self.opt.batch_size=len(self.items)
            self.opt.num_workers=0
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        return len(self.items)
    
    def get_random_crop(self, batch_masks, minsize):
        n, h, w = batch_masks.shape
        
        crop_topleft_points = torch.full((n, 4), -1, dtype=torch.int)

        for i, mask in enumerate(batch_masks):
            
            nonzero_coords = torch.nonzero(mask, as_tuple=False)
            if nonzero_coords.size(0) == 0:
                crop_topleft_points[i] = torch.tensor([0, 0, minsize, minsize])
                continue  
           
            min_coords = torch.min(nonzero_coords, dim=0)[0]
            max_coords = torch.max(nonzero_coords, dim=0)[0]
            y_min, x_min = min_coords
            y_max, x_max = max_coords

            y_center = (y_min + y_max) // 2
            x_center = (x_min + x_max) // 2

            y_min = max(0, y_center - (minsize // 2))
            y_max = min(h - 1, y_center + (minsize // 2))
            x_min = max(0, x_center - (minsize // 2))
            x_max = min(w - 1, x_center + (minsize // 2))

            if (y_max - y_min + 1) < minsize:
                y_min = max(0, y_max - minsize + 1)
                y_max = y_min + minsize - 1
            if (x_max - x_min + 1) < minsize:
                x_min = max(0, x_max - minsize + 1)
                x_max = x_min + minsize - 1

            top_y = torch.randint(y_min, y_max - minsize + 2, (1,)).item()  
            top_x = torch.randint(x_min, x_max - minsize + 2, (1,)).item()
            
            crop_topleft_points[i] = torch.tensor([top_x, top_y, minsize, minsize])

        return crop_topleft_points

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}

        images = []
        albedos = []
        normals = []
        depths = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        if self.training:
            if self.opt.is_fix_views:
                if self.opt.mvdream_or_zero123:
                    vids = [0,30,12,36,27,6,33,18][:self.opt.num_input_views] + np.random.permutation(24).tolist()
                else:
                    vids = [0,29,8,33,16,37,2,10,18,28][:self.opt.num_input_views] + np.random.permutation(24).tolist()
            else:
                vids = np.random.permutation(np.arange(0, 36))[:self.opt.num_input_views].tolist() + np.random.permutation(36).tolist()
                
        else:
            if self.opt.mvdream_or_zero123:
                vids = [0,30,12,36,27,6,33,18]
            else:
                vids = [0,29,8,33,16,37,2,10,18,28]
        
        for vid in vids:

            uid_last = uid.split('/')[1]
            
            if self.opt.rar_data:
                tar_path = os.path.join(self.opt.data_path, f"{uid}.tar")
                image_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}.png")
                meta_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}.json")
                albedo_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_albedo.png") # black bg...
                nd_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_nd.exr")
                
                with tarfile.open(tar_path, 'r') as tar:
                    with tar.extractfile(image_path) as f:
                        image = np.frombuffer(f.read(), np.uint8)
                    with tar.extractfile(albedo_path) as f:
                        albedo = np.frombuffer(f.read(), np.uint8)
                    with tar.extractfile(meta_path) as f:
                        meta = json.loads(f.read().decode())
                    with tar.extractfile(nd_path) as f:
                        nd = np.frombuffer(f.read(), np.uint8)

                image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) 
                albedo = torch.from_numpy(cv2.imdecode(albedo, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) 
            else:
                image_path = os.path.join(self.opt.data_path,uid, f"{vid:05d}/{vid:05d}.png")
                meta_path = os.path.join(self.opt.data_path,uid, f"{vid:05d}/{vid:05d}.json")
                nd_path = os.path.join(self.opt.data_path,uid, f"{vid:05d}/{vid:05d}_nd.exr")
                
                albedo_path = os.path.join(self.opt.data_path,uid, f"{vid:05d}/{vid:05d}_albedo.png")
                
                with open(image_path, 'rb') as f:
                    image = np.frombuffer(f.read(), dtype=np.uint8)
                    
                with open(albedo_path, 'rb') as f:
                    albedo = np.frombuffer(f.read(), dtype=np.uint8)
                
                with open(meta_path, 'r') as f:
                    meta = json.load(f)

                with open(nd_path, 'rb') as f:
                    nd = np.frombuffer(f.read(), np.uint8)
                
                image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) 
                albedo = torch.from_numpy(cv2.imdecode(albedo, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255)

            c2w = np.eye(4)
            c2w[:3, 0] = np.array(meta['x'])
            c2w[:3, 1] = np.array(meta['y'])
            c2w[:3, 2] = np.array(meta['z'])
            c2w[:3, 3] = np.array(meta['origin'])
            c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
            
            nd = cv2.imdecode(nd, cv2.IMREAD_UNCHANGED).astype(np.float32) 
            normal = nd[..., :3] 
            depth = nd[..., 3] 

            # rectify normal directions
            normal = normal[..., ::-1]
            normal[..., 0] *= -1
            normal = torch.from_numpy(normal.astype(np.float32)).nan_to_num_(0) # there are nans in gt normal... 
            depth = torch.from_numpy(depth.astype(np.float32)).nan_to_num_(0)
                
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            image = image.permute(2, 0, 1) 
            mask = image[3:4] 
            image = image[:3] * mask + (1 - mask) 
            image = image[[2,1,0]].contiguous() 
            
            # albdeo
            albedo = albedo.permute(2, 0, 1) 
            albedo = albedo[:3] * mask + (1 - mask) 
            albedo = albedo[[2,1,0]].contiguous() 
            

            normal = normal.permute(2, 0, 1) 
            normal = normal * mask 

            images.append(image)
            albedos.append(albedo)
            normals.append(normal)
            depths.append(depth)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            normals = normals + [normals[-1]] * n
            depths = depths + [depths[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) 
        albedos = torch.stack(albedos, dim=0) 
        normals = torch.stack(normals, dim=0) 
        depths = torch.stack(depths, dim=0) 
        masks = torch.stack(masks, dim=0) 
        cam_poses = torch.stack(cam_poses, dim=0) 

        # normalized camera feats as in LGM (transform the first pose to a fixed position)
        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.opt.cam_radius / radius
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        images = F.interpolate(images, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        albedos = F.interpolate(albedos, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False)
        
        
        if self.opt.is_crop and self.training:
            increase_size= np.maximum((self.cur_epoch/self.total_epoch-self.original_scale),0)/(1-self.original_scale) * (self.opt.input_size-self.opt.output_size) 
            increase_size= np.maximum(self.opt.output_size*0.5,increase_size)
            max_scale_input_size = int(self.opt.output_size + increase_size)
        else:
            max_scale_input_size=self.opt.output_size
        
        if max_scale_input_size > self.opt.output_size:
            scaled_input_size = np.random.randint(self.opt.output_size, max_scale_input_size+1)
        else:
            scaled_input_size = self.opt.output_size
            
        target_images = v2.functional.resize(
            images, scaled_input_size, interpolation=3, antialias=True).clamp(0, 1)
        
        target_albedos = v2.functional.resize(
            albedos, scaled_input_size, interpolation=3, antialias=True).clamp(0, 1)

        target_alphas = v2.functional.resize(
            masks.unsqueeze(1), scaled_input_size, interpolation=0, antialias=True)


        crop_params = self.get_random_crop(target_alphas[:,0], self.opt.output_size )
        
        target_images = torch.stack([v2.functional.crop(target_images[i], *crop_params[i]) for i in range(target_images.shape[0])],0)
        target_albedos = torch.stack([v2.functional.crop(target_albedos[i], *crop_params[i]) for i in range(target_albedos.shape[0])],0)
        target_alphas = torch.stack([v2.functional.crop(target_alphas[i], *crop_params[i]) for i in range(target_alphas.shape[0])],0)
        
        results['images_output']=target_images
        results['albedos_output']=target_albedos
        results['masks_output']=target_alphas

    
        # data augmentation condition input image
        images_input = images[:self.opt.num_input_views].clone()
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        #input view images, unused by tranformer based model
        results['input']=images_input 


        images_input_vit = F.interpolate(images_input, size=(224, 224), mode='bilinear', align_corners=False)
        results['input_vit']=images_input_vit

        all_rays_o=[]
        all_rays_d=[]
        for i in range(vid_cnt):
            rays_o, rays_d = get_rays(cam_poses[i], scaled_input_size, scaled_input_size, self.opt.fovy) # [h, w, 3]
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)
        all_rays_o=torch.stack(all_rays_o, dim=0)
        all_rays_d=torch.stack(all_rays_d, dim=0)
        
        if crop_params is not None:
            all_rays_o_crop=[]
            all_rays_d_crop=[]
            for k in range(all_rays_o.shape[0]):
                i, j, h, w = crop_params[k]
                all_rays_o_crop.append(all_rays_o[k][i:i+h, j:j+w, :])
                all_rays_d_crop.append(all_rays_d[k][i:i+h, j:j+w, :])
            
            all_rays_o=torch.stack(all_rays_o_crop, dim=0)
            all_rays_d=torch.stack(all_rays_d_crop, dim=0)
        
        results['all_rays_o']=all_rays_o
        results['all_rays_d']=all_rays_d

        # c2w
        cam_poses[:, :3, 1:3] *= -1 
        

        cam_view = torch.inverse(cam_poses).transpose(1, 2)
        cam_view_proj = cam_view @ self.proj_matrix 
        cam_pos = - cam_poses[:, :3, 3] 
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        results['source_camera']=cam_poses_input

        return results