import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mcubes
import kiui
from kiui.lpips import LPIPS
from core.svd_reconstructor import LDM_SVD_Net
from core.options import Options
from core.tensorsdf.tensorSDF import TensorVMSplit_Mesh,TensorVMSplit_SDF
from torchvision.transforms import v2
from core.geometry.camera.perspective_camera import PerspectiveCamera
from core.geometry.render.neural_render import NeuralRender
from core.geometry.rep_3d.flexicubes_geometry import FlexiCubesGeometry
import nvdiffrast.torch as dr
from core.utils import xatlas_uvmap
from core.utils import normalize_depth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Regulrarization loss for FlexiCubes
def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               F.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


#tensorSDF + transformer + SDF + Mesh
class LDM_Mesh(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt
        
        # attributes
        self.grid_res = 128 
        self.grid_scale = 2.0 
        self.deformation_multiplier = 4.0
        
        
        self.init_flexicubes_geometry(device, self.opt)

        self.vsd_net = LDM_SVD_Net(
            camera_embed_dim=opt.camera_embed_dim,
            transformer_dim=opt.transformer_dim,
            transformer_layers=opt.transformer_layers,
            transformer_heads=opt.transformer_heads,
            triplane_low_res=opt.triplane_low_res,
            triplane_high_res=opt.triplane_high_res,
            encoder_freeze=opt.encoder_freeze,
            encoder_type=opt.encoder_type,
            encoder_model_name=opt.encoder_model_name,
            encoder_feat_dim=opt.encoder_feat_dim,
            app_n_comp=opt.app_n_comp,
            density_n_comp=opt.density_n_comp,
        )
           
        aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]]).to(device)
        grid_size = torch.tensor([opt.splat_size, opt.splat_size, opt.splat_size]).to(device)
        near_far =torch.tensor([opt.znear, opt.zfar]).to(device)

        self.tensorRF = TensorVMSplit_Mesh(aabb, grid_size, density_n_comp=opt.density_n_comp,appearance_n_comp=opt.app_n_comp,\
            near_far=near_far, shadingMode=opt.shadingMode, pos_pe=opt.pos_pe, view_pe=opt.view_pe, fea_pe=opt.fea_pe)

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
            
        # load ckpt
        if opt.ckpt_nerf is not None:
            sd = torch.load(opt.ckpt_nerf, map_location='cpu')['model']
            sd_fc = {}
            for k, v in sd.items():
                k=k.replace('module.', '')
                if k.startswith('vsd.renderModule.'):
                    continue
                else:
                    sd_fc[k] = v
            sd_fc = {k.replace('vsd_net.', ''): v for k, v in sd_fc.items()}
            sd_fc = {k.replace('tensorRF.', ''): v for k, v in sd_fc.items()}
            self.vsd_net.load_state_dict(sd_fc, strict=False)
            self.tensorRF.load_state_dict(sd_fc, strict=False)
            print(f'Loaded weights from {opt.ckpt_nerf}')


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

        
    # predict svd_volume
    def forward_svd_volume(self, images, data):
        B, V, C, H, W = images.shape
        
        source_camera=data['source_camera']
        images_vit=data['input_vit'] 
        source_camera=source_camera.reshape(B,V,-1) 
        app_planes,app_lines,density_planes,density_lines = self.vsd_net(images_vit,source_camera) 

        
        app_planes=app_planes.view(B,3,self.opt.app_n_comp,self.opt.splat_size,self.opt.splat_size)
        app_lines=app_lines.view(B,3,self.opt.app_n_comp,self.opt.splat_size,1)
        density_planes=density_planes.view(B,3,self.opt.density_n_comp,self.opt.splat_size,self.opt.splat_size)
        density_lines=density_lines.view(B,3,self.opt.density_n_comp,self.opt.splat_size,1)

        results = {
            'app_planes': app_planes,
            'app_lines': app_lines,
            'density_planes':density_planes,
            'density_lines':density_lines
        }

        return results

    
    def init_flexicubes_geometry(self, device, opt):
        camera = PerspectiveCamera(opt, device=device)
        renderer = NeuralRender(device, camera_model=camera)
        self.geometry = FlexiCubesGeometry(
            grid_res=self.grid_res, 
            scale=self.grid_scale, 
            renderer=renderer, 
            render_type='neural_render',
            device=device,
        )


    def get_sdf_deformation_prediction(self, planes):

        B = planes['app_lines'].shape[0]
        init_position = self.geometry.verts.unsqueeze(0).expand(B, -1, -1)
        
        sdf, deformation, weight = self.tensorRF.get_geometry_prediction(planes,init_position,self.geometry.indices)
        

        deformation = 1.0 / (self.grid_res * self.deformation_multiplier) * torch.tanh(deformation)
        sdf_reg_loss = torch.zeros(sdf.shape[0], device=sdf.device, dtype=torch.float32)

        sdf_bxnxnxn = sdf.reshape((sdf.shape[0], self.grid_res + 1, self.grid_res + 1, self.grid_res + 1))
        sdf_less_boundary = sdf_bxnxnxn[:, 1:-1, 1:-1, 1:-1].reshape(sdf.shape[0], -1)
        pos_shape = torch.sum((sdf_less_boundary > 0).int(), dim=-1)
        neg_shape = torch.sum((sdf_less_boundary < 0).int(), dim=-1)
        zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        if torch.sum(zero_surface).item() > 0:
            update_sdf = torch.zeros_like(sdf[0:1])
            max_sdf = sdf.max()
            min_sdf = sdf.min()
            update_sdf[:, self.geometry.center_indices] += (1.0 - min_sdf)  # greater than zero
            update_sdf[:, self.geometry.boundary_indices] += (-1 - max_sdf)  # smaller than zero
            new_sdf = torch.zeros_like(sdf)
            for i_batch in range(zero_surface.shape[0]):
                if zero_surface[i_batch]:
                    new_sdf[i_batch:i_batch + 1] += update_sdf
            update_mask = (new_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            sdf_reg_loss = torch.abs(sdf).mean(dim=-1).mean(dim=-1)
            sdf_reg_loss = sdf_reg_loss * zero_surface.float()
            sdf = sdf * update_mask + new_sdf * (1 - update_mask)

        final_sdf = []
        final_def = []
        for i_batch in range(zero_surface.shape[0]):
            if zero_surface[i_batch]:
                final_sdf.append(sdf[i_batch: i_batch + 1].detach())
                final_def.append(deformation[i_batch: i_batch + 1].detach())
            else:
                final_sdf.append(sdf[i_batch: i_batch + 1])
                final_def.append(deformation[i_batch: i_batch + 1])
        sdf = torch.cat(final_sdf, dim=0)
        deformation = torch.cat(final_def, dim=0)
        return sdf, deformation, sdf_reg_loss, weight
    
    def get_geometry_prediction(self, planes=None):

        sdf, deformation, sdf_reg_loss, weight = self.get_sdf_deformation_prediction(planes)

        
        v_deformed = self.geometry.verts.unsqueeze(dim=0).expand(sdf.shape[0], -1, -1) + deformation
        tets = self.geometry.indices
        n_batch = planes['app_planes'].shape[0]
        v_list = []
        f_list = []
        flexicubes_surface_reg_list = []
        
    
        for i_batch in range(n_batch):
            verts, faces, flexicubes_surface_reg = self.geometry.get_mesh(
                v_deformed[i_batch], 
                sdf[i_batch].squeeze(dim=-1),
                with_uv=False, 
                indices=tets, 
                weight_n=weight[i_batch].squeeze(dim=-1),
                is_training=self.training,
            )
            flexicubes_surface_reg_list.append(flexicubes_surface_reg)
            v_list.append(verts)
            f_list.append(faces)
        
        flexicubes_surface_reg = torch.cat(flexicubes_surface_reg_list).mean()
        flexicubes_weight_reg = (weight ** 2).mean()
        
        return v_list, f_list, sdf, deformation, v_deformed, (sdf_reg_loss, flexicubes_surface_reg, flexicubes_weight_reg)
    
    def get_texture_prediction(self, planes, tex_pos, hard_mask=None):

        B = planes['app_planes'].shape[0]
        tex_pos = torch.cat(tex_pos, dim=0)
        if not hard_mask is None:
            tex_pos = tex_pos * hard_mask.float()
        batch_size = tex_pos.shape[0]
        tex_pos = tex_pos.reshape(batch_size, -1, 3)

        if hard_mask is not None:
            n_point_list = torch.sum(hard_mask.long().reshape(hard_mask.shape[0], -1), dim=-1)
            sample_tex_pose_list = []
            max_point = n_point_list.max()
            if max_point==0:  
                max_point=max_point+1
            expanded_hard_mask = hard_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3) > 0.5
            for i in range(tex_pos.shape[0]):
                tex_pos_one_shape = tex_pos[i][expanded_hard_mask[i]].reshape(1, -1, 3)
                if tex_pos_one_shape.shape[1] < max_point:
                    tex_pos_one_shape = torch.cat(
                        [tex_pos_one_shape, torch.zeros(
                            1, max_point - tex_pos_one_shape.shape[1], 3,
                            device=tex_pos_one_shape.device, dtype=torch.float32)], dim=1)
                sample_tex_pose_list.append(tex_pos_one_shape)
            tex_pos = torch.cat(sample_tex_pose_list, dim=0)

    
        tex_feat = self.tensorRF.get_texture_prediction(tex_pos,svd_volume=planes)

        if hard_mask is not None:
            final_tex_feat = torch.zeros(
                B, hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1], device=tex_feat.device)
            expanded_hard_mask = hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5
            for i in range(B):
                final_tex_feat[i][expanded_hard_mask[i]] = tex_feat[i][:n_point_list[i]].reshape(-1)
            tex_feat = final_tex_feat

        return tex_feat.reshape(B, hard_mask.shape[1], hard_mask.shape[2], tex_feat.shape[-1])
    
    def render_mesh(self, mesh_v, mesh_f, cam_mv, render_size=256):

        return_value_list = []
        for i_mesh in range(len(mesh_v)):
            return_value = self.geometry.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=render_size,
                hierarchical_mask=False
            )
            return_value_list.append(return_value)

        return_keys = return_value_list[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in return_value_list]
            return_value[k] = value

        mask = torch.cat(return_value['mask'], dim=0)
        hard_mask = torch.cat(return_value['hard_mask'], dim=0)
        tex_pos = return_value['tex_pos']
        depth = torch.cat(return_value['depth'], dim=0)
        normal = torch.cat(return_value['normal'], dim=0)
        return mask, hard_mask, tex_pos, depth, normal
    
    def forward_geometry(self, planes, render_cameras, render_size=256):

        B, NV = render_cameras.shape[:2]

        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(planes)

        
        cam_mv = render_cameras
        run_n_view = cam_mv.shape[1]
        antilias_mask, hard_mask, tex_pos, depth, normal = self.render_mesh(mesh_v, mesh_f, cam_mv, render_size=render_size)

        tex_hard_mask = hard_mask
        tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]
        tex_hard_mask = torch.cat(
            [torch.cat(
                [tex_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
                 for i_view in range(run_n_view)], dim=2)
                for i in range(B)], dim=0)

        tex_feat = self.get_texture_prediction(planes, tex_pos, tex_hard_mask)
        background_feature = torch.ones_like(tex_feat)      

        img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

        
        img_feat = torch.cat(
            [torch.cat(
                [img_feat[i:i + 1, :, render_size * i_view: render_size * (i_view + 1)]
                 for i_view in range(run_n_view)], dim=0) for i in range(len(tex_pos))], dim=0)

        img = img_feat.clamp(0, 1).permute(0, 3, 1, 2).unflatten(0, (B, NV))
        
        albedo=img[:,:,3:6,:,:]
        img=img[:,:,0:3,:,:]
        
        antilias_mask = antilias_mask.permute(0, 3, 1, 2).unflatten(0, (B, NV))
        depth = -depth.permute(0, 3, 1, 2).unflatten(0, (B, NV))        # transform negative depth to positive
        depth = normalize_depth(depth)
        normal = normal.permute(0, 3, 1, 2).unflatten(0, (B, NV))

        out = {
            'image': img,
            'albedo': albedo,
            'mask': antilias_mask,
            'depth': depth,
            'normal': normal,
            'sdf': sdf,
            'mesh_v': mesh_v,
            'mesh_f': mesh_f,
            'sdf_reg_loss': sdf_reg_loss,
        }
        return out
    
    def forward(self, data, step_ratio=1):
        
        results = {}
        loss = 0

        images = data['input'] 
        
        svd_volume = self.forward_svd_volume(images,data) 
        results['svd_volume'] = svd_volume
        results = self.forward_geometry(svd_volume, data['w2c'], self.opt.output_size)


        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32).to(device)
                
        
        pred_shading = results['image'] 
        pred_alphas = results['mask'] 
        pred_albedos = results['albedo'] 
        pred_depths = results['depth']
        
        pred_images=pred_shading*pred_albedos

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['pred_albedos'] = pred_albedos
        results['pred_shading'] = pred_shading
        

        gt_images = data['images_output'] 
        gt_albedos = data['albedos_output'] 
        gt_masks = data['masks_output'] 
        gt_depths = data['depth_output']
        
        
        sdf_reg_loss = results['sdf_reg_loss']
        sdf = results['sdf']
        sdf_reg_loss_entropy = sdf_reg_loss_batch(sdf, self.geometry.all_edges).mean() * 0.01
        _, flexicubes_surface_reg, flexicubes_weights_reg = sdf_reg_loss
        flexicubes_surface_reg = flexicubes_surface_reg.mean() * 0.5
        flexicubes_weights_reg = flexicubes_weights_reg.mean() * 0.1

        loss_reg = sdf_reg_loss_entropy + flexicubes_surface_reg + flexicubes_weights_reg

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
        gt_albedos = gt_albedos * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
        
        depth_loss = F.mse_loss(pred_depths,gt_depths)
        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks) + F.mse_loss(pred_albedos, gt_albedos) 
        
        loss = loss + loss_mse + loss_reg + 0.5*depth_loss 

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results
    
    
    def render_frame(self, data):

        results = {}

        images = data['input_vit'] 
        
        # use the first view to predict gaussians
        svd_volume = self.forward_svd_volume(images,data) 
        
        results['svd_volume'] = svd_volume
        
        # return the rendered images
        results = self.forward_geometry(svd_volume, data['w2c'], self.opt.infer_render_size)


        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32).to(device)
                
        
        pred_shading = results['image'] 
        pred_alphas = results['mask'] 
        pred_albedos = results['albedo'] 
        
        pred_images=pred_shading*pred_albedos

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['pred_albedos'] = pred_albedos
        results['pred_shading'] = pred_shading

        return results

    def extract_mesh(
        self, 
        planes: torch.Tensor, 
        use_texture_map: bool = False,
        texture_resolution: int = 1024,
        **kwargs,
    ):

        assert planes['app_planes'].shape[0] == 1
        device = planes['app_planes'].device
        

        # predict geometry first
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(planes)
        vertices, faces = mesh_v[0], mesh_f[0]

        if not use_texture_map:
            vertices_tensor = vertices.unsqueeze(0)
            rgb_colors = self.tensorRF.predict_color(planes, vertices_tensor)['rgb'].clamp(0, 1).squeeze(0).cpu().numpy()
            rgb_colors = (rgb_colors * 255).astype(np.uint8)
            
            albedob_colors = self.tensorRF.predict_color(planes, vertices_tensor)['albedo'].clamp(0, 1).squeeze(0).cpu().numpy()
            albedob_colors = (albedob_colors * 255).astype(np.uint8)
            
            shading_colors = self.tensorRF.predict_color(planes, vertices_tensor)['shading'].clamp(0, 1).squeeze(0).cpu().numpy()
            shading_colors = (shading_colors * 255).astype(np.uint8)
            

            return vertices.cpu().numpy(), faces.cpu().numpy(), [rgb_colors,albedob_colors,shading_colors]

        # use x-atlas to get uv mapping for the mesh
        ctx = dr.RasterizeCudaContext(device=device)
        uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
            self.geometry.renderer.ctx, vertices, faces, resolution=texture_resolution)
        
        tex_hard_mask = tex_hard_mask.float().cpu()

        query_vertices=gb_pos.view(1,texture_resolution*texture_resolution,3)
        
        vertices_colors = self.tensorRF.predict_color(
                planes, query_vertices)['rgb'].squeeze(0).cpu()
        
        vertices_colors=vertices_colors.reshape(1,texture_resolution,texture_resolution,3)
        
        background_feature = torch.zeros_like(vertices_colors)
        img_feat = torch.lerp(background_feature, vertices_colors, tex_hard_mask)
        texture_map = img_feat.permute(0, 3, 1, 2).squeeze(0)

        return vertices, faces, uvs, mesh_tex_idx, [texture_map]


#tensoSDF + transformer + SDF + volume_rendering
class LDM_SDF(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt
        #predict svd using transformer
        self.vsd_net = LDM_SVD_Net(
            camera_embed_dim=opt.camera_embed_dim,
            transformer_dim=opt.transformer_dim,
            transformer_layers=opt.transformer_layers,
            transformer_heads=opt.transformer_heads,
            triplane_low_res=opt.triplane_low_res,
            triplane_high_res=opt.triplane_high_res,
            encoder_freeze=opt.encoder_freeze,
            encoder_type=opt.encoder_type,
            encoder_model_name=opt.encoder_model_name,
            encoder_feat_dim=opt.encoder_feat_dim,
            app_n_comp=opt.app_n_comp,
            density_n_comp=opt.density_n_comp,
        )
           
        aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]]).to(device)
        grid_size = torch.tensor([opt.splat_size, opt.splat_size, opt.splat_size]).to(device)
        near_far =torch.tensor([opt.znear, opt.zfar]).to(device)
        
        self.tensorRF = TensorVMSplit_SDF(aabb, grid_size, density_n_comp=opt.density_n_comp,appearance_n_comp=opt.app_n_comp,\
            near_far=near_far, shadingMode=opt.shadingMode, pos_pe=opt.pos_pe, view_pe=opt.view_pe, fea_pe=opt.fea_pe)

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict
    
    def set_beta(self,t):
        self.tensorRF.lap_density.set_beta(t)
        

        
    # predict svd_volume
    def forward_svd_volume(self, images, data):

        B, V, C, H, W = images.shape
        source_camera=data['source_camera']
        # for transformer
        images_vit=data['input_vit'] 
        source_camera=source_camera.reshape(B,V,-1) 
        app_planes,app_lines,density_planes,density_lines = self.vsd_net(images_vit,source_camera) 

        app_planes=app_planes.view(B,3,self.opt.app_n_comp,self.opt.splat_size,self.opt.splat_size)
        app_lines=app_lines.view(B,3,self.opt.app_n_comp,self.opt.splat_size,1)
        density_planes=density_planes.view(B,3,self.opt.density_n_comp,self.opt.splat_size,self.opt.splat_size)
        density_lines=density_lines.view(B,3,self.opt.density_n_comp,self.opt.splat_size,1)

        results = {
            'app_planes': app_planes,
            'app_lines': app_lines,
            'density_planes':density_planes,
            'density_lines':density_lines
        }

        return results
    
    def extract_mesh(self, 
        planes: torch.Tensor, 
        mesh_resolution: int = 256, 
        mesh_threshold: int = 0.005, 
        use_texture_map: bool = False, 
        texture_resolution: int = 1024,):
        
        device = planes['app_planes'].device
        
        grid_size = mesh_resolution
        points = torch.linspace(-1, 1, steps=grid_size).half()

        x, y, z = torch.meshgrid(points, points, points)

        xyz_samples = torch.stack((x, y, z), dim=0).unsqueeze(0).to(device)
        xyz_samples=xyz_samples.permute(0,2,3,4,1)
        xyz_samples=xyz_samples.view(1,-1,1,3)
        

        grid_out = self.tensorRF.predict_sdf(planes,xyz_samples)
        grid_out['sigma']=grid_out['sigma'].view(grid_size,grid_size,grid_size).float()
        
        vertices, faces = mcubes.marching_cubes(
            grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(), 
            mesh_threshold,
        )
        vertices = vertices / (mesh_resolution - 1) * 2 - 1

        if not use_texture_map:
            # query vertex colors
            vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device).unsqueeze(0)
            rgb_colors = self.tensorRF.predict_color(
                planes, vertices_tensor)['rgb'].squeeze(0).cpu().numpy()
            rgb_colors = (rgb_colors * 255).astype(np.uint8)
            
            albedob_colors = self.tensorRF.predict_color(
                planes, vertices_tensor)['albedo'].squeeze(0).cpu().numpy()
            albedob_colors = (albedob_colors * 255).astype(np.uint8)
            
            shading_colors = self.tensorRF.predict_color(
                planes, vertices_tensor)['shading'].squeeze(0).cpu().numpy()
            shading_colors = (shading_colors * 255).astype(np.uint8)

            return vertices, faces, [rgb_colors,albedob_colors,shading_colors]
        
        # use x-atlas to get uv mapping for the mesh
        vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(faces.astype(int), dtype=torch.long, device=device)

        ctx = dr.RasterizeCudaContext(device=device)
        uvs, mesh_tex_idx, gb_pos, tex_hard_mask = xatlas_uvmap(
            ctx, vertices, faces, resolution=texture_resolution)
        tex_hard_mask = tex_hard_mask.float().cpu()

        query_vertices=gb_pos.view(1,texture_resolution*texture_resolution,3)
        
        vertices_colors = self.tensorRF.predict_color(
                planes, query_vertices)['rgb'].squeeze(0).cpu()
        
        vertices_colors=vertices_colors.reshape(1,texture_resolution,texture_resolution,3)
        
        background_feature = torch.zeros_like(vertices_colors)
        img_feat = torch.lerp(background_feature, vertices_colors, tex_hard_mask.half())
        texture_map = img_feat.permute(0, 3, 1, 2).squeeze(0)
        #albedo
        vertices_colors_albedo = self.tensorRF.predict_color(
                planes, query_vertices)['albedo'].squeeze(0).cpu()
        
        vertices_colors_albedo=vertices_colors_albedo.reshape(1,texture_resolution,texture_resolution,3)
        
        background_feature = torch.zeros_like(vertices_colors_albedo)
        img_feat = torch.lerp(background_feature, vertices_colors_albedo, tex_hard_mask.half())
        texture_map_albedo = img_feat.permute(0, 3, 1, 2).squeeze(0)

        return vertices, faces, uvs, mesh_tex_idx, [texture_map,texture_map_albedo]

    
    def forward(self, data, step_ratio=1):
        results = {}
        loss = 0

        images = data['input'] 
        
        svd_volume = self.forward_svd_volume(images,data) 

        results['svd_volume'] = svd_volume

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32).to(device)
        
        results = self.tensorRF(svd_volume, data['all_rays_o'], data['all_rays_d'],is_train=True, bg_color=bg_color, N_samples=self.opt.n_sample)
        pred_shading = results['image'] 
        pred_alphas = results['alpha'] 
        pred_albedos = results['albedo'] 
        
        pred_images = pred_shading*pred_albedos

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['pred_albedos'] = pred_albedos
        results['pred_shading'] = pred_shading
        

        gt_images = data['images_output'] 
        gt_albedos = data['albedos_output'] 
        gt_masks = data['masks_output'] 

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
        gt_albedos = gt_albedos * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks) + F.mse_loss(pred_albedos, gt_albedos)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results
    
    
    def render_frame(self, data):
        results = {}
        loss = 0

        images = data['input_vit'] 
        
        svd_volume = self.forward_svd_volume(images,data)

        results['svd_volume'] = svd_volume

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32).to(device)
        
        results = self.tensorRF(svd_volume, data['all_rays_o'], data['all_rays_d'],is_train=True, bg_color=bg_color, N_samples=self.opt.n_sample)
        pred_shading = results['image'] 
        pred_alphas = results['alpha'] 
        pred_albedos = results['albedo'] 
        
        pred_images = pred_shading*pred_albedos

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        results['pred_albedos'] = pred_albedos
        results['pred_shading'] = pred_shading
    

        return results