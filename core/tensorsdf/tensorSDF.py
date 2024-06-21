from .tensorBase import *
import torch.nn as nn
import itertools

class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density): 

    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return self.beta
    
    # t for 0-1
    def set_beta(self,t):
        self.beta = self.beta0 * (1 + ((self.beta0 - self.beta1) / self.beta1) * (t**0.8)) ** -1
        return self.beta

   
    
class TensorVMSplit_Mesh(TensorBase):
    def __init__(self, aabb, gridSize, **kargs):
        super(TensorVMSplit_Mesh, self).__init__(aabb, gridSize, **kargs)
        
        hidden_dim = 64
        num_layers = 4
        activation = nn.ReLU
        
        n_comp=self.density_n_comp+self.app_n_comp
            
        self.decoder = nn.Sequential(
            nn.Linear(n_comp*3, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 6),
        )
        
        self.net_sdf = nn.Sequential(
                nn.Linear(n_comp*3, hidden_dim),
                activation(),
                *itertools.chain(*[[
                    nn.Linear(hidden_dim, hidden_dim),
                    activation(),
                ] for _ in range(num_layers - 2)]),
                nn.Linear(hidden_dim, 1),
            )
        
        hidden_dim_min = 64
        num_layers_min = 2
            
        self.net_deformation = nn.Sequential(
                nn.Linear(n_comp*3, hidden_dim_min),
                activation(),
                *itertools.chain(*[[
                    nn.Linear(hidden_dim_min, hidden_dim_min),
                    activation(),
                ] for _ in range(num_layers_min - 2)]),
                nn.Linear(hidden_dim_min, 3),
            )
        
        self.net_weight = nn.Sequential(
            nn.Linear(n_comp*3*8, hidden_dim_min),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim_min, hidden_dim_min),
                activation(),
            ] for _ in range(num_layers_min - 2)]),
            nn.Linear(hidden_dim_min, 21),
        )
        
         # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                
    def init_render_func(self,shadingMode, pos_pe, view_pe, fea_pe, featureC):
        pass
    

    def compute_mixfeature(self, xyz_sampled):

        B, N_point, _=xyz_sampled.shape
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, B, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, B, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(3):
            
            app_plane=self.app_plane[:,idx_plane]
            app_line=self.app_line[:,idx_plane]

            plane_coef_point.append(F.grid_sample(app_plane, coordinate_plane[idx_plane],
                                                align_corners=True).view(B, -1, N_point))
            line_coef_point.append(F.grid_sample(app_line, coordinate_line[idx_plane],
                                            align_corners=True).view(B, -1, N_point))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point,dim=1), torch.cat(line_coef_point,dim=1)
        plane_coef=plane_coef_point * line_coef_point
        plane_coef=plane_coef.permute(0,2,1)

        return plane_coef


    def geometry_feature_decode(self, sampled_features, flexicubes_indices):

        sdf = self.net_sdf(sampled_features)
        deformation = self.net_deformation(sampled_features)

        grid_features = torch.index_select(input=sampled_features, index=flexicubes_indices.reshape(-1), dim=1)
        grid_features = grid_features.reshape(
            sampled_features.shape[0], flexicubes_indices.shape[0], flexicubes_indices.shape[1] * sampled_features.shape[-1])
        weight = self.net_weight(grid_features) * 0.1

        return sdf, deformation, weight


    def get_geometry_prediction(self, svd_volume, sample_coordinates, flexicubes_indices):
        
        self.svd_volume=svd_volume
        self.app_plane=svd_volume['app_planes']
        self.app_line=svd_volume['app_lines']
        self.density_plane=svd_volume['density_planes']
        self.density_line=svd_volume['density_lines']
        
        self.app_plane=torch.cat([self.app_plane,self.density_plane],2)
        self.app_line=torch.cat([self.app_line,self.density_line],2)
        
        sample_coordinates = self.normalize_coord(sample_coordinates)
        
        sampled_features = self.compute_mixfeature(sample_coordinates)
        
        sdf, deformation, weight = self.geometry_feature_decode(sampled_features, flexicubes_indices)
        
        return sdf, deformation, weight
        
    def get_texture_prediction(self,texture_pos, svd_volume=None):\
        
        features = self.compute_mixfeature(texture_pos)
        texture_rgb=self.decoder(features)
        texture_rgb = torch.sigmoid(texture_rgb)*(1 + 2*0.001) - 0.001
        
        return texture_rgb
    
    
    
    def predict_color(self, svd_volume, xyz_sampled, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):

        self.svd_volume=svd_volume
        self.app_plane=svd_volume['app_planes']
        self.app_line=svd_volume['app_lines']
        self.density_plane=svd_volume['density_planes']
        self.density_line=svd_volume['density_lines']
        
        self.app_plane=torch.cat([self.app_plane,self.density_plane],2)
        self.app_line=torch.cat([self.app_line,self.density_line],2)
        

        chunk_size: int = 2**20
        outs = []
        for i in range(0, xyz_sampled.shape[2], chunk_size):
            xyz_sampled_chunk = self.normalize_coord(xyz_sampled[:,i:i+chunk_size])
            
            features = self.compute_mixfeature(xyz_sampled_chunk)           
            chunk_out = self.decoder(features)
            
            chunk_out = torch.sigmoid(chunk_out)*(1 + 2*0.001) - 0.001 
            
            rgbs = chunk_out.clamp(0,1)
            outs.append(chunk_out)
            
        rgbs=torch.cat(outs,1)
        
        albedo=rgbs[:,:,3:6]
        rgb=rgbs[:,:,0:3]
    
        results = {
            'shading':rgb,
            'albedo':albedo,
            'rgb':rgb*albedo,
        }
        return results 
        



class TensorVMSplit_SDF(TensorBase):
    def __init__(self, aabb, gridSize, **kargs):
        super(TensorVMSplit_SDF, self).__init__(aabb, gridSize, **kargs)
        
        hidden_dim = 64
        num_layers = 4
        activation = nn.ReLU
        
        self.lap_density = LaplaceDensity(params_init={ 'beta' : 0.1})

        n_comp=self.density_n_comp+self.app_n_comp
        

        self.net_sdf = nn.Sequential(
            nn.Linear(n_comp*3, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 1),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(n_comp*3, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 6),
        )
        
         # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                
    def init_render_func(self,shadingMode, pos_pe, view_pe, fea_pe, featureC):
        pass
    
    


    def compute_mixfeature(self, xyz_sampled):

        B, N_pixel, N_sample, _=xyz_sampled.shape
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, B, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, B, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(3):
            
            app_plane=self.app_plane[:,idx_plane]
            app_line=self.app_line[:,idx_plane]

            plane_coef_point.append(F.grid_sample(app_plane, coordinate_plane[idx_plane],
                                                align_corners=True).view(B, -1, N_pixel, N_sample))
            line_coef_point.append(F.grid_sample(app_line, coordinate_line[idx_plane],
                                            align_corners=True).view(B, -1, N_pixel, N_sample))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point,dim=1), torch.cat(line_coef_point,dim=1)
        plane_coef=plane_coef_point * line_coef_point
        plane_coef=plane_coef.permute(0,2,3,1)

        return plane_coef
    
    def forward(self, svd_volume, rays_o, rays_d, bg_color, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        
        self.svd_volume=svd_volume
        self.app_plane=svd_volume['app_planes']
        self.app_line=svd_volume['app_lines']
        self.density_plane=svd_volume['density_planes']
        self.density_line=svd_volume['density_lines']
        
        self.app_plane=torch.cat([self.app_plane,self.density_plane],2)
        self.app_line=torch.cat([self.app_line,self.density_line],2)

        B,V,H,W,_=rays_o.shape
        rays_o=rays_o.reshape(B,-1, 3)
        rays_d=rays_d.reshape(B,-1, 3)
        if ndc_ray:
            pass
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_o, rays_d, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[..., 1:] - z_vals[..., :-1], torch.zeros_like(z_vals[..., :1])), dim=-1)
        rays_d = rays_d.unsqueeze(-2).expand(xyz_sampled.shape)
        
        xyz_sampled = self.normalize_coord(xyz_sampled)
        mix_feature = self.compute_mixfeature(xyz_sampled)
        sdf = self.net_sdf(mix_feature)
        
        sigma= self.lap_density(sdf)
        sigma=sigma[...,0]  
        alpha, weight, bg_weight = raw2alpha(sigma, dists)
        
        rgbs = self.decoder(mix_feature)
        rgbs = torch.sigmoid(rgbs)*(1 + 2*0.001) - 0.001

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgbs, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clamp(0,1)
        rgb_map=rgb_map.view(B,V,H,W,6).permute(0,1,4,2,3)
        
        albedo_map=rgb_map[:,:,3:6,:,:]
        rgb_map=rgb_map[:,:,0:3,:,:]

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
        depth_map=depth_map.view(B,V,H,W,1).permute(0,1,4,2,3)
        acc_map=acc_map.view(B,V,H,W,1).permute(0,1,4,2,3)

        results = {
            'image':rgb_map,
            'albedo':albedo_map,
            'alpha':acc_map,
            'depth_map':depth_map
        }

        return results 
    
    
    def predict_sdf(self, svd_volume, xyz_sampled, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):

        self.svd_volume=svd_volume
        self.app_plane=svd_volume['app_planes']
        self.app_line=svd_volume['app_lines']
        self.density_plane=svd_volume['density_planes']
        self.density_line=svd_volume['density_lines']

        self.app_plane=torch.cat([self.app_plane,self.density_plane],2)
        self.app_line=torch.cat([self.app_line,self.density_line],2)
        
        chunk_size: int = 2**20
        outs = []
        for i in range(0, xyz_sampled.shape[1], chunk_size):
            xyz_sampled_chunk = self.normalize_coord(xyz_sampled[:,i:i+chunk_size]).half()
            features = self.compute_mixfeature(xyz_sampled_chunk)
            chunk_out = self.net_sdf(features)
            outs.append(chunk_out)
        
        sdf=torch.cat(outs,1)
        results = {
            'sigma':sdf
        }
        return results 
    
    
    def predict_color(self, svd_volume, xyz_sampled, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):

        self.svd_volume=svd_volume
        self.app_plane=svd_volume['app_planes']
        self.app_line=svd_volume['app_lines']
        self.density_plane=svd_volume['density_planes']
        self.density_line=svd_volume['density_lines']
        
        self.app_plane=torch.cat([self.app_plane,self.density_plane],2)
        self.app_line=torch.cat([self.app_line,self.density_line],2)
        
        xyz_sampled=xyz_sampled.unsqueeze(2)

        chunk_size: int = 2**20
        outs = []
        for i in range(0, xyz_sampled.shape[2], chunk_size):
            xyz_sampled_chunk = self.normalize_coord(xyz_sampled[:,i:i+chunk_size]).half()
            
            features = self.compute_mixfeature(xyz_sampled_chunk)
            chunk_out = self.decoder(features)
            chunk_out = torch.sigmoid(chunk_out)*(1 + 2*0.001) - 0.001 
            
            rgbs = chunk_out.clamp(0,1)
            outs.append(chunk_out)
            
        rgbs=torch.cat(outs,1)
        rgbs=rgbs[:,:,0,:]
        
        albedo=rgbs[:,:,3:6]
        rgb=rgbs[:,:,0:3]
    
        results = {
            'shading':rgb,
            'albedo':albedo,
            'rgb':rgb*albedo,
        }
        return results 


    