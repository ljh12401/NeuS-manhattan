import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, SemanticNetwork
from models.renderer import NeuSRenderer
from models.data_utils import to_cuda
import wandb


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None
        
        #########################################################
        self.joint_weight = self.conf.get_float('train.joint_weight')
        self.joint_start = self.conf.get_float('train.joint_start')
        
        self.ce_weight=self.conf.get_float('train.ce_weight')
        self.ce_weight_decay=self.conf.get_float('train.ce_weight_decay')
        self.ce_weight_decay_iters=self.conf.get_list('train.ce_weight_decay_iters')
        self.non_plane_weight=self.conf.get_float('train.non_plane_weight')
        #########################################################

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
                
        #############################################
        self.semantic_network=SemanticNetwork(**self.conf['model.semantic_network']).to(self.device)
        self.theta = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        ############################################# 
        
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        
        #################################################
        params_to_train += list(self.semantic_network.parameters())
        params_to_train.append(self.theta)
        #################################################

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.semantic_network,
                                     self.theta,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask, semantic_mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10: 13]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)
            
            semantic=torch.ones_like(mask)*10
            
            #记录semantic mask, opencv读的顺序是bgr
            ##########################################
            for i in range(self.batch_size):
                if (semantic_mask[i][0].data < 0.5 and semantic_mask.data[i][1] < 0.5 and semantic_mask.data[i][2] > 0.5 and mask.data[i] == 1):    #平面
                    semantic[i] = 1
                elif (semantic_mask.data[i][0] > 0.5 and semantic_mask.data[i][1] < 0.5 and semantic_mask.data[i][2] < 0.5 and mask.data[i] == 1):    #立面
                    semantic[i] = 2
                else: 
                    semantic[i] = 0
            ###########################################
                    
            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            
            ###################################
            #计算semantic score
            semantic_logit=render_out['semantic_logit']
            semantic_score_log = F.log_softmax(semantic_logit, dim=-1)
            semantic_score = torch.exp(semantic_score_log)
            
            #计算法向量
            surface_normals = render_out['surface_normals']
            surface_normals_normalized = F.normalize(surface_normals, dim=-1).clamp(-1., 1.)
            ###################################
            
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
            
            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight
            
            #joint_loss
            ##############################################
            floor_mask=(semantic==1).squeeze()
            wall_mask=(semantic==2).squeeze()
            
            joint_loss = 0.
            
            is_joint_start=((self.joint_start<=iter_i) or (res_step<=self.end_iter-self.joint_start))   #是否开始联合训练
            
            if is_joint_start:
                bg_score, wall_score, floor_score = semantic_score.split(dim=-1, split_size=1)
                           
                if floor_mask.sum() > 0:
                    floor_normals = surface_normals_normalized[floor_mask]
                    
                    if(self.iter_step % 2500 == 0):
                        wandb.log({"value/floor_normals":floor_normals.mean(dim=0).tolist()})
                        
                    floor_loss = (1 - floor_normals[..., 2]) # Eq.8
                    joint_floor_loss = (floor_score[floor_mask][..., 0] * floor_loss).mean() # Eq.13
                    joint_loss += joint_floor_loss
            
                if wall_mask.sum() > 0:
                    wall_normals = surface_normals_normalized[wall_mask]
                    wall_loss_vertical = wall_normals[..., 2].abs()
                    
                    if(self.iter_step % 2500 == 0):
                        wandb.log({"value/wall_normals_vertical":wall_normals[..., 2].mean(dim=0).tolist()})

                    theta = self.theta
                    cos = wall_normals[..., 0] * torch.cos(theta) + wall_normals[..., 1] * torch.sin(theta)
                    wall_loss_horizontal = torch.min(cos.abs(), torch.min((1 - cos).abs(), (1 + cos).abs())) # Eq.9
                    wall_loss = wall_loss_vertical + wall_loss_horizontal
                    joint_wall_loss = (wall_score[wall_mask][..., 0] * wall_loss).mean() # Eq.13
                    joint_loss += joint_wall_loss
            
                if floor_mask.sum() > 0 or wall_mask.sum() > 0:                
                    loss += self.joint_weight * joint_loss
            
            else: # 在早期的训练阶段语义指标不可靠，直接把平面约束相加得到joint loss

                if floor_mask.sum() > 0:
                    floor_normals = surface_normals_normalized[floor_mask]
                    floor_loss = (1 - floor_normals[..., 2]).mean()
                    joint_loss += floor_loss
            
                if wall_mask.sum() > 0:
                    wall_normals = surface_normals_normalized[wall_mask]
                    wall_loss_vertical = wall_normals[..., 2].abs().mean()
                    joint_loss += wall_loss_vertical

                if floor_mask.sum() > 0 or wall_mask.sum() > 0:
                    loss += self.joint_weight * joint_loss
            ##############################################
            
            
            
            #cross entropy loss，记录体渲染得到的semantic score与真实的semantic mask之间的差异
            ##############################################
            cross_entropy_loss = F.nll_loss(
                semantic_score_log.reshape(-1, 3),
                semantic.reshape(-1).long(),
                weight=to_cuda(torch.tensor([self.non_plane_weight,1.0,1.0]))
            ) # Eq.14

            for decay_iters in self.ce_weight_decay_iters:
                if self.iter_step==decay_iters:
                    self.ce_weight*=self.ce_weight_decay
                       
            loss+=self.ce_weight * cross_entropy_loss
            ##############################################

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            
            #使用wandb记录
            if(self.iter_step % 100 == 0):
                wandb.log({"Loss/loss": loss, "Loss/color_fine_loss": color_fine_loss, "Loss/eikonal_loss": eikonal_loss, "Loss/mask_loss":mask_loss, "Loss/joint_loss":joint_loss, "Loss/cross_entropy_loss":cross_entropy_loss},self.iter_step)
                wandb.log({"value/theta": self.theta.item(),"value/psnr":psnr},self.iter_step)
                wandb.log({"weight/ce_weight": self.ce_weight},self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq  == 0: 
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])        
        ##############################################
        self.semantic_network.load_state_dict(checkpoint['semantic_network'])
        ##############################################
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            'semantic_network': self.semantic_network.state_dict()    #保存semantic network
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_semantic_fine=[]    #验证semantic mask

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            if feasible('semantic_logit'):
                out_semantic_fine.append(render_out['semantic_logit'].detach().cpu().numpy())
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
        
        ############################################
        semantic_img=None
        if len(out_semantic_fine) > 0:
            semantic_score = np.concatenate(out_semantic_fine, axis=0)
            #将每个像素logit最大的通道作为预测的semantic mask
            semantic_idx = semantic_score.argmax(-1)
            semantic_img = np.zeros(semantic_score.shape, dtype=np.uint8)
            rows = np.arange(semantic_img.shape[0])
            semantic_img[rows, semantic_idx] = 255
            #最后一个维度转换为[1,0,2],对应semantic mask
            semantic_img = semantic_img[..., [1, 0, 2]]
            semantic_img=semantic_img.reshape([H,W,3,-1])
        ############################################
                
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
                wandb.log({"vis/validations_fine": [wandb.Image(np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]), caption='{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx))]})
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])
            if len(out_semantic_fine) > 0:
                wandb.log({"vis/validations_semantic": [wandb.Image(semantic_img[..., i], caption='{:0>8d}_{}_{}_semantic.png'.format(self.iter_step, i, idx))]})

    ###########################################################
    def validate_semantic(self, resolution_level=1):
        os.makedirs(os.path.join(self.base_exp_dir, 'semantic'), exist_ok=True)
        
        print('Infer semantic mask')
        
        for idx in tqdm(range(self.dataset.n_images),desc="Processing images"):
            rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
            
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
            
            out_semantic_fine=[]
            
            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background =None

                render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background)
                
                def feasible(key): return (key in render_out) and (render_out[key] is not None)
                
                if feasible('semantic_logit'):
                    semantic_score_log = F.log_softmax(render_out['semantic_logit'], dim=-1)
                    semantic_score = torch.exp(semantic_score_log)
                    out_semantic_fine.append(semantic_score.detach().cpu().numpy())    
                del render_out
                
            semantic_img=None
            if len(out_semantic_fine) > 0:
                semantic_score = np.concatenate(out_semantic_fine, axis=0)
                #将每个像素logit最大的通道作为预测的semantic mask
                semantic_idx = semantic_score.argmax(-1)
                semantic_img = np.zeros(semantic_score.shape, dtype=np.uint8)
                rows = np.arange(semantic_img.shape[0])
                semantic_img[rows, semantic_idx] = 255
                #最后一个维度转换为[1,0,2],对应semantic mask
                semantic_img = semantic_img[..., [2, 0, 1]]
                semantic_img=semantic_img.reshape([H,W,3])
            
            cv.imwrite(os.path.join(self.base_exp_dir,'semantic','{}.png'.format(idx)), semantic_img)   
        
        print('Infer semantic mask end')
    ############################################################

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('NeuS-manhattan by Liu Jiahao')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    
    run = wandb.init(project="NeuS-manhattan", name="run-3")

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
        runner.validate_semantic(resolution_level=1)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
    
    wandb.finish()
