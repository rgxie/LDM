import tyro
import time
import random
import numpy as np
import os
import shutil

import torch
from core.options import AllConfigs
from core.models import LDM_Mesh,LDM_SDF
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file


import kiui


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():    
    opt = tyro.cli(AllConfigs)

    if opt.over_fit:
        opt.num_epochs=opt.num_epochs*1000
    
    directory = os.path.dirname(opt.workspace)
    if not os.path.exists(opt.workspace):
        os.makedirs(opt.workspace)
        print(f"Directory created: {opt.workspace}")

    try:
        config_path='./core/options.py'
        shutil.copy(config_path, opt.workspace)
        print(f"File copied successfully from {config_path} to {opt.workspace}")
    except Exception as e:
        print(f"Error occurred while copying file: {e}")


    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    seed = 6868
    set_seed(seed)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    # model
    if opt.volume_mode == 'TRF_Mesh':
        model = LDM_Mesh(opt)
    elif opt.volume_mode == 'TRF_SDF':
        model = LDM_SDF(opt)
    else:
        raise NotImplementedError

    accelerator.print(f'[INFO] volume mode: {opt.volume_mode}  lr:{opt.lr}  num_epochs:{opt.num_epochs}')

    # data
    if opt.data_mode == 's5': 
        from core.dataset.provider_gobjaverse_crop import GobjaverseDataset as Dataset
    elif opt.data_mode == 's6': 
        from core.dataset.provider_gobjaverse_mesh import GobjaverseDataset as Dataset
    else:
        raise NotImplementedError

    train_dataset = Dataset(opt, training=True)
    train_dataset.total_epoch = opt.num_epochs
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    total_steps = opt.num_epochs * len(train_dataloader)
    if opt.lr_scheduler=='cosine':
        from core.scheduler import CosineWarmupScheduler
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_iters=opt.warmup_real_iters,
            max_iters=total_steps,
        )
    elif opt.lr_scheduler=='OneCycleLR':
        pct_start =  opt.warmup_real_iters / total_steps   
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)
    else:
        raise NotImplementedError(f"Scheduler type {opt.lr_scheduler} not implemented")
        
    # resume
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
            
            #load pretrained openlrm model
            state_dict = model.state_dict()
            for k, v in ckpt.items():
                if 'synthesizer' in k:
                    k=k.replace('synthesizer.decoder.net', 'tensorRF.decoder')
                else:
                    k='vsd_net.'+k
                
                if 'upsampler.weight' in k:
                    v=v[:,0:40,:,:]
                
                if 'upsampler.bias' in k:
                    v=v[0:40]
                    
                if k in state_dict: 
                    if state_dict[k].shape == v.shape:
                        state_dict[k].copy_(v)
                    else:
                        if 'pos_embed' in k:
                            state_dict[k][:,0:3072,:].copy_(v)
                else:
                    accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
            accelerator.print(f'[INFO] load resume success!')
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
                        accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
                else:
                    accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
            accelerator.print(f'[INFO] load resume success!')

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    # loop
    for epoch in range(opt.num_epochs):
        train_dataset.cur_epoch = epoch
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        print_ieration = 100
        start_time = time.time()  
        for i, data in enumerate(train_dataloader):
            train_dataset.cur_itrs = epoch*len(train_dataloader)+i
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                out = model(data, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                accelerator.backward(loss)
                
                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

            if opt.over_fit and epoch% print_ieration != 0:
                continue

            if accelerator.is_main_process:
                # logging
                if i % print_ieration == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}")
                
                    gt_images = data['images_output'].detach().cpu().numpy() 
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) 
                    kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)
                    
                    gt_albedos = data['albedos_output'].detach().cpu().numpy() 
                    gt_albedos = gt_albedos.transpose(0, 3, 1, 4, 2).reshape(-1, gt_albedos.shape[1] * gt_albedos.shape[3], 3) 
                    kiui.write_image(f'{opt.workspace}/train_gt_albedos_{epoch}_{i}.jpg', gt_albedos)

                    pred_images = out['images_pred'].detach().cpu().numpy() 
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images)
                    
                    pred_albedos = out['pred_albedos'].detach().cpu().numpy() 
                    pred_albedos = pred_albedos.transpose(0, 3, 1, 4, 2).reshape(-1, pred_albedos.shape[1] * pred_albedos.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_pred_albedos_{epoch}_{i}.jpg', pred_albedos)
                    
                    pred_shading = out['pred_shading'].detach().cpu().numpy() 
                    pred_shading = pred_shading.transpose(0, 3, 1, 4, 2).reshape(-1, pred_shading.shape[1] * pred_shading.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_pred_shading_{epoch}_{i}.jpg', pred_shading)
                    
                    if 'depth' in out:
                        pred_depth = out['depth'].detach().cpu().numpy() 
                        pred_depth = pred_depth.transpose(0, 3, 1, 4, 2).reshape(-1, pred_depth.shape[1] * pred_depth.shape[3], 1)
                        kiui.write_image(f'{opt.workspace}/train_pred_depth_{epoch}_{i}.jpg', pred_depth)
                        
                        gt_depth = data['depth_output'].detach().cpu().numpy() 
                        gt_depth = gt_depth.transpose(0, 3, 1, 4, 2).reshape(-1, gt_depth.shape[1] * gt_depth.shape[3], 1)
                        kiui.write_image(f'{opt.workspace}/train_gt_depth_{epoch}_{i}.jpg', gt_depth)

                    end_time = time.time()  
                    
                    print(f"Takes {(end_time - start_time)/print_ieration:.3f} seconds per iteration")
                    start_time = time.time()

        if opt.over_fit and epoch% print_ieration != 0:
                continue
        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
        
        accelerator.wait_for_everyone()

        accelerator.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, os.path.join(opt.workspace, "last.ckpt"))

        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            for i, data in enumerate(test_dataloader):

                out = model(data)
    
                psnr = out['psnr']
                total_psnr += psnr.detach()
                
                if accelerator.is_main_process:
                    gt_images = data['images_output'].detach().cpu().numpy() 
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) 
                    kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() 
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")



if __name__ == "__main__":
    main()
