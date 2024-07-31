import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import wandb
from data.db import SPC, collate_fn
from nets.refine import REFINE
from utils import io
from utils.misc import dec2bin, to_cuda


def train(cfg):
    # Wandb Logger
    wandb.init(project=cfg.wandb, entity='tri', mode=os.getenv('WANDB_MODE', 'run'),
               config=cfg)
    cfg = wandb.config

    # Prepare data
    trainset = SPC(cfg, num_samples=cfg.num_samples)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
                                              num_workers=cfg.cpu_threads, pin_memory=True, collate_fn=collate_fn)

    # Load base model
    onet = REFINE(cfg).to(cfg.device)

    num_models = len(trainset.models)
    feats = nn.Embedding(num_models, cfg.latent_size)
    feats.weight = nn.Parameter(torch.randn(num_models, cfg.latent_size) / math.sqrt(cfg.latent_size / 2))
    feats = feats.to(cfg.device)

    # Optimizer
    optimizer = optim.AdamW(
        [
            {'params': onet.parameters(), 'lr': cfg.learning_rate},
            {'params': feats.parameters(), 'lr': cfg.learning_rate_latent}
        ],
        lr=cfg.learning_rate)

    # Scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=cfg.lr_decay)

    # Recover model and features
    if cfg.path_net:
        onet_dict = torch.load(os.path.join(cfg.path_net, 'onet.pt'))

        # DDP -> single GPU format
        new_state_dict = OrderedDict()
        for k, v in onet_dict['model'].items():
            if 'module' in k:
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        onet.load_state_dict(new_state_dict, strict=False)
        feats.load_state_dict(onet_dict['feats'])
        print('Network restored!')

    # Losses
    loss_ce = torch.nn.CrossEntropyLoss()

    # Scaler
    scaler = GradScaler()
    loss_dict = {}

    # Training
    for epoch in range(cfg.epochs_max):
        onet.train()

        # Training loop
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for i, gt in pbar:

            with autocast(enabled=True):

                # Bring GT to device
                gt = to_cuda(gt, cfg.device)

                # Zero gradients
                optimizer.zero_grad()

                # Get full object
                pred_lod, octree, pyramid, exsum, point_hierarchies = onet.get_object_kal(feats, gt)

                # GT
                occ_gt = dec2bin(gt['octree'][:, :gt['pyramid'][:, :, 1, cfg.lods]]).bool().view(1, -1)
                sdf_gt, xyz_gt = gt['sdf'], gt['xyz']
                if cfg.rgb_type:
                    rgb_gt, xyz_gt_rgb = gt['rgb'], gt['xyz']
                    if cfg.data_type == 'sdf':
                        xyz_gt_rgb = gt['xyz_rgb']
                        xyz_gt = torch.cat([xyz_gt, xyz_gt_rgb], dim=1)

                # Extract features for query points
                feats_3d = onet.extract_features(point_hierarchies, pred_lod, pyramid, xyz_gt, lods=cfg.lods_interp)

                # Recover surface information
                if cfg.data_type == 'sdf':
                    if cfg.rgb_type:
                        feats_3d_sdf = feats_3d[:, :(feats_3d.shape[1] // 2)]
                    else:
                        feats_3d_sdf = feats_3d
                    sdf_pred = onet.get_sdf(feats_3d_sdf)
                elif cfg.data_type == 'nerf':
                    feats_3d_sdf = feats_3d
                    sdf_pred = onet.get_density(feats_3d_sdf)

                # Compute losses
                loss_dict['SDF'] = (sdf_pred - sdf_gt).norm(p=2, dim=-1).mean() * cfg.w_sdf
                loss_dict['OCC'] = loss_ce(pred_lod['occ'][:, 1:].permute(0, 2, 1), occ_gt.long()) * cfg.w_occ

                # Add color if available
                if cfg.rgb_type:
                    feats_3d_rgb = feats_3d
                    if cfg.data_type == 'sdf':
                        feats_3d_rgb = feats_3d[:, -(feats_3d.shape[1] // 2):]
                        rgb_pred = onet.get_color(feats_3d_rgb)
                    elif cfg.data_type == 'nerf':
                        rgb_pred = onet.get_color(feats_3d_rgb, ray_d=gt['ray_d'])

                    mask_loss_rgb = (feats_3d_rgb.sum(-1) != 0)
                    loss_dict['RGB'] = (rgb_pred[mask_loss_rgb] - rgb_gt[mask_loss_rgb]).norm(p=2, dim=-1).mean() * cfg.w_rgb

                # Combine losses
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            log_str = 'Epoch {}, Loss: '.format(epoch)
            for text, val in loss_dict.items():
                log_str += '{} - {:.6f}, '.format(text, val)
                # wandb logger
                if i % cfg.iter_log == 0:
                    wandb.log({text: val})
            pbar.set_description(log_str)

        # Scheduler step
        lr_scheduler.step()
        wandb.log({'lr': lr_scheduler.get_last_lr()[0]})

        # Store model
        if epoch > 0 and epoch % cfg.epoch_analyze == 0:

            sv_file = {
                'model': onet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'feats': feats.state_dict(),
            }
            if cfg.path_output:
                os.makedirs(cfg.path_output, exist_ok=True)
                torch.save(sv_file, os.path.join(cfg.path_output, 'onet.pt'))
            else:
                torch.save(sv_file, os.path.join(wandb.run.dir, 'onet.pt'))


def main():
    # Parse input
    args = io.parse_input()

    # Save config
    os.makedirs(args.path_output, exist_ok=True)
    with open(os.path.join(args.path_output, 'cfg.yaml'), 'w') as yamlfile:
        _ = yaml.dump(vars(args), yamlfile)
        print("Config saved")
        yamlfile.close()

    # Start training
    train(args)


if __name__ == '__main__':
    main()
