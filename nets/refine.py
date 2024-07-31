import kaolin as kal
import torch
import torch.nn.functional as F
from torch import nn

from nets.siren import SirenBaseNet, SirenDecoder
from utils.misc import subdivide, dec2bin, bin2dec


class REFINE(nn.Module):
    def __init__(self, cfg):
        super(REFINE, self).__init__()

        # Save parameters to a dict
        self.data_type = cfg.data_type
        self.rgb_type = cfg.rgb_type
        self.lods = cfg.lods
        self.multiscale_type = cfg.multiscale_type
        self.latent_decoder = cfg.latent_size * len(cfg.lods_interp)\
            if self.multiscale_type == 'cat' else cfg.latent_size

        # Base network
        self.lodnet = nn.ModuleList()
        self.lodnet_base = SirenBaseNet(
            dim_in=cfg.latent_size,  # input dimension, ex. 2d coor
            dim_hidden=cfg.base_dim,
            dim_out=cfg.latent_size,  # output dimension, ex. rgb value
            num_layers=cfg.base_layers,  # number of layers
            final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
            decoder_layers=cfg.occ_layers,
            decoder_dim=cfg.occ_dim,
            w0_initial=30.  # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )
        for _ in range(cfg.lods):
            self.lodnet.append(self.lodnet_base)

        # RGB decoder
        if cfg.rgb_type:
            self.rgb_net = SirenDecoder(
                dim_in=self.latent_decoder if (cfg.data_type == 'sdf') else self.latent_decoder + 3,
                dim_hidden=cfg.rgb_dim,
                dim_out=3,  # output dimension, ex. rgb value
                num_layers=cfg.rgb_layers,  # number of layers
                final_activation=nn.Sigmoid(),  # activation of final layer (nn.Identity() for direct output)
                w0_initial=30.  # different signals may require different omega_0 in the first layer - this is a hyperparameter
            )

        if cfg.data_type == 'sdf':
            self.sdf_net = SirenDecoder(
                    dim_in=self.latent_decoder,
                    dim_hidden=cfg.geom_dim,
                    dim_out=1,  # output dimension, ex. rgb value
                    num_layers=cfg.geom_layers,  # number of layers
                    final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
                    w0_initial=30.  # different signals may require different omega_0 in the first layer - this is a hyperparameter
                )
        elif cfg.data_type == 'nerf':
            self.density_net = SirenDecoder(
                    dim_in=self.latent_decoder,
                    dim_hidden=cfg.geom_dim,
                    dim_out=1,  # output dimension, ex. rgb value
                    num_layers=cfg.geom_layers,  # number of layers
                    final_activation=nn.ReLU(),  # activation of final layer (nn.Identity() for direct output)
                    w0_initial=30.  # different signals may require different omega_0 in the first layer - this is a hyperparameter
                )

    def inference(self, feats, gt, lod_current=None):
        # First LoD
        pred_lod = []
        pred = {}
        pred['lat'] = feats(gt['idx']).unsqueeze(-2)
        pred['xyz'] = torch.zeros(pred['lat'].shape[0], 1, 3).to(gt[0]['occ'].device)
        pred['occ'] = torch.tensor([[[0., 1.]]]).repeat(pred['lat'].shape[0], 1, 1).to(gt[0]['occ'].device)
        pred_lod.append(pred)

        # Traverse through further LoDs to extract object
        lods = lod_current if lod_current else self.lods
        for lod in range(lods):
            pred = {}

            lat, pred['occ'] = self.lodnet[lod](pred_lod[lod]['lat'])
            pred['lat'] = lat.gather(dim=1, index=gt[lod + 1]['ids'][..., None].expand(-1, -1, lat.shape[-1]))
            pred['xyz'] = subdivide(pred_lod[lod]['xyz'], level=lod).gather(dim=1, index=gt[lod + 1]['ids'][..., None].expand(-1, -1, 3))
            pred_lod.append(pred)

        return pred_lod

    def get_object_kal(self, feat, gt=None, lod_current=None):
        device = feat(gt['idx']).device
        # First LoD
        pred_lod = {}
        pred_lod['lat'] = feat(gt['idx']).unsqueeze(-2)
        pred_lod['xyz'] = torch.zeros(pred_lod['lat'].shape[0], 1, 3).to(device)
        pred_lod['occ'] = torch.tensor([[[0., 1.]]]).repeat(pred_lod['lat'].shape[0], 1, 1).to(device)
        batch_size = pred_lod['lat'].shape[0]

        # Kaolin stats
        lods = lod_current if lod_current else self.lods
        pyramid = torch.zeros([2, lods + 2]).int()
        pyramid[0, 0], pyramid[1, 1] = 1, 1
        octree = torch.tensor([]).to(device)
        exsum = torch.tensor([0]).to(device)
        point_hierarchies = torch.tensor([[[0, 0, 0]]]).expand(batch_size, 1, 3).to(device)

        # Traverse through further LoDs to extract object
        lods = lod_current if lod_current else self.lods
        for lod in range(lods):
            lat, occ_net = self.lodnet[lod](pred_lod['lat'][:, pyramid[1, lod]:])

            if gt:
                if batch_size > 1:
                    occ = dec2bin( gt['octree'][:1][:, gt['pyramid'][:1, :, 1, lod]:gt['pyramid'][:1, :, 1, lod + 1]]).bool().view(1, -1)
                    occ = occ.expand(batch_size, -1)
                else:
                    occ = dec2bin(gt['octree'][:, gt['pyramid'][:, :, 1, lod]:gt['pyramid'][:, :, 1, lod + 1]]).bool().view(1, -1)
            else:
                occ = occ_net.softmax(-1).max(-1, keepdims=False).indices.bool()

            # Extend dict
            xyz = subdivide(pred_lod['xyz'][:, pyramid[1, lod]:], level=lod)[occ].view(batch_size, -1, 3)
            pred_lod['xyz'] = torch.cat([pred_lod['xyz'], xyz], dim=1)
            pred_lod['lat'] = torch.cat([pred_lod['lat'], lat[occ].view(batch_size, -1, lat.shape[-1])], dim=1)
            pred_lod['occ'] = torch.cat([pred_lod['occ'], occ_net], dim=1)

            # Add Kaolin stats
            pyramid[0, lod + 1] = occ.sum().item()
            pyramid[1, lod + 2] = pyramid[0, lod + 1] + pyramid[1, lod + 1]
            octree = torch.cat([octree, bin2dec(occ.view(-1, 8).int())])
            exsum = torch.cat([exsum, exsum[-1] + occ.view(-1, 8).int().sum(-1).cumsum(-1)])
            point_hierarchies = torch.cat([point_hierarchies, kal.ops.spc.quantize_points(xyz, level=lod + 1)], dim=1)

        return pred_lod, octree.byte(), pyramid, exsum.int(), point_hierarchies[0].short()

    def get_object_kal_feat(self, feat, lod_current=None):
        device = feat.device
        # First LoD
        pred_lod = {}
        pred_lod['lat'] = feat
        pred_lod['xyz'] = torch.zeros(pred_lod['lat'].shape[0], 1, 3).to(device)
        pred_lod['occ'] = torch.tensor([[[0., 1.]]]).repeat(pred_lod['lat'].shape[0], 1, 1).to(device)

        # Kaolin stats
        lods = lod_current if lod_current else self.lods
        pyramid = torch.zeros([2, lods + 2]).int()
        pyramid[0, 0], pyramid[1, 1] = 1, 1
        octree = torch.tensor([]).to(device)
        exsum = torch.tensor([0]).to(device)
        point_hierarchies = torch.tensor([[[0, 0, 0]]]).to(device)

        # Traverse through further LoDs to extract object
        lods = lod_current if lod_current else self.lods
        for lod in range(lods):
            lat, occ_net = self.lodnet[lod](pred_lod['lat'][:, pyramid[1, lod]:])
            occ = occ_net.softmax(-1).max(-1, keepdims=False).indices.bool()

            # Extend dict
            xyz = subdivide(pred_lod['xyz'][:, pyramid[1, lod]:], level=lod)[occ].view(1, -1, 3)
            pred_lod['xyz'] = torch.cat([pred_lod['xyz'], xyz], dim=1)
            pred_lod['lat'] = torch.cat([pred_lod['lat'], lat[occ].view(1, -1, lat.shape[-1])], dim=1)
            pred_lod['occ'] = torch.cat([pred_lod['occ'], occ_net], dim=1)

            # Add Kaolin stats
            pyramid[0, lod + 1] = occ.sum().item()
            pyramid[1, lod + 2] = pyramid[0, lod + 1] + pyramid[1, lod + 1]
            octree = torch.cat([octree, bin2dec(occ.view(-1, 8).int())])
            exsum = torch.cat([exsum, exsum[-1] + occ.view(-1, 8).int().sum(-1).cumsum(-1)])
            point_hierarchies = torch.cat([point_hierarchies, kal.ops.spc.quantize_points(xyz, level=lod + 1)], dim=1)

        return pred_lod, octree.byte(), pyramid, exsum.int(), point_hierarchies[0].short()

    def get_color(self, feats_3d, ray_d=None):
        if self.data_type == 'nerf':
            if torch.is_tensor(ray_d):
                rgb_pred = self.rgb_net(torch.cat([feats_3d, ray_d], dim=-1))
            else:
                rgb_pred = self.rgb_net(torch.cat([feats_3d, torch.zeros_like(feats_3d[..., :3])], dim=-1))
        else:
            rgb_pred = self.rgb_net(feats_3d)
        return rgb_pred

    def get_sdf(self, feats_3d):
        return self.sdf_net(feats_3d)

    def get_density(self, feats_3d):
        return self.density_net(feats_3d)

    def extract_features(self, point_hierarchies, pred_lod, pyramid, xyz_gt, lods):
        feats = []
        for lod in lods:
            feats_tmp = pred_lod['lat'][0][pyramid[1, lod]:pyramid[1, lod + 1]]
            feats_tmp = kal.ops.spc.to_dense(point_hierarchies, pyramid.unsqueeze(0), feats_tmp, level=lod)

            # Transform to pytorch coordinate system
            xyz_tmp = xyz_gt.flip(-1)
            feats_interp = F.grid_sample(feats_tmp, xyz_tmp.unsqueeze(2).unsqueeze(2),
                                         padding_mode="zeros", align_corners=False, mode='bilinear').squeeze(-1).squeeze(-1)
            feats.append(feats_interp)

        if self.multiscale_type == 'mean':
            features = torch.cat(feats, dim=0).mean(0, keepdims=True).permute(0, 2, 1)
        elif self.multiscale_type == 'sum':
            features = torch.cat(feats, dim=0).sum(0, keepdims=True).permute(0, 2, 1)
        elif self.multiscale_type == 'cat':
            features = torch.cat(feats, dim=1).permute(0, 2, 1)
        return features
