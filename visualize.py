import argparse
import os
from collections import OrderedDict

import open3d as o3d
import torch
import yaml
from torch.cuda.amp import autocast
from tqdm import tqdm
from wisp.ops.differential import autodiff_gradient

from nets.refine import REFINE
from utils.misc import get_cell_size, subdivide


def visualize(cfg, path_net=None, lod_inc=1):
    # Initialize net
    onet = REFINE(cfg).to(cfg.device)

    # Recover model and features
    onet_dict = torch.load(path_net)

    # DDP -> single GPU format
    new_state_dict = OrderedDict()
    for k, v in onet_dict['model'].items():
        if 'module' in k:
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    onet.load_state_dict(new_state_dict, strict=False)
    feats = onet_dict['feats']['weight']
    print('Network restored!')

    # Evaluation
    onet.eval()

    # Loop through the dataset
    pbar = tqdm(enumerate(feats))
    for i, feat in pbar:

        with autocast():

            # Bring GT to device
            feat = feat.cuda()

            # Get full object
            pred_lod, octree, pyramid, exsum, point_hierarchies = onet.get_object_kal_feat(feat[None, None])

            # Extract XYZ coordinates at the last LoD
            xyz = pred_lod['xyz'][:, pyramid[1, cfg.lods]:pyramid[1, cfg.lods + 1]]

            # Get to a higher LoD
            lod_top = cfg.lods + lod_inc + 1
            for lod in range(cfg.lods, lod_top):

                # Extract features for query points
                feats_3d = onet.extract_features(point_hierarchies, pred_lod, pyramid, xyz, lods=cfg.lods_interp)

                # Recover surface information
                sdf_pred = onet.get_sdf(feats_3d)
                occ = sdf_pred.abs() <= get_cell_size(lod)
                xyz = xyz[occ[..., 0]][None]
                sdf_pred = sdf_pred[occ[..., 0]][None]
                if lod != lod_top - 1:
                    xyz = subdivide(xyz, level=lod)

            # Recover surface information
            sdf_func = lambda x: onet.get_sdf(onet.extract_features(point_hierarchies, pred_lod, pyramid, x, lods=cfg.lods_interp))
            nrm_pred = torch.nn.functional.normalize(autodiff_gradient(xyz, sdf_func), dim=-1)
            xyz_pred = xyz - sdf_pred * nrm_pred

            # Recover color
            if cfg.rgb_type:
                feats_3d = onet.extract_features(point_hierarchies, pred_lod, pyramid, xyz_pred, lods=cfg.lods_interp)
                rgb_pred = onet.get_color(feats_3d)
            else:
                rgb_pred = (nrm_pred + 1) / 2

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_pred[0].detach().cpu())
            pcd.normals = o3d.utility.Vector3dVector(nrm_pred[0].detach().cpu())
            pcd.colors = o3d.utility.Vector3dVector(rgb_pred[0].detach().cpu())
            o3d.visualization.draw_geometries([pcd], width=500, height=500)


def main():
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_net', help='Path to a trained net')
    parser.add_argument('--lod_inc', type=int, default=1, help='LoD increment')
    args = parser.parse_args()

    # Load config
    with open(os.path.join(args.path_net, 'cfg.yaml'), "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfg = argparse.Namespace(**cfg)

    # Visualize
    visualize(cfg, path_net=os.path.join(args.path_net, 'onet.pt'), lod_inc=args.lod_inc)


if __name__ == '__main__':
    main()
