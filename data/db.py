import os

import kaolin as kal
import numpy as np
import torch
from kaolin.ops.spc.points import quantize_points
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class SPC(Dataset):
    def __init__(self, args, num_samples=16384):

        self.path = args.path_data
        self.offline = False
        self.data_type = args.data_type

        # Create octgrid
        self.lods = args.lods
        self.lod_current = args.lods
        self.rgb = args.rgb_type
        self.num_samples = num_samples

        # Load data list
        self.models = []
        self.octree_common = set()
        for root, directories, filenames in os.walk(self.path):
            filenames.sort()
            pbar = tqdm(filenames)
            for file in pbar:
                file_npz = os.path.join(root, file)
                model = dict(np.load(file_npz))

                model['name'] = os.path.splitext(file)[0]
                self.models.append(model)

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):

        if self.offline:
            model_npz = np.load(self.models[idx]['path'], allow_pickle=True).item()
        else:
            model_npz = self.models[idx]

        # Model dictionary
        model = {}
        model['idx'] = idx
        model['octree'] = torch.from_numpy(model_npz['octree'])
        max_level, pyramids, exsum = kal.ops.spc.scan_octrees(model['octree'].cuda(), torch.tensor([len(model['octree'])]))
        model['pyramid'] = pyramids

        # Subsample points
        if self.data_type == 'sdf':

            if self.rgb:
                num_samples = self.num_samples // 2
                lod_ids_rgb = np.random.choice(np.arange(model_npz['colors'].shape[0]), size=num_samples, replace=True)
                model['rgb'] = torch.from_numpy(model_npz['colors'])[lod_ids_rgb] / 255
                model['xyz_rgb'] = torch.from_numpy(model_npz['xyz'])[lod_ids_rgb]
                model['rgb_surface'] = torch.from_numpy(model_npz['colors'])[:131072] / 255  # evaluation
            else:
                num_samples = self.num_samples

            # Subsample points
            lod_ids = np.random.choice(np.arange(model_npz['xyz_sdf'].shape[0]), size=num_samples, replace=True)
            model['xyz'] = torch.from_numpy(model_npz['xyz_sdf'])[lod_ids]
            model['sdf'] = torch.from_numpy(model_npz['sdf'])[lod_ids]

        elif self.data_type == 'nerf':
            # Subsample points
            lod_ids = np.random.choice(np.arange(model_npz['xyz'].shape[0]), size=self.num_samples, replace=True)
            model['xyz'] = torch.from_numpy(model_npz['xyz'])[lod_ids]
            model['sdf'] = torch.from_numpy(model_npz['density'])[lod_ids] / 255
            model['rgb'] = torch.from_numpy(model_npz['colors'])[lod_ids] / 255
            model['ray_d'] = torch.from_numpy(model_npz['ray_d'])[lod_ids]

        return model


def collate_fn(batch):
    """ Collate function for dataloader """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
