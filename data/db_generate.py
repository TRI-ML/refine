import os

import open3d as o3d
import torch

torch.multiprocessing.set_start_method("spawn", force=True)
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import numpy as np
import kaolin as kal
from kaolin.ops.mesh import sample_points
from kaolin.ops.conversions.pointcloud import unbatched_pointcloud_to_spc
from wisp.ops.mesh.compute_sdf import compute_sdf

from utils.misc import get_cell_size, dilate, normalize_vertices, quantized_to_cube
from utils import io


class BOP(Dataset):
    def __init__(self, args, num_samples=300000):

        self.path = args.path_data.replace('spc', '')
        self.num_samples = num_samples

        # Create octgrid
        self.lods = args.lods
        self.color = True

        # Load data list
        self.models = []
        self.spc_dir = os.path.join('/'.join(self.path.split('/')[:-1]), 'spc')
        os.makedirs(self.spc_dir, exist_ok=True)

        for root, directories, filenames in os.walk(self.path):
            for file in filenames:
                if 'ply' in file in file:
                    file_npz = os.path.join(self.spc_dir, os.path.basename(file))[:-3] + 'npz'
                    if not os.path.isfile(file_npz):
                        self.models.append(os.path.join(root, file))

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):

        file = self.models[idx]
        file_npz = os.path.join(self.spc_dir, file.split('/')[-1])[:-3] + 'npz'
        if os.path.isfile(file_npz):
            return torch.eye(3)
        else:
            # load model
            try:
                mesh_o3d = o3d.io.read_triangle_mesh(file)
                mesh_vertices, mesh_faces = torch.from_numpy(np.array(mesh_o3d.vertices)), torch.from_numpy(np.array(mesh_o3d.triangles)).long()
            except:
                return torch.eye(3)

            # Normalize vertices
            try:
                vertices = normalize_vertices(mesh_vertices)
            except:
                return torch.eye(3)
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.numpy())

            # Kaolin assumes an exact batch format, we make sure to convert from:
            batched_vertices = vertices.unsqueeze(0)

            print('File: {}'.format(file))
            try:
                sampled_pcd = mesh_o3d.sample_points_uniformly(number_of_points=self.num_samples)
                sampled_verts = torch.from_numpy(np.array(sampled_pcd.points))
                sampled_uvs = torch.from_numpy(np.array(sampled_pcd.colors))
            except:
                print('Skipped {}'.format(file))
                return torch.eye(3)

            # Unbatch
            vertices = sampled_verts.squeeze(0).cuda()
            vertex_colors = sampled_uvs.squeeze(0).cuda()

            spc = unbatched_pointcloud_to_spc(vertices, self.lods, features=vertex_colors)
            print(f'SPC generated with {spc.point_hierarchies.shape[0]} cells.')

            # Dilate an octree
            points_old = quantized_to_cube(spc.point_hierarchies[spc.pyramids[:, 1, self.lods]:spc.pyramids[:, 1, self.lods + 1]], self.lods)
            points, colors = dilate(points_old, self.lods, spc.features)
            mask_points = (points[:, 0].abs() < 1) & (points[:, 1].abs() < 1) & (points[:, 2].abs() < 1)
            points = points[mask_points, :].view(-1, 3)
            colors = colors[mask_points, :].view(-1, 3)
            spc = unbatched_pointcloud_to_spc(points, self.lods, features=colors)

            # Compute SDF using SPC coordinates
            points_spc = quantized_to_cube(spc.point_hierarchies[spc.pyramids[:, 1, self.lods]:spc.pyramids[:, 1, self.lods + 1]], self.lods)
            sdf_spc = compute_sdf(batched_vertices[0].float(), mesh_faces, points_spc.float())
            octrees_spc = spc.octrees
            _, pyramids_spc, _ = kal.ops.spc.scan_octrees(octrees_spc, torch.tensor([len(octrees_spc)]))
            colors_spc = 255 * spc.features[:, :3]

            # Sample points for finetuning
            points_coarse = sampled_verts + (torch.rand_like(sampled_verts).float() * 2. - 1.) * get_cell_size(self.lods - 1)
            points_dense = sampled_verts + (torch.rand_like(sampled_verts).float() * 2. - 1.) * get_cell_size(self.lods + 2)
            points = torch.cat([points_coarse, points_dense], dim=0)
            sdf = compute_sdf(batched_vertices[0].float(), mesh_faces, points.float())

            vertex_surface = vertices
            colors_surface = vertex_colors[:, :3] * 255

            model = dict(
                octree=octrees_spc.cpu().numpy().astype(np.uint8),
                pyramids=pyramids_spc.cpu().numpy(),
                sdf_spc=sdf_spc.cpu().numpy().astype(np.float32),
                colors_spc=colors_spc.cpu().numpy().astype(np.uint8),
                xyz=vertex_surface.cpu().numpy().astype(np.float16),
                colors=colors_surface.cpu().numpy().astype(np.uint8),
                xyz_sdf=points.cpu().numpy().astype(np.float32),
                sdf=sdf.cpu().numpy().astype(np.float16),
            )

            np.savez(file_npz, **model)
        return torch.eye(3)


if __name__ == '__main__':

    # Parse input
    args = io.parse_input()

    # Prepare data
    trainset = BOP(args)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=3)

    pbar = tqdm(trainloader, total=len(trainloader))
    for _ in pbar:
        continue
