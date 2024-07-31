import torch


def subdivide(centers, level):
    """ Subdivide voxels to generate new voxel centers at a higher level of detail.

    Args:
        centers: Voxel centers (N,3)
        level: Level of detail

    Returns:
        centers_new: New voxel centers (N*8,3)
    """
    offset_size = (1 / (2 ** level)) * 2 / 4
    offsets = torch.tensor([(-1, -1, -1), (-1, -1, 1), (-1, +1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]).to(centers.device) * offset_size
    centers_new = centers.repeat_interleave(8, dim=-2) + offsets.unsqueeze(0).repeat(centers.shape[0], centers.shape[1], 1)
    return centers_new


def bin2dec(b, bits=8):
    """ Convert binary representation of an octree node to a decimal number.

    Args:
        b: A tensor containing binary representations (N, bits), where each row is a binary vector.
        bits: Number of bits in the binary representation.

    Returns:
        A tensor of shape (N,) where each element is the decimal equivalent of the corresponding binary vector in `b`.
    """
    mask = 2 ** torch.arange(bits).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def dec2bin(x, bits=8):
    """ Convert a decimal number to its binary representation with specified bit width.

    Args:
        x: A tensor containing decimal numbers (B, N) to be converted.
        bits: Number of bits for the binary representation.

    Returns:
        A tensor of shape (N, bits) where each row is the binary representation of the corresponding decimal number in `x`.
    """
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def get_cell_size(lod):
    """ Get an octree cell size at given LoD """
    return (1 / (2 ** lod)) * 2


def dilate(centers, level, rgb=None):
    """
    Dilate a voxel grid at a given Level of Detail (LoD).

    Args:
        centers: A tensor of shape (N, 3) containing the coordinates of the original voxel centers, where `N` is the number of voxels.
        level: The level of detail (LoD) that determines the size of the dilation. It is used to compute the offset size.
        rgb: A tensor of shape (N, 3) containing RGB values for each voxel.

    Returns:
        - If `rgb` is `None`, returns a tensor of shape (N * 27, 3) containing the new voxel centers after dilation.
        - If `rgb` is provided, returns a tuple (centers_new, rgb_new), where:
          - `centers_new` is a tensor of shape (N * 27, 3) containing the new voxel centers after dilation.
          - `rgb_new` is a tensor of shape (N * 27, C) containing the dilated RGB values corresponding to the new voxel centers.
    """
    offset_size = get_cell_size(level)
    offsets = torch.tensor([(-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0), (-1, 0, 1), (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
                            (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1), (1, 1, 0), (1, 1, 1),
                            (0, -1, -1), (0, -1, 0), (0, -1, 1), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1)]).to(centers.device) * offset_size

    centers_new = centers.repeat_interleave(27, dim=-2) + offsets.repeat(centers.shape[0], 1)
    if rgb is None:
        return centers_new
    else:
        rgb_new = rgb.repeat_interleave(27, dim=-2)
        return centers_new, rgb_new


def normalize_vertices(vertices, scale=0.9):
    """
    Normalizes vertices to fit an [-1...1] bounding box,
    common during training, but not necessary for visualization.

    Args:
        vertices: Model vertices (N, 3)
        scale: Scale multiplier

    Returns:
        Normalized model vertices (N, 3)
    """
    result = vertices - torch.mean(vertices, dim=0).unsqueeze(0)
    span = torch.maximum(torch.max(result, dim=0).values, torch.min(result, dim=0).values.abs())
    return (result / torch.max(span)) * scale


def quantized_to_cube(xyz_surface, level):
    """
    Quantize float points to a unit cube at a given LoD
    """
    xyz_surface = (xyz_surface / (2 ** level)) * 2 - 1
    return xyz_surface + get_cell_size(level) / 2


def to_cuda(gt, device):
    """ Move ground truth values to GPU """
    for k in gt.keys():
        gt[k] = gt[k].to(device)
    return gt
