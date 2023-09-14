import torch

from src.utils.constants import BOX_OFFSETS
from src.utils.fast_hash import fast_hash


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    """
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    """
    box_min, box_max = bounding_box

    keep_mask = xyz == torch.max(torch.min(xyz, box_max), box_min)
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution

    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0]) * grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = fast_hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask
