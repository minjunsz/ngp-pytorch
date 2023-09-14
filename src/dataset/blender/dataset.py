import json
from pathlib import Path
from typing import Literal

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.spherical_pose import get_pose_spherical
from src.ray_utils import get_rays_from_parameter


class BlenderDataset(Dataset):
    def __init__(
        self,
        basedir: str,
        split: Literal["train", "val", "test"] = "train",
        half_res: bool = False,
        testskip: int = 1,
        precrop: bool = True,
        precrop_frac: float = 1.0,
        N_rand: int = 1024,
    ):
        super(BlenderDataset, self).__init__()
        self.basedir = basedir
        self.precrop = precrop
        self.precrop_frac = precrop_frac
        self.N_rand = N_rand

        self.meta: dict
        with Path(basedir, f"transforms_{split}.json").open("r") as fp:
            self.meta = json.load(fp)

        skip = testskip
        if split == "train" or testskip == 0:
            skip = 1

        imgs, poses = [], []
        for frame in self.meta["frames"][::skip]:
            fname = Path(basedir, f"{frame['file_path']}.png")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
        # keep all 4 channels (RGBA)
        self.imgs = np.array(imgs, dtype=np.float32) / 255.0
        self.poses = np.array(poses, dtype=np.float32)

        self.H, self.W = self.imgs[0].shape[:2]
        camera_angle_x = float(self.meta["camera_angle_x"])
        focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)

        self.render_poses = torch.stack(
            [
                get_pose_spherical(angle, -30.0, 4.0)
                for angle in np.linspace(-180, 180, 40 + 1)[:-1]
            ],
            0,
        )

        if half_res:
            self.H = self.H // 2
            self.W = self.W // 2
            self.focal = focal / 2.0

            imgs_half_res = np.zeros((self.imgs.shape[0], self.H, self.W, 4))
            for i, img in enumerate(self.imgs):
                imgs_half_res[i] = cv2.resize(
                    img, (self.W, self.H), interpolation=cv2.INTER_AREA
                )
            self.imgs = imgs_half_res

        # White Background
        self.imgs = self.imgs[..., :3] * self.imgs[..., -1:] + (1 - self.imgs[..., -1:])

    @property
    def K(self):
        return np.array(
            [
                [self.focal, 0, 0.5 * self.W],
                [0, self.focal, 0.5 * self.H],
                [0, 0, 1],
            ]
        )

    def __getitem__(self, index):
        target = self.imgs[index]
        target = torch.tensor(target)
        pose = self.poses[index, :3, :4]
        rays_o, rays_d = get_rays_from_parameter(
            self.H, self.W, self.K, torch.tensor(pose)
        )

        if self.precrop:
            dH = int(self.H // 2 * self.precrop_frac)
            dW = int(self.W // 2 * self.precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.H // 2 - dH, self.H // 2 + dH - 1, 2 * dH),
                    torch.linspace(self.W // 2 - dW, self.W // 2 + dW - 1, 2 * dW),
                ),
                -1,
            )
        else:
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, self.H - 1, self.H),
                    torch.linspace(0, self.W - 1, self.W),
                ),
                -1,
            )  # (H, W, 2)

        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        select_inds = np.random.choice(
            coords.shape[0], size=[self.N_rand], replace=False
        )  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        return batch_rays, target_s

    def __len__(self):
        return self.imgs.shape[0]
