import os
import pickle
import time
from pathlib import Path
from pprint import pprint

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical
from tqdm import tqdm, trange

from src.dataset.blender.bounding_box import get_bbox3d_for_blenderobj
from src.dataset.blender.dataset import BlenderDataset
from src.loss import sigma_sparsity_loss, total_variation_loss
from src.radam import RAdam
from src.ray_utils import get_rays_from_parameter
from src.run_nerf_helpers import *
from src.utils.config import config_parser
from src.utils.constants import declare_globals

DEVICE_ID = 0
DEBUG = False

device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
np.random.seed(100)


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded, keep_mask = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    _outputs = []
    for i in range(0, embedded.size(0), netchunk):
        _outputs.append(fn(embedded[i : i + netchunk]))
    outputs_flat = torch.cat(_outputs, dim=0)

    outputs_flat[~keep_mask, -1] = 0  # set sigma to 0 for invalid points
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(
    H,
    W,
    K,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    c2w_staticcam=None,
    **kwargs,
):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays_from_parameter(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    viewdirs = rays_d
    if c2w_staticcam is not None:
        # special case to visualize effect of viewdirs
        rays_o, rays_d = get_rays_from_parameter(H, W, K, c2w_staticcam)
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ["rgb_map", "depth_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(
    render_poses,
    H,
    W,
    focal,
    K,
    chunk,
    render_kwargs,
    gt_imgs=None,
    savedir=None,
    render_factor=0,
):
    near, far = render_kwargs["near"], render_kwargs["far"]

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    depths = []
    psnrs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, depth, acc, _ = render(
            H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs
        )
        rgbs.append(rgb.cpu().numpy())
        # normalize depth to [0,1]
        depth = (depth - near) / (far - near)
        depths.append(depth.cpu().numpy())
        if i == 0:
            print(rgb.shape, depth.shape)

        if gt_imgs is not None and render_factor == 0:
            try:
                gt_img = gt_imgs[i].cpu().numpy()
            except:
                gt_img = gt_imgs[i]
            p = -10.0 * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img)))
            print(p)
            psnrs.append(p)

        if savedir is not None:
            # save rgb and depth as a figure
            fig = plt.figure(figsize=(25, 15))
            ax = fig.add_subplot(1, 2, 1)
            rgb8 = to8b(rgbs[-1])
            ax.imshow(rgb8)
            ax.axis("off")
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(depths[-1], cmap="plasma", vmin=0, vmax=1)
            ax.axis("off")
            filename = os.path.join(savedir, "{:03d}.png".format(i))
            # save as png
            plt.savefig(filename, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            # imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    depths = np.stack(depths, 0)
    if gt_imgs is not None and render_factor == 0:
        avg_psnr = sum(psnrs) / len(psnrs)
        print("Avg PSNR over Test set: ", avg_psnr)
        with open(
            os.path.join(savedir, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb"
        ) as fp:
            pickle.dump(psnrs, fp)

    return rgbs, depths


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args, i=1)
    embedding_params = list(embed_fn.parameters())

    embeddirs_fn = None
    embeddirs_fn, input_ch_views = get_embedder(args, i=2)

    model = NeRFSmall(
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        input_ch=input_ch,
        input_ch_views=input_ch_views,
    ).to(device)

    grad_vars = list(model.parameters())

    model_fine = NeRFSmall(
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        input_ch=input_ch,
        input_ch_views=input_ch_views,
    ).to(device)

    grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,
    )

    optimizer = RAdam(
        [
            {"params": grad_vars, "weight_decay": 1e-6},
            {"params": embedding_params, "eps": 1e-15},
        ],
        lr=args.lrate,
        betas=(0.9, 0.99),
    )

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_importance": args.N_importance,
        "network_fine": model_fine,
        "N_samples": args.N_samples,
        "network_fn": model,
        "embed_fn": embed_fn,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1
    )  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # sigma_loss = sigma_sparsity_loss(raw[...,3])
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = (
        alpha
        * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1), -1
        )[:, :-1]
    )
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    # Calculate weights sparsity loss
    try:
        # Simplex check's tolerance is 1e-6, but cuda computation leads to higher numerical errors.
        # To remove the numerical instability, I enlarged the auxiliary value 1e-6 to 2e-6
        entropy = Categorical(
            probs=torch.cat(
                [weights, 1.0 - weights.sum(-1, keepdim=True) + 2e-6], dim=-1
            )
        ).entropy()
    except:
        pdb.set_trace()
    sparsity_loss = entropy

    return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    embed_fn=None,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    verbose=False,
    pytest=False,
):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.0:
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = (
        rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    )  # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
    )

    if N_importance > 0:
        rgb_map_0, depth_map_0, acc_map_0, sparsity_loss_0 = (
            rgb_map,
            depth_map,
            acc_map,
            sparsity_loss,
        )

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            N_importance,
            det=(perturb == 0.0),
            pytest=pytest,
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
        )

    ret = {
        "rgb_map": rgb_map,
        "depth_map": depth_map,
        "acc_map": acc_map,
        "sparsity_loss": sparsity_loss,
    }
    if retraw:
        ret["raw"] = raw
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["depth0"] = depth_map_0
        ret["acc0"] = acc_map_0
        ret["sparsity_loss0"] = sparsity_loss_0
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    dataset = BlenderDataset(
        args.datadir,
        "train",
        half_res=args.half_res,
        precrop_frac=args.precrop_frac,
        N_rand=args.N_rand,
    )
    test_dataset = BlenderDataset(
        args.datadir,
        "test",
        testskip=args.testskip * 2,
        half_res=args.half_res,
        precrop=False,
        N_rand=args.N_rand,
    )

    pprint(
        {
            "Image shape": dataset.imgs.shape,
            "Render Pose shape": dataset.render_poses.shape,
            "H,W,focal": (dataset.H, dataset.W, dataset.focal),
            "datadir": dataset.basedir,
        }
    )

    near = 2.0
    far = 6.0
    args.bounding_box = get_bbox3d_for_blenderobj(dataset, near=near, far=far)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    log_dir = Path(basedir, expname)
    log_dir.mkdir(parents=True, exist_ok=True)
    with Path(log_dir, "args.txt").open("w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, grad_vars, optimizer = create_nerf(args)
    start = 0
    global_step = start

    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.tensor(dataset.render_poses, device=device)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    poses = torch.tensor(dataset.poses, device=device)

    N_iters = 5000 + 1
    print("Train Start")

    loss_list = []
    psnr_list = []
    time_list = []
    start = start + 1
    time0 = time.time()
    for i in trange(start, N_iters):
        if i == args.precrop_iters:
            dataset.precrop = False

        # Random from one image
        img_i = np.random.randint(0, len(dataset))
        batch_rays, target_s = dataset[img_i]

        #####  Core optimization loop  #####
        rgb, depth, acc, extras = render(
            dataset.H,
            dataset.W,
            dataset.K,
            chunk=args.chunk,
            rays=batch_rays,
            verbose=i < 10,
            retraw=True,
            **render_kwargs_train,
        )

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras["raw"][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        img_loss0 = img2mse(extras["rgb0"], target_s)
        loss = loss + img_loss0
        psnr0 = mse2psnr(img_loss0)

        sparsity_loss = args.sparse_loss_weight * (
            extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum()
        )
        loss = loss + sparsity_loss

        # add Total Variation loss
        n_levels = render_kwargs_train["embed_fn"].n_levels
        min_res = render_kwargs_train["embed_fn"].base_resolution
        max_res = render_kwargs_train["embed_fn"].finest_resolution
        log2_hashmap_size = render_kwargs_train["embed_fn"].log2_hashmap_size
        TV_loss = sum(
            total_variation_loss(
                render_kwargs_train["embed_fn"].embeddings[i],
                min_res,
                max_res,
                i,
                log2_hashmap_size,
                n_levels=n_levels,
            )
            for i in range(n_levels)
        )
        loss = loss + args.tv_loss_weight * TV_loss
        if i > 1000:
            args.tv_loss_weight = 0.0

        loss.backward()
        # pdb.set_trace()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        ################################

        t = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i))
            if args.i_embed == 1:
                torch.save(
                    {
                        "global_step": global_step,
                        "network_fn_state_dict": render_kwargs_train[
                            "network_fn"
                        ].state_dict(),
                        "network_fine_state_dict": render_kwargs_train[
                            "network_fine"
                        ].state_dict(),
                        "embed_fn_state_dict": render_kwargs_train[
                            "embed_fn"
                        ].state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    path,
                )
            else:
                torch.save(
                    {
                        "global_step": global_step,
                        "network_fn_state_dict": render_kwargs_train[
                            "network_fn"
                        ].state_dict(),
                        "network_fine_state_dict": render_kwargs_train[
                            "network_fine"
                        ].state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    path,
                )
            print("Saved checkpoints at", path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(
                    render_poses,
                    dataset.H,
                    dataset.W,
                    dataset.focal,
                    dataset.K,
                    args.chunk,
                    render_kwargs_test,
                )
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, "{}_spiral_{:06d}_".format(expname, i)
            )
            imageio.mimwrite(moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(
                moviebase + "disp.mp4", to8b(disps / np.max(disps)), fps=30, quality=8
            )

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, "testset_{:06d}".format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", test_dataset.poses.shape)
            with torch.no_grad():
                render_path(
                    torch.tensor(test_dataset.poses),
                    test_dataset.H,
                    test_dataset.W,
                    test_dataset.focal,
                    test_dataset.K,
                    args.chunk,
                    render_kwargs_test,
                    gt_imgs=test_dataset.imgs,
                    savedir=testsavedir,
                )
            print("Saved test set")

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            loss_list.append(loss.item())
            psnr_list.append(psnr.item())
            time_list.append(t)
            loss_psnr_time = {"losses": loss_list, "psnr": psnr_list, "time": time_list}
            with open(os.path.join(basedir, expname, "loss_vs_time.pkl"), "wb") as fp:
                pickle.dump(loss_psnr_time, fp)

        global_step += 1


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    # TODO: config 추출을 main block 안에서 하고, 그거 기반으로 default device 설정까지 여기서 해버리자.
    # TODO: config logging 하는것도 여기서 바로 해버리면 될듯?
    train()
