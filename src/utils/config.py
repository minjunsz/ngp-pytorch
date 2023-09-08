def config_parser():
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir",
        type=str,
        default="./logs/",
        help="where to store ckpts and logs",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./data/nerf_synthetic/lego",
        help="input data directory",
    )

    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 64,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )

    # rendering options
    parser.add_argument(
        "--N_samples",
        type=int,
        default=64,
        help="number of coarse samples per ray",
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=0,
        help="number of additional fine samples per ray",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=1,
        help="set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical",
    )
    parser.add_argument(
        "--i_embed_views",
        type=int,
        default=2,
        help="set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )

    # training options
    parser.add_argument(
        "--precrop_iters",
        type=int,
        default=0,
        help="number of steps to train on central crops",
    )
    parser.add_argument(
        "--precrop_frac",
        type=float,
        default=0.5,
        help="fraction of img taken for central crops",
    )

    # dataset options
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="blender",
        help="options: llff / blender / deepvoxels",
    )
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )

    ## blender flags
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )
    parser.add_argument(
        "--half_res",
        action="store_true",
        help="load blender synthetic data at 400x400 instead of 800x800",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=100,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_img", type=int, default=500, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--i_testset", type=int, default=1000, help="frequency of testset saving"
    )
    parser.add_argument(
        "--i_video",
        type=int,
        default=5000,
        help="frequency of render_poses video saving",
    )

    parser.add_argument(
        "--finest_res",
        type=int,
        default=512,
        help="finest resolultion for hashed embedding",
    )
    parser.add_argument(
        "--log2_hashmap_size", type=int, default=19, help="log2 of hashmap size"
    )
    parser.add_argument(
        "--sparse_loss_weight", type=float, default=1e-10, help="learning rate"
    )
    parser.add_argument(
        "--tv_loss_weight", type=float, default=1e-6, help="learning rate"
    )

    return parser


if __name__ == "__main__":
    from pathlib import Path

    parser = config_parser()
    args = parser.parse_args()
    with Path("args.yaml").open("w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            help_str = parser._option_string_actions[f"--{arg}"].help
            file.write(f"{arg}: {attr} # {help_str}\n")
