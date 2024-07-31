import argparse

import yaml


def parse_input():
    """ Parse input arguments """
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml', help='config file')

    # Network parameters
    parser.add_argument('--latent_size', help='net number of layers', type=int, required=False)
    parser.add_argument('--lods', help='Number of LoDs in the base network', type=int, required=False)
    parser.add_argument('--lods_interp', help='LoD used for interpolation', type=int, required=False, nargs="+")
    parser.add_argument('--multiscale_type', help='type of latent vector combination at interpolation: sum, cat, mean', type=str, required=False)
    parser.add_argument('--device', help='Device: cuda, cpu', type=str, required=False)
    parser.add_argument('--path_net', help='Path to the network if available', type=str, required=False)

    parser.add_argument('--base_dim', help='net hidden dimensionality', type=int, required=False)
    parser.add_argument('--base_layers', help='net number of layers', type=int, required=False)

    parser.add_argument('--occ_layers', help='net number of occ layers', type=int, required=False)
    parser.add_argument('--occ_dim', help='net hidden dimensionality of occ layers', type=int, required=False)
    parser.add_argument('--geom_layers', help='net number of sdf layers', type=int, required=False)
    parser.add_argument('--geom_dim', help='net hidden dimensionality of sdf layers', type=int, required=False)
    parser.add_argument('--rgb_layers', help='net number of rgb layers', type=int, required=False)
    parser.add_argument('--rgb_dim', help='net hidden dimensionality of rgb layers', type=int, required=False)
    parser.add_argument('--rgb_type', help='rgb type: 3d, null', type=str, required=False)

    # Optimizer
    parser.add_argument('--learning_rate', help='Learning rate', type=float, required=False)
    parser.add_argument('--learning_rate_latent', help='Learning rate latent', type=float, required=False)
    parser.add_argument('--epochs_max', help='Number of epochs', type=int, required=False)
    parser.add_argument('--lr_decay', help='LR multiplier when switching to the next LoD', type=float, required=False)

    # Loss weights
    parser.add_argument('--w_occ', help='Occupancy loss weight', type=float, required=False)
    parser.add_argument('--w_sdf', help='SDF loss weight', type=float, required=False)
    parser.add_argument('--w_rgb', help='RGB loss weight', type=float, required=False)

    # Data
    parser.add_argument('--path_data', help='Path to training data', type=str, required=False)
    parser.add_argument('--data_type', help='nerf, sdf', type=str, required=False)
    parser.add_argument('--cpu_threads', help='CPU threads for the dataloader', type=int, required=False)
    parser.add_argument('--batch_size', help='Batch size', type=int, required=False)
    parser.add_argument('--num_samples', help='Number of sampling points', type=int, required=False)

    # Evaluation
    parser.add_argument('--iter_log', help='Analyze iter', type=int, required=False)
    parser.add_argument('--epoch_analyze', help='Analyze epoch', type=int, required=False)
    parser.add_argument('--path_output', help='Output path', type=str, required=False)
    parser.add_argument('--wandb', help='Wandb project name', type=str, required=False)

    # Combine inline input with config input (inline has a priority)
    args_inline = vars(parser.parse_args())
    args_config = yaml.load(open(args_inline['config']), Loader=yaml.FullLoader)
    args = {k: (v if v is not None else args_config[k]) for (k, v) in args_inline.items()}
    args = argparse.Namespace(**args)

    return args
