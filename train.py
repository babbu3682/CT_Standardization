import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    # cudnn.benchmark = True
    cudnn.benchmark = False  # for solving error

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.    
    simens_loader  = get_loader(config.batch_size, 'Simens',  config.mode, config.num_workers)
    ge_loader      = get_loader(config.batch_size, 'GE',      config.mode, config.num_workers)
    
    # philips_loader = get_loader(config.batch_size, 'Philips', config.mode, config.num_workers)
    # toshiba_loader = get_loader(config.batch_size, 'TOSHIBA', config.mode, config.num_workers)
    philips_loader = get_loader(1, 'Philips', config.mode, config.num_workers)
    toshiba_loader = get_loader(1, 'TOSHIBA', config.mode, config.num_workers)

    # valid for patch learning
    valid_simens_loader  = get_loader(1, 'Simens',  'valid', 0)
    valid_ge_loader      = get_loader(1, 'GE',      'valid', 0)

    # Solver for training and testing StarGAN.
    solver = Solver(simens_loader, ge_loader, philips_loader, toshiba_loader, valid_simens_loader, valid_ge_loader, config)

    if config.mode == 'train':
        solver.train_multi()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # multi gpus
    parser.add_argument('--multi-gpu-mode', default='DataParallel', choices=['DataParallel', 'DistributedDataParallel', 'Single'], type=str, help='multi-gpu-mode')          

    # Model configuration.
    parser.add_argument('--c1_dim', type=int, default=3, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=3, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--c3_dim', type=int, default=2, help='dimension of domain labels (3nd dataset)')
    parser.add_argument('--c4_dim', type=int, default=2, help='dimension of domain labels (4nd dataset)')

    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    # parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    # parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset', default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    # parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    # parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    # parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)