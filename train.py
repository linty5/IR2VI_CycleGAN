import argparse
import os
import torch

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'trianing stage 1 or 2')
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--task_name', type = str, default = 'recon', help = 'task name for loading networks, saving, and log')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 5, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 10000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--finetune_path', type = str, default = '', help = 'load the pre-trained model with certain epoch, None for pre-training')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 41, help = 'number of epochs of training')
    parser.add_argument('--train_batch_size', type = int, default = 1, help = 'size of the training batches for single GPU')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'size of the validation batches for single GPU')
    parser.add_argument('--lr_g', type = float, default = 0.0002, help = 'Adam: learning rate for G')
    parser.add_argument('--lr_d', type = float, default = 0.0002, help = 'Adam: learning rate for D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0.0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode, by_epoch or by_iter')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 100000, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_pixel', type = float, default = 1, help = 'coefficient for L1 / L2 Loss, please fix it to 1')
    parser.add_argument('--lambda_gan', type = float, default = 0.1, help = 'coefficient for GAN Loss')
    parser.add_argument('--lambda_percep', type = float, default = 1, help = 'coefficient for perceptual Loss')
    parser.add_argument('--lambda_color', type = float, default = 0, help = 'coefficient for perceptual Loss')
    # GAN parameters
    parser.add_argument('--gan_mode', type = str, default = 'WGAN', help = 'type of GAN: [LSGAN | WGAN], WGAN is recommended')
    parser.add_argument('--additional_training_d', type = int, default = 1, help = 'number of training D more times than G')
    # Network initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of networks')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 32, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Dataset parameters
    parser.add_argument('--train_path', type = str, \
        default = 'E:\\dataset, task related\\Multi Spectral\\KAIST dataset processed\\train', help = 'train baseroot')
    parser.add_argument('--val_path', type = str, \
        default = 'E:\\dataset, task related\\Multi Spectral\\KAIST dataset processed\\val', help = 'val baseroot')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #                  Train
    # ----------------------------------------
    '''
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    '''
    '''
    seed = 1546
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    '''

    if opt.pre_train:
        trainer.Trainer(opt)
    else:
        trainer.Trainer_GAN(opt)
