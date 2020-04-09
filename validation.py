import argparse
import os
import torch

import utils
import dataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--finetune_path', type = str, \
        default = './trained_models/all_loss_lr0.0001_b10_iso6400/G_iso6400_iter10000_bs1.pth', \
        #./models/all_loss_GAN/G2_iso100_epoch2000_bs12.pth
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the testing batches for single GPU')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--val_path', type = str, default = './val_results', help = 'saving path that is a folder')
    parser.add_argument('--task_name', type = str, default = 'val_320patch', help = 'task name for loading networks, saving, and log')
    # Network initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Dataset parameters
    parser.add_argument('--in_path_val', type = str, \
        default = 'E:\\SenseTime\\collect_data_v2\\val\\qbayer_input_16bit', \
            help = 'input baseroot')
    parser.add_argument('--RGBout_path_val', type = str, \
        default = 'E:\\SenseTime\\collect_data_v2\\val\\srgb_target_8bit', \
            help = 'target baseroot')
    parser.add_argument('--Qbayerout_path_val', type = str, \
        default = 'E:\\SenseTime\\collect_data_v2\\val\\qbayer_target_16bit', \
            help = 'target baseroot')
    parser.add_argument('--shuffle', type = bool, default = False, help = 'the training and validation set should be shuffled')
    parser.add_argument('--color_bias_aug', type = bool, default = False, help = 'color_bias_aug')
    parser.add_argument('--color_bias_level', type = bool, default = 0.05, help = 'color_bias_level')
    parser.add_argument('--noise_aug', type = bool, default = True, help = 'noise_aug')
    parser.add_argument('--iso', type = int, default = 6400, help = 'noise_level, according to ISO value')
    parser.add_argument('--random_crop', type = bool, default = True, help = 'random_crop')
    parser.add_argument('--crop_size', type = int, default = 320, help = 'single patch size')
    parser.add_argument('--extra_process_train_data', type = bool, default = True, help = 'recover short exposure data')
    parser.add_argument('--cover_long_exposure', type = bool, default = False, help = 'set long exposure to 0')
    parser.add_argument('--short_expo_per_pattern', type = int, default = 2, help = 'the number of exposure pixel of 2*2 square')
    opt = parser.parse_args()

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator_val(opt).cuda()
    namelist = utils.get_jpgs(opt.in_path_val)
    test_dataset = dataset.Qbayer2RGB_dataset(opt, 'val', namelist)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    sample_folder = os.path.join(opt.val_path, opt.task_name)
    utils.check_path(sample_folder)

    # forward
    val_PSNR = 0
    for i, (in_img, RGBout_img, Qbayerout_img, path) in enumerate(test_loader):
        # To device
        # A is for input image, B is for target image
        in_img = in_img.cuda()
        RGBout_img = RGBout_img.cuda()
        Qbayerout_img = Qbayerout_img.cuda()
        #print(path)
        # Forward propagation
        with torch.no_grad():
            out = generator(in_img)
        # Sample data every iter
        img_list = [out, RGBout_img]
        name_list = ['pred', 'gt']
        utils.save_sample_png(sample_folder = sample_folder, sample_name = '%d' % (i), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
        # PSNR
        val_PSNR_this = utils.psnr(out, RGBout_img, 1) * in_img.shape[0]
        print('The %d-th image PSNR %.4f' % (i, val_PSNR_this))
        val_PSNR = val_PSNR + val_PSNR_this
    val_PSNR = val_PSNR / len(namelist)
    print('The average PSNR equals to', val_PSNR)

