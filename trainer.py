import os
import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import autograd
from tensorboardX import SummaryWriter

import utils
import dataset

def Trainer(opt):
    # ----------------------------------------
    #              Initialization
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark
    
    # configurations
    save_folder = os.path.join(opt.save_path, opt.task_name)
    sample_folder = os.path.join(opt.sample_path, opt.task_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Initialize networks
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    #opt.val_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Define the dataset
    train_dataset = dataset.NIR_Colorization_dataset(opt, 'train')
    val_dataset = dataset.NIR_Colorization_dataset(opt, 'val')
    print('The overall number of training images:', len(train_dataset))
    print('The overall number of validation images:', len(val_dataset))

    # Define the dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = opt.val_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    class ColorLoss(nn.Module):
        def __init__(self):
            super(ColorLoss, self).__init__()
            self.L1loss = nn.L1Loss()
        
        def RGB2YUV(self, RGB):
            YUV = RGB.clone()
            YUV[:,0,:,:] = 0.299 * RGB[:,0,:,:] + 0.587 * RGB[:,1,:,:] + 0.114 * RGB[:,2,:,:]
            YUV[:,1,:,:] = -0.14713 * RGB[:,0,:,:] - 0.28886 * RGB[:,1,:,:] + 0.436 * RGB[:,2,:,:]
            YUV[:,2,:,:] = 0.615 * RGB[:,0,:,:] - 0.51499 * RGB[:,1,:,:] - 0.10001 * RGB[:,2,:,:]
            return YUV

        def forward(self, x, y):
            yuv_x = self.RGB2YUV(x)
            yuv_y = self.RGB2YUV(y)
            return self.L1loss(yuv_x, yuv_y)
    
    yuv_loss = ColorLoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = 'G_epoch%d_bs%d.pth' % (epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = 'G_iter%d_bs%d.pth' % (iteration, opt.train_batch_size)
        save_model_path = os.path.join(opt.save_path, opt.task_name, model_name)
        # Save model
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # Tensorboard
    writer = SummaryWriter()

    # For loop training
    for epoch in range(opt.epochs):
        
        # Record learning rate
        for param_group in optimizer_G.param_groups:
            writer.add_scalar('data/lr', param_group['lr'], epoch)
            print('learning rate = ', param_group['lr'])
        
        if epoch == 0:
            iters_done = 0

        ### Training
        for i, (in_img, RGBout_img) in enumerate(train_loader):

            # To device
            # A is for input image, B is for target image
            in_img = in_img.cuda()
            RGBout_img = RGBout_img.cuda()

            # Forward propagation
            out = generator(in_img)

            # Pixel loss
            L_pixel = opt.lambda_pixel * criterion_L1(out, RGBout_img)

            # Sum up to total loss
            loss = L_pixel

            # Record losses
            writer.add_scalar('data/L_pixel', L_pixel.item(), iters_done)
            writer.add_scalar('data/L_total', loss.item(), iters_done)

            # Backpropagate gradients
            optimizer_G.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm(generator.parameters(), opt.grad_clip_norm)
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i + 1
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()
            
            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Total Loss: %.4f] [L_pixel: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(train_loader), loss.item(), L_pixel.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), iters_done, len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), iters_done, optimizer_G)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [in_img, out, RGBout_img]
            name_list = ['input', 'output', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        ### Validation
        val_PSNR = 0
        num_of_val_image = 0

        for j, (in_img, RGBout_img) in enumerate(val_loader):
            
            # To device
            # A is for input image, B is for target image
            in_img = in_img.cuda()
            RGBout_img = RGBout_img.cuda()

            # Forward propagation
            with torch.no_grad():
                out = generator(in_img)

            # Accumulate num of image and val_PSNR
            num_of_val_image += in_img.shape[0]
            val_PSNR += utils.psnr(out, RGBout_img, 1) * in_img.shape[0]

        val_PSNR = val_PSNR / num_of_val_image

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [in_img, out, RGBout_img]
            name_list = ['input', 'output', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'val_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Record average PSNR
        writer.add_scalar('data/val_PSNR', val_PSNR, epoch)
        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))

    writer.close()

def Trainer_GAN(opt):
    # ----------------------------------------
    #              Initialization
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark
    
    # configurations
    save_folder = os.path.join(opt.save_path, opt.task_name)
    sample_folder = os.path.join(opt.sample_path, opt.task_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Initialize networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet()

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    #opt.val_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Define the dataset
    train_dataset = dataset.NIR_Colorization_dataset(opt, 'train')
    val_dataset = dataset.NIR_Colorization_dataset(opt, 'val')
    print('The overall number of training images:', len(train_dataset))
    print('The overall number of validation images:', len(val_dataset))

    # Define the dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = opt.val_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    class ColorLoss(nn.Module):
        def __init__(self):
            super(ColorLoss, self).__init__()
            self.L1loss = nn.L1Loss()
        
        def RGB2YUV(self, RGB):
            YUV = RGB.clone()
            YUV[:,0,:,:] = 0.299 * RGB[:,0,:,:] + 0.587 * RGB[:,1,:,:] + 0.114 * RGB[:,2,:,:]
            YUV[:,1,:,:] = -0.14713 * RGB[:,0,:,:] - 0.28886 * RGB[:,1,:,:] + 0.436 * RGB[:,2,:,:]
            YUV[:,2,:,:] = 0.615 * RGB[:,0,:,:] - 0.51499 * RGB[:,1,:,:] - 0.10001 * RGB[:,2,:,:]
            return YUV

        def forward(self, x, y):
            yuv_x = self.RGB2YUV(x)
            yuv_y = self.RGB2YUV(y)
            return self.L1loss(yuv_x, yuv_y)
    
    yuv_loss = ColorLoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(generator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer, lr_gd):
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = lr_gd * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = lr_gd * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = 'G_epoch%d_bs%d.pth' % (epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = 'G_iter%d_bs%d.pth' % (iteration, opt.train_batch_size)
        save_model_path = os.path.join(opt.save_path, opt.task_name, model_name)
        # Save model
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # Tensorboard
    writer = SummaryWriter()

    # For loop training
    for epoch in range(opt.epochs):
        
        # Record learning rate
        for param_group in optimizer_G.param_groups:
            writer.add_scalar('data/lr', param_group['lr'], epoch)
            print('learning rate = ', param_group['lr'])
        
        if epoch == 0:
            iters_done = 0

        ### Training
        for i, (in_img, RGBout_img) in enumerate(train_loader):

            # To device
            # A is for input image, B is for target image
            in_img = in_img.cuda()
            RGBout_img = RGBout_img.cuda()

            ## Train Discriminator
            # Forward propagation
            out = generator(in_img)

            optimizer_D.zero_grad()
            # Fake samples
            fake_scalar_d = discriminator(in_img, out.detach())
            true_scalar_d = discriminator(in_img, RGBout_img)
            # Overall Loss and optimize
            loss_D = - torch.mean(true_scalar_d) + torch.mean(fake_scalar_d)
            loss_D.backward()
            #torch.nn.utils.clip_grad_norm(discriminator.parameters(), opt.grad_clip_norm)
            optimizer_D.step()

            ## Train Generator
            # Forward propagation
            out = generator(in_img)

            # GAN loss
            fake_scalar = discriminator(in_img, out)
            L_gan = - torch.mean(fake_scalar) * opt.lambda_gan

            # Perceptual loss features
            fake_B_fea = perceptualnet(out)
            true_B_fea = perceptualnet(RGBout_img)
            L_percep = opt.lambda_percep * criterion_L1(fake_B_fea, true_B_fea)

            # Pixel loss
            L_pixel = opt.lambda_pixel * criterion_L1(out, RGBout_img)

            # Color loss
            L_color = opt.lambda_color * yuv_loss(out, RGBout_img)

            # Sum up to total loss
            loss = L_pixel + L_percep + L_gan + L_color

            # Record losses
            writer.add_scalar('data/L_pixel', L_pixel.item(), iters_done)
            writer.add_scalar('data/L_percep', L_percep.item(), iters_done)
            writer.add_scalar('data/L_color', L_color.item(), iters_done)
            writer.add_scalar('data/L_gan', L_gan.item(), iters_done)
            writer.add_scalar('data/L_total', loss.item(), iters_done)
            writer.add_scalar('data/loss_D', loss_D.item(), iters_done)

            # Backpropagate gradients
            optimizer_G.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm(generator.parameters(), opt.grad_clip_norm)
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i + 1
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Total Loss: %.4f] [L_pixel: %.4f]" %
                ((epoch + 1), opt.epochs, i, len(train_loader), loss.item(), L_pixel.item()))
            print("\r[L_percep: %.4f] [L_color: %.4f] [L_gan: %.4f] [loss_D: %.4f] Time_left: %s" %
                (L_percep.item(), L_color.item(), L_gan.item(), loss_D.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), iters_done, len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), iters_done, optimizer_G, opt.lr_g)
            adjust_learning_rate(opt, (epoch + 1), iters_done, optimizer_D, opt.lr_d)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [in_img, out, RGBout_img]
            name_list = ['input', 'output', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        ### Validation
        val_PSNR = 0
        num_of_val_image = 0

        for j, (in_img, RGBout_img) in enumerate(val_loader):
            
            # To device
            # A is for input image, B is for target image
            in_img = in_img.cuda()
            RGBout_img = RGBout_img.cuda()

            # Forward propagation
            with torch.no_grad():
                out = generator(in_img)
            
            # Accumulate num of image and val_PSNR
            num_of_val_image += in_img.shape[0]
            val_PSNR += utils.psnr(out, RGBout_img, 1) * in_img.shape[0]
        val_PSNR = val_PSNR / num_of_val_image
        
        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [in_img, out, RGBout_img]
            name_list = ['input', 'output', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'val_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Record average PSNR
        writer.add_scalar('data/val_PSNR', val_PSNR, epoch)
        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))

    writer.close()
