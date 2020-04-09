python train.py \
--pre_train False \
--train_path '/mnt/lustre/share/zhaoyuzhi/Quad-Bayer-data/slrgb2rgb/train' \
--val_path '/mnt/lustre/share/zhaoyuzhi/Quad-Bayer-data/slrgb2rgb/val' \
--finetune_path "./models/all_loss/G_iso6400_epoch100_bs1.pth" \
--task_name "train_all" \
--save_mode 'epoch' \
--save_by_epoch 20 \
--save_by_iter 10000 \
--lr_g 0.0001 \
--lr_d 0.0001 \
--b1 0.5 \
--b2 0.999 \
--weight_decay 0.0 \
--train_batch_size 1 \
--val_batch_size 1 \
--epochs 501 \
--lr_decrease_mode "epoch" \
--lr_decrease_epoch 100 \
--lr_decrease_iter 100000 \
--lr_decrease_factor 0.5 \
--lambda_pixel 1 \
--lambda_gan 0.1 \
--lambda_percep 5 \
--lambda_color 0.5 \
--num_workers 8 \
--multi_gpu True \
--save_path "./models" \
--sample_path "./samples" \
--pad "reflect" \
--activ "lrelu" \
--norm "none" \
--in_channels 3 \
--out_channels 3 \
--start_channels 64 \
--init_type "xavier" \
--init_gain 0.02 \
--noise_aug True \
--noise_level 0.03 \
--random_crop True \
--crop_size 320 \