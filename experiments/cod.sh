#!/usr/bin/env bash



python -m torch.distributed.launch --nproc_per_node=4 --master_port 8219 train_distributed.py --gpu_id '0,1,2,3' \
--gray_aug --gray_p 0.1 --scale_aug --scale_type 1 --scale_p 0.3 --epochs 1500 --lr_step 1200 --lr 1e-5 \
--batch_size 8 --num_patch 1 --threshold 0.35 --test_per_epoch 20 --num_queries 700 \
--dataset cod --crop_size 256 --pre None --test_per_epoch 20  --test_patch --save --save_path twobranch_merge_way2_lr1e-5-w_dilation-4g-all-loss-wtv0-wot0-wdm0.5_transformer_flag_merge3_bs8 \
 --dm_count  --dilation --branch_merge --branch_merge_way 2 --transformer_flag merge3 



python -m torch.distributed.launch --nproc_per_node=1 --master_port 8219 train_distributed.py --gpu_id '0' \
--gray_aug --gray_p 0.1 --scale_aug --scale_type 1 --scale_p 0.3 --epochs 1500 --lr_step 1200 --lr 1e-5 \
--batch_size 8 --num_patch 1 --threshold 0.35 --test_per_epoch 20 --num_queries 700 \
--dataset cod --crop_size 256 --pre None --test_per_epoch 20  --test_patch --save --save_path test \
 --dm_count  --dilation --branch_merge --branch_merge_way 2 --transformer_flag merge3 