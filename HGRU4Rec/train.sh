#!/bin/bash
python train_hgru4rec.py 172 100 --hdf_path=$1 --test_hdf_path=$2 --early_stopping \
   --checkpoint_dir $3 --n_epochs 200 --batch_size 512 --momentum 0.1 --loss top1 --learning_rate 0.001 \
   --user_key user_id --item_key item_id --session_key session_id --time_key created_at 