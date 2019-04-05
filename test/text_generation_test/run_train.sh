#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python /home/LAB/zhangzy/ProgrammingAlpha/test/text_generation_test/train.py \
                   -data /home/LAB/zhangzy/ProjectData/openNMT/knowledgeData \
                   -save_model /home/LAB/zhangzy/ProjectModels/knowledgeComprehension \
                   -layers 1 \
                   -rnn_size 768 \
                   -word_vec_size 768 \
                   -max_grad_norm 0 \
                   -optim adam \
                   -encoder_type transformer \
                   -decoder_type transformer \
                   -position_encoding \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -learning_rate 1e-5 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 4 \
                   -batch_type tokens \
                   -normalization tokens \
                   -max_generator_batches 2 \
                   -train_steps 100000 \
                   -valid_steps 1000 \
                   -save_checkpoint_steps 1000 \
                   -keep_checkpoint 100 \
                   -report_every 10 \
                   -accum_count 8 \
                   -copy_attn \
                   -param_init_glorot \
                   -world_size 1 \
                   #-gpu_ranks -1 \
                   #-share_embeddings \
