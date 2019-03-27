#!/usr/bin/env bash
python /home/LAB/zhangzy/ProgrammingAlpha/OpenNMT-py-master/translate.py \
                    -batch_size 4 \
                    -beam_size 10 \
                    -model /home/LAB/zhangzy/ProjectModels/knowledgeComprehension/translate_model.pt \
                    -src /home/LAB/zhangzy/ProjectData/seq2seq/valid-src \
                    -tgt /home/LAB/zhangzy/ProjectData/seq2seq/valid-dst \
                    -output predict.txt \
                    -min_length 35 \
                    -max_length 512 \
                    -verbose \
                    -stepwise_penalty \
                    -coverage_penalty summary \
                    -beta 5 \
                    -length_penalty wu \
                    -alpha 0.9 \
                    -verbose \
                    -block_ngram_repeat 2 \
                    -n_best 1\
                    -ignore_when_blocking "." "</t>" "<t>" \
                    -report_bleu \
                    -report_rouge \
                    -share_vocab \
                    -dynamic_dict \
                    #-gpu 1 \
