from programmingalpha.models.TextGenModels import TextGeneratorModel
from pytorch_pretrained_bert import optimization as bertOptimizer
import numpy as np
import torch
from torch import nn
import random
import onmt
import argparse
import programmingalpha
import logging
#from programmingalpha.models.openNMT_utils import paralleltrainer
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def trainModel(textGen:TextGeneratorModel,train_data_files=None,valid_data_files=None):
    model,vocab_fields=textGen.transformer,textGen.vocab_fields
    #init random state
    #define some hyper-paprameters
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    device = "cuda" if args.n_gpu>0 and torch.cuda.is_available() else "cpu"
    if args.fp16:
        model.half()
        model_type="fp16"
    else:
        model_type="fp32"

    if args.n_gpu>1:
        model=nn.DataParallel(model)
        args.n_gpu=1
    if args.n_gpu<1:
        args.gpu_rank=-1


    model.to(torch.device(device))

    # extract fields
    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab

    #define loss
    loss=onmt.modules.CopyGeneratorLossCompute(
        criterion=onmt.modules.CopyGeneratorLoss(vocab_size=len(tgt_vocab), force_copy=False,
                    unk_index=tgt_vocab.stoi[tgt_text_field.unk_token],ignore_index=tgt_vocab.stoi[tgt_text_field.pad_token], eps=1e-20),
        generator=(model.module if hasattr(model, 'module') else model).generator,
        tgt_vocab=tgt_vocab, normalize_by_length=True
    )


    '''
    model.generator=nn.Sequential(
            nn.Linear(768, len(tgt_vocab)),
            nn.LogSoftmax(dim=-1))
    loss = onmt.utils.loss.NMTLossCompute(
        criterion=nn.NLLLoss(ignore_index=tgt_vocab.stoi[tgt_text_field.pad_token], reduction="sum"),
        generator=model.generator)
    '''

    #configure optimizer
    lr = args.learning_rate

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]


    bert_optimizer = bertOptimizer.BertAdam(params=optimizer_grouped_parameters, lr=lr, warmup=args.warmup_proportion,
                                            t_total=args.num_train_steps
                                            )

    optim = onmt.utils.optimizers.Optimizer(
        bert_optimizer, learning_rate=lr, max_grad_norm=2)

    logger.info(model)



    train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=train_data_files,
                                                         fields=vocab_fields,
                                                         batch_size=args.train_batch_size,
                                                         batch_size_multiple=1,
                                                         batch_size_fn=None,
                                                         device=device,
                                                         is_train=True,
                                                         repeat=True)

    valid_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=valid_data_files,
                                                         fields=vocab_fields,
                                                         batch_size=args.eval_batch_size,
                                                         batch_size_multiple=1,
                                                         batch_size_fn=None,
                                                         device=device,
                                                         is_train=False,
                                                         repeat=False)

    report_manager = onmt.utils.ReportMgr(
        report_every=args.eval_steps, start_time=-1, tensorboard_writer=None)

    saver=onmt.models.ModelSaver(base_path=textGen.modelPath,
                                 model=model.module if hasattr(model, 'module') else model,
                                 model_opt=args,
                                 fields=textGen.vocab_fields,
                                 optim=optim,keep_checkpoint=args.keep_checkpoints)

    trainer = onmt.Trainer(model=model,
                           train_loss=loss,
                           valid_loss=loss,
                           optim=optim,shard_size=32,grad_accum_count=args.gradient_accumulation_steps,
                           report_manager=report_manager,
                           model_saver=saver,n_gpu=args.n_gpu,gpu_rank=args.gpu_rank,
                           model_dtype=model_type)

    trainer.train(train_iter=train_iter,
                  train_steps=args.num_train_steps,
                  valid_iter=valid_iter,
                  valid_steps=args.eval_steps,
                  save_checkpoint_steps=args.save_steps)


def generateText(textGen:TextGeneratorModel,test_data_file):
    from onmt import translate
    from onmt import inputters

    src_reader = onmt.inputters.str2reader["text"]
    tgt_reader = onmt.inputters.str2reader["text"]
    scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7,
                                             beta=0.,
                                             length_penalty="avg",
                                             coverage_penalty="none")

    device = "cuda" if args.n_gpu>0 and torch.cuda.is_available() else "cpu"

    model=textGen.transformer
    vocab_fields=textGen.vocab_fields
    batch_size=args.eval_batch_size
    gpu = args.gpu_rank if device=="cuda" else -1

    model.to(torch.device(device))

    translator = translate.Translator(model=model,
                                           fields=vocab_fields,
                                           src_reader=src_reader,
                                           tgt_reader=tgt_reader,
                                           global_scorer=scorer,
                                           copy_attn=True,
                                           gpu=gpu)

    builder = translate.TranslationBuilder(data=torch.load(test_data_file),
                                                fields=vocab_fields)

    valid_iter = inputters.inputter.DatasetLazyIter(dataset_paths=[test_data_file],
                                                     fields=vocab_fields,
                                                     batch_size=batch_size,
                                                     batch_size_multiple=1,
                                                     batch_size_fn=None,
                                                     device=device,
                                                     is_train=False,
                                                     repeat=False)

    for batch in valid_iter:
        trans_batch = translator.translate_batch(
            batch=batch, src_vocabs=batch.dataset.src_vocabs,
            attn_debug=False)
        translations = builder.from_batch(trans_batch)
        for trans in translations:
            print(trans.log(0))


def runPrediction():
    data_dir=args.data_dir
    TextGeneratorModel.vocab_data_file=data_dir+"/knowledgeData.vocab.pt"
    TextGeneratorModel.vocab_file=data_dir+"/vocab.txt"
    TextGeneratorModel.modelPath=args.save_path
    textGen=TextGeneratorModel()
    textGen.loadModel(6000)

    # Load some data
    validate_data_files=[ "/home/LAB/zhangzy/ProjectData/openNMT/knowledgeData.valid.0.pt" ]


    '''
    #load vocabs
    vocab_fields = textGen.vocab_fields

    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    
    print(src_text_field.pad_token,src_text_field.eos_token,src_text_field.init_token,src_text_field.unk_token)
    print(tgt_text_field.pad_token,tgt_text_field.eos_token,tgt_text_field.init_token,tgt_text_field.unk_token)

    vocabs=textGen.tokenizer.vocab
    print(type(src_vocab.stoi),type(tgt_vocab.stoi),type(vocabs))
    count=0
    conflicts=[]
    for k in tgt_vocab.stoi:
        idx1=tgt_vocab.stoi[k]
        idx2=src_vocab.stoi[k]
        idx3=vocabs[k]
        print(k,idx1,idx2,idx3)
        try:
            assert idx1==idx2 and idx1==idx3
        except:
            count+=1
            conflicts.append((k,idx1,idx2,idx3))
        print(k,tgt_vocab.stoi[k],src_vocab.stoi[k],vocabs[k])

    print(count)
    print(conflicts)
    '''

    generateText(textGen,validate_data_files[0])

def runTrain():
    import os
    data_dir=args.data_dir

    train_data_files=[]
    validate_data_files=[]

    for filename in os.listdir(data_dir):
        if "train" in filename:
            train_data_files.append(os.path.join(data_dir,filename))
        elif "valid" in filename:
            validate_data_files.append(os.path.join(data_dir,filename))

    logger.info("train files:{}".format(train_data_files))
    logger.info("validate files:{}".format(validate_data_files))

    TextGeneratorModel.vocab_data_file=data_dir+"/knowledgeData.vocab.pt"
    TextGeneratorModel.vocab_file=data_dir+"/vocab.txt"
    TextGeneratorModel.modelPath=args.save_path

    textGen=TextGeneratorModel()

    trainModel(textGen,train_data_files=train_data_files,valid_data_files=validate_data_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=programmingalpha.DataPath+"openNMT/",
                        type=str,
                        help="The input data dir.")


    parser.add_argument("--save_path",
                        default=programmingalpha.ModelPath+"knowledgeComprehension/model",
                        type=str,help="path dir to save the model")

    ## Other parameters


    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=10,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_steps",
                        default=500,type=int)
    parser.add_argument("--save_steps",
                        default=500,type=int)
    parser.add_argument("--keep_checkpoints",
                        default=50,type=int)
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_steps",
                        default=100000,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--n_gpu",
                        type=int,
                        default=1,
                        help="how many gpus to use")
    parser.add_argument("--gpu_rank",
                        type=int,
                        default=0,
                        help="which gpu to use")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")



    args = parser.parse_args()
    args.fp16=False
    logger.info("args:{}".format(args))

    runTrain()
    #runPrediction()
