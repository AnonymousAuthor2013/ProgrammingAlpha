import os

import torch

from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from programmingalpha.models.TextGenModels import TextGeneratorModel
from pytorch_pretrained_bert import optimization as bertOptimizer
import numpy as np
import torch
import random
import onmt


#my model
def buildModelForTrain(opts,device_id):
    vocab_data,model_save_path=opts.data+".vocab.pt",opts.save_model

    TextGeneratorModel.layer_num=opts.layers
    TextGeneratorModel.drop_out=opts.dropout
    textGen=TextGeneratorModel()

    model,vocab_fields=textGen.transformer,textGen.vocab_fields

    random.seed(1237)
    np.random.seed(7453)
    torch.manual_seed(13171)


    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab

    #define loss
    loss=onmt.modules.CopyGeneratorLossCompute(
        criterion=onmt.modules.CopyGeneratorLoss(vocab_size=len(tgt_vocab), force_copy=False,
                    unk_index=tgt_vocab.stoi[tgt_text_field.unk_token],ignore_index=tgt_vocab.stoi[tgt_text_field.pad_token], eps=1e-20),
        generator=(model.module if hasattr(model, 'module') else model).generator,
        tgt_vocab=tgt_vocab, normalize_by_length=True
    )

    #configure optimizer
    lr = opts.learning_rate

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

    warmup_proportion=opts.warmup_steps/opts.train_steps

    bert_optimizer = bertOptimizer.BertAdam(params=optimizer_grouped_parameters, lr=lr, warmup=warmup_proportion,
                                            t_total=opts.train_steps
                                            )

    optim = onmt.utils.optimizers.Optimizer(
        bert_optimizer, learning_rate=lr, max_grad_norm=2)


    model_saver=onmt.models.ModelSaver(base_path=textGen.modelPath,
                                 model=model.module if hasattr(model, 'module') else model,
                                 model_opt=opts,
                                 fields=textGen.vocab_fields,
                                 optim=optim,keep_checkpoint=opts.keep_checkpoint)

    trunc_size = opts.truncated_decoder  # Badly named...
    shard_size = opts.max_generator_batches if opts.model_dtype == 'fp32' else 0
    norm_method = opts.normalization
    grad_accum_count = opts.accum_count
    n_gpu = opts.world_size
    average_decay = opts.average_decay
    average_every = opts.average_every
    if device_id >= 0:
        gpu_rank = opts.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opts.gpu_verbose_level

    report_manager = onmt.utils.build_report_manager(opts)
    if torch.cuda.is_available() and len(opts.gpu_ranks)>0:
        model.to(torch.device("cuda"))
    else:
        model.to(torch.device("cpu"))

    trainer = onmt.Trainer(model, loss, loss, optim, trunc_size,
                           shard_size, norm_method,
                           grad_accum_count, n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opts.model_dtype)

    # Build NMTModel(= encoder + decoder).
    if opts.model_dtype == 'fp16':
        model.half()

    return model, optim,model_saver,trainer


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def main(opt, device_id):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)

        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    if checkpoint:
        model = build_model(model_opt, opt, fields, checkpoint)
        n_params, enc, dec = _tally_parameters(model)
        logger.info('encoder: %d' % enc)
        logger.info('decoder: %d' % dec)
        logger.info('* number of parameters: %d' % n_params)
        _check_save_model_path(opt)

        # Build optimizer.
        optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

        # Build model saver
        model_saver = build_model_saver(model_opt, opt, model, fields, optim)

        trainer = build_trainer(
            opt, device_id, model, fields, optim, model_saver=model_saver)
    else:
        model, optim,saver,trainer=buildModelForTrain(opt,device_id)
        logger.info(model)

    train_iter = build_dataset_iter("train", fields, opt)
    valid_iter = build_dataset_iter(
        "valid", fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')

    train_steps = opt.train_steps

    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()
