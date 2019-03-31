from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import os
import importlib
from torchtext import data
import tensorflow as tf
import texar as tx
import numpy as np

from programmingalpha.models.rl_utility import  utils,data_utils
from programmingalpha.models.rl_utility.preprocess_alpha import eos_token_id
from programmingalpha.models.RL_Transformer import BertRLTransformer
from programmingalpha.models.RL_Transformer import VocabWrapper
from bert import modeling
import argparse

# Uses the best sample by beam search
best_results = {'score': 0, 'epoch': -1}

def computeScore(epoch,sess, hypotheses,references,mode="eval"):
    hypotheses_text=[]
    references_text=[]

    for i in range(len(eval_data)):
        hypotheses_text .append( tx.utils.map_ids_to_strs(
                        hypotheses[i], vocab,strip_bos="[BOS]",strip_pad="[PAD]",
                        strip_eos="[EOS]", join=True)
                    )
        references_text.append( tx.utils.map_ids_to_strs(
                        references[i], vocab,strip_bos="[BOS]",strip_pad="[PAD]",
                        strip_eos="[EOS]", join=True)
        )

    print("hypo",len(hypotheses_text))
    [print(h) for h in hypotheses_text[:3]]
    print("refs",len(references_text))
    [print(r) for r in references_text[:3]]
    fname = os.path.join(FLAGS.model_dir, 'tmp.{}.{}'.format(machine_host,mode))
    tx.utils.write_paired_text(
        hypotheses_text, references_text, fname, mode='s',src_fname_suffix="predict",tgt_fname_suffix="truth")

    # Computes score
    bleu_scores=[]
    for ref, hyp in zip(references_text, hypotheses_text):
        bleu_one = tx.evals.sentence_bleu([ref], hyp, smooth=True)
        bleu_scores.append(bleu_one)

    eval_bleu = np.mean(bleu_scores)
    logger.info('epoch: %d, eval_bleu %.4f', epoch, eval_bleu)
    print('epoch: %d, eval_bleu %.4f' % (epoch, eval_bleu))

    if eval_bleu > best_results['score']:
        logger.info('epoch: %d, best bleu: %.4f', epoch, eval_bleu)
        best_results['score'] = eval_bleu
        best_results['epoch'] = epoch
        model_path = os.path.join(FLAGS.model_dir, model_name)
        logger.info('saving model to %s', model_path)
        print('saving model to %s' % model_path)
        #saver.save(sess, model_path)
        BertRLTransformer.saveModel(sess,model_name)



def testModel(epoch,test_data):
    
    references, hypotheses = [], []
    bsize = config_data.test_batch_size
    
    beam_width = config_model.beam_width
    encoder_input, predictions=BertRLTransformer.createInferenceModel()
    beam_search_ids = predictions['sample_id'][:, :, 0]
    # Uses the best sample by beam search
    print("evaluating epoch:{} with beam size={}".format(epoch,beam_width))

    with tf.Session() as sess:
        print("init variables !")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        BertRLTransformer.loadModel(sess,model_name)
        
        #computing evaluation output
        for i in range(0, len(test_data), bsize):
            sources, targets = zip(*test_data[i:i+bsize])
            
            x_block = data_utils.source_pad_concat_convert(sources)
            feed_dict = {
                encoder_input: x_block,
            }
                
            fetches = {
                'beam_search_ids': beam_search_ids,
            }
            
            fetches_ = sess.run(fetches, feed_dict=feed_dict)
    
            hypotheses.extend(h.tolist() for h in fetches_['beam_search_ids'])
            references.extend(r.tolist() for r in targets)
            hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
            references = utils.list_strip_eos(references, eos_token_id)

        computeScore(epoch,sess,hypotheses,references,mode="test")


def train_model():

    (predictions,encoder_input,decoder_input, labels, global_step, learning_rate, avg_loss, train_op, summary_merged) \
        = BertRLTransformer.createModelForTrain()
    
    
    def _eval(epoch, sess:tf.Session):
        hypotheses,references=[],[]
        
        bsize=config_data.eval_batch_size
        for i in range(0,len(eval_data),bsize):
            in_arrays=data_utils.seq2seq_pad_concat_convert(eval_data[i:i+bsize])
            
            feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                labels: in_arrays[2],
                learning_rate: 0.1,
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches={
                "sample_ids":predictions
            }
            fetches_ = sess.run(fetches, feed_dict=feed_dict)
    
            hypotheses.extend(h.tolist() for h in fetches_['beam_search_ids'])
            references.extend(r.tolist() for r in in_arrays[1])
            hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
            references = utils.list_strip_eos(references, eos_token_id)
            
        computeScore(epoch,sess, hypotheses,references)
        
    #begin train or eval
    def _train_epoch(sess, epoch, step, smry_writer):
        print("training epoch:{}".format(epoch))

        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            config_data.train_batch_size,
            key=lambda x: (len(x[0]), len(x[1])),
            random_shuffler=data.iterator.RandomShuffler()
        )

        for train_batch in train_iter:
            #print("batch",len(train_batch),)
            in_arrays = data_utils.seq2seq_pad_concat_convert(train_batch)
            #print(in_arrays[0].shape,in_arrays[1].shape,in_arrays[2].shape)

            feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                labels: in_arrays[2],
                learning_rate: utils.get_lr(step, config_model.lr),
                #tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            }
            fetches = {
                'step': global_step,
                'train_op': train_op,
                'smry': summary_merged,
                'loss': avg_loss,
            }

            fetches_ = sess.run(fetches, feed_dict=feed_dict)

            step, loss = fetches_['step'], fetches_['loss']

            if step and step % config_data.display_steps == 0:
                logger.info('step: %d, loss: %.4f', step, loss)
                print('step: %d, loss: %.4f' % (step, loss))
                smry_writer.add_summary(fetches_['smry'], global_step=step)


        return step


    # Run the graph
    with tf.Session() as sess:
        print("init variables !")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        BertRLTransformer.initBert(sess)


        smry_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

       
        logger.info('Begin running with train_and_evaluate mode')

        step = 0
        for epoch in range(config_data.max_train_epoch):
            if step>=config_data.train_steps:
                break

            step = _train_epoch(sess, epoch, step, smry_writer)
            _eval(epoch,sess)

def train_model_rl():

    (predictions, agent, encoder_input,decoder_input, global_step) \
        = BertRLTransformer.createRLForTrain()

    def _eval(epoch, sess:tf.Session):
        hypotheses,references=[],[]
        
        bsize=config_data.eval_batch_size
        for i in range(0,len(eval_data),bsize):
            in_arrays=data_utils.seq2seq_pad_concat_convert(eval_data[i:i+bsize])
            
            feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches={
                "sample_ids":predictions
            }
            fetches_ = sess.run(fetches, feed_dict=feed_dict)
    
            hypotheses.extend(h.tolist() for h in fetches_['beam_search_ids'])
            references.extend(r.tolist() for r in in_arrays[1])
            hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
            references = utils.list_strip_eos(references, eos_token_id)
            
        computeScore(epoch,sess, hypotheses,references)
        
    def _train_epoch(epoch, step):
        print("training epoch:{}".format(epoch))

        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            config_data.train_batch_size,
            key=lambda x: (len(x[0]), len(x[1])),
            random_shuffler=data.iterator.RandomShuffler()
        )

        #rl train
        for train_batch in train_iter:

            in_arrays = data_utils.seq2seq_pad_concat_convert(train_batch)
            feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            }

            # Samples
            extra_fetches = {
                'step': global_step,
            }

            fetches = agent.get_samples(
                extra_fetches=extra_fetches, feed_dict=feed_dict)

            sample_text = tx.utils.map_ids_to_strs(
                fetches['samples'], vocab,
                strip_pad="[PAD]",strip_bos="[BOS]",strip_eos="[EOS]",
                join=False)
            truth_text = tx.utils.map_ids_to_strs(
                in_arrays[1], vocab,
                strip_pad="[PAD]",strip_bos="[BOS]",strip_eos="[EOS]",
                join=False)

            # Computes rewards
            reward = []
            for ref, hyp in zip(truth_text, sample_text):
                r = tx.evals.sentence_bleu([ref], hyp, smooth=True)
                reward.append(r)

            # Updates
            loss = agent.observe(reward=reward)


            # Displays
            step = fetches['step']
            if step and step % config_data.display_steps == 0:
                logger.info("rl: step={}, loss={:.4f}, reward={:.4f}".format(
                    step, loss, np.mean(reward)))
                print("rl: step={}, loss={:.4f}, reward={:.4f}".format(
                    step, loss, np.mean(reward)))
                smry_writer.add_summary(fetches['smry'], global_step=step)


        return step


    # Run the graph
    with tf.Session() as sess:
        print("init variables !")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        BertRLTransformer.initBert(sess)


        agent.sess=sess

        smry_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        logger.info('Begin running with train_and_evaluate mode')

        step = 0

        for epoch in range(config_data.max_train_epoch):
            if step>=config_data.train_steps:
                break

            step = _train_epoch( epoch, step)
            if epoch==0:
                BertRLTransformer.saveModel(sess,model_name)

            _eval(epoch,sess)

def main():
    """Entrypoint.
    """
    import pickle
    from texar.modules.decoders import TransformerDecoder
    from texar.utils import transformer_utils
    # Load data
    train_data, dev_data = data_utils.load_data_numpy(
        config_data.input_dir, config_data.filename_prefix)
    with open(config_data.vocab_file, 'rb') as f:
        id2w = pickle.load(f)
    vocab_size = len(id2w)


    # Create logging
    tx.utils.maybe_create_dir(FLAGS.model_dir)
    logging_file = os.path.join(FLAGS.model_dir, 'logging.txt')
    logger = utils.get_logger(logging_file)
    print('logging file is saved in: %s', logging_file)

    # Build model graph
    encoder_input = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input = tf.placeholder(tf.int64, shape=(None, None))
    # (text sequence length excluding padding)
    encoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(encoder_input, 0)), axis=1)
    decoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(decoder_input, 0)), axis=1)

    labels = tf.placeholder(tf.int64, shape=(None, None))
    is_target = tf.to_float(tf.not_equal(labels, 0))

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')


    encoder_output, emb_tabel=BertRLTransformer.bertTransformerEncoder(True,encoder_input)
    tgt_embedding=emb_tabel
    def __computeEmbedding(embedding_table,input_ids):

            if input_ids.shape.ndims == 2:
                input_ids = tf.expand_dims(input_ids, axis=[-1])

            flat_decoder_input_ids = tf.reshape(input_ids, [-1])
            embedded = tf.gather(embedding_table, flat_decoder_input_ids)
            input_shape = modeling.get_shape_list(input_ids)
            embedded = tf.reshape(embedded,
                                  input_shape[0:-1] + [input_shape[-1] * BertRLTransformer.bert_config.hidden_size])

            return embedded

    decoder_emb_input=__computeEmbedding(emb_tabel,decoder_input)

    decoder = TransformerDecoder(embedding=tgt_embedding,
                                 hparams=config_model.decoder)
    # For training
    outputs = decoder(
        memory=encoder_output,
        memory_sequence_length=encoder_input_length,
        inputs=decoder_emb_input, #embedder(decoder_input),
        sequence_length=decoder_input_length,
        decoding_strategy='train_greedy',
        mode=tf.estimator.ModeKeys.TRAIN
    )

    mle_loss = transformer_utils.smoothing_cross_entropy(
        outputs.logits, labels, vocab_size, config_model.loss_label_confidence)
    mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)

    train_op = tx.core.get_train_op(
        mle_loss,
        learning_rate=learning_rate,
        global_step=global_step,
        hparams=config_model.opt)

    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('mle_loss', mle_loss)
    summary_merged = tf.summary.merge_all()

    # Uses the best sample by beam search

    saver = tf.train.Saver(max_to_keep=5)


    def _train_epoch(sess, epoch, step, smry_writer):
        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            config_data.train_batch_size,
            key=lambda x: (len(x[0]), len(x[1])),
            #batch_size_fn=utils.batch_size_fn,
            random_shuffler=data.iterator.RandomShuffler())
        for train_batch in train_iter:
            print("batch size",len(train_batch))
            in_arrays = data_utils.seq2seq_pad_concat_convert(train_batch)
            feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                labels: in_arrays[2],
                learning_rate: utils.get_lr(step, config_model.lr)
            }
            fetches = {
                'step': global_step,
                'train_op': train_op,
                'smry': summary_merged,
                'loss': mle_loss,
            }

            fetches_ = sess.run(fetches, feed_dict=feed_dict)

            step, loss = fetches_['step'], fetches_['loss']
            if step and step % config_data.display_steps == 0:
                logger.info('step: %d, loss: %.4f', step, loss)
                print('step: %d, loss: %.4f' % (step, loss))
                smry_writer.add_summary(fetches_['smry'], global_step=step)


        return step

    # Run the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        smry_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        if FLAGS.run_mode == 'train_and_evaluate':
            logger.info('Begin running with train_and_evaluate mode')

            if tf.train.latest_checkpoint(FLAGS.model_dir) is not None:
                logger.info('Restore latest checkpoint in %s' % FLAGS.model_dir)
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))

            step = 0
            for epoch in range(config_data.max_train_epoch):
                step = _train_epoch(sess, epoch, step, smry_writer)

        elif FLAGS.run_mode == 'test':
            logger.info('Begin running with test mode')

            logger.info('Restore latest checkpoint in %s' % FLAGS.model_dir)
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))


        else:
            raise ValueError('Unknown mode: {}'.format(FLAGS.run_mode))

if __name__ == '__main__':
    flags=argparse.ArgumentParser()

    flags.add_argument("--use_rl", action="store_true",
                        help="wthether or not use reinforcement learning.")
    flags.add_argument("--model_name", default="transoformer-rl",
                        help="name of the ouput model file.")
    flags.add_argument("--run_mode", default="train_and_evaluate",
                        help="Either train_and_evaluate or test.")
    flags.add_argument("--model_dir", default="/home/LAB/zhangzy/ProjectModels/rlmodel",
                        help="Directory to save the trained model and logs.")

    flags.add_argument("--bert_config", default="/home/LAB/zhangzy/ShareModels/uncased_L-12_H-768_A-12/bert_config.json",
                        help="Directory to bert config json file.")
    flags.add_argument("--bert_ckpt", default="/home/LAB/zhangzy/ShareModels/uncased_L-12_H-768_A-12/bert_model.ckpt",
                        help="Directory to bert model dir.")

    FLAGS = flags.parse_args()

    config_model = importlib.import_module("programmingalpha.models.rl_utility.config_model")
    config_data = importlib.import_module("programmingalpha.models.rl_utility.config_data")

    utils.set_random_seed(config_model.random_seed)

    BertRLTransformer.config_data=config_data
    BertRLTransformer.config_model=config_model
    BertRLTransformer.bert_config=modeling.BertConfig.from_json_file(FLAGS.bert_config)
    BertRLTransformer.bert_model_ckpt=FLAGS.bert_ckpt
    BertRLTransformer.transformer_model_dir=FLAGS.model_dir

    #get host name of the running machine
    import socket
    machine_host=socket.gethostname()
    # Create logging
    tx.utils.maybe_create_dir(FLAGS.model_dir)
    logging_file = os.path.join(FLAGS.model_dir, 'logging.{}.txt'.format(machine_host))
    logger = utils.get_logger(logging_file)
    print('logging file is saved in: %s', logging_file)

    # Load data
    train_data, eval_data = data_utils.load_data_numpy(
        config_data.input_dir, config_data.filename_prefix)
    #eval_data=eval_data[:100]
    #train_data=eval_data
    
    # Load vocab
    vocab=VocabWrapper(config_data.vocab)

    model_name=FLAGS.model_name+".ckpt"
    FLAGS.run_mode="test"
    if FLAGS.run_mode=="train_and_evaluate":
        if FLAGS.use_rl:
            print("training use rl")
            train_model_rl()
        else:
            print("traditional training method")
            train_model()
            
    elif FLAGS.run_mode=="test":
        testModel(0,eval_data[:50])
    
    else:
        raise ValueError("run mode: {} =>not defined!".format(FLAGS.run_mode))
    
