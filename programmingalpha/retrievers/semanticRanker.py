# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import logging
import numpy as np


import programmingalpha
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


from programmingalpha.tokenizers.bert_tokenizer import BertTokenizer
from programmingalpha.models.modeling import BertForSequenceClassification,BertForSemanticPrediction

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,value=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.value=value

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,value):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.simValue=value


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self,sourcePair=""):
        self.sourcePair=sourcePair

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_values(self):
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a json file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines=[]
            for line in f.readlines():
                lines.append(json.loads(line))
            return lines


class SemanticPairProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    value2label={
            0:"0",
            0.5:"1",
            0.7:"2",
            1:"3",
        }
    def get_train_examples(self, data_dir):
        """See base class."""
        data_file="train-"+self.sourcePair+".txt"
        if self.sourcePair is None or self.sourcePair=="":
            data_file="train.txt"

        logger.info("LOOKING AT {}".format(os.path.join(data_dir, data_file)))
        return self._create_examples(
            self._read_json(os.path.join(data_dir, data_file)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_file="test-"+self.sourcePair+".txt"
        if self.sourcePair is None or self.sourcePair=="":
            data_file="test.txt"

        return self._create_examples(
            self._read_json(os.path.join(data_dir, data_file)), "dev")

    def get_labels(self):
        """See base class."""
        return ["0","1","2","3"]

    def get_values(self):
        return (0,1)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []



        for (i, line) in enumerate(lines):

            guid = "%s-%s" % (set_type, i)
            titles=line["title"].split('|||')
            bodies=line["body"].split('|||')
            text_a = titles[0]+bodies[0]
            text_b = titles[1]+bodies[1]
            value = line["simValue"]
            label=self.value2label[value]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,value=value)
            )

        return examples



def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def mseError(out,values):
    mse=np.sum(np.square(out-values))
    return mse

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


class SemanticRanker(object):
    ## model parameters
    max_seq_length=128
    do_lower_case=True
    eval_batch_size=8
    use_cuda=True
    cuda_rank=0
    fp16=False
    __default_label="0"
    __default_value=0

    def __init__(self,model_dir):

        self.device=torch.device("cuda")
        model_state_dict = torch.load(model_dir)
        model = BertForSemanticPrediction.from_pretrained(programmingalpha.BertBasePath,
                                                          state_dict=model_state_dict,
                                                          num_labels=4)
        if self.fp16:
            model.half()
        model.to(self.device)
        model = torch.nn.DataParallel(model,device_ids=[0,1])
        self.model=model

        self.tokenizer=BertTokenizer.from_pretrained(programmingalpha.BertBasePath, do_lower_case=self.do_lower_case)


    def getSemanticPair(self,query_doc,docs,doc_ids):
        examples=[]
        for (i,doc) in enumerate(docs):
            guid = "%s-%s" % (doc_ids[i], i)
            text_a = query_doc
            text_b = doc
            value = self.__default_value
            label=self.__default_label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,value=value)
            )

        return examples

    def closest_docs(self,query_doc,docs,k=1):
        doc_texts=[f["text"] for f in docs]
        doc_ids=[f["Id"] for f in docs]
        doc_ids=np.array(doc_ids,dtype=int).reshape((-1,1))

        eval_examples=self.getSemanticPair(query_doc,doc_texts,doc_ids)


        eval_features = self.convert_examples_to_features(
            eval_examples, [self.__default_label], self.max_seq_length, self.tokenizer,0)
        #logger.info("***** Running evaluation *****")
        #logger.info("  Num examples = %d", len(eval_examples))
        #logger.info("  Batch size = %d", self.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        #print("eval",all_sim_values.size(),all_label_ids.size(),all_segment_ids.size(),all_input_mask.size(),all_input_ids.size())
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)


        logits,simValues=self.computeSimvalue(eval_dataloader)

        results=np.concatenate((doc_ids,simValues),axis=1).tolist()
        results.sort(key=lambda x:x[1])

        return results[:k]

    def computeSimvalue(self,eval_dataloader:DataLoader):
        self.model.eval()
        device=self.device
        logits=[]
        simValues=[]

        for input_ids, input_mask, segment_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                b_logits,b_simValues = self.model(input_ids, segment_ids, input_mask)

            b_logits = b_logits.detach().cpu().numpy()

            b_simValues=b_simValues.detach().cpu().numpy()

            logits.append(b_logits)
            simValues.append(b_simValues)

        logits=np.concatenate(logits,axis=0)
        simValues=np.concatenate(simValues,axis=0)
        return logits,simValues

    @staticmethod
    def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,verbose=2):
        """Loads a data file into a list of `InputBatch`s."""

        label_map = {label : i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a).words()

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b).words()
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]
            if ex_index < 5 and verbose>1:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label_id))
                logger.info("sim value: %d"%example.value)

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id,
                                  value=example.value))

            if verbose>0:
                logger.info("loaded {} features".format(len(features)))

        return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
