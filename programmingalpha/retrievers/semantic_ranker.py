from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import logging
import numpy as np
import random

import programmingalpha
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from programmingalpha.tokenizers import BertTokenizer
from programmingalpha.models.InferenceModels import BertForSemanticPrediction

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,verbose=2):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
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

        if verbose>0 and len(features)%100==0:
            logger.info("loaded {} features".format(len(features)))

    return features


class SemanticRanker(object):
    #running paprameters
    server_ip=None
    server_port=None

    device="cuda"
    no_cuda=False
    gpu=-1
    local_rank=-1
    fp16=False
    seed=42

    ## model parameters
    max_seq_length=128
    do_lower_case=True
    batch_size=8

    __default_label="0"
    __default_value=0

    def initRunningConfig(self,model:BertForSemanticPrediction):
        if self.server_ip and self.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(self.server_ip, self.server_port), redirect_output=True)
            ptvsd.wait_for_attach()


        if self.local_rank == -1 or self.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(self.local_rank != -1), self.fp16))



        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

        if self.fp16:
            model.half()
        model.to(device)
        if self.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        return model

    def __init__(self,model_dir):

        model_state_dict = torch.load(model_dir)
        model = BertForSemanticPrediction.from_pretrained(programmingalpha.BertBasePath,
                                                          state_dict=model_state_dict,
                                                          num_labels=4)
        self.model=self.initRunningConfig(model)

        #print(25*"*"+"loaded model"+"*"*25)

        self.tokenizer=BertTokenizer.from_pretrained(programmingalpha.BertBasePath, do_lower_case=self.do_lower_case)
        logger.info("ranker model init finished!!!")

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
        #logger.warning("closest method excuting")
        doc_texts=[f["text"] for f in docs]
        doc_ids=[f["Id"] for f in docs]
        doc_ids=np.array(doc_ids,dtype=str).reshape((-1,1))

        eval_examples=self.getSemanticPair(query_doc,doc_texts,doc_ids)
        #print("find {} pairs to compute".format(len(eval_examples)))

        eval_features = convert_examples_to_features(
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
        results.sort(key=lambda x:float(x[1]),reverse=True)
        return results[:k]

    def computeSimvalue(self,eval_dataloader:DataLoader):
        self.model.eval()
        device=self.device
        logits=[]
        simValues=[]

        for input_ids, input_mask, segment_ids in eval_dataloader:
            #logger.info("batch predicting")
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




