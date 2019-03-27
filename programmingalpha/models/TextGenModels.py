import torch
import torch.nn as nn
import onmt.utils
from pytorch_pretrained_bert.modeling import BertModel
import programmingalpha
import logging
import numpy as np
from onmt.modules import PositionalEncoding
from pytorch_pretrained_bert.modeling import BertEmbeddings

from copy import deepcopy

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class OnmtBertEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, vocab_size,embedinng_dim,max_posistion_size,drop_out,padding_idx,bertEmb:BertEmbeddings):
        super(OnmtBertEmbedding, self).__init__()
        #self.word_embeddings = nn.Embedding(vocab_size, embedinng_dim)
        #self.position_embeddings =  nn.Embedding(max_posistion_size, embedinng_dim)
        #self.LayerNorm = BertLayerNorm(embedinng_dim, eps=1e-12)
        #share embeddings between tgt and src
        self.word_embeddings=bertEmb.word_embeddings
        self.position_embeddings=bertEmb.position_embeddings
        self.LayerNorm = bertEmb.LayerNorm

        #self.position_embeddings=PositionalEncoding(dropout=drop_out,dim=embedinng_dim,max_len=max_posistion_size)

        self.word_padding_idx=padding_idx

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.dropout = nn.Dropout(drop_out)

        self.max_position_size=max_posistion_size
        self.vocab_size=vocab_size
        self.type_num=2
        self.dropout_prob=drop_out
        self.embeddings_size=embedinng_dim

    def forward(self, input_ids,step=None):
        #print("O input",input_ids.size())
        input_ids=input_ids.transpose(0,1).squeeze(2)
        #print("input",input_ids.size())

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print("before",embeddings.size())
        embeddings=embeddings.transpose(0,1)
        #print("after",embeddings.size())
        return embeddings

class BertAsEmbedding(nn.Module):
    def __init__(self,bert:BertModel,padding_idx):
        super(BertAsEmbedding, self).__init__()
        self.embeddings=bert.embeddings
        self.encoder=bert.encoder
        self.word_padding_idx=padding_idx

    def forward(self, input_ids, token_type_ids=None, step=None, lengths=None,attention_mask=None, output_all_encoded_layers=False):
        #print("before input ids",input_ids.size())
        input_ids=input_ids.transpose(0,1).squeeze(2)
        #print("after input ids",input_ids.size())
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids=torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.encoder.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        #print("embedding output:",embedding_output.size(),embedding_output.dtype)

        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        #sequence_output = encoded_layers[-1]
        #pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        #print(output_all_encoded_layers,"size of encoded layers, embeddings",encoded_layers.size(),embedding_output.size())
        return encoded_layers.transpose(0,1)


class OnmtBertTransformerEncoder(BertModel):
    def forward(self, input_ids, lengths=None,token_type_ids=None, attention_mask=None, output_all_encoded_layers=False):
        #print("before input ids",input_ids.size())
        input_ids=input_ids.transpose(0,1).squeeze(2)
        #print("after input ids",input_ids.size())
        #print("inpu ids",input_ids)
        if attention_mask is None:
            attention_mask=torch.ones_like(input_ids)
            #print("att before",attention_mask)
            #indices=input_ids==1
            #attention_mask[indices]=0
            #print("att after",attention_mask)
        if token_type_ids is None:
            token_type_ids=torch.zeros_like(input_ids)
            #print("tok before",token_type_ids)
            #token_type_ids[:,100:]=1
            #print("tok after",token_type_ids)
        #exit(120)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        #sequence_output = encoded_layers[-1]
        #pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        #print("size of encoded layers, embeddings",encoded_layers.size(),embedding_output.size())
        return embedding_output.transpose(0,1),encoded_layers.transpose(0,1),lengths



class TransformerModel(nn.Module):
    def __init__(self, encoder:OnmtBertTransformerEncoder, decoder:onmt.decoders.TransformerDecoder):
        nn.Module.__init__(self)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        #print("lengths, src and tgt size is ",lengths.size(),src.size(),tgt.size())
        #src_enc=src.squeeze(2)

        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state,memory_bank,lengths = self.encoder(input_ids=src,lengths=lengths)
        #print("size of encoded layers",memory_bank.size())

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        #print("decoded result",dec_out.size(),attns["std"].size())
        return dec_out, attns

class TextGeneratorModel(object):


    hidden_size=768
    max_tgt_seq=512
    max_src_seq=512
    feed_forwad_size=3072
    heads_num=12
    layer_num=4
    drop_out=0.1

    tgt_padding=1
    tgt_vocab_size=30522


    def __init__(self):
        self.__iniModelConifg()


    def __iniModelConifg(self):
        #encoder
        bert=OnmtBertTransformerEncoder.from_pretrained(programmingalpha.BertBasePath)

        def __copyEmbeddings(embeddings:nn.Embedding,index1,index2):
            #print(embeddings.weight.size())
            weight=embeddings.weight.detach().numpy()

            weight[index2]=deepcopy(weight[index1])

            weight=torch.tensor(weight,requires_grad=True)
            embeddings.weight=nn.Parameter(weight,requires_grad=True)

        __copyEmbeddings(bert.embeddings.word_embeddings,0,1)
        __copyEmbeddings(bert.embeddings.word_embeddings,100,0)

        #self.__iniVocab()


        #tgt embeddings
        transformerEmb=OnmtBertEmbedding(self.tgt_vocab_size,self.hidden_size,self.max_tgt_seq,self.drop_out,self.tgt_padding,bert.embeddings)

        #transformerEmb=BertAsEmbedding(bert,self.tgt_padding)


        #decoder
        transformerDecoder=onmt.decoders.TransformerDecoder(
            num_layers=self.layer_num, d_model=self.hidden_size, heads=self.heads_num, d_ff=self.feed_forwad_size,
                         copy_attn=True, self_attn_type="scaled-dot", dropout=self.drop_out, embeddings=transformerEmb,
                         max_relative_positions=self.max_tgt_seq
        )

        self.transformer=TransformerModel(bert,transformerDecoder)
        self.transformer.generator=onmt.modules.CopyGenerator(
            input_size=self.hidden_size,output_size=self.tgt_vocab_size,pad_idx=self.tgt_padding)



    def loadModel(self,model_file=None,checkpoint=None):
        if checkpoint is None:
            model_dict=torch.load(model_file)
        else:
            model_dict=checkpoint

        weight_dict=model_dict["model"]
        generator_dict=model_dict["generator"]
        #print(weight_dict.keys())
        #print(generator_dict.keys())
        for k in generator_dict:
            weight_dict["generator."+k]=generator_dict[k]

        print("decoder layer num",self.layer_num)
        self.transformer.load_state_dict(weight_dict)

        if model_file:
            logger.info("init model weight with "+model_file)
        else:
            logger.info("init model with checkpoint")
