from . import utils
import os
from ..DataSet.DBLoader import MongoStackExchange
from .tfidf_doc_ranker import TfidfDocRanker
from .semanticRanker import SemanticRanker

WorkingDocCollection="QAPForAI"

def getTF_IDF_Data(data_source,ngram=2,hash_size=16777216,tokenizer_name='bert'):
    dataName='tf_idf_hash/{}-docs-tfidf-ngram={}-hash={}-tokenizer={}.npz'.format(data_source,ngram,hash_size,tokenizer_name)
    return dataName


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'mongoDB':
        return MongoStackExchange
    if name=='semantic':
        return SemanticRanker
    raise RuntimeError('Invalid retriever class: %s' % name)

