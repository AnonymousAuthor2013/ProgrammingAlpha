from . import utils
from ..DataSet.DBLoader import MongoStackExchange
from .tfidf_doc_ranker import TfidfDocRanker
from .bert_doc_ranker import SemanticRanker


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

