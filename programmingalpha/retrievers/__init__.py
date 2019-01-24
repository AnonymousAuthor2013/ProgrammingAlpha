from . import utils
import os
from ..DataSet.DBLoader import MongoStackExchange
from .tfidf_doc_ranker import TfidfDocRanker
from .semantic_ranker import SemanticRanker

WorkingDocCollection="QAPForAI"


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

