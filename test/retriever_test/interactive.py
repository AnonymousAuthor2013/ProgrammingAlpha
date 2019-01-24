#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf DrQA retriever module."""

import argparse
import code
import prettytable
import logging
from programmingalpha import retrievers
from programmingalpha.DataSet import DBLoader
import heapq
import programmingalpha

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
args = parser.parse_args()

logger.info('Initializing ranker...')


rankers={}
rankers['stackoverflow']=retrievers.get_class('tfidf')('stackoverflow')
rankers['AI']=retrievers.get_class('tfidf')('AI')
rankers['datascience']=retrievers.get_class('tfidf')('datascience')
rankers['crossvalidated']=retrievers.get_class('tfidf')('crossvalidated')
KBSource={'stackoverflow','datascience','crossvalidated','AI'}
# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------
docDB=DBLoader.MongoStackExchange(**DBLoader.MongodbAuth)

sranker=retrievers.get_class('semantic')(programmingalpha.ModelPath+"/pytorch_model.bin")

def process(query, k=5):
    results=[]
    docs=[]
    for dbName in KBSource:
        ranker=rankers[dbName]
        doc_names, doc_scores = ranker.closest_docs(query, k)
        #print("found {}/{} in {}".format(len(doc_names),k,dbName))
        for i in range(len(doc_names)):
            results.append(
                {"Id":doc_names[i],
                 "score":doc_scores[i],
                 "db":dbName}
            )
            docs.append(
                {"Id":doc_names[i]+"|||"+dbName,
                 "text":docDB.get_doc_text(doc_names[i],chunk_answer=0)
                 }
                 )


    results=heapq.nlargest(k,key=lambda doc:doc[1],iterable=results)

    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score','Doc']
    )
    for i in range(len(results)):
        #print([ i + 1, results[i][0], '%.5g' % results[i][1], results[i][2] ])
        docDB.useDB(results[i][2])
        docDB.setDocCollection(retrievers.WorkingDocCollection)
        table.add_row([ i + 1, "{}-{}".format(results[i]["Id"],results[i]["db"]),
                        '%.5g' % results[i]["score"], docDB.get_doc_text(results[i]["Id"],0,0) ])
    print(table)


    logger.info("using semantic ranker to resort {} entries".format(len(docs)))
    sresults=sranker.closest_docs(query,docs,k)

    table.clear()
    for i in range(len(sresults)):
        r=sresults[i]
        doc_id,dbName=r[0].split("|||")
        doc_id=int(doc_id)
        score=r[1]

        #print([ i + 1, results[i][0], '%.5g' % results[i][1], results[i][2] ])

        docDB.useDB(dbName)
        docDB.setDocCollection(retrievers.WorkingDocCollection)
        table.add_row([ i + 1, r, '%.5g' % score, docDB.get_doc_text(doc_id,0,0) ])

    print(table)

banner = """
Interactive TF-IDF Programming Alpha For AI Retriever
>> process(question, k=1)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
