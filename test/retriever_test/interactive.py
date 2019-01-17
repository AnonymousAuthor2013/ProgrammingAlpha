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
from programmingalpha.DataSet.DBLoader import connectToDB

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
args = parser.parse_args()

logger.info('Initializing ranker...')


dbName='AI'
ranker = retrievers.get_class('tfidf')(dbName)


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------
docDB=connectToDB()
docDB.useDB(dbName)
docDB.setDocCollection(retrievers.WorkingDocCollection)

def process(query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score','Doc']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i], docDB.get_doc_text(doc_names[i]) ])
    print(table)


banner = """
Interactive TF-IDF DrQA Retriever
>> process(question, k=1)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
