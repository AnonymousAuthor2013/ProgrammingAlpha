#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import argparse
import torch
from onmt.utils.logging import init_logger, logger

def get_vocabs(embedding_file,dim):
    vocabs=[]
    embeddings=[]
    index=0
    def parseStr(s):
        try:
            x=float(s)
        except:
            print(s)
            if s=='.':
                x=0.0
            else:
                raise ValueError("cannot parse %s"%s)

        return x

    with open(embedding_file,"r") as f:
        for line in f:
            try:
                fields=line.split()
                word=fields[:len(fields)-dim]
                vocabs.append(' '.join(word))
                embeddings.append(list(map(parseStr,fields[len(fields)-dim:])))
                index+=1
                if index%10000==0:
                    logger.info('loaded {} words'.format(index))
            except:
                print(line)

    logger.info("loaded embeddings from {}, size={}".format(embedding_file,(len(embeddings),len(embeddings[0]))))
    logger.info("vocab: %d words" % len(vocabs))

    return vocabs, embeddings

def main():

    parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
    parser.add_argument('-emb_file', required=True,
                        help="source Embeddings from this file")
    parser.add_argument('-output_file', required=True,
                        help="Output file for the prepared data")
    parser.add_argument('-dict_file', required=True,
                        help="Dictionary file")
    parser.add_argument('-dim',required=True,type=int,help="embedding size")
    opt = parser.parse_args()

    vocabs, embeddings = get_vocabs(opt.emb_file,opt.dim)


    torch.save(embeddings, opt.output_file+".pt")
    torch.save(vocabs,opt.dict_file+".pt")
    with open(opt.dict_file+".txt","w") as f:

        f.writelines(map(lambda s:s+'\n',vocabs))

    logger.info("\nDone.")


if __name__ == "__main__":
    init_logger('embeddings_to_torch.log')
    main()
