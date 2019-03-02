from programmingalpha.Utility.PostPreprocessing import PreprocessPostContent
from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
from programmingalpha.tokenizers.tokenizer import CoreNLPTokenizer
import logging
import argparse
import tqdm
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--db', type=str, default="stackoverflow")

    args = parser.parse_args()
    dbName=args.db

    processor=PreprocessPostContent()
    database=MongoStackExchange(host='10.1.1.9',port=50000)
    database.useDB(dbName)

    def generatorDB(data):
        for record in data:
            yield record

    #questions
    f=open(programmingalpha.DataPath+"Corpus/"+dbName.lower()+".txt","w")
    cache=[]

    questions=database.questions.find().batch_size(args.batch_size)
    n_questions=questions.count()
    for _ in tqdm.trange(n_questions,desc="retrieving questions"):

        q=next(questions)
        title=q["Title"]
        body=q["Body"]
        cache.append(processor.getPlainTxt(title+"\n"+body)+"\n")
        if len(cache)>args.batch_size*2:
            f.writelines(cache)
            cache.clear()

    #answers
    answers=database.answers.find().batch_size(args.batch_size)
    n_answers=answers.count()
    for _ in tqdm.trange(n_answers,desc="retrieving answers"):
        ans=next(answers)
        body=ans["Body"]
        cache.append(processor.getPlainTxt(body)+"\n")
        if len(cache)>args.batch_size*2:
            f.writelines(cache)
            cache.clear()
    if len(cache)>0:
        f.writelines(cache)
        cache.clear()

    f.close()
