from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
import logging
import argparse
import tqdm
import json
import multiprocessing
from programmingalpha.tokenizers import CoreNLPTokenizer
from programmingalpha.Utility.TextPreprocessing import QuestionTextInformationExtractor

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)

def init(questionsData_G):
    global tokenizer,textExtractor
    tokenizer=CoreNLPTokenizer()#SpacyTokenizer()
    textExtractor=QuestionTextInformationExtractor(args.maxLength,tokenizer)
    global questionsData
    questionsData=questionsData_G.copy()
    logger.info("process {} init".format(multiprocessing.current_process()))

def fetchAnswerData():
    answersData={}

    db=MongoStackExchange(host='10.1.1.9',port=50000)
    db.useDB(args.db)
    for ans in tqdm.tqdm(db.answers.find().batch_size(args.batch_size),desc="loading answers"):

        Id=ans["Id"]
        del ans["_id"]
        answersData[Id]=ans

    logger.info("loaded: answers({})".format(len(answersData)))

    return answersData

def fetchQuestionData():
    questionsData={}

    db=MongoStackExchange(host='10.1.1.9',port=50000)
    db.useDB(args.db)
    for question in tqdm.tqdm(db.questions.find().batch_size(args.batch_size),desc="loading questions"):

        Id=question["Id"]
        del question["_id"]
        questionsData[Id]=question

    logger.info("loaded: questions({})".format(len(questionsData)))

    return questionsData


def _genCore(link):

    label=link["label"]
    q1,q2=link["pair"]

    if not (q1 in questionsData and q2 in questionsData):
        #if label=='duplicate':
        #    print(link,q1 in questionsData, q2 in questionsData)
        return None


    q1=textExtractor.clipText(questionsData[q1])
    q2=textExtractor.clipText(questionsData[q2])

    record={"q1":q1,"q2":q2,"label":label}

    return record
    #return json.dumps(record)+"\n"

def generateQuestionCorpus(labelData):

    cache=[]
    batch_size=args.batch_size
    batches=[labelData[i:i+batch_size] for i in range(0,len(labelData),batch_size)]

    workers=multiprocessing.Pool(args.workers,initializer=init,initargs=(questionsDataGlobal,))

    counter={'unrelated': 0, 'direct': 0, 'transitive': 0, 'duplicate': 0}

    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+".json","w") as f:
        for batch_labels in tqdm.tqdm(batches,desc="processing documents"):
            for record in workers.map(_genCore,batch_labels):
                if record is not None:
                    counter[record["label"]]+=1
                    #cache.append(record)
                    cache.append(json.dumps(record)+"\n")

            f.writelines(cache)
            cache.clear()

        workers.close()
        workers.join()

    logger.info("after extratcing informatiev paragraphs: {}".format(counter))

def main():

    labelData=[]
    with open(programmingalpha.DataPath+"/linkData/"+args.db.lower()+"-labelPair.json","r") as f:
        for line in f:
            labelData.append(json.loads(line))

    labels=map(lambda ll:ll["label"],labelData)
    import collections
    logger.info(collections.Counter(labels))

    generateQuestionCorpus(labelData)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--db', type=str, default="stackoverflow")
    parser.add_argument('--maxLength', type=int, default=250)
    parser.add_argument('--workers', type=int, default=5)

    args = parser.parse_args()

    questionsDataGlobal=fetchQuestionData()
    #answersData=fetchAnswerData()

    main()
