from programmingalpha.Utility.TextPreprocessing import PreprocessPostContent
from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
from programmingalpha.tokenizers.tokenizer import CoreNLPTokenizer
from programmingalpha.Utility.TextPreprocessing import AnswerTextInformationExtractor,QuestionTextInformationExtractor
import numpy as np
import json
import logging
import argparse
import tqdm
import multiprocessing

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


precessor=PreprocessPostContent()


def init(questionsData_G,answersData_G,indexData_G,linksData_G):
    global tokenizer,answerExtractor,questionExtractor
    tokenizer=CoreNLPTokenizer()#SpacyTokenizer()
    answerExtractor=AnswerTextInformationExtractor(args.answerLen,tokenizer)
    questionExtractor=QuestionTextInformationExtractor(args.questionLen,tokenizer)
    global questionsData,answersData,indexData,linksData
    questionsData=questionsData_G.copy()
    answersData=answersData_G.copy()
    indexData=indexData_G.copy()
    linksData=linksData_G.copy()
    logger.info("process {} init".format(multiprocessing.current_process()))

def filerLowQualityQuestion(question):
    if ("AcceptedAnswerId" not in question or not question["AcceptedAnswerId"]) and \
                ("AnswerCount" not in question or question["AnswerCount"]<10) and \
                ("FavoriteCount" not in question or question["FavoriteCount"]):
        return False

    return True

def fetchLinkData():
    links=docDB.stackdb.get_collection("postlinks")

    allLinks=list(links.find().batch_size(args.batch_size))

    myG={}
    for link in tqdm.tqdm(allLinks,desc="loading links"):
        id_a,id_b=link["PostId"],link["RelatedPostId"]
        r=link["LinkTypeId"]
        if r==3:
            w=0
        elif r==1:
            w=1
        else:
            raise ValueError("unexpected value {} for link type".format(r))

        if id_a in myG:
            myG[id_a][id_b]=w
        else:
            myG[id_a]={id_b:w}

        if id_b in myG:
            myG[id_b][id_a]=w
        else:
            myG[id_b]={id_a:w}

    logger.info("finished loading {} sublinks".format(len(allLinks)))

    return myG

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
        if filerLowQualityQuestion(question)==False:
            continue

        Id=question["Id"]
        del question["_id"]
        questionsData[Id]=question

    logger.info("loaded: questions({})".format(len(questionsData)))

    return questionsData

def fetchQAIndex():
    indexData={}

    db=MongoStackExchange(host='10.1.1.9',port=50000)
    db.useDB(args.db)
    indexes=db.stackdb.get_collection("QAIndexer")
    for index in tqdm.tqdm(indexes.find().batch_size(args.batch_size),desc="loading indexes"):

        Id=index["Id"]
        del index["_id"]
        indexData[Id]=index

    logger.info("loaded: indexes({})".format(len(indexData)))

    return indexData


#generate Core
def _getBestAnswer(q_id):
    ans_id=None
    if "AcceptedAnswerId" in questionsData[q_id]:
        ans_id=questionsData[q_id]["AcceptedAnswerId"]
    else:
        score=-100000
        for can_id in indexData[q_id]["Answers"]:
            if answersData[can_id]["Score"]>score:
                ans_id=can_id
                score=answersData[can_id]["Score"]

    return ans_id

def _genCore(distances):

    q_id=distances["id"]

    #get question
    if q_id not in questionsData:
        return None

    question=questionExtractor.clipText(questionsData[q_id])

    #get answer
    ans_id=_getBestAnswer(q_id)

    if ans_id is  None or ans_id not in answersData:
        return None

    answerExtractor.keepdOrder=True
    answer=answerExtractor.clipText(answersData[ans_id]["Body"])
    answerExtractor.keepdOrder=False

    #get context
    relative_q_ids=[]
    if q_id in linksData:
        dists=linksData[q_id]
    else:
        dists=distances["distances"]

    for id in dists:
        if len(relative_q_ids)>=5:
            break
        if dists[id]==1:
            relative_q_ids.append(id)
        elif dists[id]==0:
            relative_q_ids.insert(0,id)
        else:
            pass

    if len(relative_q_ids)==0:
        return None

    answers=[]
    for q_id in relative_q_ids:
        if q_id not in questionsData:
            continue
        ans_idx=indexData[q_id]

        if ans_idx["AcceptedAnswerId"] != -1:
            ans_id=ans_idx["AcceptedAnswerId"]
        else:
            ans_id=_getBestAnswer(q_id)

        if ans_id is not None and ans_id in answersData:
            answers.append(answersData[ans_id])

    context=[]
    count=0
    for ans in answers:
        ans_txt=answerExtractor.clipText(ans["Body"])
        context.extend(ans_txt)

        count+=sum(map(lambda s:len(s),ans_txt))
        if count>args.contextLen:
            break

    if len(context)==0:
        return None

    record={"question":question,"context":context,"answer":answer}

    return record

def generateContextAnswerCorpus(distanceData):

    cache=[]
    batch_size=args.batch_size
    batches=[distanceData[i:i+batch_size] for i in range(0,len(distanceData),batch_size)]

    workers=multiprocessing.Pool(args.workers,initializer=init,
                                 initargs=(questionsDataGlobal,answersDataGlobal,indexDataGlobal,linksDataGlobal))



    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-seq2seq.json","w") as f:
        for batch_links in tqdm.tqdm(batches,desc="processing documents"):
            for record in workers.map(_genCore,batch_links):
                if record is not None:

                    cache.append(json.dumps(record)+"\n")

            f.writelines(cache)
            cache.clear()

        workers.close()
        workers.join()


def main():
    logger.info("loading distance data")
    distance_file=programmingalpha.DataPath+"linkData/"+dbName.lower()+'-2graph.json'
    distance_data=[]
    nodes=[]
    with open(distance_file,"r") as f:
        for line in f:
            path=json.loads(line)
            distance_data.append(path)
            nodes.append(path["id"])

    np.random.shuffle(distance_data)

    generateContextAnswerCorpus(distance_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--db', type=str, default="stackoverflow")
    parser.add_argument('--questionLen', type=int, default=250)
    parser.add_argument('--answerLen', type=int, default=300)
    parser.add_argument('--contextLen', type=int, default=600)

    parser.add_argument('--workers', type=int, default=25)

    args = parser.parse_args()

    docDB=MongoStackExchange(host='10.1.1.9',port=50000)
    dbName=args.db
    docDB.useDB(dbName)

    questionsDataGlobal=fetchQuestionData()
    answersDataGlobal=fetchAnswerData()
    indexDataGlobal=fetchQAIndex()
    linksDataGlobal=fetchLinkData()

    main()
