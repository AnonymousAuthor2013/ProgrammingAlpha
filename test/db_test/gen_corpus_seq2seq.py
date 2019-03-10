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


def init(questionsData_G,answersData_G,linksData_G,copy=True):
    global tokenizer,answerExtractor,questionExtractor
    tokenizer=CoreNLPTokenizer()#SpacyTokenizer()
    answerExtractor=AnswerTextInformationExtractor(args.answerLen,tokenizer)
    questionExtractor=QuestionTextInformationExtractor(args.questionLen,tokenizer)
    global questionsData,answersData,indexData,linksData
    if copy:
        questionsData=questionsData_G.copy()
        answersData=answersData_G.copy()
        linksData=linksData_G.copy()
    else:
        questionsData=questionsData_G
        answersData=answersData_G
        linksData=linksData_G

    logger.info("process {} init".format(multiprocessing.current_process()))

def filerLowQualityQuestion(question,mode=1):
    cond1=lambda q:("AcceptedAnswerId" not in q or not q["AcceptedAnswerId"])
    cond2=lambda q:("AnswerCount" not in q or q["AnswerCount"]<10)
    cond3=lambda q:("FavoriteCount" not in q or q["FavoriteCount"])

    if  mode==1 and cond1(question) :
        'accepted answer'
        return False

    if mode==2 and ( cond2(question) or cond3(question) ):
        'high quality question only'
        return False

    return True



def fetchQuestionData():
    questionsData={}

    needed_answerIds=set()

    query={"AcceptedAnswerId":{"$exists":True,"$ne":''}}

    for question in tqdm.tqdm(docDB.questions.find(query).batch_size(args.batch_size),desc="loading questions"):
        if filerLowQualityQuestion(question,mode=1)==False:
            continue

        Id=question["Id"]
        del question["_id"]
        questionsData[Id]={"Title":question["Title"],"Body":question["Body"],"AcceptedAnswerId":question["AcceptedAnswerId"]}

        needed_answerIds.add(questionsData["AcceptedAnswerId"])

    logger.info("loaded: questions({})".format(len(questionsData)))

    return questionsData, needed_answerIds

def fetchAnswerData():
    answersData={}

    for ans in tqdm.tqdm(docDB.answers.find().batch_size(args.batch_size),desc="loading answers"):

        Id=ans["Id"]

        if  Id not in ansIdxGlobal or ans["ParentId"] not in questionsDataGlobal:
            continue


        answersData[Id]={"Body":ans["Body"],"Score":ans["Score"]}

    logger.info("loaded: answers({})".format(len(answersData)))

    return answersData

def fetchLinkData():
    links=docDB.stackdb.get_collection("postlinks")

    allLinks=list(links.find().batch_size(args.batch_size))

    myG={}
    for link in tqdm.tqdm(allLinks,desc="loading links"):
        id_a,id_b=link["PostId"],link["RelatedPostId"]

        if id_a not in questionsDataGlobal or id_b not in questionsDataGlobal:
            continue

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

#generate Core
def _getBestAnswer(q_id):
    try:
        ans_id=questionsData[q_id]["AcceptedAnswerId"]
        answer=answersData[ans_id]
    except:
        return None

    return answer

def _genCore(distances):
    try:
        q_id=distances["id"]

        #get question
        if q_id not in questionsData:
            return None

        question=questionExtractor.clipText(questionsData[q_id])

        #get answer
        answer=_getBestAnswer(q_id)

        if answer is  None:
            return None

        answerExtractor.keepdOrder=True
        answer=answerExtractor.clipText(answer["Body"])
        answerExtractor.keepdOrder=False

        #get context
        relative_q_ids=[]
        if q_id in linksData:
            dists=linksData[q_id]
        else:
            dists=distances["distances"]

        for id in dists:
            if id not in questionsData:
                continue

            if len(relative_q_ids)>=5:
                break

            if dists[id]==1:
                relative_q_ids.append(id)
            elif dists[id]==0:
                relative_q_ids.insert(0,id)
            elif dists[id]==2:
                if filerLowQualityQuestion(questionsData[id],mode=2)==True:
                    relative_q_ids.append(id)
            else:
                pass

        if len(relative_q_ids)==0:
            return None



        context=[]
        count=0
        for q_id in relative_q_ids:
            if q_id not in questionsData:
                continue

            ans=_getBestAnswer(q_id)
            if ans is None:
                continue

            ans_txt=answerExtractor.clipText(ans["Body"])
            context.extend(ans_txt)

            count+=sum(map(lambda s:len(s.split()),ans_txt))
            if count>args.contextLen:
                break

        if len(context)==0:
            return None

        record={"question":question,"context":context,"answer":answer}

        return record
    except :
        logger.warning("except triggered for distance data: {}".format(distances))
        return None


def generateContextAnswerCorpusParallel(distanceData):

    cache=[]
    batch_size=args.batch_size
    batches=[distanceData[i:i+batch_size] for i in range(0,len(distanceData),batch_size)]

    workers=multiprocessing.Pool(args.workers,initializer=init,
                                 initargs=(questionsDataGlobal,answersDataGlobal,linksDataGlobal))

    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-seq2seq.json","w") as f:
        for batch_links in tqdm.tqdm(batches,desc="processing documents"):

            for record in workers.map(_genCore,batch_links):
                if record is not None:

                    cache.append(json.dumps(record)+"\n")

            f.writelines(cache)
            cache.clear()


        workers.close()
        workers.join()

def generateContextAnswerCorpus(distanceData):

    cache=[]

    init(questionsDataGlobal,answersDataGlobal,linksDataGlobal,copy=False)

    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-seq2seq.json","w") as f:
        for link in tqdm.tqdm(distanceData,desc="processing documents"):
            record =_genCore(link)
            if record is not None:
                cache.append(json.dumps(record)+"\n")

            if len(cache)>args.batch_size:
                f.writelines(cache)
                cache.clear()

        if len(cache)>0:
            f.writelines(cache)
            cache.clear()

def main():
    logger.info("loading distance data")
    distance_file=programmingalpha.DataPath+"linkData/"+dbName.lower()+'-2graph.json'
    distance_data=[]
    with open(distance_file,"r") as f:
        for line in f:
            path=json.loads(line)
            if path["id"] not in questionsDataGlobal:
                continue
            distance_data.append(path)
    logger.info("loaded {} links data".format(len(distance_data)))

    #generateContextAnswerCorpus(distance_data)
    generateContextAnswerCorpusParallel(distance_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--db', type=str, default="stackoverflow")
    parser.add_argument('--questionLen', type=int, default=250)
    parser.add_argument('--answerLen', type=int, default=300)
    parser.add_argument('--contextLen', type=int, default=600)

    parser.add_argument('--workers', type=int, default=5)

    args = parser.parse_args()

    docDB=MongoStackExchange(host='10.1.1.9',port=50000)
    dbName=args.db
    docDB.useDB(dbName)

    questionsDataGlobal, ansIdxGlobal=fetchQuestionData()
    answersDataGlobal=fetchAnswerData()
    linksDataGlobal=fetchLinkData()

    main()
