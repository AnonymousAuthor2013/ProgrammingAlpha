import logging,argparse
import programmingalpha
import multiprocessing
from programmingalpha.Utility.TextPreprocessing import PreprocessPostContent
from programmingalpha.DataSet.DBLoader import MongoStackExchange
import tqdm

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def init(questionsData_G,answersData_G,indexData_G,copy=True):
    global preprocessor
    preprocessor=PreprocessPostContent()
    global questionsData,answersData,indexData,linksData
    if copy:
        questionsData=questionsData_G.copy()
        answersData=answersData_G.copy()
        indexData=indexData_G.copy()
    else:
        questionsData=questionsData_G
        answersData=answersData_G
        indexData=indexData_G

    logger.info("process {} init".format(multiprocessing.current_process()))

def fetchAnswerData():
    answersData={}

    for ans in tqdm.tqdm(docDB.answers.find().batch_size(args.batch_size),desc="loading answers"):

        Id=ans["Id"]
        del ans["_id"]
        answersData[Id]=ans

    logger.info("loaded: answers({})".format(len(answersData)))

    return answersData

def fetchQuestionData():
    questionsData={}

    for question in tqdm.tqdm(docDB.questions.find().batch_size(args.batch_size),desc="loading questions"):
        Id=question["Id"]
        del question["_id"]
        questionsData[Id]=question

    logger.info("loaded: questions({})".format(len(questionsData)))

    return questionsData

def fetchQAIndex():
    indexData={}

    indexes=docDB.stackdb.get_collection("QAIndexer")
    for index in tqdm.tqdm(indexes.find().batch_size(args.batch_size),desc="loading indexes"):

        Id=index["Id"]
        del index["_id"]
        indexData[Id]=index

    logger.info("loaded: indexes({})".format(len(indexData)))

    return indexData

def genCorpus(q_id):
    answer_ids=indexDataGlobal[q_id]["Answers"]
    answers=[]
    for ans_id in answer_ids:
        ans_paragraphs=preprocessor.getPlainTxt( answersDataGlobal[ans_id]["Body"])
        answers.append(" ".join(ans_paragraphs))

    q=questionsDataGlobal[q_id]
    question_paragraphs=(preprocessor.getPlainTxt(q["Body"])+preprocessor.getPlainTxt(q["Title"]))
    question=" ".join(question_paragraphs)
    #print(answers)
    #print(question_paragraphs)

    text="\n".join([question]+answers)+"\n"

    return text

def generateKnowledgeCorpus():
    q_ids=questionsDataGlobal.keys()
    init(questionsDataGlobal,answersDataGlobal,indexDataGlobal,copy=False)
    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-unit.json","w") as f:
        cache=[]
        for q_id in tqdm.tqdm(q_ids,desc="generating corpus data"):
            raw_text=genCorpus(q_id)
            cache.append(raw_text)
            if len(cache)>args.batch_size:
                f.writelines(cache)
                cache.clear()
        if len(cache)>0:
            f.writelines(cache)

def generateKnowledgeCorpusParallel():
    q_ids=list(questionsDataGlobal.keys())
    batches=[q_ids[i:i+args.batch_size] for i in range(0,len(q_ids),args.batch_size)]
    workers=multiprocessing.Pool(args.workers,initializer=init,initargs=(questionsDataGlobal,answersDataGlobal,indexDataGlobal))

    with open(programmingalpha.DataPath+"Corpus/"+args.db.lower()+"-unit.json","w") as f:
        cache=[]
        for batch in tqdm.tqdm(batches,desc="generating corpus data in parallel"):
             for raw_text in workers.map(genCorpus,batch):
                cache.append(raw_text)

             f.writelines(cache)
             cache.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--db', type=str, default="stackoverflow")

    args = parser.parse_args()

    docDB=MongoStackExchange(host='10.1.1.9',port=50000)
    dbName=args.db
    docDB.useDB(dbName)

    questionsDataGlobal=fetchQuestionData()
    answersDataGlobal=fetchAnswerData()
    indexDataGlobal=fetchQAIndex()

    if len(questionsDataGlobal)<args.batch_size*args.workers and args.workers==1:
        generateKnowledgeCorpus()
    else:
        generateKnowledgeCorpusParallel()

