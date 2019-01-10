from tqdm import tqdm
import argparse
from programmingalpha.DataSet.DBLoader import MongoDBConnector
from multiprocessing.dummy import Pool as ThreadPool
from programmingalpha.Utility.DataGenerator import batchGenerator
####################data handler###################


def initPool():
    global db

    db=MongoDBConnector(args.mongodb)

def constructQA(q):

        ans=db.searchForAnswers(q["Id"])

        if ans is None:
            return None

        qa={"Id":q["Id"],
            "question_title":q["Title"],"question_body":q["Body"],
            "answer_body":ans["Body"],
            "Tags":q["Tags"],
            "QuestionDate":q["CreationDate"],"AnswerDate":ans["CreationDate"]}

        return qa

def getAllIds(collection):
    reults=collection.find().batch_size(args.batch_size)
    allIds=set()
    for x in reults:
        allIds.add(x["Id"])

    return allIds


def generateQAPairs():

    workers=ThreadPool(num_workers)

    if collection_name in db.stackdb.list_collection_names():
        QACollection=db.stackdb.get_collection(collection_name)
        currentIds=getAllIds(QACollection)
        print("starting to add to existing {} pairs".format(len(currentIds)))
    else:
        print("create {} and insert qa pair data with batch size={}".format(collection_name,batch_size))
        QACollection=db.stackdb.create_collection(collection_name)
        currentIds=set()

    query={"Tags":{"$in":m_tag}}

    acceptedQuestions=[]
    for batch_q in db.getBatchAcceptedQIds(batch_size=batch_size,query=query):
        acceptedQuestions+=batch_q

    search_needed=[]
    for q in acceptedQuestions:
        if q["Id"] in currentIds:
            continue

        search_needed.append(q)

    print("increasing search for {} questions".format(len(search_needed)))
    qa_pairs=[]

    for batch_q in batchGenerator(search_needed,args.batch_size):

        for qa in workers.map(constructQA,batch_q):

            if qa is not None:
                qa_pairs.append(qa)

            if  len(qa_pairs)>0 and len(qa_pairs)%batch_size==0:

                QACollection.insert_many(qa_pairs)
                qa_pairs.clear()
                print("constructed {} qa pairs".format(QACollection.count()))


    workers.close()
    workers.join()
    if len(qa_pairs)>0:
        QACollection.insert_many(qa_pairs)
    print("finished inserting data({} q/a pair)".format(QACollection.count()))
    QACollection.create_index("Id")

if __name__ == '__main__':
    #settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongodb', type=str, help='mongodb host, e.g. mongodb://10.1.1.9:27017/',default='mongodb://10.1.1.9:27017/')
    parser.add_argument('--batch_size', type=str, default=100)
    parser.add_argument('--collection', type=str, default="QAPForAI")

    parser.add_argument('--num_workers', type=int, default=20,
                        help='Number of threads (for tokenizing, etc)')

    args = parser.parse_args()


    initPool()

    m_tag=["<keras>","<tensorflow>","<caffe>","<pytorch>","<artificial-intelligence>","<nlp>","<computer-vision>",
           "<deep-learning>","<neural-network>","<machine-learning>","<reinforcement-learning>","<scikit-learn>"]

    #m_tag=frozenset(m_tag)

    print("using tags",m_tag)

    batch_size=args.batch_size
    num_workers=args.num_workers
    collection_name=args.collection

    initPool()

    generateQAPairs()

