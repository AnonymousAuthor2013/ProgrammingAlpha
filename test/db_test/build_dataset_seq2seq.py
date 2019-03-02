from programmingalpha.retrievers import ESEngine
from programmingalpha.Utility.PostPreprocessing import PreprocessPostContent
from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
from programmingalpha.tokenizers.tokenizer import SpacyTokenizer
import os
import json
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

stackSearcher=ESEngine.SearchStackExchange(host='10.1.1.9',port=9200)
wikiSearcher=ESEngine.SearchWiki(host='10.1.1.9',port=9200)
precessor=PreprocessPostContent()
tokenizer=SpacyTokenizer()

docDB=MongoStackExchange(host='10.1.1.9',port=36666)
dbName='datascience'
docDB.useDB(dbName)
batch_size=10

def getPalinTxt(txt):
    txt=precessor.getPlainTxt(txt)
    txt=precessor.filterNILStr(txt)
    txt="\n".join(txt)
    txt=" ".join(tokenizer.tokenize(txt))
    return txt

def constructPairs():
    questions=docDB.questions.find().batch_size(batch_size)
    n_size=questions.count()
    filepath=os.path.join(programmingalpha.DataPath,"seq2seq/train-{}.json".format(dbName))
    cache=[]
    size=5
    count=0
    with open(filepath,"w") as f:
        for q in questions:

            data={"question":getPalinTxt(q["Title"]+"\n"+q["Body"]),"context":[]}

            answer=stackSearcher.getAnswers(q["Id"],dbName)

            if len(answer)>0:
                data["answer"]=getPalinTxt(answer[0]["content"])
            else:
                continue

            sim_qs=stackSearcher.retriveSimilarQuestions(data["question"],dbName,size*2)
            if len(sim_qs)>0:
                sim_qs=sim_qs[1:]
            for sim_q in sim_qs:
                ans=stackSearcher.getAnswers(sim_q["Id"],dbName)
                if len(ans)>0:
                    data["context"].append(getPalinTxt(sim_q["content"]+"\n"+ans[0]["content"]))
                if len(data["context"])>=size:
                    break
            if len(data["context"])>0:
                cache.append(json.dumps(data)+"\n")
            else:
                continue
            if len(cache)%batch_size==0:
                count+=1
                f.writelines(cache)
                cache.clear()
                logger.info("process {}/{}".format(count*batch_size,n_size))

if __name__ == '__main__':
    constructPairs()

