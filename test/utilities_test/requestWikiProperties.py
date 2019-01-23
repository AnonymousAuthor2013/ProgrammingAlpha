import requests
from programmingalpha.DataSet.DBLoader import MongoWikiDoc,MongodbAuth
from multiprocessing.dummy import Pool as ThreadPool
from programmingalpha.Utility.WebCrawler import AgentProxyCrawler
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

db=MongoWikiDoc(**MongodbAuth)
db.useDB('wikidocs')
db.setDocCollection('articles')

#text=db.get_doc_text(18585770)
#print(text)
#exit(10)
crawler=AgentProxyCrawler()

def requestData(id=None):

    url="https://en.wikipedia.org/w/api.php"
    params={
        "action":"query",
        "format":"json",
        "pageids":id,
        #"titles":"Gradient descent",
        "prop":"categories"
    }

    R=crawler.get(url=url,params=params,timeout=3)

    data=R.json()
    try:
      #  print(data)
        data=data["query"]["pages"][str(id)]
        cats=[]
        for cat in data["categories"]:
     #       print(cat)
            cats.append(cat["title"][9:])
    except KeyError as e:
        #e.with_traceback()

        print("data content",data)
        #exit(1)

    return {"Id":id,"Title":data["title"],"categories":cats}


#cats=requestData(238493);print(cats);exit(10)

wikidoc_ids=db.get_doc_ids()
print("init with {} doc ids".format(len(wikidoc_ids)))
batch_size=10

batch_doc_ids=[wikidoc_ids[i:i+batch_size] for i in range(0,len(wikidoc_ids),batch_size)]

workers=ThreadPool(batch_size)
collection_tag_tmp=db.wikidb["tags_tmp"]
for i in range(len(batch_doc_ids)):
    logger.info(25*"*"+"requesting batches {}/{}".format(i+1,len(batch_doc_ids))+"*"*25)
    batch_results=workers.map(requestData,batch_doc_ids[i])

    collection_tag_tmp.insert_many(batch_results)

workers.close()
workers.close()
