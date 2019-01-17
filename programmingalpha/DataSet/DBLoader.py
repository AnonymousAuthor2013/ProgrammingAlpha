import pymongo
from urllib.parse import quote_plus
from programmingalpha.DataSet import *

class MongoDBConnector(object):

    def __init__(self,host,port,user,passwd):
        url = "mongodb://%s:%s@%s:%d" % (
                quote_plus(user), quote_plus(passwd), host,port)

        self.client=pymongo.MongoClient(url)
        self.stackdb=self.client["stackoverflow"]
        self.initData()


    def initData(self):
        self.questions=self.stackdb["questions"]
        self.answers=self.stackdb["answers"]
        self.tags=self.stackdb["tags"]
        self.postlinks=self.stackdb["postlinks"]

    def useDB(self,dbName):
        if dbName is None or dbName not in self.client.list_database_names():
            return False

        self.stackdb=self.client[dbName]
        self.initData()

        return True

    def close(self):
        self.client.close()

    #for doc retrival interface
    def setDocCollection(self,collectionName):
        self.docs=self.stackdb[collectionName]
    def get_doc_text(self,doc_id):
        doc_json=self.docs.find_one({"Id":doc_id})
        return 'Title=>\n{}\n Body=>\n{}'.format(doc_json["question_title"],doc_json["question_body"])
    def get_doc_ids(self):
        doc_ids=[]
        for doc in self.docs.find():
            doc_ids.append(doc["Id"])
        return doc_ids


    def getPopTags(self,ratio=0.1,need_num=2000):
        all_tags=self.tags.find().sort("Count",pymongo.DESCENDING)
        pop_tags=[]
        need_num=min(ratio*all_tags.count(),need_num)
        i=0
        for x in all_tags:
            pop_tags.append(u"<"+x["TagName"]+u">")
            i+=1
            if i>=need_num:
                break
        return set(pop_tags)


    def getBatchAcceptedQIds(self,batch_size=1000,query=None):
        # generator for question ids with accpted answers
        batch=[]

        if not query:
            results=self.questions.find().batch_size(batch_size)
        else:
            results=self.questions.find(query).batch_size(batch_size)

        for x in results:
            #print(x)
            if 'AcceptedAnswerId' not in x or x["AcceptedAnswerId"]=='':
                continue

            batch.append(x)
            if len(batch)%batch_size==0:
                yield batch
                batch.clear()

        if len(batch)>0:
            yield batch

    def searchForAnswers(self,question_id):
        ans=self.answers.find_one({"ParentId":question_id})
        return ans

def connectToDB():
    dbauth=MongodbAuth
    db=MongoDBConnector(dbauth["host"],dbauth["port"],dbauth["user"],dbauth["passwd"])
    return db
