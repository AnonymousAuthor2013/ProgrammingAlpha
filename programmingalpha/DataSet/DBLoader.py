import pymongo
from urllib.parse import quote_plus
from programmingalpha.DataSet import *
from programmingalpha.DataSet.PostPreprocessing import PreprocessPostContent

class MongoDbConnector(object):
    def __init__(self,host,port,user,passwd):
        url = "mongodb://%s:%s@%s:%d" % (
                quote_plus(user), quote_plus(passwd), host,port)

        self.client=pymongo.MongoClient(url)

    def close(self):
        self.client.close()

    def useDB(self,dbName):
        if dbName is None or dbName not in self.client.list_database_names():
            raise KeyError("{} is not found".format(dbName))


class DocDB(object):
    #for doc retrival interface
    def initData(self):
        raise NotImplementedError

    def setDocCollection(self,collectionName):
        raise NotImplementedError

    def get_doc_text(self,doc_id,**kwargs):
        raise NotImplementedError

    def get_doc_ids(self):
        raise NotImplementedError


class MongoStackExchange(MongoDbConnector,DocDB):
    textExtractor=PreprocessPostContent()
    def __init__(self,host,port,user,passwd):

        MongoDbConnector.__init__(self,host,port,user,passwd)

    def initData(self):
        self.questions=self.stackdb["questions"]
        self.answers=self.stackdb["answers"]
        self.tags=self.stackdb["tags"]
        self.postlinks=self.stackdb["postlinks"]

    def useDB(self,dbName):
        super(MongoStackExchange,self).useDB(dbName)

        self.stackdb=self.client[dbName]
        self.initData()

        return True



    #for doc retrival interface
    def setDocCollection(self,collectionName):
        self.docs=self.stackdb[collectionName]
    def get_doc_text(self,doc_id,chunk_title=int(1e6),chunk_answer=int(1e6)):
        doc_json=self.docs.find_one({"Id":doc_id})
        if doc_json is None:
            print("error found none",self.stackdb.name,self.docs.name,doc_id)
            return None

        doc=self.textExtractor.getPlainTxt(doc_json["question_title"])
        doc="Title=>\n "+doc+" \n"

        if chunk_title>0:
            doc_title=self.textExtractor.getPlainTxt(doc_json["question_body"])
            doc+="Decription=>\n "+doc_title[:chunk_title]+" \n"
        if chunk_answer>0:
            doc_answer=self.textExtractor.getPlainTxt(doc_json["answer"])
            doc+="Answer=>\n "+doc_answer[:chunk_answer]
        return doc

    def get_doc_ids(self):
        doc_ids=[]
        for doc in self.docs.find():
            doc_ids.append(doc["Id"])
        return doc_ids

#stackexchange site methods
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


class MongoWikiDoc(MongoDbConnector,DocDB):
    def __init__(self,host,port,user,passwd):
        MongoDbConnector.__init__(self,host,port,user,passwd)
        self.useDB()

    def useDB(self,dbName="wikidocs"):
        super(MongoWikiDoc, self).useDB(dbName)

        self.wikidb=self.client["wikidocs"]
    def initData(self):
        self.docs=self.wikidb["articles"]
        self.tags=self.wikidb[""]

    def setDocCollection(self,collectionName):
        self.docDB=self.wikidb[collectionName]

    def get_doc_text(self,doc_id,text_chunk=int(1e+6)):
        doc_json=self.docDB.find_one({"Id":doc_id})
        doc="Title=> "+doc_json["Title"]+"\n"
        if text_chunk>0:
            doc+="Text=> "+doc_json["text"][:text_chunk]

        return doc

    def get_doc_ids(self):
        doc_ids=[]
        for doc in self.docDB.find().batch_size(10000):
            doc_ids.append(doc["Id"])
        return doc_ids

def connectToMongoDB(connector_DB:MongoDbConnector.__class__=MongoStackExchange):
    dbauth=MongodbAuth
    db=connector_DB(dbauth["host"], dbauth["port"], dbauth["user"], dbauth["passwd"])
    return db
