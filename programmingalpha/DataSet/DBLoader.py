import pymongo

class MongoDBConnector(object):

    def __init__(self,url):
        self.client=pymongo.MongoClient(url)
        self.stackdb=self.client["stackoverflow"]

        self.questions=self.stackdb["questions"]
        self.answers=self.stackdb["answers"]
        self.tags=self.stackdb["tags"]
        self.postlinks=self.stackdb["postlinks"]


    def useDB(self,dbName):
        if dbName is None or dbName not in self.client.list_database_names():
            return False

        self.stackdb=self.client[dbName]

        self.questions=self.stackdb["questions"]
        self.answers=self.stackdb["answers"]
        self.tags=self.stackdb["tags"]
        self.postlinks=self.stackdb["postlinks"]

        return True

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
            if x["AcceptedAnswerId"]=='':
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

