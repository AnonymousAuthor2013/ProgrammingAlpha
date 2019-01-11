import argparse
from programmingalpha.DataSet.DBLoader import MongoDBConnector
from programmingalpha.Utility.CorrelationAnalysis import Aprio

class TagCounter(object):
    def __init__(self,mongodb:MongoDBConnector,dbName:str):
        self.mongoDB=mongodb
        self.mongoDB.useDB(dbName)
        self.transactions=[]
        questions=self.mongoDB.questions.find().batch_size(args.batch_size)

        for q in questions:
            self.transactions.append(q["Tags"])
            if len(self.transactions)%args.batch_size==0:
                print("scanned %d questions"%(len(self.transactions)))

            if len(self.transactions)>=tuneMaxClipNum:
                break

        print("scanned %d questions"%(len(self.transactions)))

        self.transactions=list(map(frozenset,self.transactions))

    def getRelatedTags(self,itemSeeds):

        apriori=Aprio()
        apriori.maxK=2
        apriori.minSupport=0
        apriori.minConfidence=0.7


        results=apriori.stepSearch(self.transactions,itemSeeds,2)

        if results is None:
            return None


        _,_,confData=results

        return confData


    def getTagCounter(self,relatedTags):
        tag_collection=self.mongoDB.tags
        counter={}

        for tagName in relatedTags:
            tagName=tagName[1:][:-1]
            tag=tag_collection.find_one({"TagName":tagName})

            if tag:
                count=tag["Count"]
            else:
                #print("not found",tag)
                #exit(10)
                continue

            #print(tagName,count)

            counter[tagName]=count

        return counter

if __name__ == '__main__':

    #settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongodb', type=str, help='mongodb host, e.g. mongodb://10.1.1.9:27017/',default='mongodb://10.1.1.9:27017/')
    parser.add_argument('--batch_size', type=str, default=10000)
    parser.add_argument('--collection', type=str, default="QAPForAI")

    parser.add_argument('--num_workers', type=int, default=20,
                        help='Number of threads (for tokenizing, etc)')

    args = parser.parse_args()

    m_tag=["<keras>","<tensorflow>","<caffe>","<pytorch>","<artificial-intelligence>","<nlp>","<computer-vision>",
           "<deep-learning>","<neural-network>","<machine-learning>","<reinforcement-learning>","<scikit-learn>"]

    tuneMaxClipNum=10000

    itemSeeds=[]
    for tag in m_tag:
        itemSeeds.append({tag})
    itemSeeds=frozenset(map(frozenset,itemSeeds))

    mongodb=MongoDBConnector(args.mongodb)
    dbname="stackoverflow"
    tagCounter=TagCounter(mongodb,dbname)

    confData=tagCounter.getRelatedTags(itemSeeds)

    print("related tags from seeds={}".format(itemSeeds))
    for k in confData:
        confidence=confData[k]
        if confidence>0.7:
            print(k, confidence)

    relatedTags=map(list,confData.keys())

    tagIndex=lambda tag: 1 if tag[0] in m_tag else 0
    relatedTags=map(lambda x:x[tagIndex(x)],relatedTags)
    relatedTags=set(relatedTags)
    print(relatedTags)
    counter=tagCounter.getTagCounter(relatedTags)

    print("tags counter info")
    for k in counter.keys():
        print(k, counter[k])

