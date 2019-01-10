import argparse
from programmingalpha.DataSet.DBLoader import MongoDBConnector
from multiprocessing.dummy import Pool as ThreadPool

class TagCounter(object):
    def __init__(self,mongodb:MongoDBConnector,dbName:str):
        self.mongoDB=mongodb
        self.mongoDB.useDB(dbName)

    def getRelatedTags(self):

        questions = self.mongoDB.questions.find({"Tags":{"$in":m_tag}}).batch_size(args.batch_size)

        relatedTags=set()

        count=0
        for q in questions:
            for tag in q["Tags"]:
                relatedTags.add(tag)
            count+=1
            if count%args.batch_size==0:
                print("scanned %d questions, current tags set size=%d"%(count,len(relatedTags)))

        print("scanned %d questions, current tags set size=%d"%(count,len(relatedTags)))

        return relatedTags

    def getTagCounter(self):
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

    mongodb=MongoDBConnector(args.mongodb)
    dbname="stackoverflow"
    tagCounter=TagCounter(mongodb,dbname)

    relatedTags=tagCounter.getRelatedTags()
    #print(None in relatedTags)
    #exit(10)
    counter=tagCounter.getTagCounter()

    saveF=[]
    with open("testdata/counterTags.txt","w") as f:
        for k in counter.keys():
            v=counter[k]
            if v>1000:
                print(k,v)
            saveF.append("%s,%d\n"%(k,v))
        f.writelines(saveF)

