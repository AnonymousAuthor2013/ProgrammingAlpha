import argparse
import collections
from programmingalpha.DataSet.DBLoader import MongoDBConnector
import programmingalpha
from pytorch_pretrained_bert import BertForMaskedLM,BertModel,BertTokenizer

#load pretrained model
tokenizer=BertTokenizer.from_pretrained(programmingalpha.BertBasePath,do_lower_case=True)

s = "I am  a strong and powerful man! I love puppeteer."
print("model loaded")
print(tokenizer.tokenize(s))
#exit(10)

if __name__ == '__main__':

    #settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongodb', type=str, help='mongodb host, e.g. mongodb://10.1.1.9:27017/',default='mongodb://10.1.1.9:27017/')
    parser.add_argument('--collection', type=str, default="QAPForAI")
    parser.add_argument('--batch_size', type=str, default=10000)

    args = parser.parse_args()

    db=MongoDBConnector(args.mongodb)

    qaAI=db.stackdb.get_collection("QAPForAI")

    allIDs=set()
    records=qaAI.find().batch_size(args.batch_size)

    for x in records:
        allIDs.add(x["Id"])

    print("finished gathering {} q/a pair ids".format(len(allIDs)))

    links=db.stackdb.get_collection("postlinks")

    allLinks=links.find().batch_size(args.batch_size)

    subGraph=[]
    graphDict=collections.OrderedDict()
    count=0
    conflicts=0
    for link in allLinks:
        count+=1
        if count%args.batch_size==0:
            print("scanned %d links "%(count))

        #make sure id_a<id_b
        id_a,id_b=link["PostId"],link["RelatedPostId"]
        if id_a> id_b:
            tmp=id_b
            id_b=id_a
            id_a=tmp

        r=link["LinkTypeId"]

        #concerns only those in required Ids
        if id_a in allIDs and id_b in allIDs:

            subGraph.append((id_a,id_b,r))

            if (id_a,id_b) in graphDict and graphDict[(id_a,id_b)]!=r:

                #print("conflict",id_a,id_b,r,graphDict[(id_a,id_b)])

                conflicts+=1

                if r==3:
                    graphDict[(id_a,id_b)]=r

            else:
                graphDict[(id_a,id_b)]=r




    print("finished finding {} sublinks".format(len(subGraph)))
    print("finished finding {} sublinks".format(len(graphDict)))
    print("finished with {} conflicts".format(conflicts))

    with open("testdata/graphDict.pkl","wb") as f:
        import pickle
        pickle.dump(graphDict,f)

