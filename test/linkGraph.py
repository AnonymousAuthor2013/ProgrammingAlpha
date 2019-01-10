import networkx as nx
import pickle
import collections
import argparse
from programmingalpha.DataSet.DBLoader import MongoDBConnector
import numpy as np


def createLinkDict():

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
        #if count%args.batch_size==0:
        #    print("scanned %d links "%(count))

        #make sure id_a<id_b
        id_a,id_b=link["PostId"],link["RelatedPostId"]
        if id_a> id_b:
            tmp=id_b
            id_b=id_a
            id_a=tmp

        r=link["LinkTypeId"]

        #concerns only those in required Ids
        if id_a in allIDs or id_b in allIDs:

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

    return graphDict


def buildGraph(graphDict:collections.OrderedDict):
    id_to_index={}
    index_to_id={}
    graphIds=set()
    for k in graphDict.keys():
        graphIds.add(k[0])
        graphIds.add(k[1])
    for k in graphIds:
        index=len(id_to_index)
        index_to_id[index]=k
        id_to_index[k]=index
    G=nx.Graph()
    edges=[]

    for e in graphDict.keys():
        w=graphDict[e]

        if w==3:
            w=0

        u=id_to_index[e[0]]
        v=id_to_index[e[1]]
        edges.append((u,v,w))
        #edges.append((v,u,w))

    G.add_weighted_edges_from(edges)

    print(len(list(G.edges)),len(list(G.nodes)))
    print(len(graphDict),len(graphIds))

    return G,id_to_index,index_to_id

def statistics():

    data=np.load("testdata/shortestLinkPath.npz")
    nodelist,distance=data["nodelist"],data["distance"]

    print(np.max(distance),np.min(distance),np.mean(distance),np.var(distance))

    #np.savez("testdata/shortestLinkPath",nodelist=nodelist,distance=distance)
    print(nodelist)

    #print(collections.Counter(np.reshape(distance,newshape=-1)))

    exit(10)

def buildAIGraph():
    with open("graphData/graphDict.pkl","rb") as f:
        data=pickle.load(f)
        id_to_index,index_to_id=data["id_to_index"],data["index_to_id"]

    db=MongoDBConnector(args.mongodb)

    qaAI=db.stackdb.get_collection("QAPForAI")

    allIDs=[]
    records=qaAI.find().batch_size(args.batch_size)

    for x in records:
        allIDs.append(x["Id"])

    print("finished gathering {} q/a pair ids".format(len(allIDs)))
    nodelist=[]
    isolated=[]

    for id in allIDs:
        if id in id_to_index:
            nodelist.append(id_to_index[id])
        else:
            isolated.append(id)

    nodelist.sort()

    data=np.load("graphData/shortestLinkPath.npz")
    distance=data["distance"]

    distance=distance[nodelist][:,nodelist]
    print(distance)
    print(np.max(distance),np.min(distance),np.mean(distance),np.var(distance))

    np.savez("graphData/AIPathLen",nodelist=nodelist,distance=distance,isolated=isolated)

    print("count for len, with %d isolated ids"%len(isolated))

    with open("graphData/linklenArray.pkl","wb") as f:
        linkCount=len(nodelist)
        totalCount=len(allIDs)
        linkLen={}
        for i in range(linkCount):
            for j in range(i):
                plen=distance[i,j]
                u=index_to_id[i]
                v=index_to_id[j]
                if plen in linkLen:

                    linkLen[plen].append((v,u))
                else:
                    linkLen[plen]=[(v,u)]

        if -1 not in linkLen:
            linkLen[-1]=[]

        connectedIds=[index_to_id[i] for i in range(linkCount)]

        for i in range(linkCount,totalCount):

            addIds=isolated[:i-linkCount]

            selected=connectedIds+addIds

            u=isolated[i-linkCount]

            for v in selected:

                linkLen[-1].append((v,u))

        pickle.dump(linkLen,f)

    exit(10)

def generateSemanticPairData(output="testdata/Smpair"):
    with open("graphData/linklenArray.pkl","rb") as f:
        linkLen=pickle.load(f)

    for k in linkLen.keys():
        v=linkLen[k]
        print(k,len(v))

    duplicates=[]
    directs=[]
    connects=[]
    unrelated=[]

    for k in linkLen.keys():
        if k==0:
            duplicates+=linkLen[k]
        elif k==1:
            directs+=linkLen[k]
        elif k>1 and k< 6:
            connects+=linkLen[k]
        else:
            unrelated+=linkLen[k]

    sampleNum=4000

    duplicates=np.array(duplicates)
    directs=np.array(directs)
    connects=np.array(connects)
    unrelated=np.array(unrelated)

    print("begin to sample %d records for each class"%sampleNum)

    duplicates=duplicates[np.random.choice(np.arange(duplicates.shape[0]),sampleNum,replace=sampleNum>len(duplicates))]
    directs=directs[np.random.choice(np.arange(directs.shape[0]),sampleNum,replace=sampleNum>len(directs))]
    connects=connects[np.random.choice(np.arange(connects.shape[0]),sampleNum,replace=sampleNum>len(connects))]
    unrelated=unrelated[np.random.choice(np.arange(unrelated.shape[0]),sampleNum,replace=sampleNum>len(unrelated))]

    np.savez(output,duplicates=duplicates,directs=directs,
             connects=connects,unrelated=unrelated)

    exit(2)

def constructLabledData():
    from programmingalpha.DataSet.PostPreprocessing import PreprocessPostContent
    processTxt=PreprocessPostContent()

    data=np.load("testdata/Smpair.npz")
    duplicates,directs,connects,unrelated=data["duplicates"],data["directs"],data["connects"],data["unrelated"]

    mongodb=MongoDBConnector(args.mongodb)
    questions=mongodb.stackdb.get_collection("QAPForAI")
    data=[]
    for (u,v) in duplicates:
        print(u,v)
        q1=questions.find_one({"Id":u})
        q2=questions.find_one({"Id":v})
        processTxt.raw_txt=q1["question_title"]
        q1_title=processTxt.getPlainTxt()
        processTxt.raw_txt=q2["question_title"]
        q2_title=processTxt.getPlainTxt()

        data.append((q1_title,q2_title,1))
    print("finished 1/4")
    for (u,v) in directs:
        q1=questions.find_one({"Id":u})
        q2=questions.find_one({"Id":v})
        processTxt.raw_txt=q1["question_title"]
        q1_title=processTxt.getPlainTxt()
        processTxt.raw_txt=q2["question_title"]
        q2_title=processTxt.getPlainTxt()

        data.append((q1_title,q2_title,2))
    print("finished 2/4")

    for (u,v) in connects:
        q1=questions.find_one({"Id":u})
        q2=questions.find_one({"Id":v})
        processTxt.raw_txt=q1["question_title"]
        q1_title=processTxt.getPlainTxt()
        processTxt.raw_txt=q2["question_title"]
        q2_title=processTxt.getPlainTxt()

        data.append((q1_title,q2_title,3))
    print("finished 3/4")

    for (u,v) in unrelated:
        q1=questions.find_one({"Id":u})
        q2=questions.find_one({"Id":v})
        processTxt.raw_txt=q1["question_title"]
        q1_title=processTxt.getPlainTxt()
        processTxt.raw_txt=q2["question_title"]
        q2_title=processTxt.getPlainTxt()

        data.append((q1_title,q2_title,4))
    print("finished 4/4")

    np.random.shuffle(data)
    trainSize=int(0.8*len(data))

    with open("testdata/train.txt","w") as f:
        f.writelines(data[:trainSize])
    with open("testdata/test.txt","w") as f:
        f.writelines(data[trainSize:])

    exit(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongodb', type=str, help='mongodb host, e.g. mongodb://10.1.1.9:27017/',default='mongodb://10.1.1.9:27017/')
    parser.add_argument('--collection', type=str, default="QAPForAI")
    parser.add_argument('--batch_size', type=str, default=10000)

    args = parser.parse_args()
    constructLabledData()
    #generateSemanticPairData()
    #statistics()
    #buildAIGraph()

    graphDict=createLinkDict()

    G,id_to_index,index_to_id=buildGraph(graphDict)
    with open("graphData/graphDict.pkl","wb") as f:
        data={"graphDict":graphDict,"id_to_index":id_to_index,"index_to_id":index_to_id}
        pickle.dump(data,f)
    #exit(1)

    nodelist=list(G.nodes)
    nodelist.sort()
    distance=nx.floyd_warshall_numpy(G,nodelist=nodelist)
    print(distance.shape)
    print(distance)

    np.savez("graphData/shortestLinkPath",nodelist=nodelist,distance=distance)
