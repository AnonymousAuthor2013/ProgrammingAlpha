import networkx as nx
import pickle
import collections
import argparse
from programmingalpha.DataSet.DBLoader import MongoStackExchange,connectToMongoDB
import programmingalpha
import numpy as np

def createLinkDict(dbName,m_tags=None):
    db=connectToMongoDB()
    db.useDB(dbName)
    questions=db.questions

    query=None
    if m_tags:
        query={"Tags":{"$in":m_tags}}

    allIDs=set()
    if query:
        records=questions.find(query).batch_size(args.batch_size)
    else:
        records=questions.find().batch_size(args.batch_size)

    for x in records:
        allIDs.add(x["Id"])

    print("finished gathering {} q/a pair ids".format(len(allIDs)))

    links=db.stackdb.get_collection("postlinks")

    allLinks=links.find().batch_size(args.batch_size)

    graphDict=collections.OrderedDict()
    count=0
    conflicts=0
    for link in allLinks:
        count+=1

        #make sure id_a<id_b
        id_a,id_b=link["PostId"],link["RelatedPostId"]
        if id_a> id_b:
            tmp=id_b
            id_b=id_a
            id_a=tmp

        r=link["LinkTypeId"]

        #concerns only those in required Ids
        if id_a in allIDs or id_b in allIDs:


            if (id_a,id_b) in graphDict and graphDict[(id_a,id_b)]!=r:


                conflicts+=1

                if r==3:
                    graphDict[(id_a,id_b)]=r

            else:
                graphDict[(id_a,id_b)]=r

    print("finished finding {} sublinks".format(len(graphDict)))
    print("finished with {} conflicts".format(conflicts))

    return graphDict


def buildGraph(graphDict:collections.OrderedDict):

    G=nx.Graph()
    edges=[]

    for e in graphDict.keys():
        w=graphDict[e]
        u,v=e
        if w==3:
            w=0
        elif w==1:
            w=1

        edges.append((u,v,w))

    G.add_weighted_edges_from(edges)

    print("graph size of edges and nodes",len(list(G.edges)),len(list(G.nodes)))

    return G

def computeShortestPath(G:nx.Graph,maxLength=2):

    distanceData=nx.all_pairs_dijkstra_path_length(G,cutoff=maxLength)

    return distanceData


def pairRelation(distanceData:dict,maxSize):

    def maxClip(data):
        if len(data)>2*maxSize:
            np.random.shuffle(data)
            data=data[:maxSize]
        return data

    duplicateLink=[]
    directLink=[]
    transitiveLink=[]
    unrelatedLink=[]

    nodes=list(distanceData.keys())
    for i in range(len(nodes)):
        source=nodes[i]
        srcNodes=distanceData[source]
        for j in range(i):
            target=nodes[j]
            if target in srcNodes:
                if srcNodes[target]==0:
                    duplicateLink.append((source,target))
                elif srcNodes[target]==1:
                    directLink.append((source,target))
                elif srcNodes[target]==2:
                    transitiveLink.append((source,target))
            else:
                unrelatedLink.append((source,target))

            duplicateLink=maxClip(duplicateLink)
            directLink=maxClip(directLink)
            transitiveLink=maxClip(transitiveLink)
            unrelatedLink=maxClip(unrelatedLink)

        duplicateLink=maxClip(duplicateLink)
        directLink=maxClip(directLink)
        transitiveLink=maxClip(transitiveLink)
        unrelatedLink=maxClip(unrelatedLink)

    return duplicateLink,directLink,transitiveLink,unrelatedLink

def constructLabledData(dbName):

    #linkData=np.load(programmingalpha.DataPath+dbName+"-linkData.npz")
    #duplicateLink,directLink,transitiveLink,unrelatedLink=linkData['duplicateLink'],linkData['directLink'],\
    #linkData['trainsitiveLink'],linkData['unrelatedLink']
    #relation_path=programmingalpha.DataPath+dbName+'-linkData.npz'
    #np.savez(relation_path,duplicateLink=duplicateLink,directLink=directLink,
    #         transitiveLink=transitiveLink,unrelatedLink=unrelatedLink)
    #exit(1)

    from programmingalpha.DataSet.PostPreprocessing import PreprocessPostContent
    import json
    processTxt=PreprocessPostContent()

    linkData=np.load(programmingalpha.DataPath+dbName+"-linkData.npz")
    dataSource={"duplicateLink","directLink","transitiveLink","unrelatedLink"}
    labelDict={"duplicateLink":1,"directLink":0.7,"transitiveLink":0.5,"unrelatedLink":0}

    mongodb=connectToMongoDB()
    mongodb.useDB(dbName)
    questions=mongodb.questions
    dictQuestions={}
    for q in questions.find().batch_size(args.batch_size):
        dictQuestions[q["Id"]]=q
    print("finished reading all q data",len(dictQuestions))
    data=[]
    count=0
    for key in dataSource:
        label=labelDict[key]
        for (u,v) in linkData[key]:
            #print(u,v)
            #q1=questions.find_one({"Id":int(u)})
            #q2=questions.find_one({"Id":int(v)})
            #if not (q1 and q2):
            #    continue
            try:
                q1=dictQuestions[int(u)]#questions.find_one({"Id":int(u)})
                q2=dictQuestions[int(v)]#questions.find_one({"Id":int(v)})
            except:
                continue
            #print(q1)
            #print(q2)

            q1_title=processTxt.getPlainTxt(q1["Title"])
            q2_title=processTxt.getPlainTxt(q2["Title"])
            q1_body=processTxt.getPlainTxt(q1["Body"])
            q2_body=processTxt.getPlainTxt(q2["Body"])

            x={"title":q1_title+"|||"+q2_title,"body":q1_body+"|||"+q2_body,"simValue":label}
            data.append(json.dumps(x)+"\n")

        count+=1
        print("finished {}/{}".format(count,len(labelDict)))


    print("finished creating dataset for training and tuning",len(data))
    np.random.shuffle(data)
    trainSize=int(0.8*len(data))

    savePath=programmingalpha.DataPath
    with open(savePath+"train-"+dbName+".txt","w") as f:
        f.writelines(data[:trainSize])
    with open(savePath+"test-"+dbName+".txt","w") as f:
        f.writelines(data[trainSize:])

    exit(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=str, default=10000)

    args = parser.parse_args()
    dbName='stackoverflow'

    constructLabledData(dbName)

    if dbName=='stackoverflow':
        m_tag=programmingalpha.loadConfig(programmingalpha.ConfigPath+'TagSeeds.json')['Tags']
    else:
        m_tag=None

    maxSize=0
    maxPathLength=2
    graphDict=createLinkDict(dbName,m_tag)
    maxSize=collections.Counter(graphDict.values())[3]
    print("max size",maxSize)
    G=buildGraph(graphDict)
    distanceData=computeShortestPath(G,maxPathLength)
    distanceData=dict(distanceData)
    print("shortest distance computing finished")
    #print(distanceData)

    duplicateLink,directLink,transitiveLink,unrelatedLink=pairRelation(distanceData,maxSize)
    print("duplicateLink({}),directLink({}),transitiveLink({}),unrelatedLink({})".format(
        len(duplicateLink),len(directLink),len(transitiveLink),len(unrelatedLink)
    ))

    relation_path=programmingalpha.DataPath+dbName+'-linkData.npz'
    np.savez(relation_path,duplicateLink=duplicateLink,directLink=directLink,
             transitiveLink=transitiveLink,unrelatedLink=unrelatedLink)
