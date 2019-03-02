import networkx as nx
import collections
import argparse
from programmingalpha.DataSet.DBLoader import MongoStackExchange
import programmingalpha
import numpy as np
import tqdm
import json
import multiprocessing
from functools import partial
import logging
#from programmingalpha.Utility.DataStructure import UnifoldSet

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



def buildGraph(dbName):
    db=MongoStackExchange(host='10.1.1.9',port=50000)
    db.useDB(dbName)

    links=db.stackdb.get_collection("postlinks")

    allLinks=list(links.find().batch_size(args.batch_size))

    G=nx.Graph()

    for link in tqdm.tqdm(allLinks,desc="building graph from links"):
        id_a,id_b=link["PostId"],link["RelatedPostId"]
        r=link["LinkTypeId"]
        if r==3:
            w=0
        elif r==1:
            w=1
        else:
            raise ValueError("unexpected value {} for link type".format(r))

        G.add_edge(id_a,id_b,weight=w)

    logger.info("finished finding {} sublinks".format(len(allLinks)))
    logger.info("graph size of edges({}) and nodes({})".format(len(list(G.edges)),len(list(G.nodes))))

    ccs=nx.connected_components(G)
    graphs=[]

    for cc in ccs:
        g=G.subgraph(cc)
        graphs.append(g)

    graphs.sort(key=lambda g:len(g.nodes),reverse=True)

    print(len(graphs),list(map(lambda g:len(g.nodes),sorted(graphs,key=lambda g:g.nodes,reverse=True)))[:10])

    return graphs,G

def computePathCore(src,maxLength,G):
    path=nx.single_source_dijkstra_path_length(G=G,source=src,cutoff=maxLength,weight="weight")
    if src in path:
        del path[src]
    return {src:path}

def computeBatch(srcs,maxLength,G):
    batch=[]
    for src in srcs:
        batch.append(computePathCore(src,maxLength,G))
    return batch

def computeShortestPathParallel(G,maxLength=2):
    logger.info("computing path length using Pool")
    distanceData=[]
    worker_num=30 if args.workers>0 else multiprocessing.cpu_count()
    workers=multiprocessing.Pool(worker_num)
    _compute=partial(computeBatch,maxLength=maxLength,G=G)

    nodes=list(G.nodes)
    batch_size=args.batch_size
    batches=[nodes[i:i+batch_size] for i in range(0,len(nodes),batch_size)]
    for batch in workers.map(_compute,batches):
        distance_data.extend(batch)

    workers.close()
    workers.join()

    return distanceData

def computeShortestPath(G,maxLength=2):
    logger.info("computing path length")
    distanceData=[]
    _compute=partial(computePathCore,maxLength=maxLength,G=G)

    for src in tqdm.tqdm(G.nodes,desc="computing path"):
        distance_data.append(_compute(src))

    return distanceData

def _maxClip(data,maxSize):
    if len(data)>1.2*maxSize:
        np.random.shuffle(data)
        data=data[:maxSize]
    return data

def relationPCore(path_data,nodes):
    linkData=[]
    source=list(path_data.keys())[0]
    targets_path=path_data[source]

    for target in targets_path:
        if targets_path[target]==0:
            linkData.append({"duplicate":(source,target)})
        elif targets_path[target]==1:
            linkData.append({"direct":(source,target)})
        elif targets_path[target]==2:
            linkData.append({"transitive":(source,target)})
        else:
            raise ValueError("unexpected value {} for link relation".format(targets_path[target]))

    link_num=len(linkData)
    add_ons=link_num//3

    count=0
    for node in nodes:
        if count>add_ons:
            break
        if node in targets_path:
            continue
        linkData.append({"unrelated":(source,node)})
        count+=1


    return linkData

def pairRelation(distanceData:list,nodes):
    linkData=[]
    batches=[distanceData[i:args.batch_size] for i in range(0,len(distanceData),args.batch_size)]
    workers=multiprocessing.Pool(args.workers)
    _relationPCore=partial(relationPCore,nodes=nodes)
    for batch in tqdm.tqdm(batches,desc="computing in batch"):
        links_batch=workers.map(_relationPCore,batch)
        np.random.shuffle(nodes)
        for links in links_batch:
            linkData.extend(links)

    workers.close()
    workers.join()

    np.random.shuffle(linkData)

    return linkData

def constructLabledData(dbName):

    from programmingalpha.Utility.PostPreprocessing import PreprocessPostContent
    import json
    processTxt=PreprocessPostContent()

    linkData=np.load(programmingalpha.DataPath+dbName+"-linkData.npz")
    dataSource={"duplicateLink","directLink","transitiveLink","unrelatedLink"}
    labelDict={"duplicateLink":0,"directLink":1,"transitiveLink":2,"unrelatedLink":3}

    mongodb=MongoStackExchange(host='10.1.1.9',port=50000)
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

            try:
                q1=dictQuestions[int(u)]
                q2=dictQuestions[int(v)]
            except:
                continue

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
    with open(savePath+"train-"+dbName+".json","w") as f:
        f.writelines(data[:trainSize])
    with open(savePath+"test-"+dbName+".json","w") as f:
        f.writelines(data[trainSize:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--db', type=str, default="crossvalidated")
    parser.add_argument('--workers', type=int, default=20)

    args = parser.parse_args()
    dbName=args.db

    #constructLabledData(dbName);exit(3)


    maxPathLength=2
    graphs,G=buildGraph(dbName)
    distance_data=[]

    for G in tqdm.tqdm(graphs,desc="computing sub graph"):
        logger.info("subgraph nodes: {}".format(len(G.nodes)))
        if len(G.nodes)>args.batch_size:
            distance_data_one=computeShortestPathParallel(G,maxPathLength)
        else:
            distance_data_one=computeShortestPath(G,maxPathLength)

        distance_data.extend(distance_data_one)

    logger.info("shortest distance computing finished")

    distance_file=programmingalpha.DataPath+"linkData/"+dbName.lower()+'-%dgraph.json'%maxPathLength
    with open(distance_file,"w") as f:
        for path in distance_data:
            path=json.dumps(path)
            f.write(path+"\n")

    labeled_link_data=pairRelation(distance_data,list(G.nodes))
    labels=list(map(lambda l:list(l.keys())[0],labeled_link_data))
    logger.info(collections.Counter(labels))

    labeled_link_file=programmingalpha.DataPath+"linkData/"+dbName.lower()+'-linkData.json'
    with open(labeled_link_file,"w") as f:
        for labeled in labeled_link_data:
            labeled=json.dumps(labeled)
            f.write(labeled+"\n")
