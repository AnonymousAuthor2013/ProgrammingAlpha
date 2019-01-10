from programmingalpha.DataSet.DBLoader import MongoDBConnector

dbUrl="mongodb://10.1.1.9:27017"

def AlterMove(src,dst,alter):
    db=MongoDBConnector(dbUrl)
    srcC=db.stackdb[src]
    dstC=db.stackdb[dst]

    srcData=srcC.find().batch_size(10000)

    cache=[]
    for x in srcData:
        del x[alter]

        cache.append(x)

        if len(cache)%10000==0:
            dstC.insert_many(cache)
            cache.clear()

    if len(cache)>0:
        dstC.insert_many(cache)


if __name__ == '__main__':
    AlterMove("QAPForAI","QAPForAItmp","answer_title")
