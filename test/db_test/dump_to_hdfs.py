import pyspark
import pymongo
import json
import sys

def loadData(dbName):
    from urllib.parse import quote_plus
    url = "mongodb://{}:{}@{}:{}".format (
                quote_plus("zzy123"), quote_plus("abc123"), "10.1.1.9",50000)
    client=pymongo.MongoClient(url)

    db=client.get_database(dbName)
    query=db[collectionName].find().batch_size(10000)

    links=[]
    for link in query:
        del link["_id"]
        links.append(link)

    with open(sys.argv[3],"w") as f:
        for link in links:
            f.write(json.dumps(link)+"\n")

if __name__ == '__main__':

    dbName=sys.argv[1]
    collectionName=sys.argv[2]
    dumpFile=sys.argv[3]

    spark = pyspark.sql.SparkSession\
        .builder\
        .appName("PostLinksRelation")\
        .getOrCreate()
    loadData(dbName)
    data=spark.read.json(dumpFile)
    data.write.json("/user/zhangzy/"+dumpFile.split("/")[-1])
