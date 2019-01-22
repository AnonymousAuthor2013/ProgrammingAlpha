from programmingalpha.DataSet.DBLoader import MongoStackExchange,connectToMongoDB
import argparse
import regex as re


def initPool():
    global db
    db=connectToMongoDB()

def splitTags():
    db.useDB("crossvalidated")

    batch_q=[]
    if "tmpQuestions" in db.stackdb.list_collection_names():
        db.stackdb.drop_collection("tmpQuestions")


    questions=db.stackdb.create_collection("tmpQuestions")

    for q in db.questions.find():
        tags=q["Tags"]
        if type(tags)!=list:
            q["Tags"]=tag_matcher.findall(tags)

        batch_q.append(q)
        if len(batch_q)%args.batch_size==0:
            questions.insert_many(batch_q)
            batch_q.clear()
            print("insert a batch,current size {}/{}".format(questions.count(),db.questions.count()))

    if len(batch_q)>0:
        questions.insert_many(batch_q)
        batch_q.clear()
        print("insert a batch,current size {}/{}".format(questions.count(),db.questions.count()))

    name=db.questions.name
    db.stackdb.drop_collection(db.questions.name)
    questions.rename(name)

def createkeysForWikiDocs():
    db.useDB("wikidocs")
    articles=db.stackdb.get_collection('articles')
    tmp_collection=db.stackdb.create_collection('tmp_articles')
    cache_docs=[]
    for doc in articles.find().batch_size(args.batch_size):
        newdoc={"Id":int(doc['id']),"text":doc["text"],"Title":doc["title"],"url":doc["url"]}
        cache_docs.append(newdoc)
        if len(cache_docs)%args.batch_size==0:
            tmp_collection.insert_many(cache_docs)
            print("inserted %d docs"%tmp_collection.count())
            cache_docs.clear()

    if len(cache_docs)>0:
        tmp_collection.insert_many(cache_docs)
        print("inserted %d docs"%tmp_collection.count())

    tmp_collection.create_index("Id")


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mongodb', type=str, help='mongodb host, e.g. mongodb://10.1.1.9:27017/',default='mongodb://10.1.1.9:27017/')
    parser.add_argument('--batch_size', type=str, default=10000)
    args=parser.parse_args()

    initPool()
    tag_matcher=re.compile(r"<.*?>",re.I)

    createkeysForWikiDocs()
    #splitTags()
