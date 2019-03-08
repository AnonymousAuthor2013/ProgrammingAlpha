import numpy as np
def test1():
    x=np.array([
        [1,2],
        [3,4],
        [5,6],
    ])


    y=np.array([
        [0,1,2],
        [2,3,4]
    ])

    print(x,'\n',y)

    print('-*'*50)

    z=y.dot(x)

    print(z)

    xm=np.sum(np.square(x),0)
    ym=np.sum(np.square(y),1)

    print(np.sqrt(xm))
    print(np.sqrt(ym))

    print(xm)
    print(ym)

    xm=np.linalg.norm(x,axis=0)
    ym=np.linalg.norm(y,axis=1)

    print(xm)
    print(ym)

    print(np.divide(x,xm))
    print((y.T/ym).T)
    print(
        ((y.T/ym).T).dot(np.divide(x,xm))
    )

    from programmingalpha.Utility.TextPreprocessing import PreprocessPostContent
    extractor=PreprocessPostContent()
    txt=extractor.getPlainTxt('<p> is is it good?</p><code></code>')
    print(txt)

    'How to compute machine learning training computation time and what are reference values? Ask Question in ' \
    'many forums and documents on the internet we hear about "short" and "long" learning and prediction computation time ' \
    'for machine learning algorithms. For example the Decision Tree Algorithm has a short computation time as ' \
    'compared to Neural Networks. But what it is never mentioned is what is "short" and what is "long".' \
    'Could you please clarify which unit you would use to measure computation time? Maybe \"seconds per sample"? ' \
    'And what are reference values, so that I can predict if it takes 1h, 1day or 1Week?'

def test2():
    import programmingalpha,os
    dataSource=["AI","datascience","crossvalidated","stackoverflow"]
    data=[]
    for ds in dataSource:
        data_dir=os.path.join(programmingalpha.DataPath,"inference_pair/test-"+ds+".txt")
        with open(data_dir,"r") as f:
            data.extend(f.readlines())
        data_dir=os.path.join(programmingalpha.DataPath,"inference_pair/train-"+ds+".txt")
        with open(data_dir,"r") as f:
            data.extend(f.readlines())

        print("extended {}, current size={}".format(ds,len(data)))

    n_samples=len(data)

    train=data[:int(0.9*n_samples)]
    test=data[int(0.9*n_samples):]

    print("all data records size=",n_samples)

    data_dir=os.path.join(programmingalpha.DataPath,"inference_pair/train.txt")
    with open(data_dir,"w") as f:
        f.writelines(train)
    data_dir=os.path.join(programmingalpha.DataPath,"inference_pair/test.txt")
    with open(data_dir,"w") as f:
        f.writelines(test)
    data_dir=os.path.join(programmingalpha.DataPath,"inference_pair/train-all.txt")
    with open(data_dir,"w") as f:
        f.writelines(data)

def test3():
    from programmingalpha.DataSet.DBLoader import MongoStackExchange
    from programmingalpha.Utility.TextPreprocessing import PreprocessPostContent
    processor=PreprocessPostContent()
    db=MongoStackExchange(host='10.1.1.9',port=50000)
    dbName='stackoverflow'
    db.useDB(dbName)
    count=0
    threshold=0.2
    verbose=0
    for q in db.questions.find().batch_size(10000):
        txt=q['Title']+q['Body']
        codes=' '.join(processor.getCodeSnippets(txt))
        if len(codes) and verbose<10:
            print(len(codes),len(txt))
            verbose+=1

        if len(codes)/len(txt)>threshold:
            count+=1
    print("code question is {}/{}".format(count,db.questions.count()))

    count=0
    for ans in db.answers.find().batch_size(10000):
        txt=ans['Body']
        codes=' '.join(processor.getCodeSnippets(txt))
        if len(codes) and verbose<10:
            print(len(codes),len(txt))
            verbose+=1

        if len(codes)/len(txt)>threshold:
            count+=1
    print('code answer is {}/{}'.format(count,db.answers.count()))
#test2()

def test4():
    path="/home/LAB/zhangzy/BertModels/Embeddings/glove.6B/"
    f1="glove.6B.100d.txt"
    f2="glove.6B.200d.txt"
    f3="glove.6B.300d.txt"
    f4="glove.6B.50d.txt"

    with open(path+f1,"r") as f:
        lines1=f.readlines()
    with open(path+f2,"r") as f:
        lines2=f.readlines()
    with open(path+f3,"r") as f:
        lines3=f.readlines()
    with open(path+f4,"r") as f:
        lines4=f.readlines()
    assert len(lines1)==len(lines2)
    assert len(lines1)==len(lines3)
    assert len(lines1)==len(lines4)

    n=len(lines1)

    for i in range(n):
        if lines1[i][0]==lines2[i][0] and lines1[i][0]==lines3[i][0] and lines1[i][0]==lines4[i][0]:
            pass
        else:
            raise ValueError("not equal, {}, {}, {}, {}".format(lines1[i][0],lines2[i][0],lines3[i][0],lines4[i][0]))

    print("equal")
if __name__ == '__main__':
    test4()
