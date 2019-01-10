from programmingalpha.DataSet.MyApriori import Aprio

def loadData():
    data=[[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
    data=frozenset(map(frozenset,data))

    return data

dataSet = loadData()

apriori=Aprio()
apriori.minConfidence=0.8
apriori.minSupport=0.5
apriori.maxK=2
a,b = apriori.mineSupportSet(dataSet)

print(a)
print(b)
