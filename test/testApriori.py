from programmingalpha.Utility.CorrelationAnalysis import Aprio

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

itemSeed=[{1},{3},{4}]
itemSeed=list(map(frozenset,itemSeed))
print("=="*20)
results=apriori.stepSearch(dataSet,itemSeed,2)
if results:
    a,b,c=results
    print(a)
    print(b)
    print(c)

    print()
    for i in a:
        print("*"*20)

        for s in itemSeed:
            if s.issubset(i):
                print(i,"<=",s,b[i]/b[s])
