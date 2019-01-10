import numpy as np



class Aprio(object):
    def __init__(self):
        self.maxK=np.inf
        self.minSupport=0.5
        self.minConfidence=0.8

    def initSingleItem(self,dataSet):
        c1 = []
        for transaction in dataSet:
            for item in transaction:
                if  [item] not in c1:
                    c1.append([item])

        return list(map(frozenset,c1))

    def scanDataset(self,D,Ck):
        supportCount = {}
        def countForCan(can):
            count=0
            for tid in D:
                if can.issubset(tid):
                    count+=1
            supportCount[can]=count

        vectorizerSearch=np.vectorize(countForCan)
        #print(type(Ck),len(Ck))
        vectorizerSearch(Ck)


        numItems = float(len(D))
        retList = []

        for key in supportCount:
            support = supportCount[key] / numItems
            if support >= self.minSupport:
                retList.append(key)
            supportCount[key] = support

        return retList,supportCount

    def aprioriGen(self,Lk,k):
        retList = []
        lenLk = len(Lk)

        #print("current K",k)

        for i in range(lenLk):
            for j in range(i+1,lenLk):
                L1 = Lk[i]
                L2 = Lk[j]
                L=L1|L2
                #print("L1/L2",L1,L2)
                if len(L)>k:
                    #print("to skip", len(L),L)
                    continue

                symL=L1.symmetric_difference(L2)

                if k>2 and symL not in Lk:
                    #print("symetric to skip",L1.symmetric_difference(L2))
                    continue
                retList.append(L)

        return retList

    def mineSupportSet(self,dataSet):
        C1 = self.initSingleItem(dataSet)

        L1,supportData = self.scanDataset(dataSet,C1)

        L = [L1]

        k = 2

        while (len(L[k-2]) > 0) and k<=self.maxK :
            Ck = self.aprioriGen(L[k-2],k)
            if len(Ck)<1:
                break
            Lk,supK = self.scanDataset(dataSet,Ck)
            if len(Lk)>0:
                supportData.update(supK)
                L.append(Lk)
            else:
                break
            k += 1
        return L,supportData
