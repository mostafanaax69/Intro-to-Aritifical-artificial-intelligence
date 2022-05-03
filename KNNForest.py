from ID3 import ID3
import numpy as np
import pandas as pd
import random
from math import *
from sklearn.model_selection import KFold


def classifyForest(knnForestArray, test, k):
    mCnt = 0
    bCnt = 0

    EcuDist = [(Ecudistance(tree[1], test[1:]), tree) for tree in knnForestArray]
    EcuDist.sort(key=lambda x: x[0])
    topK = EcuDist[:k]
    for dist, tree in topK:
        res = tree[0].classify(test)
        if (res == 'M'):
            mCnt += 1
        else:
            bCnt += 1

    if mCnt > bCnt:
        return 'M'
    else:
        return 'B'


def testFunc(knnForestArray, tests, k):
    cnt = 0
    for test in tests:
        if classifyForest(knnForestArray, test, k) == test[0]:
            cnt += 1
    return cnt / len(tests)


def Ecudistance(data, test):
    return sqrt(sum((it[0] - it[1]) ** 2 for it in list(zip(data, test))))


def get_examples(train_indexes, train):
    return [train[e] for e in train_indexes]


#def exp(N, examples, features, p, k):
#    kf = KFold(n_splits=5, shuffle=True, random_state=318695293)
#    arr = []
#    for trainIndex, testIndex in kf.split(examples):  # or examples idk
#        trainSet = get_examples(trainIndex, examples)  # here i think should be example
#        testSet = get_examples(testIndex, examples)
#        knnForest = KNNforestTrial(N, trainSet, features, p)
#        cell = testFunc(knnForest, testSet, k)
#        arr.append(cell)

#    sum1 = sum(arr)
#    avgCell = sum1 / 5
#    print("accuracy= ", avgCell, "K=", k, "N=", N, "P=", p)
#    return avgCell


def KNNforestTrial(N, examples, features, p):
    numExamplesSize = int(p * len(examples))
    treeArr = []
    for i in range(N):
        newExamples = random.choices(examples, k=numExamplesSize)
        data_first_column = [z[0] for z in newExamples]
        ID3Tree = ID3(data_first_column, newExamples, features, flag=1)
        centroArr = calc_centroid(newExamples, features)
        treeArr.append((ID3Tree, centroArr))

    return treeArr


def calc_centroid(examples, features):
    centroid = []
    sum = 0
    for i in features:
        for j in range(len(examples)):
            sum += examples[j][i]
        avg = sum / len(examples)
        centroid.append(avg)
    return centroid


def main():
    train = pd.read_csv("train.csv")
    examples = np.array(train)
    features = np.arange(1, 31)
    exam = pd.read_csv("test.csv")
    tests = np.array(exam)
    cnt = 0
    res = 0
   # N_list = [10, 16, 25, 50,100]
   # p_arr = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
   # k_list = [5,18,12,16,25,50]
    # for N in N_list:
    #    for k in k_list:
    #        for p in p_arr:
    #            cnt+=1
    #            res += exp(N, examples, features, p, k)



   # for i in range(10):
    knnForestArray = KNNforestTrial(10, examples, features, 0.7)
    res1 = float(testFunc(knnForestArray, tests, 6))
   #     res += res1
   #     cnt+=1
   #     print(res1)

    print("final accuracy ", res1)

if __name__ == "__main__": main()
