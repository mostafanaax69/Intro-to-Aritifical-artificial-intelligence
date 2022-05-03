import numpy as np
import pandas as pd
from numpy import log2 as log
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt




class buildTree:
    def __init__(self, feature, subtree, classfication, tmp):
        self.feautre = feature
        self.subtree = subtree
        self.classfictation = classfication
        self.tmp = tmp

    def classify(self, test):
        if len(self.subtree) == 0:
            return self.classfictation

        if test[self.feautre] >= self.tmp:
            return self.subtree[0].classify(test)

        else:
            return self.subtree[1].classify(test)


def find_entropy(data, d1, num,par):  # target is the whole column that we do entropy to
    entropy = 0
    elements = []
    count = []
    if num == 1:
        elements, count = np.unique(d1, return_counts=True)
    if num == 0:
        # print(data)
        elements, count = np.unique(data["diagnosis"], return_counts=True)

    for i in range(len(elements)):
        if elements[i] == 'M':
            entropy += par*(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count))

        if elements[i] == 'B':
            entropy += float(1-par)*(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count))

    # entropy = np.sum([(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count))
    #                   for i in range(
    #     len(elements))])  # calc the entropy as we learned in the tut , (-counts[i] / np.sum(counts) is Pi ..
    return entropy


def find_InfoGain(data, feature, tmp, examples, flag,par):  # look again
    # elements, indexes = np.unique(data["diagnosis"], return_counts=True)

    c1 = []
    c2 = []
    d1 = []
    d2 = []
    for row in examples:
        if row[feature] >= tmp:
            c1.append(row)
            d1.append(row[0])
        else:
            c2.append(row)
            d2.append(row[0])

    # print(d1)
    c1_ent = (len(c1) / len(examples) * find_entropy(data, d1, 1,par))
    c2_ent = (len(c2) / len(examples) * find_entropy(data, d2, 1,par))

    if flag == 0:

        totalEntropy = find_entropy(data, d1, 0,par)  # think about it

    else:

        totalEntropy = find_entropy(data, data, 1,par)  # think about it

    infoGain = totalEntropy - (c1_ent + c2_ent)  # calc infogain as we learned in tut (IG(F,E))
    return infoGain


def MaxIG(features, data, examples, flag,par):  # change here
    maxIG = 0
    tmp = float('-inf')
    bestemp = None
    bestFeature = None
    max_binary = None
    for feature in features:
        inner_max = float('-inf')
        vals = []
        for row in examples:
            vals.append(row[feature])

        vals = np.array(vals)
        vals = np.sort(vals)

        for i in range(len(vals) - 1):
            tmp = float((vals[i] + vals[i + 1]) / 2)
            currIgVal = find_InfoGain(data, feature, tmp, examples, flag,par)
            if inner_max < currIgVal:
                inner_max = currIgVal
                max_binary = tmp

        if inner_max >= maxIG:
            maxIG = inner_max
            bestFeature = feature
            bestemp = max_binary

    #  bestFeatureIndex = np.argmax(vals)
    #  bestFeature = features[bestFeatureIndex]
    return bestFeature, bestemp


def MajorityClass(data, flag):  # take a look
    healthycnt = 0
    elements = []
    count = []
    arr = []
    if flag == 1:
        elements, count = np.unique(data, return_counts=True)
        arr = data
    if flag == 0:
        elements, count = np.unique(data["diagnosis"], return_counts=True)
        arr = np.array(data)

    for i in range(len(elements)):
        if (elements[i] == 'B'):
            healthycnt = count[i]

    if healthycnt >= len(arr) / 2: return 'B'
    return 'M'


def ID3(data, examples, features,par=0.9, min=30, flag=0):  # Here we build our decision tree
    c = MajorityClass(data, flag)
    return TDIDIT(data, examples, features, c, MaxIG, par,flag, min)


def TDIDIT(data, examples, features, deafult, selectFeature, par,flag, min):
    if len(examples) == 0:
        return buildTree(None, [], deafult, 0)

    if len(examples) < min:
        return buildTree(None, [], deafult, 0)

    c = MajorityClass(data, flag)

    # check_consistent
    if (flag == 0):
        if len(np.unique(data["diagnosis"])) <= 1 or len(features) == 0:
            return buildTree(None, [], c, 0)

    if (flag == 1):
        if len(np.unique(data)) <= 1 or len(features) == 0:
            return buildTree(None, [], c, 0)

    c1 = []
    c2 = []
    d1 = []
    d2 = []
    bestFeature, besttmp = MaxIG(features, data, examples, flag,par)
    for row in examples:
        if row[bestFeature] >= besttmp:
            c1.append(row)
            d1.append(row[0])
        else:
            c2.append(row)
            d2.append(row[0])

    # Remove the feature with the best inforamtion gain from the feature space
    # features = [i for i in features if i != bestFeature]

    leftSon = TDIDIT(d1, c1, features, c, MaxIG, par,1, min)
    rightSon = TDIDIT(d2, c2, features, c, MaxIG, par,1, min)

    return buildTree(bestFeature, [leftSon, rightSon], c, besttmp)


def testFunc(ID3tree, tests):
    cnt = 0
    FPcnt = 0
    FNcnt = 0
    for test in tests:
        if ID3tree.classify(test) == 'M' and test[0] == 'B':
            FPcnt += 1
        if ID3tree.classify(test) == 'B' and test[0] == 'M':
            FNcnt += 1

    return ((0.1 * FPcnt) + FNcnt) / len(tests)


def get_examples(train_indexes, train):
    return [train[e] for e in train_indexes]


# i used this function to do a cross valadtion on the paramters than i am going to multiply them
#in the find entropy function !
#
#def expNumThree(mVals, train, data, examples, features):
#    kf = KFold(n_splits=10, shuffle=True, random_state=318695293)
#    avgArr = []
#    minLoss = 1
#    pars = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#    for par in pars:
#        arr = []
#        for trainIndex, testIndex in kf.split(examples):  # or examples idk
#            trainSet = get_examples(trainIndex, examples)  # here i think should be example
#            testSet = get_examples(testIndex, examples)
#            data_first_column = [i[0] for i in trainSet]
#            # test_first_column = [i[0] for i in testSet]
#            ID3Tree = ID3(data_first_column, trainSet, features,par,30, flag= 1)
#            cell = testFunc(ID3Tree, testSet)
#            arr.append(cell)

#        sum1 = sum(arr)

#        avgCell = sum1 / 10
        #avgArr.append(avgCell)
#        if avgCell < minLoss:
#            minLoss = avgCell
#            print("minLose" , minLoss , "first par" , par ,"sec par" , 1-par)

    #plt.plot(mVals, avgArr)
    #plt.xlabel('Min Number')
    #plt.ylabel('Accuracy')
    #plt.show()
    #print(avgArr)


def main():
    train = pd.read_csv("train.csv")
    data = pd.read_csv("train.csv")
    examples = np.array(train)
    features = np.arange(1, 31)
    id3Tree = ID3(data, examples, features)
    exam = pd.read_csv("test.csv")
    tests = np.array(exam)
    ret = testFunc(id3Tree, tests)
    print(ret)
    #i choose M = 30
    #M = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]

    #expNumThree(M, train, data, examples, features)


if __name__ == "__main__": main()
