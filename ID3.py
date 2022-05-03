import numpy as np
import pandas as pd
from numpy import log2 as log
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import datetime




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


def find_entropy(data, d1, num):  # target is the whole column that we do entropy to
    if num == 1:
        elements, count = np.unique(d1, return_counts=True)
    if num == 0:
        # print(data)
        elements, count = np.unique(data["diagnosis"], return_counts=True)

    entropy = np.sum([(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count)) for i in range(
        len(elements))])  # calc the entropy as we learned in the tut , (-counts[i] / np.sum(counts) is Pi ..
    return entropy


def find_InfoGain(data, feature, tmp, examples, flag):  # look again
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
    c1_ent = (len(c1) / len(examples) * find_entropy(data, d1, 1))
    c2_ent = (len(c2) / len(examples) * find_entropy(data, d2, 1))

    if flag == 0:

        totalEntropy = find_entropy(data, d1, 0)  # think about it

    else:

        totalEntropy = find_entropy(data, data, 1)  # think about it

    infoGain = totalEntropy - (c1_ent + c2_ent)  # calc infogain as we learned in tut (IG(F,E))
    return infoGain


def MaxIG(features, data, examples, flag):  # change here
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
            currIgVal = find_InfoGain(data, feature, tmp, examples, flag)
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


def ID3(data, examples, features, min=float("-inf"), flag=0):  # Here we build our decision tree
    c = MajorityClass(data, flag)
    return TDIDIT(data, examples, features, c, MaxIG, flag, min)


def TDIDIT(data, examples, features, deafult, selectFeature, flag, min):
    if len(examples) == 0:
        return buildTree(None, [], deafult, 0)

    if len(examples) < min:
        return buildTree(None, [], deafult, 0)

    c = MajorityClass(data, flag)

    # check_consistent
    #i used this flag because sometimes i had to use the data as a tuple with the feartures row and sometimes
    #i had it as a tuple with out the feature rows ...
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
    bestFeature, besttmp = MaxIG(features, data, examples, flag)
    for row in examples:
        if row[bestFeature] >= besttmp:
            c1.append(row)
            d1.append(row[0])
        else:
            c2.append(row)
            d2.append(row[0])

    # Remove the feature with the best inforamtion gain from the feature space
    # features = [i for i in features if i != bestFeature]

    leftSon = TDIDIT(d1, c1, features, c, MaxIG, 1, min)
    rightSon = TDIDIT(d2, c2, features, c, MaxIG, 1, min)

    return buildTree(bestFeature, [leftSon, rightSon], c, besttmp)


def testFunc(ID3tree, tests):
    cnt = 0
    for test in tests:
        if ID3tree.classify(test) == test[0]:
            cnt += 1
    return cnt / len(tests)


def get_examples(train_indexes, train):
    return [train[e] for e in train_indexes]

#This is question 3 implemnation " experiment " to run this go to the main and remove the # from the call of this function.
# also disable the lines which has " #disable to run exp3 " and thats it , i always send min = float("-inf")
#in default way in that case i do not need to remove the line that checks the min examples , if i want to run exp 3 i send another
#vals instead of float("-inf") therefore it runs as  M = min from the array !
def experiment(mVals, train, data, examples, features):
    kf = KFold(n_splits=5, shuffle=True, random_state=318695293)
    avgArr = []
    data_set = np.array(train)
    for vals in mVals:
        arr = []
        for trainIndex, testIndex in kf.split(examples):  # or examples idk
            trainSet = get_examples(trainIndex, examples)  # here i think should be example
            testSet = get_examples(testIndex, examples)
            data_first_column = [i[0] for i in trainSet]
            # test_first_column = [i[0] for i in testSet]
            ID3Tree = ID3(data_first_column, trainSet, features, vals, 1)
            cell = testFunc(ID3Tree, testSet)
            arr.append(cell)

        sum1 = sum(arr)

        avgCell = sum1 / 5
        avgArr.append(avgCell)

   # plt.plot(mVals, avgArr)
   # plt.xlabel('Min Number')
   # plt.ylabel('Accuracy')
   # plt.show()
   # print(avgArr)


def main():
    train = pd.read_csv("train.csv")
    data = pd.read_csv("train.csv")
    examples = np.array(train)
    features = np.arange(1, 31)
    id3Tree = ID3(data, examples, features) #disable when running exp3
    exam = pd.read_csv("test.csv")#disable when running exp3
    tests = np.array(exam)#disable when running exp3
    ret = testFunc(id3Tree, tests)#disable when running exp3
    print(ret)#disable when running exp3
    #M = [15, 25, 50, 100, 200]

    #experiment(M, train, data, examples, features)


if __name__ == "__main__":
    # now = datetime.datetime.now()
    # print(now)
    main()
    # now = datetime.datetime.now()
    # print(now)
