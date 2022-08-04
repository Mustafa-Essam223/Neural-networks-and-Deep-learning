from tkinter import *
from tkinter import messagebox
from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
import random


def getTrueValue(i):
    fClass = []
    data = np.genfromtxt("IrisData.txt", delimiter=',', dtype=str)
    for row in data:
        if row[0] != "X1":
            if str(row[4]) == "Iris-setosa":
                fClass.append(0)
            elif str(row[4]) == "Iris-versicolor":
                fClass.append(1)
            elif str(row[4]) == "Iris-virginica":
                fClass.append(2)
    return fClass[i]


def extractData(v1, v2):
    X1 = []
    X2 = []
    fClass = []
    data = np.genfromtxt("IrisData.txt", delimiter=',', dtype=str)
    for row in data:
        if row[0] != "X1":
            X1.append(float(row[v1]))
            X2.append(float(row[v2]))

        if str(row[4]) == "Iris-setosa":
            fClass.append(0)
        elif str(row[4]) == "Iris-versicolor":
            fClass.append(1)
        elif str(row[4]) == "Iris-virginica":
            fClass.append(2)
    return X1, X2, fClass


def Modified_ExtractData(v1, v2, className, Learnig_Testing, flag):
    global cntr
    feature1 = []
    feature2 = []
    classID = []

    if Learnig_Testing == 1:
        cntr = 0  # to take learning tuples only
    elif Learnig_Testing == 0:
        cntr = 30  # to take testing tuples only
    Data = np.genfromtxt("IrisData.txt", delimiter=',', dtype=str)
    for tuple in Data:
        if tuple[0] != "X1" and className == tuple[4] and cntr < 30:
            feature1.append(tuple[v1])
            feature2.append(tuple[v2])
            cntr += 1
        if tuple[0] != "X1" and className == tuple[4] and cntr >= 30:
            feature1.append(tuple[v1])
            feature2.append(tuple[v2])
            classID.append(flag)
            cntr += 1

    return feature1, feature2


def drawData(x1, x2):
    f1c1 = []
    f2c1 = []
    f1c2 = []
    f2c2 = []
    f1c3 = []
    f2c3 = []
    if x1 == x2:
        return messagebox.showinfo(title="Error Message", message="Can't choose same features")
    else:
        feat1, feat2, fClass = extractData(x1, x2)
    for i in range(len(feat1)):
        if fClass[i] == 0:
            f1c1.append(feat1[i])
            f2c1.append(feat2[i])
        elif fClass[i] == 1:
            f1c2.append(feat1[i])
            f2c2.append(feat2[i])
        elif fClass[i] == 2:
            f1c3.append(feat1[i])
            f2c3.append(feat2[i])
    plt.figure("Iris Data")
    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.scatter(f1c1, f2c1)
    plt.scatter(f1c2, f2c2)
    plt.scatter(f1c3, f2c3)
    plt.show()


def updateData(feat1, feat2, fClass, c1, c2):
    updatedFeat1 = []
    updatedFeat2 = []
    nFClass = []
    for i in range(len(feat1)):
        if fClass[i] == c1:
            updatedFeat1.append(feat1[i])
            updatedFeat2.append(feat2[i])
            nFClass.append(fClass[i])
        elif fClass[i] == c2:
            updatedFeat1.append(feat1[i])
            updatedFeat2.append(feat2[i])
            nFClass.append(fClass[i])
    nFeat1 = updatedFeat1[:30]
    nFeat2 = updatedFeat2[:30]
    nFClass1 = nFClass[:30]
    nFClass2 = nFClass[50:80]
    nFClass = nFClass1 + nFClass2
    for i in range(len(nFClass)):
        if nFClass[i] == 0 | nFClass[i] == 2:
            nFClass[i] = -1
    return nFeat1, nFeat2, nFClass


def signum(x):
    if x >= 1:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def learningAlgorithm(X1, X2, C1, C2, eta, m, biasF):
    # implement learning algorithm
    b = random.uniform(0, 1)
    if biasF == 0:
        w = np.array([random.uniform(0, 1), random.uniform(0, 1)]).transpose()
    else:
        w = np.random.uniform([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]).transpose()
    feat1, feat2, fClass = extractData(X1, X2)
    nFeat1, nFeat2, nFClass = updateData(feat1, feat2, fClass, C1, C2)
    for epoch in range(int(m)):
        for i in range(len(nFeat1)):

            if biasF == 0:
                X = np.array([nFeat1[i], nFeat2[i]])
                yPred = signum(w.dot(X)) + b
            else:
                X = np.array([nFeat1[i], nFeat2[i], 1])
                yPred = signum(w.dot(X))
            if yPred != nFClass[i]:
                L = nFClass[i] - yPred
                for i in range(len(w)):
                    w[i] = float(w[i]) + float(eta) * float(L) * float(X[i])
    # Testing_LearningAlgorithm(X1, X2, C1, C2, w, biasF)
    plotGraph(X1, X2, C1, C2, w, b)


def Testing_LearningAlgorithm(X1, X2, C1, C2, w, biasF):
    correct = 0
    tst_feat1, tst_feat2 = Modified_ExtractData(X1, X2, C1, 0, 1)
    tst_featt1, tst_featt2 = Modified_ExtractData(X1, X2, C2, 0, -1)
    b = random.uniform(0, 1)
    for i in range(len(tst_feat1)):
        if biasF == 0:
            X = np.array([tst_feat1[i], tst_feat2[i]])
            yPred = signum(w.dot(X)) + b
        else:
            X = np.array([tst_feat1[i], tst_feat2[i]], 1)
            yPred = signum(w.dot(X))
        if yPred == 1:
            correct += 1

    if biasF == 0:
        w = np.array([random.uniform(0, 1), random.uniform(0, 1)]).transpose()
    else:
        w = np.random.uniform([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]).transpose()
    for i in range(len(tst_featt1)):
        if biasF == 0:
            X = np.array([tst_featt1[i], tst_featt2[i]])
            yPred = signum(w.dot(X)) + b
        else:
            X = np.array([tst_featt1[i], tst_featt2[i]], 1)
            yPred = signum(w.dot(X))

        if yPred == -1:
            correct += 1
        print(yPred,"***********")
    print("Accuracy = ", (correct / 40.0) * 100)


def plotGraph(X1, X2, C1, C2, w, b):
    plt.figure("Iris Data")
    plt.xlabel("X1")
    plt.ylabel("X2")
    Lrn_Feat1, Lrn_Feat2 = Modified_ExtractData(X1, X2, names[C1], 1, 1)
    plt.scatter(Lrn_Feat1, Lrn_Feat2, color="black")
    Lrn_Feat1, Lrn_Feat2 = Modified_ExtractData(X1, X2, names[C2], 1, -1)
    plt.scatter(Lrn_Feat1, Lrn_Feat2, color="red")
    point1 = [0, (-b / w[1])]
    point2 = [(5 * w[1] - b) / w[0], 10]

    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.plot(x_values, y_values)
    print(point1)
    print(point2)

    for i in range(len(w)):
        print(w[i])

    plt.show()
