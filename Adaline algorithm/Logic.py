import random
from tkinter import messagebox
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

Sitosa = []
VersiColor = []
Verginica = []
General_Class = []


def ReadFile():
    data = np.genfromtxt("IrisData.txt", delimiter=',', dtype=str)

    for tuple in data:
        if tuple[0] == "X1":
            continue
        temp = [float(tuple[0]), float(tuple[1]), float(tuple[2]), float(tuple[3])]
        if tuple[-1] == "Iris-setosa":
            tuple[-1] = 0
            Sitosa.append(temp)
        elif tuple[-1] == "Iris-versicolor":
            tuple[-1] = 1
            VersiColor.append(temp)
        elif tuple[-1] == "Iris-virginica":
            tuple[-1] = 2
            Verginica.append(temp)

    General_Class.append(Sitosa)
    General_Class.append(VersiColor)
    General_Class.append(Verginica)


def DrawData(x1, x2):
    if x1 == x2:
        return messagebox.showinfo(title="Error Message", message="Can't choose same features")
    else:
        X1, X2, X3 = [], [], []
        Y1, Y2, Y3 = [], [], []

        for row in General_Class[0]:
            X1.append(row[x1])
            Y1.append(row[x2])
        for row in General_Class[1]:
            X2.append(row[x1])
            Y2.append(row[x2])
        for row in General_Class[2]:
            X3.append(row[x1])
            Y3.append(row[x2])

        X1_normalized = preprocessing.normalize([X1])
        Y1_normalized = preprocessing.normalize([Y1])

        X2_normalized = preprocessing.normalize([X2])
        Y2_normalized = preprocessing.normalize([Y2])

        X3_normalized = preprocessing.normalize([X3])
        Y3_normalized = preprocessing.normalize([Y3])
        plt.figure("Iris-Dataset")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.scatter(X1_normalized, Y1_normalized, color="red")
        plt.scatter(X2_normalized, Y2_normalized, color="green")
        plt.scatter(X3_normalized, Y3_normalized, color="blue")
        plt.show()


def signum(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


def Adaline_LearningAlgorithm(X1, X2, C1, C2, input_MSE, eta, m, biasF):
    random.shuffle(Sitosa)
    random.shuffle(VersiColor)
    random.shuffle(Verginica)
    # generate learning,testing samples mapped to 1 or -1:

    Learning_Samples = []
    Testing_Samples = []

    for tuple1 in General_Class[C1]:
        if Learning_Samples.count == 30:
            break
        else:
            Learning_Samples.append([tuple1[X1], tuple1[X2], 1])

    for tuple2 in General_Class[C2]:
        if Learning_Samples.count == 60:
            break
        Learning_Samples.append([tuple2[X1], tuple2[X2], -1])

    for idx in range(len(General_Class[C1])):
        if idx >= 30:
            tuple1 = General_Class[C1][idx]
            tuple2 = General_Class[C2][idx]
            Testing_Samples.append([tuple1[X1], tuple1[X2], 1])
            Testing_Samples.append([tuple2[X1], tuple2[X2], -1])
            idx += 1

    random.shuffle(Learning_Samples)
    random.shuffle(Testing_Samples)

    if int(biasF) == 0:
        Weight = np.array([0, 0])
    else:
        Weight = np.array([random.uniform(0, 1), 0, 0])
    for epoch in range(int(m * 50)):
        for i in range(
                len(Learning_Samples)):  # row[0]=X1 , row[1]=X2 , row[2]= (1 or -1) according to class1 or class2
            if int(biasF) == 1:
                X = np.array([1, Learning_Samples[i][0], Learning_Samples[i][1]])
            else:
                X = np.array([Learning_Samples[i][0], Learning_Samples[i][1]])
            Y_hat = (Weight.transpose()).dot(X)
            Error = Learning_Samples[i][2] - Y_hat
            Weight = Weight + float(eta) * Error * X
        sum = 0.0
        for j in range(len(Learning_Samples)):
            if int(biasF) == 1:
                X = np.array([1, Learning_Samples[j][0], Learning_Samples[j][1]])
            else:
                X = np.array([Learning_Samples[j][0], Learning_Samples[j][1]])
            Final_Y_Hat = (Weight.transpose()).dot(X)
            E = Learning_Samples[j][2] - Final_Y_Hat
            E = round(E, 2) ** 2
            sum += E

        MSE = round(sum, 2) / float(len(Learning_Samples))
        MSE = round(MSE, 2)
        # print(MSE)
        if MSE <= float(input_MSE):
            break
        else:
            continue
    plotGraph(Learning_Samples, Weight, biasF)
    Testing_LearningAlgorithm(Testing_Samples, Weight, biasF)


def plotGraph(learning_data, Weight, Bias):
    plt.figure("Iris Data")
    plt.xlabel("X1")
    plt.ylabel("X2")
    X1, X2 = [], []
    Y1, Y2 = [], []
    for row in learning_data:
        if row[2] == 1:
            X1.append(row[0])
            Y1.append(row[1])
        elif row[2] == -1:
            X2.append(row[0])
            Y2.append(row[1])
    plt.scatter(X1, Y1, color="black")
    plt.scatter(X2, Y2, color="cyan")
    point_01 = [4, (-1 * (Bias + Weight[0])) / 4 * Weight[1]]
    point_02 = [(-1 * (Bias + 7 * Weight[1])) / Weight[0], 7]
    x_values = [point_01[0], point_02[0]]
    y_values = [1.5 + point_01[1], 1.5 + point_02[1]]
    plt.plot(x_values, y_values, color="green", linewidth=3)
    plt.show()


def Testing_LearningAlgorithm(testing_data, Weight, Bias_Flag):
    correct = 0
    ACTUAL_VALUES = []
    PREDICTED_VALUES = []
    for i in range(
            len(testing_data)):
        if int(Bias_Flag) == 1:
            X = np.array([1, testing_data[i][0], testing_data[i][1]])
        else:
            X = np.array([testing_data[i][0], testing_data[i][1]])
        predicted = (Weight.transpose()).dot(X)
        Y_Predicted = signum(predicted)
        Y_actual = testing_data[i][2]
        ACTUAL_VALUES.append(Y_actual)
        PREDICTED_VALUES.append(Y_Predicted)
        if Y_Predicted == Y_actual:
            correct += 1
    print("Accuracy=", round(((correct / 40.0) * 100), 2), "%")
    print("Confusion Matrix : "+confusion_matrix(ACTUAL_VALUES,PREDICTED_VALUES))
