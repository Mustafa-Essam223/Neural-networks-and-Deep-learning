from tkinter import *
from tkinter import messagebox

import Logic
from Logic import drawData


def createMenu(mOptions, x, y, dIndex, defaultV, window):
    defaultV.set(mOptions[dIndex])
    m = OptionMenu(window, defaultV, *mOptions)
    m.place(x=x, y=y)


def createLabel(window, text, x, y):
    menuL = StringVar()
    ml = Label(window, textvariable=menuL)
    menuL.set(text)
    ml.place(x=x, y=y)


def getIndex(option):
    if option.get() == "X1":
        return 0
    elif option.get() == "X2":
        return 1
    elif option.get() == "X3":
        return 2
    else:
        return 3


def getClassIndex(option):
    if option.get() == "C1":
        return 0
    elif option.get() == "C2":
        return 1
    else:
        return 2


top = Tk()
top.geometry("600x600")

title = StringVar()
l1 = Label(top, textvariable=title)
title.set("Iris Dataset")

l1.pack()

options = ["X1", "X2", "X3", "X4"]
default1 = StringVar(top)
default2 = StringVar(top)

createLabel(top, "Choose feature 1", 70, 40)
createMenu(options, 180, 37, 0, default1, top)
createLabel(top, "Choose feature 2", 250, 40)
createMenu(options, 360, 37, 1, default2, top)

btn1 = Button(top, text="Draw Dataset", command=lambda: Logic.drawData(getIndex(default1), getIndex(default2)))
btn1.place(x=480, y=40)

createLabel(top, "Choose feature 1", 70, 150)
default3 = StringVar(top)
createMenu(options, 180, 147, 0, default3, top)

createLabel(top, "Choose feature 2", 250, 150)
default4 = StringVar(top)
createMenu(options, 360, 147, 1, default4, top)

# btn10 = Button(top, text="Evaluate Perceptron", command=lambda: Logic.Perceptron())
# btn10.place(x=450, y=200)


default5 = StringVar(top)
cOptions = ["C1", "C2", "C3"]
createLabel(top, "Choose class1", 70, 210)
createMenu(cOptions, 180, 200, 0, default5, top)

default6 = StringVar(top)
createLabel(top, "Choose class2", 250, 210)
createMenu(cOptions, 360, 200, 1, default6, top)

createLabel(top, "Enter learning rate", 50, 253)
eta = Entry(top)
eta.place(x=160, y=253)

createLabel(top, "Enter number of epochs", 300, 253)
m = Entry(top)
m.place(x=440, y=253)

bias = IntVar()
cb1 = Checkbutton(top, text="bias", variable=bias)
cb1.place(x=200, y=295)

btn2 = Button(top, text="Test",
              command=lambda: Logic.learningAlgorithm(getIndex(default3), getIndex(default4), getClassIndex(default5),
                                                      getClassIndex(default6), eta.get(), m.get(), bias.get()))
btn2.place(x=300, y=295)


top.mainloop()
