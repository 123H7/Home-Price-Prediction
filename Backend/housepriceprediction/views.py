from django.shortcuts import render

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def home(req):
    return render(req, "home.html")


def predict(req):
    return render(req, "predict.html")


def result(req):
    data = pd.read_csv(r"E:\All in One Learning\Django\MachineLearning\USA_Housing.csv")
    data = data.drop(["Address"], axis=1)
    X = data.drop("Price", axis=1)
    Y = data["Price"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    var1 = float(req.GET["v1"])
    var2 = float(req.GET["v2"])
    var3 = float(req.GET["v3"])
    var4 = float(req.GET["v4"])
    var5 = float(req.GET["v5"])

    pred = model.predict(np.array([var1, var2, var3, var4, var5]).reshape(1, -1))
    pred = round(pred[0])

    price = "The predicted price is $" + str(pred)

    return render(req, "predict.html", {"result2": price})
