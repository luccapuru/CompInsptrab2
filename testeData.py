import pandas
import numpy as np

#df = pandas.read_csv("iris.data")
df = pandas.read_csv("iris.data", header=None, names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])
print(type(df["Class"][0]))

def SaidaDesejada(classeStr):
    saidaDesejada = {
        "Iris-setosa": np.array([1, 0, 0]),
        "Iris-versicolor": np.array([0, 1, 0]),
        "Iris-virginica": np.array([0, 0, 1])
    }
    return saidaDesejada.get(classeStr, -1)

x = 92
print(df["Class"][x])
print(SaidaDesejada(df["Class"][x]))
print(df["SepalLength"].count())