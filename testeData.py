import pandas
import numpy as np

#df = pandas.read_csv("iris.data")
df = pandas.read_csv("iris.data", header=None, names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])
print(type(df["Class"][0]))
df2 = pandas.read_csv("wine.data", header=None, names=["Class", "Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium", 
    "TotalPhenol", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensity", "Hue", "OD280/OD315", "Proline"])

# def SaidaDesejada(classeStr):
#     saidaDesejada = {
#         "Iris-setosa": np.array([1, 0, 0]),
#         "Iris-versicolor": np.array([0, 1, 0]),
#         "Iris-virginica": np.array([0, 0, 1])
#     }
#     return saidaDesejada.get(classeStr, -1)

def SaidaDesejada(classeStr):
    saidaDesejada = {
        1: np.array([1, 0, 0]),
        2: np.array([0, 1, 0]),
        3: np.array([0, 0, 1])
    }
    return saidaDesejada.get(classeStr, -1)

x = 177
#print(df["Class"][x])
print(SaidaDesejada(df2["Class"][x]))
#print(df["SepalLength"].count())
print(df2["Class"].count(), "AAAA")
print(df2)