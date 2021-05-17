import random
import pandas



df = pandas.read_csv("iris.data", header=None, names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])

def AbrirDados(fileName):
    return pandas.read_csv(fileName, header=None, 
    names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])

def Perceptron(fileName, maxt):
    w = [0, 0, 0, 0, 0]
    df = AbrirDados(fileName)
    t = 0
    E = 1
    while t < maxt and E > 0:
        



