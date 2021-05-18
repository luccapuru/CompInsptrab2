import random
import pandas
import numpy as np
import copy


#df = pandas.read_csv("iris.data", header=None, names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])

def SaidaDesejada(classeStr):
    saidaDesejada = {
        "Iris-setosa": np.array([1, 0, 0]),
        "Iris-versicolor": np.array([0, 1, 0]),
        "Iris-virginica": np.array([0, 0, 1])
    }
    return saidaDesejada.get(classeStr, -1)

def AbrirDados(fileName):
    return pandas.read_csv(fileName, header=None, 
    names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])

def IniciaPesos():
    w = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    for i in range(len(w)):
        for j in range(len(w[i])):
            #print(w[i, j])
            w[i, j] = random.uniform(-0.01, 0.01)
            #w[i, j] = 1
            #print(w[i, j])
    return w

def Perceptron(fileName, maxt, taxaApren):
    w = IniciaPesos()
    df = AbrirDados(fileName)
    train, validate, test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.85*len(df))])
    test = test.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    train = train.reset_index(drop=True)
    t = 0
    E = 1000
    bestt = t
    bestw = copy.deepcopy(w)
    bestE = E
    #while t < maxt and E > 0:
    while E > 0:
        E = 0
        for i in range(train["SepalLength"].count()):
            x = np.array([[1], [train["SepalLength"][i]], [train["SepalWidth"][i]], 
            [train["PetalLength"][i]], [train["PetalWidth"][i]]])
            y = w.dot(x)
            for j in range(len(y)):
                if y[j] > 0:
                    y[j] = 1
                else:
                    y[j] = 0
            y = y.transpose()
            e = np.subtract(SaidaDesejada(train["Class"][i]), y)
            e = e.transpose()
            w = np.add(w, taxaApren * e.dot(x.transpose()))
            for i in range(len(e)):
                E += float(e[i]**2)
        t += 1
        print("E", E)
        if E < bestE:
            bestE = E
            bestt = t
            bestw = copy.deepcopy(w)
            bestE = E
        print("Epoch", t)
    return w, bestw, E, bestE, t, bestt

print(Perceptron("iris.data", 1000, 0.01))


