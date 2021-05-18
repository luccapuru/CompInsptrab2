import pandas
import numpy as np
import random

def IniciaPesos():
    w = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    for i in range(len(w)):
        for j in range(len(w[i])):
            #print(w[i, j])
            w[i, j] = random.uniform(-0.01, 0.01)
            #w[i, j] = 1
            #print(w[i, j])
    return w

def SaidaDesejada(classeStr):
    saidaDesejada = {
        "Iris-setosa": np.array([1, 0, 0]),
        "Iris-versicolor": np.array([0, 1, 0]),
        "Iris-virginica": np.array([0, 0, 1])
    }
    return saidaDesejada.get(classeStr, -1)

#df = pandas.read_csv("iris.data")
df = pandas.read_csv("iris.data", header=None, names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])
#print(df, df.count())
#print(df.sample(frac=0.7).reset_index(drop=True))
train, validate, test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.85*len(df))])
print("train\n", train)
#print("validate", validate.count(), validate)
#print("test", test.count(), test)
print("--------------------")
train = train.reset_index(drop=True)
print("train\n", train)
x = np.array([[1], [train["SepalLength"][0]], [train["SepalWidth"][0]], 
[train["PetalLength"][0]], [train["PetalWidth"][0]]])
print(x)
print(x[0, 0])

taxaApren = 0.5
w = IniciaPesos()
print("w", w)
y = w.dot(x)
print("y", y)
for j in range(len(y)):
    if y[j] > 0:
        y[j] = 1
    else:
        y[j] = 0
print("y2", y)
y = y.transpose()
e = np.subtract(SaidaDesejada(train["Class"][0]), y)
print(SaidaDesejada(train["Class"][0]))
print("e", e)
e = e.transpose()
print("e2", e.transpose())
aux = taxaApren * e.dot(x.transpose())
print("aux", aux)
w = np.add(w, taxaApren * e.dot(x.transpose()))
print("w2", w)
E = 0
for i in range(len(e)):
    E += float(e[i]**2)
print("E", E)