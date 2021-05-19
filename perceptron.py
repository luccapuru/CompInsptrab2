import random
import pandas
import numpy as np
import copy
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

#df = pandas.read_csv("iris.data", header=None, names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])

def SaidaDesejada(classeStr):
    saidaDesejada = {
        "Iris-setosa": np.array([1, 0, 0]),
        "Iris-versicolor": np.array([0, 1, 0]),
        "Iris-virginica": np.array([0, 0, 1])
    }
    return saidaDesejada.get(classeStr, -1)

def AbrirDados(fileName):
    df = pandas.read_csv(fileName, header=None, 
    names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])
    min_max_scaler = preprocessing.MinMaxScaler()
    df[["SepalLength", "SepalWidth","PetalLength", "PetalWidth"]] = min_max_scaler.fit_transform(df[["SepalLength", "SepalWidth","PetalLength", "PetalWidth"]])
    return df

def IniciaPesos():
    w = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    for i in range(len(w)):
        for j in range(len(w[i])):
            #print(w[i, j])
            w[i, j] = random.uniform(-0.01, 0.01)
            #w[i, j] = 1
            #print(w[i, j])
    return w

def SeperarDados(df):
    train, validate, test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.85*len(df))])
    test = test.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    train = train.reset_index(drop=True)
    return train, validate, test

def Treino(maxt, taxaApren, train, validate):
    k = 100
    kaux = k
    w = IniciaPesos()
    t = 0
    E = 1000
    Ev = 0 #erro de validacao
    bestt = t
    bestw = copy.deepcopy(w)
    bestE = E
    Egraph = []
    bestactrein = 0
    bestacval = 0
    acertos = 0
    #while E > 0:
    #Treino
    while t < maxt and E > 0 and k >= 0:
        print("Epoch", t)
        E = 0
        Ev = 0
        acertos = 0
        for i in range(train["Class"].count()):
            somaErro = 0
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
            #print("e", e)
            w = np.add(w, taxaApren * e.dot(x.transpose()))
            for i in range(len(e)):
                somaErro += float(e[i]**2)
                E += float(e[i]**2)
            if somaErro == 0:
                acertos += 1
        Egraph.append(E)
        acuracia = acertos/train["Class"].count()
        print("acu1", acuracia)
        print("acertos trein", acertos)
        acertos = 0
        t += 1
        if E < bestE:
            bestE = E
            bestt = t
            bestw = copy.deepcopy(w)
            bestE = E
        #Fim treino
        
        if acuracia > bestactrein:
            bestactrein = acuracia 

        #Validacao
        for i in range(validate["Class"].count()):
            somaErro = 0
            x = np.array([[1], [validate["SepalLength"][i]], [validate["SepalWidth"][i]], 
            [validate["PetalLength"][i]], [validate["PetalWidth"][i]]])
            y = w.dot(x)
            for j in range(len(y)): #Limiar
                if y[j] > 0:
                    y[j] = 1
                else:
                    y[j] = 0
            y = y.transpose()
            e = np.subtract(SaidaDesejada(validate["Class"][i]), y)
            e = e.transpose()
            #print("e", e)
            #w = np.add(w, taxaApren * e.dot(x.transpose()))
            for i in range(len(e)):
                somaErro += float(e[i]**2)
                Ev += float(e[i]**2)
            #print("somaErro", somaErro)
            if somaErro == 0:
                #print("Acerto!")
                acertos += 1
        print("EV:", Ev)
        print("E", bestE)
        acuracia = acertos/validate["Class"].count()
        print("acertos val", acertos)
        print("acu2", acuracia)
        if acuracia > bestacval:
            bestacval = acuracia
            k = kaux
        else: 
            k -= 1
        print("k", k)
    #return w, bestw, E, bestE, t, bestt
    print("bestactrein", bestactrein, "bestacval", bestacval)

    #plotar grafico do erro
    fig, ax = plt.subplots()
    ax.plot(range(len(Egraph)), Egraph, label = "Erro acumulado")
    #ax.plot(range(len(hMax)), hMax, label = "Aptidão Máxima")
    ax.set_xlabel("Épocas")  
    ax.set_ylabel("Erro")
    ax.legend()
    matplotlib.pyplot.savefig("teste")
    return bestw

def Teste(w, test):
    E = 0
    acertos = 0
    for i in range(test["Class"].count()):
        somaErro = 0
        x = np.array([[1], [test["SepalLength"][i]], [test["SepalWidth"][i]], 
        [test["PetalLength"][i]], [test["PetalWidth"][i]]])
        y = w.dot(x)
        for j in range(len(y)):
            if y[j] > 0:
                y[j] = 1
            else:
                y[j] = 0
        y = y.transpose()
        e = np.subtract(SaidaDesejada(test["Class"][i]), y)
        e = e.transpose()
        #print("e", e)
        #w = np.add(w, taxaApren * e.dot(x.transpose()))
        for i in range(len(e)):
            somaErro += float(e[i]**2)
            E += float(e[i]**2)
        if somaErro == 0:
            acertos += 1
    acuracia = acertos/test["Class"].count()
    print("acu3", acuracia)
    print("acertos teste", acertos)


#print(Perceptron("iris.data", 1000, 0.01))

maxt = input("Digite o numero maximo de iteracoes: ")
#fileName = input("Digite o nome do arquivo: ")
fileName = "iris.data"
taxaApren = input("Digite a taxa de Aprendizado: ")
df = AbrirDados(fileName)
train, validate, test = SeperarDados(df)
bestw = Treino(int(maxt), float(taxaApren), train, validate)
Teste(bestw, test)
