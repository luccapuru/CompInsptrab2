import random
import pandas
import numpy as np
import copy
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib

#df = pandas.read_csv("iris.data", header=None, names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])

def SaidaDesejada(classeStr):
    saidaDesejada = {
        1: np.array([1, 0, 0]),
        2: np.array([0, 1, 0]),
        3: np.array([0, 0, 1])
    }
    return saidaDesejada.get(classeStr, -1)

def AbrirDados(fileName):
    df = pandas.read_csv(fileName, header=None, names=["Class", "Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium", 
    "TotalPhenol", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensity", "Hue", "OD280/OD315", "Proline"])
    min_max_scaler = preprocessing.MinMaxScaler()
    df[["Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium", 
    "TotalPhenol", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", 
    "ColorIntensity", "Hue", "OD280/OD315", "Proline"]] = min_max_scaler.fit_transform(
    df[["Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium", 
    "TotalPhenol", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", 
    "ColorIntensity", "Hue", "OD280/OD315", "Proline"]])    
    return df

#def AbrirDados(fileName):
    #return pandas.read_csv(fileName, header=None, names=["Class", "Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium", 
    #"TotalPhenol", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensity", "Hue", "OD280/OD315", "Proline"])

def IniciaPesos():
    w = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
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
    #Treino
    while t < maxt and E > 0:
        print("Epoch", t)
        E = 0
        Ev = 0
        acertos = 0
        for i in range(train["Class"].count()):
            somaErro = 0
            x = np.array([[1], [train["Alcohol"][i]], [train["MalicAcid"][i]],  [train["Ash"][i]], [train["AlcalinityAsh"][i]], 
            [train["Magnesium"][i]], [train["TotalPhenol"][i]], [train["Flavanoids"][i]], [train["NonflavanoidPhenols"][i]], 
            [train["Proanthocyanins"][i]], [train["ColorIntensity"][i]], [train["Hue"][i]], [train["OD280/OD315"][i]], 
            [train["Proline"][i]]])
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
        print("E", E)
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
            x = np.array([[1], [validate["Alcohol"][i]], [validate["MalicAcid"][i]],  [validate["Ash"][i]], [validate["AlcalinityAsh"][i]], 
            [validate["Magnesium"][i]], [validate["TotalPhenol"][i]], [validate["Flavanoids"][i]], [validate["NonflavanoidPhenols"][i]], 
            [validate["Proanthocyanins"][i]], [validate["ColorIntensity"][i]], [validate["Hue"][i]], [validate["OD280/OD315"][i]], 
            [validate["Proline"][i]]])
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
    matplotlib.pyplot.savefig("teste2")

    return bestw

def Teste(w, test):
    E = 0
    acertos = 0
    for i in range(test["Class"].count()):
        somaErro = 0
        x = np.array([[1], [test["Alcohol"][i]], [test["MalicAcid"][i]],  [test["Ash"][i]], [test["AlcalinityAsh"][i]], 
            [test["Magnesium"][i]], [test["TotalPhenol"][i]], [test["Flavanoids"][i]], [test["NonflavanoidPhenols"][i]], 
            [test["Proanthocyanins"][i]], [test["ColorIntensity"][i]], [test["Hue"][i]], [test["OD280/OD315"][i]], 
            [test["Proline"][i]]])
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

maxt = input("Digite o numero maximo de iteracoes: ")
#fileName = input("Digite o nome do arquivo: ")
fileName = "wine.data"
taxaApren = input("Digite a taxa de Aprendizado: ")
df = AbrirDados(fileName)
print(df)
train, validate, test = SeperarDados(df)
bestw = Treino(int(maxt), float(taxaApren), train, validate)
Teste(bestw, test)
