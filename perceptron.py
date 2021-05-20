import random
import pandas
import numpy as np
import copy
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib

#df = pandas.read_csv("iris.data", header=None, names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])

def SaidaDesejada(classe):
    saidaDesejada = {
        "Iris-setosa": np.array([1, 0, 0]),
        "Iris-versicolor": np.array([0, 1, 0]),
        "Iris-virginica": np.array([0, 0, 1]),
        1: "Iris-setosa",
        2: "Iris-versicolor",
        3: "Iris-virginica"
    }
    return saidaDesejada.get(classe, -1)

# def SaidaInvertida(classe):
#     saidaInvertida = {
#         "Iris-setosa": 1,
#         "Iris-versicolor": 2,
#         "Iris-virginica": 3,
#         # 1: "Iris-setosa",
#         # 2: "Iris-versicolor",
#         # 3: "Iris-virginica"
#     }
#     return saidaInvertida.get(classe, -1)


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
            w[i, j] = random.uniform(-0.01, 0.01)
    return w

def SeperarDados(df):
    train, validate, test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.85*len(df))])
    test = test.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    train = train.reset_index(drop=True)
    return train, validate, test

def Treino(maxt, taxaApren, train, validate):
    #Iniciando parametros
    k = 100 #contador para o criterio de parada da validacao
    kaux = k
    w = IniciaPesos() #iniciando pesos
    t = 0 #num de iteracoes passadas
    E = 1000 #iniciar erro com numero absurdamente alto
    Ev = 0 #erro de validacao
    ypred = []
    ytrue = [] #listas para construção da matriz de confusao
    
    #salvar os melhores valores encontrados durante o treinamento
    bestt = t
    bestw = copy.deepcopy(w)
    bestE = E
    bestacval = 0
    bestactrein = 0
    

    Egraph = [] #lista para a posterior criacao do grafico de convergencia do erro
    acertos = 0 #contador de acertos para calculo da acuracia
    
    #Treino
    while t < maxt and E > 0 and k >= 0:
        #print("Epoch", t)
        E = 0
        Ev = 0
        acertos = 0
        ypred = []
        ytrue = []
        for i in range(train["Class"].count()):
            somaErro = 0
            x = np.array([[1], [train["SepalLength"][i]], [train["SepalWidth"][i]], 
            [train["PetalLength"][i]], [train["PetalWidth"][i]]])
            y = w.dot(x)

            #matriz de confusao
            ymax  = np.argmax(y)
            ypred.append(SaidaDesejada(ymax+1))
            ytrue.append(train["Class"][i])

            for j in range(len(y)): #limiar
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
        # print("acu1", acuracia)
        # print("acertos trein", acertos)
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
            # print("ypred", ypred)
            # print("ytrue", ytrue) 
            disptrain = ConfusionMatrixDisplay(confusion_matrix(ytrue, ypred), display_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]) 

        #Validacao
        ypred = []
        ytrue = []
        for i in range(validate["Class"].count()):
            somaErro = 0
            x = np.array([[1], [validate["SepalLength"][i]], [validate["SepalWidth"][i]], 
            [validate["PetalLength"][i]], [validate["PetalWidth"][i]]])
            y = w.dot(x)

            #matriz de confusao
            ymax  = np.argmax(y)
            ypred.append(SaidaDesejada(ymax+1))
            ytrue.append(validate["Class"][i])

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
        # print("EV:", Ev)
        # print("E", bestE)
        acuracia = acertos/validate["Class"].count()
        # print("acertos val", acertos)
        # print("acu2", acuracia)
        if acuracia > bestacval:
            bestacval = acuracia
            k = kaux
            dispval = ConfusionMatrixDisplay(confusion_matrix(ytrue, ypred), display_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
        else: 
            k -= 1
        # print("k", k)
    #return w, bestw, E, bestE, t, bestt
    # print("bestactrein", bestactrein, "bestacval", bestacval)

    #plotar grafico do erro
    # fig, ax = plt.subplots()
    # ax.plot(range(len(Egraph)), Egraph, label = "Erro acumulado")
    # ax.set_xlabel("Épocas")  
    # ax.set_ylabel("Erro")
    # ax.legend()
    # matplotlib.pyplot.savefig("teste")

    #Matriz Confusao
    # disptrain.plot()
    # matplotlib.pyplot.savefig("train1")
    # dispval.plot()
    # matplotlib.pyplot.savefig("val1")
    return bestw, bestactrein, bestacval

def Teste(w, test):
    E = 0
    acertos = 0
    ypred = []
    ytrue = []
    for i in range(test["Class"].count()):
        somaErro = 0
        x = np.array([[1], [test["SepalLength"][i]], [test["SepalWidth"][i]], 
        [test["PetalLength"][i]], [test["PetalWidth"][i]]])
        y = w.dot(x)

        ymax  = np.argmax(y)
        ypred.append(SaidaDesejada(ymax+1))
        ytrue.append(test["Class"][i])

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
    # print("acu3", acuracia)
    # print("acertos teste", acertos)

    #Matriz de confusao
    # disptest = ConfusionMatrixDisplay(confusion_matrix(ytrue, ypred), display_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    # disptest.plot()
    # matplotlib.pyplot.savefig("test1")
    return acuracia


#print(Perceptron("iris.data", 1000, 0.01))

maxt = input("Digite o numero maximo de iteracoes: ")
#fileName = input("Digite o nome do arquivo: ")
fileName = "iris.data"
#taxaApren = input("Digite a taxa de Aprendizado: ")
df = AbrirDados(fileName)
train, validate, test = SeperarDados(df)
#bestw, actrein, acval = Treino(int(maxt), float(taxaApren), train, validate)
#Teste(bestw, test)


#Teste de variacao de parametros
taxaApren = [0.01, 0.1, 0.5]
actreinlist = []
acvallist = []
actestlist = []
actreinMedia = []
acvalMedia = []
actestMedia = []
for i in taxaApren:
    print("teste")
    for j in range(100):
        bestw, actrein, acval = Treino(int(maxt), i, train, validate)
        actest = Teste(bestw, test)
        actreinlist.append(actrein)
        acvallist.append(acval)
        actestlist.append(actest)
    actreinMedia.append(sum(actreinlist)/len(actreinlist))
    acvalMedia.append(sum(acvallist)/len(acvallist))
    actestMedia.append(sum(actestlist)/len(actestlist))

f = open("exp1.txt", "a")
f.write("Acuracia Treino, " + str(actreinMedia) + "\t")
f.write("Acuracia Validacao, " + str(acvalMedia) + "\n")
f.write("Acuracia Teste, " + str(actestMedia) + "\n")
f.close()
    