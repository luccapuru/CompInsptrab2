import random
import pandas
import numpy as np
import copy
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Retorna o vetor com o valor esperado de y para cada classe
def SaidaDesejada(classe):
    saidaDesejada = {
        1: np.array([1, 0, 0]),
        2: np.array([0, 1, 0]),
        3: np.array([0, 0, 1])
    }
    return saidaDesejada.get(classe, -1)

#Abre o arquivo do dataset e normaliza as colunas
def AbrirDados(fileName):
    #Abrindo o arquivo e armazenando em um dataframe
    df = pandas.read_csv(fileName, header=None, names=["Class", "Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium", 
    "TotalPhenol", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensity", "Hue", "OD280/OD315", "Proline"])
    #normaliza os dados do dataframe 
    min_max_scaler = preprocessing.MinMaxScaler()
    df[["Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium", 
    "TotalPhenol", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", 
    "ColorIntensity", "Hue", "OD280/OD315", "Proline"]] = min_max_scaler.fit_transform(
    df[["Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium", 
    "TotalPhenol", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", 
    "ColorIntensity", "Hue", "OD280/OD315", "Proline"]])    
    return df

#Inicia a matriz de pesos com valores entre -0.01 e 0.01
def IniciaPesos():
    #Inicia os pesos com 0
    w = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    #gera valores aleatorios e atribui a cada posicao da matriz
    for i in range(len(w)):
        for j in range(len(w[i])):
            w[i, j] = random.uniform(-0.01, 0.01)
    return w

#Separa o dataset em conjuntos de treinamento, validacao e teste
def SeperarDados(df):
    #Dividindo o dataset de maneira aleatoria
    train, validate, test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.85*len(df))])
    #resetando os indices de cada subconjunto
    test = test.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    train = train.reset_index(drop=True)
    return train, validate, test

#Realiza o treinamento e a validacao
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
    bestactrein = 0
    bestacval = 0
    bestt = t
    bestw = copy.deepcopy(w)
    bestE = E 
    
    Egraph = [] #lista para a posterior criacao do grafico de convergencia do erro
    acertos = 0 #contador de acertos para calculo da acuracia

    #Treino
    while t < maxt and E > 0:
        # print("Epoch", t)
        E = 0
        Ev = 0
        acertos = 0
        ypred = []
        ytrue = [] #zernado valores para a iteracao
        for i in range(train["Class"].count()):
            somaErro = 0
            x = np.array([[1], [train["Alcohol"][i]], [train["MalicAcid"][i]],  [train["Ash"][i]], [train["AlcalinityAsh"][i]], 
            [train["Magnesium"][i]], [train["TotalPhenol"][i]], [train["Flavanoids"][i]], [train["NonflavanoidPhenols"][i]], 
            [train["Proanthocyanins"][i]], [train["ColorIntensity"][i]], [train["Hue"][i]], [train["OD280/OD315"][i]], 
            [train["Proline"][i]]]) #montagem do vetor x
            y = w.dot(x) #calculo de y pela multiplicacao de w por x 


            #matriz de confusao
            ymax  = np.argmax(y)
            ypred.append(ymax+1)
            ytrue.append(train["Class"][i])

            for j in range(len(y)): #Limiar
                if y[j] > 0:
                    y[j] = 1
                else:
                    y[j] = 0
            y = y.transpose()
            e = np.subtract(SaidaDesejada(train["Class"][i]), y) #calculo do erro
            e = e.transpose()
            w = np.add(w, taxaApren * e.dot(x.transpose())) #atualizacao dos pesos
            #calculo do erro acumulado
            for i in range(len(e)):
                somaErro += float(e[i]**2)
                E += float(e[i]**2)
            #incrementa o contador de acertos
            if somaErro == 0:
                acertos += 1
        Egraph.append(E) #salvando valor dos erros para o grafico
        acuracia = acertos/train["Class"].count() #calculo da acuracia do treinamento desta epoca
        acertos = 0
        t += 1
        #salvando a melhor matriz de peso
        if E < bestE:
            # bestt = t
            bestw = copy.deepcopy(w)
            bestE = E
        #Fim treino

        #montagem da matriz de confusao caso a acuracia tenha melhorado (salvar a melhor)
        if acuracia > bestactrein:
            bestactrein = acuracia
            disptrain = ConfusionMatrixDisplay(confusion_matrix(ytrue, ypred), display_labels = ["1", "2", "3"]) 
        
        #Validacao
        ypred = []
        ytrue = []
        for i in range(validate["Class"].count()):
            somaErro = 0
            x = np.array([[1], [validate["Alcohol"][i]], [validate["MalicAcid"][i]],  [validate["Ash"][i]], [validate["AlcalinityAsh"][i]], 
            [validate["Magnesium"][i]], [validate["TotalPhenol"][i]], [validate["Flavanoids"][i]], [validate["NonflavanoidPhenols"][i]], 
            [validate["Proanthocyanins"][i]], [validate["ColorIntensity"][i]], [validate["Hue"][i]], [validate["OD280/OD315"][i]], 
            [validate["Proline"][i]]])
            y = w.dot(x)

            #matriz de confusao
            ymax  = np.argmax(y)
            ypred.append(ymax+1)
            ytrue.append(validate["Class"][i])            

            for j in range(len(y)): #Limiar
                if y[j] > 0:
                    y[j] = 1
                else:
                    y[j] = 0
            y = y.transpose()
            e = np.subtract(SaidaDesejada(validate["Class"][i]), y)
            e = e.transpose()
            for i in range(len(e)):
                somaErro += float(e[i]**2)
                Ev += float(e[i]**2)
            if somaErro == 0:
                acertos += 1
        acuracia = acertos/validate["Class"].count()
        #Se a acuracia nao melhorou, decrementa a variavel k. Se a variavel k chegar a zero a execucao eh interrompida
        if acuracia > bestacval:
            bestacval = acuracia
            k = kaux
            dispval = ConfusionMatrixDisplay(confusion_matrix(ytrue, ypred), display_labels = ["1", "2", "3"])
        else: 
            k -= 1

    #plotando grafico do erro
    fig, ax = plt.subplots()
    ax.plot(range(len(Egraph)), Egraph, label = "Erro acumulado")
    ax.set_xlabel("Épocas")  
    ax.set_ylabel("Erro")
    ax.legend()
    matplotlib.pyplot.savefig("teste2")

    #plotando matrizes de confusao
    disptrain.plot()
    matplotlib.pyplot.savefig("train2")
    dispval.plot()
    matplotlib.pyplot.savefig("val2")
    return bestw, bestactrein, bestacval

#Realiza o Teste (processo semelhante ao treinamento)
def Teste(w, test):
    E = 0
    acertos = 0
    ypred = []
    ytrue = []
    for i in range(test["Class"].count()):
        somaErro = 0
        x = np.array([[1], [test["Alcohol"][i]], [test["MalicAcid"][i]],  [test["Ash"][i]], [test["AlcalinityAsh"][i]], 
            [test["Magnesium"][i]], [test["TotalPhenol"][i]], [test["Flavanoids"][i]], [test["NonflavanoidPhenols"][i]], 
            [test["Proanthocyanins"][i]], [test["ColorIntensity"][i]], [test["Hue"][i]], [test["OD280/OD315"][i]], 
            [test["Proline"][i]]])
        y = w.dot(x)

        ymax  = np.argmax(y)
        ypred.append(ymax+1)
        ytrue.append(test["Class"][i]) 

        for j in range(len(y)):
            if y[j] > 0:
                y[j] = 1
            else:
                y[j] = 0
        y = y.transpose()
        e = np.subtract(SaidaDesejada(test["Class"][i]), y)
        e = e.transpose()
        for i in range(len(e)):
            somaErro += float(e[i]**2)
            E += float(e[i]**2)
        if somaErro == 0:
            acertos += 1
    acuracia = acertos/test["Class"].count()

    #Plotando Matriz de confusao
    disptest = ConfusionMatrixDisplay(confusion_matrix(ytrue, ypred), display_labels = ["1", "2", "3"])
    disptest.plot()
    matplotlib.pyplot.savefig("test2")
    return acuracia

maxt = input("Digite o numero maximo de iteracoes: ")
#fileName = input("Digite o nome do arquivo: ")
fileName = "wine.data"
taxaApren = input("Digite a taxa de Aprendizado: ")
df = AbrirDados(fileName)
train, validate, test = SeperarDados(df)
bestw, actrein, acval = Treino(int(maxt), float(taxaApren), train, validate)
actest = Teste(bestw, test)
print("W final:\n", bestw)
print("Acuracia do treino:", actrein)
print("Acuracia do validacao:", acval)
print("Acuracia do teste:", actest)

# #Teste de variacao de parametros
# taxaApren = [0.01, 0.1, 0.5]
# actreinlist = []
# acvallist = []
# actestlist = []
# actreinMedia = []
# acvalMedia = []
# actestMedia = []
# for i in taxaApren:
#     print("teste")
#     for j in range(100):
#         bestw, actrein, acval = Treino(int(maxt), i, train, validate)
#         actest = Teste(bestw, test)
#         actreinlist.append(actrein)
#         acvallist.append(acval)
#         actestlist.append(actest)
#     actreinMedia.append(sum(actreinlist)/len(actreinlist))
#     acvalMedia.append(sum(acvallist)/len(acvallist))
#     actestMedia.append(sum(actestlist)/len(actestlist))

# f = open("exp2.txt", "a")
# f.write("Acuracia Treino, " + str(actreinMedia) + "\t")
# f.write("Acuracia Validacao, " + str(acvalMedia) + "\n")
# f.write("Acuracia Teste, " + str(actestMedia) + "\n")
# f.close()
