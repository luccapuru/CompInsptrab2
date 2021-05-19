import pandas
import numpy as np
import random
from sklearn import preprocessing

df2 = pandas.read_csv("wine.data", header=None, names=["Class", "Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium", 
    "TotalPhenol", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensity", "Hue", "OD280/OD315", "Proline"])
x2 = df2.values #returns a numpy array
#print(x2)
min_max_scaler = preprocessing.MinMaxScaler()

df2[["Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium", 
    "TotalPhenol", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", 
    "ColorIntensity", "Hue", "OD280/OD315", "Proline"]] = min_max_scaler.fit_transform(
    df2[["Alcohol", "MalicAcid", "Ash", "AlcalinityAsh", "Magnesium", 
    "TotalPhenol", "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", 
    "ColorIntensity", "Hue", "OD280/OD315", "Proline"]])

#x_scaled = min_max_scaler.fit_transform(x2)
#df2 = pandas.DataFrame(x_scaled)
print(df2)