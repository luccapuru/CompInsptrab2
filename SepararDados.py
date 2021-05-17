import pandas
import numpy

#df = pandas.read_csv("iris.data")
df = pandas.read_csv("iris.data", header=None, names=["SepalLength", "SepalWidth","PetalLength", "PetalWidth", "Class"])
