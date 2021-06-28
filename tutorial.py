from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose = 0)
_, accuracy = model.evaluate(X, y)
# evaluate the keras model
print('Accuracy: %.2f' % (accuracy*100))
# acuracia = []

# for i in range(100):
# 	print(i)
# 	model.fit(X, y, epochs=150, batch_size=10, verbose = 0)
# 	_, accuracy = model.evaluate(X, y)
# 	acuracia.append(accuracy)
# 	print('Accuracy: %.2f' % (accuracy*100))

# acuraciaMedia = sum(acuracia)/len(acuracia)
# print('AccuracyMed: %.2f' % (acuraciaMedia*100))

print(len(y))
print(len(dataset))
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))