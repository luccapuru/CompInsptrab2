import numpy as np

A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
B = np.array([[1], [2], [3], [4]])
print(A.dot(B))
#print(B.dot(A))

x = np.array([[1, 2, 3], [4, 5, 6]])
#y = np.array([[7, 8], [9, 10]])
#print(np.subtract(x,y))
print(len(x[0]), len(x))