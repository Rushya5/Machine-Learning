import numpy as np
Matrix = np.zeros((4,3))
print("Matrix of zeros with 4 row and 3 columns:\n",Matrix)
Matrix = np.random.randint(1,100,size=(4,3))
array = Matrix.flatten()

print("Matrix filled with Random Numbers:\n",Matrix)
print("single dimensional array of the Matrix:\n",array)