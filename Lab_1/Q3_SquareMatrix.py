import numpy as np

def Matrix_Power(A, m):
    A = np.array(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")

    result = np.linalg.matrix_power(A, m)
    return result

if __name__ == "__main__":
    Matrix = [[2, 2], 
              [2, 2]]
    Power = 2
    print("Matrix Square:", Matrix_Power(Matrix, Power))
