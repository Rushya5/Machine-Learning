import numpy as np

V1 = np.random.randint(100, size = 100)
#Q1
Sorted = np.sort(V1)
#Q2
Scaled_3 = V1 * 3
#Q3
Mean = np.mean(V1)
standard_deviation = np.std(V1)


print("Original Vector:",V1)
print("Multipled V1 with 3 to scaled :",Scaled_3)
print("Sorted in increasing Order:",Sorted)
print(f"Mean : {Mean} and  standard deviation : {standard_deviation}")