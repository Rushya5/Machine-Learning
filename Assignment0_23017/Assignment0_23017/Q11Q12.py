import numpy as np
import matplotlib.pyplot as plt

V1 = np.random.rand(100)
V1_sorted = np.sort(V1)

plt.figure(figsize=(8, 4))
plt.plot(V1_sorted, color='red', label='V1 (Sorted)')
plt.title("Sorted Vector V1 (Red Color)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

V2 = V1_sorted ** 2 

plt.figure(figsize=(10, 6))
plt.plot(V1_sorted, label="V1 (Original)", color='blue')
plt.plot(V2, label="V2 (Squared)", color='orange')
plt.title("Comparison of V1 and V2")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
