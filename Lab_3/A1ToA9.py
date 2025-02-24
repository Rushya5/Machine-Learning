import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import minkowski

# Load and Clean the Dataset
file_path = r"Dataset\thyroid0387_UCI.xlsx"
df = pd.read_excel(file_path)

# Drop non-numeric and irrelevant column
df_cleaned = df.drop(columns=["Record ID", "referral source"], errors="ignore")

# Convert categorical values to numeric
df_cleaned["sex"] = df_cleaned["sex"].replace({'F': 0, 'M': 1})
df_cleaned = df_cleaned.replace({'f': 0, 't': 1, '?': np.nan})
df_cleaned["Condition"] = np.where(df_cleaned["Condition"] == "NO CONDITION", 0, 1)

# Convert all columns to numeric
df_cleaned = df_cleaned.apply(pd.to_numeric)

# Handle missing values (impute with column mean)
df_cleaned.fillna(df_cleaned.mean(), inplace=True)

# Define Features (X) and Target (Y)
X = df_cleaned.drop(columns=["Condition"])
Y = df_cleaned["Condition"]

# A1: Compute Class Centroids, Spread, and Interclass Distance
class1 = df_cleaned[df_cleaned["Condition"] == 0].drop(columns=["Condition"])
class2 = df_cleaned[df_cleaned["Condition"] == 1].drop(columns=["Condition"])

mean1, mean2 = class1.mean(), class2.mean()
spread1, spread2 = class1.std(), class2.std()
interclass_distance = np.linalg.norm(mean1 - mean2)

print("Interclass Distance:", interclass_distance)

# A2: Histogram of a Feature
feature = "age"  # Choose any numerical feature
plt.hist(df_cleaned[feature], bins=10, edgecolor="black", alpha=0.7)
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.title(f"Histogram of {feature}")
plt.show()

print(f"Mean of {feature}: {df_cleaned[feature].mean()}")
print(f"Variance of {feature}: {df_cleaned[feature].var()}")

# A3: Minkowski Distance Calculation
vec1, vec2 = X.iloc[0].values, X.iloc[1].values
minkowski_distances = [minkowski(vec1, vec2, r) for r in range(1, 11)]

plt.plot(range(1, 11), minkowski_distances, marker="o")
plt.xlabel("Minkowski Order (r)")
plt.ylabel("Distance")
plt.title("Minkowski Distance vs r")
plt.show()

# A4: Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# A5: Train k-NN Classifier (k=3)
kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(X_train, Y_train)

# A6: Compute Model Accuracy
accuracy = kNN.score(X_test, Y_test)
print("k-NN Accuracy:", accuracy)

# A7: Make Predictions
predictions = kNN.predict(X_test)
print("Predictions:", predictions)

# A8: k-NN Accuracy for Different k-values
k_values = range(1, 12)
accuracies = [KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train).score(X_test, Y_test) for k in k_values]

plt.plot(k_values, accuracies, marker="o")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("k vs Accuracy")
plt.show()

# A9: Compute Confusion Matrix and Classification Report
train_predictions = kNN.predict(X_train)
train_conf_matrix = confusion_matrix(Y_train, train_predictions)
train_report = classification_report(Y_train, train_predictions)

test_conf_matrix = confusion_matrix(Y_test, predictions)
test_report = classification_report(Y_test, predictions)

print("Confusion Matrix (Train):\n", train_conf_matrix)
print("Classification Report (Train):\n", train_report)

print("Confusion Matrix (Test):\n", test_conf_matrix)
print("Classification Report (Test):\n", test_report)
