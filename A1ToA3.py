import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Load Excel file
file_path = "Purchase data.xlsx"

df = pd.read_excel(file_path, sheet_name="Sheet1")

A = df.iloc[:, 1:-1].values
C = df.iloc[:, -1].values 

#Convert to float
A = A.astype(float)
C = C.astype(float)

rank_A = np.linalg.matrix_rank(A)
print("Rank of Matrix A:", rank_A)

#Compute pseudo-inverse of A
A_pinv = np.linalg.pinv(A)
X = np.dot(A_pinv, C)
print("Model Vector X:", X)

#Categorize customers as "RICH" or "POOR"
df["Category"] = np.where(df.iloc[:, -1] > 200, "RICH", "POOR")
print("\nUpdated Data with Categories:\n", df.head())

X_features = df.iloc[:, 1:-2].values 
y_labels = df["Category"].values

#Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

#Train a classifier (Random Forest)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#Predict and evaluate
y_pred = clf.predict(X_test)
print("Classification Accuracy:", accuracy_score(y_test, y_pred))
