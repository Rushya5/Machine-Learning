import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score

# A1: Load dataset function
def load_data(file_path):
    return pd.read_csv(file_path)

# A1: Entropy calculation function
def calculate_entropy(y):
    value, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# A2: Gini index calculation function
def calculate_gini(y):
    value, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(np.square(probabilities))
    return gini

# A3 & A4: Binning continuous attributes
def bin_continuous(data, feature, bins=4, strategy='uniform'):
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
    data[feature] = discretizer.fit_transform(data[[feature]])
    return data

def encode_categorical(data):
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category').cat.codes
    return data
# A3: Information Gain calculation function
def information_gain(data, feature, target):
    total_entropy = calculate_entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = np.sum((counts[i] / np.sum(counts)) * calculate_entropy(data[data[feature] == values[i]][target]) for i in range(len(values)))
    info_gain = total_entropy - weighted_entropy
    return info_gain

# A3: Selecting the best attribute
def select_best_attribute(data, target):
    info_gains = {col: information_gain(data, col, target) for col in data.columns if col != target}
    best_attribute = max(info_gains, key=info_gains.get)
    return best_attribute, info_gains[best_attribute]

# A5: Building a Decision Tree model
def build_decision_tree(X, y):
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf

# A6: Visualize Decision Tree
def visualize_decision_tree(clf, feature_names):
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=feature_names, filled=True)
    plt.title("Decision Tree")
    plt.show()

# A7: Visualize Decision Boundary
def visualize_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title("Decision Boundary")
    plt.show()

# Main function
def main():
    data = load_data("Normalized_thyroid_dataset.csv")
    target = data.columns[-1]

    # Preprocessing: Encoding categorical features
    data = encode_categorical(data)

    # Preprocessing: Binning continuous attributes
    for col in data.select_dtypes(include=[np.number]).columns:
        if col != target:
            data = bin_continuous(data, col)

    # Selecting best attribute
    best_attribute, info_gain = select_best_attribute(data, target)
    print(f"Best Attribute: {best_attribute}, Information Gain: {info_gain}")

    # Splitting data
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Building and visualizing the decision tree
    clf = build_decision_tree(X_train, y_train)
    visualize_decision_tree(clf, data.columns[:-1])

    # Decision boundary visualization (if applicable)
    if X.shape[1] == 2:
        visualize_decision_boundary(clf, X, y)

    # Evaluating the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
