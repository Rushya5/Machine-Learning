import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

#Load Dataset
df = pd.read_excel("thyroid0387_UCI.xlsx", sheet_name="Sheet1")
df.replace("?", pd.NA, inplace=True)

#A5: Data Exploration
print("Data Types:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe(include='all'))

categorical_columns = df.select_dtypes(include=['object']).columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print("\nRange of Numeric Variables:\n", df[numeric_columns].max() - df[numeric_columns].min())

#Study the presence of missing values in each attribute
print("\nMissing Values:\n", df.isnull().sum())

#Study presence of outliers in data
Q1 = df[numeric_columns].quantile(0.25)
Q3 = df[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).sum()
print("\nOutliers:\n", outliers)

#For numeric variables, calculate the mean and variance
print("\nMean:\n", df[numeric_columns].mean())
print("\nVariance:\n", df[numeric_columns].var())

#A6: Data Imputation
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if col in numeric_columns:
            if outliers[col] > 0:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0]).infer_objects(copy=False)

#A7: Data Normalization
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
df.to_csv('Normalized_thyroid_dataset.csv')
print("Normalization complete. Data saved as 'normalized_thyroid_data.csv'.")

#A8: Similarity Measure(JC and SMC)
binary_columns = [
    "on thyroxine", "query on thyroxine", "on antithyroid medication",
    "sick", "pregnant", "thyroid surgery", "I131 treatment",
    "query hypothyroid", "query hyperthyroid", "lithium", "goitre",
    "tumor", "hypopituitary", "psych", "TSH measured", "T3 measured",
    "TT4 measured", "T4U measured", "FTI measured", "TBG measured"
]

#Convert t= 1 and f = 0
df[binary_columns] = df[binary_columns].apply(lambda x: x.map({'t': 1, 'f': 0}))

#Select the first 2 observation vectors
vec1, vec2 = df.iloc[0][binary_columns], df.iloc[1][binary_columns]

#Compute JC and SMC
f11 = sum((vec1 == 1) & (vec2 == 1))
f00 = sum((vec1 == 0) & (vec2 == 0))
f10 = sum((vec1 == 1) & (vec2 == 0))
f01 = sum((vec1 == 0) & (vec2 == 1))

jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0
smc = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

print("\nJaccard Coefficient (JC):", jc)
print("Simple Matching Coefficient (SMC):", smc)

#A9: Cosine Similarity Measure
#Select only Numeric columns for Cosine Similarity
vec1_numeric = df.iloc[0][numeric_columns]
vec2_numeric = df.iloc[1][numeric_columns]

#Calculate cosine similarity
cosine_sim = cosine_similarity([vec1_numeric], [vec2_numeric])[0][0]
print("\nCosine Similarity:", cosine_sim)

#A10: Heatmap Plot
df_subset = df.iloc[:20][binary_columns]

#Initialize matrices for JC, SMC, and COS
n = df_subset.shape[0]
jc_matrix = np.zeros((n, n))
smc_matrix = np.zeros((n, n))
cosine_matrix = np.zeros((n, n))

#Compute Similarity Measures
for i in range(n):
    for j in range(n):
        vec1, vec2 = df_subset.iloc[i], df_subset.iloc[j]

        f11 = sum((vec1 == 1) & (vec2 == 1))
        f00 = sum((vec1 == 0) & (vec2 == 0))
        f10 = sum((vec1 == 1) & (vec2 == 0))
        f01 = sum((vec1 == 0) & (vec2 == 1))

        denominator_jc = f11 + f10 + f01
        denominator_smc = f11 + f00 + f10 + f01

        jc_matrix[i, j] = f11 / denominator_jc if denominator_jc != 0 else 0
        smc_matrix[i, j] = (f11 + f00) / denominator_smc if denominator_smc != 0 else 0

        dot_product = np.dot(vec1, vec2)
        magnitude_A = np.linalg.norm(vec1)
        magnitude_B = np.linalg.norm(vec2)
        cosine_matrix[i, j] = dot_product / (magnitude_A * magnitude_B) if magnitude_A * magnitude_B != 0 else 0

#Convert to DataFrame for Visualization
jc_df = pd.DataFrame(jc_matrix, index=df_subset.index, columns=df_subset.index)
smc_df = pd.DataFrame(smc_matrix, index=df_subset.index, columns=df_subset.index)
cosine_df = pd.DataFrame(cosine_matrix, index=df_subset.index, columns=df_subset.index)

#Plot Heatmaps
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(jc_df, annot=True, cmap="coolwarm")
plt.title("Jaccard Coefficient (JC)")

plt.subplot(1, 3, 2)
sns.heatmap(smc_df, annot=True, cmap="coolwarm")
plt.title("Simple Matching Coefficient (SMC)")

plt.subplot(1, 3, 3)
sns.heatmap(cosine_df, annot=True, cmap="coolwarm")
plt.title("Cosine Similarity (COS)")

plt.tight_layout()
plt.show()