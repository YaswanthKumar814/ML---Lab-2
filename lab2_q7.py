import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# Encode categorical variables
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Take first 20 rows
data20 = df_encoded.iloc[:20].reset_index(drop=True)

# ---------- Cosine Similarity ----------
cos_matrix = cosine_similarity(data20)

plt.figure(figsize=(10, 8))
sns.heatmap(cos_matrix, annot=False, cmap="coolwarm")
plt.title("Cosine Similarity Heatmap (First 20 Observations)")
plt.show()

# ---------- JC and SMC ----------
def jaccard_smc(vec1, vec2):
    f11 = np.sum((vec1 == 1) & (vec2 == 1))
    f00 = np.sum((vec1 == 0) & (vec2 == 0))
    f10 = np.sum((vec1 == 1) & (vec2 == 0))
    f01 = np.sum((vec1 == 0) & (vec2 == 1))

    jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
    return jc, smc

# Only binary columns
binary_cols = [col for col in data20.columns if data20[col].isin([0,1]).all()]

jc_matrix = np.zeros((20, 20))
smc_matrix = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        v1 = data20.loc[i, binary_cols].astype(int).values
        v2 = data20.loc[j, binary_cols].astype(int).values
        jc, smc = jaccard_smc(v1, v2)
        jc_matrix[i, j] = jc
        smc_matrix[i, j] = smc

# ---------- JC Heatmap ----------
plt.figure(figsize=(10, 8))
sns.heatmap(jc_matrix, annot=False, cmap="viridis")
plt.title("Jaccard Coefficient Heatmap (Binary Features)")
plt.show()

# ---------- SMC Heatmap ----------
plt.figure(figsize=(10, 8))
sns.heatmap(smc_matrix, annot=False, cmap="YlGnBu")
plt.title("Simple Matching Coefficient Heatmap (Binary Features)")
plt.show()
