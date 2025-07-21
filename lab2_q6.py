import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# Encode all categorical variables
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Select the first two vectors
vec1 = df_encoded.iloc[0].values.reshape(1, -1)
vec2 = df_encoded.iloc[1].values.reshape(1, -1)

# Calculate cosine similarity
cos_sim = cosine_similarity(vec1, vec2)[0][0]

print(f"Cosine Similarity between first two observations: {cos_sim}")
