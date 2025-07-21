import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset
df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# Fill missing numeric values temporarily for scaling
df.fillna(df.mean(numeric_only=True), inplace=True)

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Check range of each numeric column
print("Column Ranges:")
for col in numeric_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    col_range = col_max - col_min
    print(f"{col}: Min={col_min}, Max={col_max}, Range={col_range}")

# Use Min-Max Scaling where values are on very different scales
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Apply Min-Max scaling to all numeric columns (can customize if needed)
df_minmax_scaled = df.copy()
df_minmax_scaled[numeric_cols] = min_max_scaler.fit_transform(df[numeric_cols])
print("\nMin-Max Scaling Applied")

# Apply Standardization
df_standard_scaled = df.copy()
df_standard_scaled[numeric_cols] = standard_scaler.fit_transform(df[numeric_cols])
print("Z-score Standardization Applied")

# Example: View a few rows
print("\nSample - Min-Max Scaled Data:\n", df_minmax_scaled[numeric_cols].head())
print("\nSample - Standardized Data:\n", df_standard_scaled[numeric_cols].head())
