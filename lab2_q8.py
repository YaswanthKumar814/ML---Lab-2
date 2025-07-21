import pandas as pd
import numpy as np

# Load the data
df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Imputation for numeric columns
for col in numeric_cols:
    if df[col].isnull().sum() == 0:
        continue
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    has_outliers = ((df[col] < lower) | (df[col] > upper)).any()
    
    if has_outliers:
        impute_value = df[col].median()
        method = "median"
    else:
        impute_value = df[col].mean()
        method = "mean"
    
    df[col].fillna(impute_value, inplace=True)
    print(f"{col}: Filled missing with {method} = {impute_value}")

# Imputation for categorical columns using mode
for col in categorical_cols:
    if df[col].isnull().sum() == 0:
        continue
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)
    print(f"{col}: Filled missing with mode = {mode_val}")

# Confirm missing values filled
print("\nMissing values after imputation:\n", df.isnull().sum())
