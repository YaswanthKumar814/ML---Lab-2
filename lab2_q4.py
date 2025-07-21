import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load the data
# -----------------------------
df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# View the first few rows
print("🔍 Sample Data:\n", df.head())

# -----------------------------
# 2. Data types of each attribute
# -----------------------------
print("\n📊 Data Types:\n", df.dtypes)

# -----------------------------
# 3. Check for missing values
# -----------------------------
print("\n❓ Missing Values:\n", df.isnull().sum())

# -----------------------------
# 4. Detect categorical columns
# -----------------------------
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\n🗂️ Categorical Attributes:", categorical_cols)

# 💡 Encoding Scheme Hint
print("\n💡 Encoding Recommendation:")
print("- Use Label Encoding for ordinal attributes (e.g., severity levels)")
print("- Use One-Hot Encoding for nominal attributes (e.g., gender, class)")

# -----------------------------
# 5. Apply Label Encoding as placeholder
# -----------------------------
label_encoded_df = df.copy()
for col in categorical_cols:
    label_encoder = LabelEncoder()
    try:
        label_encoded_df[col] = label_encoder.fit_transform(df[col].astype(str))
    except:
        print(f"Skipping encoding for {col}")

# -----------------------------
# 6. Analyze numeric columns
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\n📈 Numeric Summary:")
for col in numeric_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    var_val = df[col].var()
    min_val = df[col].min()
    max_val = df[col].max()
    print(f"{col}: Mean = {mean_val:.2f}, Std = {std_val:.2f}, Var = {var_val:.2f}, Range = {min_val} to {max_val}")

# -----------------------------
# 7. Outlier detection using boxplot
#    (Exclude ID-type columns with very large scale)
# -----------------------------
# Filter out ID-like columns
exclude_cols = ['Record ID', 'record_id', 'id']
plot_cols = [col for col in numeric_cols if col not in exclude_cols]

# Plot the boxplot
# Plot age below 200 to ignore absurd outliers
plt.figure(figsize=(6, 4))
df[df['age'] < 200]['age'].plot.box()
plt.title("Boxplot of Age (Filtered < 200)")
plt.show()

