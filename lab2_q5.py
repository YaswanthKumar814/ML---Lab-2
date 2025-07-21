import pandas as pd

# Load dataset
df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

# Fill missing categorical values temporarily with mode for processing (won't affect binary comparison)
df.fillna(df.mode().iloc[0], inplace=True)

# Identify binary attributes (only 0 and 1)
binary_cols = [col for col in df.columns if df[col].dropna().value_counts().index.isin([0,1]).all()]

print("Binary attributes used:", binary_cols)

# Take first two observation vectors for binary attributes only
v1 = df.loc[0, binary_cols].astype(int)
v2 = df.loc[1, binary_cols].astype(int)

# Calculate f11, f00, f10, f01
f11 = ((v1 == 1) & (v2 == 1)).sum()
f00 = ((v1 == 0) & (v2 == 0)).sum()
f10 = ((v1 == 1) & (v2 == 0)).sum()
f01 = ((v1 == 0) & (v2 == 1)).sum()

# Calculate JC and SMC
jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
smc = (f11 + f00) / (f00 + f01 + f10 + f11)

print(f"f11 = {f11}, f00 = {f00}, f10 = {f10}, f01 = {f01}")
print(f"Jaccard Coefficient = {jc}")
print(f"Simple Matching Coefficient = {smc}")

# Judgment
if abs(jc - smc) > 0.1:
    print("Jaccard Coefficient focuses on shared presence (1s), better when 1s are more meaningful.")
else:
    print("Both JC and SMC are similar, but JC is preferred for sparse binary data.")
