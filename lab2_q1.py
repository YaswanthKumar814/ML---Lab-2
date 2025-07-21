import pandas as pd
import numpy as np
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
print(df.head())
A = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = df['Payment (Rs)'].values
rank_A = np.linalg.matrix_rank(A)
print(f"The dimensionality of the vector space is: {rank_A}")
print(f"The number of vectors in the vector space is: {A.shape[0]}")
print(f"The rank of Matrix A is: {rank_A}")
A_pseudo_inv = np.linalg.pinv(A)
X = np.dot(A_pseudo_inv, C)

print("Cost of each product (Candies, Mangoes, Milk Packets) in Rs:")
print(f"Candies: {X[0]:.2f} Rs")
print(f"Mangoes: {X[1]:.2f} Rs")
print(f"Milk Packets: {X[2]:.2f} Rs")
