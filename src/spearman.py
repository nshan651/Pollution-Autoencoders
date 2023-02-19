import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Compare monotonicity of norm data and pca embeddings
dfx = pd.read_csv('/home/nick/git/Pollution-Autoencoders/data/data_norm/o3_data_norm.csv')
dfy = pd.read_csv('/home/nick/git/Pollution-Autoencoders/data/vec/o3_vec_190.csv')

# Only compare first 10 dims with first 10 features
dfx = dfx.iloc[:, 3:-1]
dfy = dfy.iloc[:, 3:]
for i in range(len(dfx.columns)):
    rho, p = spearmanr(dfx.iloc[i], dfy.iloc[i])
    if p < 0.05:
        print(f'dim {i}, rho: {rho}, p: {p}')
