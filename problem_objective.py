import pandas as pd
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prices = pd.read_csv("twelve_years.csv", index_col=0)

prices = prices.iloc[:, :5]
assets = list(prices.columns)
num_assets = len(assets)

mu = torch.Tensor(np.array(prices.pct_change().mean() * 252)).to(device)
sigma = torch.Tensor(np.array(prices.pct_change().cov() * 252)).to(device)


def objective(x):
    lam = 0.5

    tot_obj = []
    for p in x:
        global mu, sigma
        risk = 0
        for s1 in range(x.shape[1]):
            for s2 in range(x.shape[1]):
                risk = risk + sigma[s1][s2] * p[s1] * p[s2]

        returns = 0
        for s in range(x.shape[1]):
            returns = returns + mu[s] * p[s]

        sum_ = 0
        for e in range(x.shape[1]):
            sum_ += p[e]

        obj = lam * risk - (1 - lam) * returns

        tot_obj.append(obj)

    return torch.Tensor(tot_obj).to(device)
