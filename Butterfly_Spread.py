
import numpy as np
from scipy import stats

def round_interval(interval, n=4):
    a, b = interval
    return round(a,n), round(b,n)

def MCAVButterfly(S0, rho, T, sigma, n, K1, K3, seed=None):
    K2 = (K1 + K3) / 2
    rng = np.random.default_rng(seed)
    sample = rng.normal(size=n)
    S_T1 = S0 * np.exp((rho - sigma ** 2 / 2) * T + sigma * np.sqrt(T) * sample)
    S_T2 = S0 * np.exp((rho - sigma ** 2 / 2) * T - sigma * np.sqrt(T) * sample)
    payoff1 = np.zeros(n)
    payoff1[(S_T1 > K1) & (S_T1 <= K2)] = S_T1[(S_T1 > K1) & (S_T1 <= K2)] - K1
    payoff1[(S_T1 > K2) & (S_T1 <= K3)] = K3 - S_T1[(S_T1 > K2) & (S_T1 <= K3)]
    payoff2 = np.zeros(n)
    payoff2[(S_T2 > K1) & (S_T2 <= K2)] = S_T2[(S_T2 > K1) & (S_T2 <= K2)] - K1
    payoff2[(S_T2 > K2) & (S_T2 <= K3)] = K3 - S_T2[(S_T2 > K2) & (S_T2 <= K3)]
    disc_payoff = np.exp(- rho * T) * (payoff1 + payoff2) / 2
    mu = disc_payoff.mean()
    CI = stats.norm.interval(.95, loc=mu, scale=stats.sem(disc_payoff))
    return mu, CI

price, CI = MCAVButterfly(60, .02, 1/2, .3, 100000, 55, 65, seed=12345)
print('Option price:', round(price,4))
print('Confidence interval (95%):', round_interval(CI))
print('Lenght of the confidence interval:', round(CI[1]-CI[0],4))
