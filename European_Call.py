
import numpy as np
from scipy import stats

def GeoBMPaths2AS(S0, nu, sigma, T, Nsteps, Npaths, seed=None):
    s = sigma * (T / Nsteps) ** .5
    n = nu * T / Nsteps
    rng = np.random.default_rng(seed)
    N = rng.normal(scale=s, size=(Nsteps,Npaths))
    incr1 = np.zeros((Nsteps+1, Npaths))
    incr1[0] = S0
    incr1[1:] = np.exp(n + N)
    Ssample1 = np.cumprod(incr1, axis=0) 
    incr2 = np.zeros((Nsteps+1, Npaths))
    incr2[0] = S0
    incr2[1:] = np.exp(n - N)
    Ssample2 = np.cumprod(incr2, axis=0) 
    return Ssample1, Ssample2

def priceCall1AS(paths1, paths2, K, rho, T):
    payoff1 = np.maximum(paths1[-1]-K,0)
    payoff2 = np.maximum(paths2[-1]-K,0)
    disc_payoff = np.exp(-rho * T) * (payoff1 + payoff2) / 2
    price = disc_payoff.mean()
    CI = stats.norm.interval(.95, loc=price, scale=stats.sem(disc_payoff))
    return price, CI 


paths1, paths2 = GeoBMPaths2AS(1, -.03, .4, 2, 1000, 10000, seed=12345)
price, CI = priceCall1AS(paths1, paths2, 1, .05, 2)

def round_interval(interval, n=4):
    a, b = interval
    return round(a,n), round(b,n) 
print('Antithetic sampling MC estimate:', round(price,4))
print('Confidence interval:', round_interval(CI))
