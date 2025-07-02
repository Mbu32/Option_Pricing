import numpy as np

def PathsHeston(S0, mu, T, alpha, beta, gamma, V0, corr, Nsteps, Npaths, seed=None):

    L = T/Nsteps
    Ls = L ** 0.5
    rng = np.random.default_rng(seed)
    # start with sampling V
    Vsample = np.zeros((Nsteps+1, Npaths))
    Vsample[0] = V0
    Bsample = rng.normal(size=(Nsteps, Npaths)) 
    for j in range(Nsteps):
       
        Vsample[j+1] = np.maximum(Vsample[j] + alpha*(beta-Vsample[j])*L + gamma*np.sqrt(Vsample[j])*Ls*Bsample[j],0)
    Ssample = np.zeros((Nsteps+1, Npaths))
    Ssample[0] = S0
    Wsample = corr*Bsample + np.sqrt(1-corr**2)*rng.normal(size=(Nsteps, Npaths))
    
    for j in range(Nsteps):
        Ssample[j+1] = Ssample[j]*np.exp(np.sqrt(Vsample[j])*Ls*Wsample[j]+(mu-Vsample[j]/2)*L)
    return Ssample, Vsample

Nsteps = 10000; Npaths=1000
Ssample, Vsample = PathsHeston(1, .02, 2, 1.5, .2, .05, .25, -.3, Nsteps, Npaths, seed=12345)

def HestonCallPrice(S0, K, r, T, alpha, beta, gamma, V0, corr, Nsteps, Npaths, seed=None):
    Ssample, Vsample = PathsHeston(S0, r, T, alpha, beta, gamma, V0, corr, Nsteps, Npaths, seed)
    ST = Ssample[-1]  
    payoff = np.maximum(ST - K, 0)  # Call payoff
    price = np.exp(-r * T) * np.mean(payoff)  
    return price

# An Example
K = 1.0  
r = 0.02  
call_price = HestonCallPrice(1, K, r, 2, 1.5, 0.2, 0.05, 0.25, -0.3, 10000, 1000, seed=12345)
print(f"European Call Price: {call_price:.4f}")
