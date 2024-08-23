import numpy as np
cimport numpy as np
from scipy.stats import norm
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double black_scholes_call_price(double S, double K, double T, double r, double sigma):
    cdef double d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    cdef double d2 = d1 - sigma*np.sqrt(T)
    cdef double call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call_price

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double black_scholes_delta(double S, double K, double T, double r, double sigma):
    cdef double d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    cdef double delta = norm.cdf(d1)
    return delta

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double black_scholes_gamma(double S, double K, double T, double r, double sigma):
    cdef double d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    cdef double gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simulate_gbm_paths(double S0, double r, double sigma, double T, double dt, int n_sim):
    cdef int N = int(T/dt)
    cdef np.ndarray[np.double_t, ndim=2] S_paths = np.zeros((n_sim, N), dtype=np.double)
    cdef int i, j
    cdef double z

    S_paths[:, 0] = S0
    for j in range(1, N):
        for i in range(n_sim):
            z = np.random.standard_normal()
            S_paths[i, j] = S_paths[i, j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return S_paths

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gamma_scalping(np.ndarray[np.double_t, ndim=2] S_paths, double K, double r, double sigma, double T, double dt):
    cdef int n_sim = S_paths.shape[0]
    cdef int N = S_paths.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] portfolio_values = np.zeros((n_sim, N), dtype=np.double)
    cdef int i, t
    cdef double cash, shares, S, tau, delta, gamma, delta_shares, gamma_scalping_pnl, payoff

    for i in range(n_sim):
        cash = 0
        shares = 0
        for t in range(N):
            S = S_paths[i, t]
            tau = T - t*dt
            
            delta = black_scholes_delta(S, K, tau, r, sigma) if tau > 0 else 0
            gamma = black_scholes_gamma(S, K, tau, r, sigma) if tau > 0 else 0
            
            if t == 0:
                shares = delta
                cash = black_scholes_call_price(S, K, tau, r, sigma) - shares * S
            else:
                shares_new = delta
                delta_shares = shares_new - shares
                cash -= delta_shares * S
                shares = shares_new

                if t > 0 and np.abs(S_paths[i, t] - S_paths[i, t-1]) > 0:
                    gamma_scalping_pnl = 0.5 * gamma * (S_paths[i, t] - S_paths[i, t-1])**2
                    cash += gamma_scalping_pnl
            
            portfolio_values[i, t] = shares * S + cash
        
        payoff = max(S_paths[i, -1] - K, 0)
        portfolio_values[i, -1] -= payoff
    
    pnl = portfolio_values[:, -1]
    return pnl

