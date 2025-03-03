import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Example: Generate synthetic market data
strikes = np.linspace(80, 120, 10)
market_vols = 0.2 + 0.1 * np.exp(-(strikes - 100)**2 / 100)  # Example market implied volatility

# Example SABR Model function for implied volatility
def sabr_volatility(alpha, beta, rho, nu, F, K):
    # Example approximation of the SABR model for implied volatility
    sigma_ATM = alpha / (F**beta)  # Assume ATM volatility approximation
    vol = sigma_ATM * (K / F) ** beta + nu * np.sqrt(K / F) * (1 + rho * (K / F))  # Simplified model
    return vol

# Objective function to minimize (sum of squared errors)
def objective(params, strikes, market_vols, F):
    alpha, beta, rho, nu = params
    model_vols = sabr_volatility(alpha, beta, rho, nu, F, strikes)
    return np.sum((model_vols - market_vols)**2)

# Initial guess for parameters: [alpha, beta, rho, nu]
initial_guess = [0.2, 0.5, 0.1, 0.5]

# Forward price (for simplicity)
F = 100

# Perform the calibration
result = minimize(objective, initial_guess, args=(strikes, market_vols, F), bounds=[(0, 1), (0, 1), (-1, 1), (0, 1)])
calibrated_params = result.x
print("Calibrated Parameters:", calibrated_params)

# Plot the results
calibrated_vols = sabr_volatility(*calibrated_params, F, strikes)

plt.plot(strikes, market_vols, label="Market Volatility", linestyle='--')
plt.plot(strikes, calibrated_vols, label="SABR Model Volatility", linestyle='-')
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.legend()
plt.show()

