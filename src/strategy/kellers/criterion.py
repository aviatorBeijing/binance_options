
def keller_fact(mu,sigma,rf):
    """
    mu = 0.15      # Expected return (e.g., 15% annual)
    sigma = 0.25   # Volatility (e.g., 25% annual)
    risk_free_rate = 0.02  # Risk-free rate (e.g., 2% annual)
    """

    optimal_fraction = (mu - risk_free_rate) / (sigma ** 2)

    #print("The optimal fraction of wealth to allocate (Keller criterion):", optimal_fraction)
    return optimal_fraction

