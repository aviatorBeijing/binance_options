import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Ref:
    https://www.interviewqs.com/blog/efficient-frontier
"""
def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *365
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(365)
    return std, returns

def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    sz = cov_matrix.shape[0]
    results = np.zeros((3,num_portfolios))
    weight_array = []
    np.random.seed(123)
    for i in range(num_portfolios):
        weights = np.random.random( sz )
        weights /= np.sum(weights)
        weight_array.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        # Sharpe ratio
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weight_array

def optimize_on_efficient_frontier(columns, mean_returns, cov_matrix, num_portfolios, risk_free_rate, do_plot=False)->dict:
    print("-- MPT Optimizing...")
    results, weights = generate_random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    max_sharpe_idx = np.argmax(results[2])
    stdev_portfolio, returns_portfolio = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    print("-"*100)
    print("  -- RF set to:", risk_free_rate )
    print("  -- Annualized Return:", round(returns_portfolio,2))
    print("  -- Annualized Volatility:", round(stdev_portfolio,2))

    print("  -- Allocation%:\n")
    print(max_sharpe_allocation)
    print("-"*100)
    
    if do_plot:
        plt.figure(figsize=(16, 9))
        # x = volatility, y = annualized return, color mapping = sharpe ratio
        plt.scatter(results[0,:],results[1,:],c=results[2,:], cmap='winter', marker='o', s=10, alpha=0.3)
        plt.colorbar()
        # Mark the portfolio w/ max Sharpe ratio
        plt.scatter(stdev_portfolio, returns_portfolio, marker='x',color='r',s=150, label='Max Sharpe ratio')
        plt.title('Simulated portfolios illustrating efficient frontier')
        plt.xlabel('annualized volatility')
        plt.ylabel('annualized returns')
        plt.legend(labelspacing=1.2)
        fn = os.getenv('USER_HOME','')+'/tmp/reversal_from_volume_hikes__mpt.pdf'
        plt.savefig(fn)
        print('-- saved:', fn)
    
    rst =  {
        'allocation_pct': max_sharpe_allocation.transpose().to_dict()['allocation'], # dict
        'best_choice':{
            'return': returns_portfolio,    # Float
            'std': stdev_portfolio  # Std
        },
    }
    return rst

def optimized_mpt( returns, num_portfolios, risk_free_rate, do_plot=False ):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    opt = optimize_on_efficient_frontier(returns.columns, mean_returns, cov_matrix, num_portfolios, risk_free_rate, do_plot)
    return opt