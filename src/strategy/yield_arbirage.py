"""
Platform	USDC Yield (APY)	USDT Yield (APY)
Aave	        1.8% - 11.2%	        2.2% - 8.3%
Compound	1.9% - 6.3%	        2.4% - 7.4%
Nexo	        7% - 10%	        7.5% - 10%
BlockFi	        6% - 7%	                6.5% - 7.5%
Binance	        2% - 6.5%       	3% - 8%
Crypto.com	4% - 8%                 5% - 9%

https://coinlive.io/blockfi/interest-rates/
https://coin360.com/learn/can-you-borrow-lend-usdc-guide
https://www.bitcoinmarketjournal.com/usdt-interest-rates/

DeFi Platforms (Aave, Compound): These platforms typically offer lower yields compared to centralized platforms but provide greater control and flexibility. Yields on Aave range from 1.8% to 11.2% for USDC and up to 8.3% for USDT, depending on demand and other factors.

Centralized Platforms (Nexo, BlockFi, Binance, Crypto.com): These platforms offer competitive yields, often higher than DeFi platforms due to additional incentives, such as loyalty tokens or longer lock-in periods. For instance, Nexo offers up to 10% APY on USDT with the NEXO token.


Yield arbitrage between platforms like Aave and Nexo involves taking advantage of the difference in interest rates offered on the same assets (such as USDC or USDT) across these platforms. Here’s a step-by-step guide on how to do this:

Step 1: Assess the Yield Differentials
Aave: Typically offers variable interest rates, which fluctuate based on supply and demand. For example, USDC yields on Aave might range from 1.8% to 11.2%.
Nexo: Provides relatively stable and often higher yields, such as 7% to 10% for USDC, depending on whether you opt for NEXO token rewards.
Step 2: Supply Funds to Aave
Deposit USDC/USDT: Start by depositing your stablecoins (USDC or USDT) into Aave to earn interest. Aave allows you to earn interest while maintaining control of your assets.
Borrow against Collateral: Use your deposited USDC/USDT as collateral to borrow another stablecoin or asset, like ETH, which you plan to use for arbitrage.
Step 3: Transfer and Deposit into Nexo
Convert or Transfer Funds: Take the borrowed funds (or convert them back to USDC/USDT) and transfer them to Nexo.
Earn Higher Interest on Nexo: Deposit the funds into your Nexo account, where you might earn a higher interest rate (e.g., 10% for USDT).
Step 4: Manage and Monitor
Monitor Interest Rates: Keep an eye on interest rate fluctuations on Aave and Nexo. If the rates on Aave rise above what Nexo offers, you may need to unwind your position.
Repay Borrowed Funds: Periodically, you’ll need to repay the borrowed amount on Aave. You can do this using the interest earned on Nexo or by withdrawing from Nexo and converting back to the original asset.
Step 5: Calculate and Harvest Arbitrage Profit
Net Profit: Your profit comes from the difference between the lower interest paid on borrowed funds (Aave) and the higher interest earned on Nexo. For example, if you borrow at 5% on Aave and earn 10% on Nexo, your net profit is the 5% difference, minus any transaction fees.
Risks and Considerations:
Market Fluctuations: Aave’s interest rates can change, reducing the profitability of the arbitrage.
Platform Risks: Nexo and Aave are both subject to platform-specific risks, such as changes in terms or issues with liquidity.
Collateral Liquidation: If the value of your collateral on Aave drops, you might face liquidation, so it's crucial to monitor the loan-to-value (LTV) ratio closely.
This strategy requires careful monitoring and quick action to capitalize on interest rate discrepancies while managing the associated risks.
"""
def yield_arbitrage(L, C, r_N, F_A, T):
    """
    Calculate the net profit from yield arbitrage between Aave and Nexo.

    Parameters:
    L (float): Initial amount of USDC/USDT deposited as collateral.
    C (float): Collateralization ratio on Aave.
    r_N (float): Interest rate (APY) on Nexo.
    F_A (float): Borrowing fee or interest rate (APY) on Aave.
    T (float): Time period for which the arbitrage is conducted, in years.

    Returns:
    float: Net arbitrage profit in USDC/USDT.
    """
    # Calculate the amount that can be borrowed
    B = L / C
    
    # Calculate interest earned on Nexo
    I_N = B * r_N * T
    
    # Calculate interest paid on Aave
    I_A = B * F_A * T
    
    # Calculate net profit
    profit = I_N - I_A
    
    return profit

def main():
    # Input parameters
    L = float(input("Enter the initial collateral in USDC/USDT (e.g., 1000): "))
    C = float(input("Enter the collateralization ratio on Aave (e.g., 1.5): "))
    r_N = float(input("Enter the interest rate (APY) on Nexo (e.g., 0.10 for 10%): "))
    F_A = float(input("Enter the borrowing fee (APY) on Aave (e.g., 0.05 for 5%): "))
    T = float(input("Enter the time period in years (e.g., 1): "))

    # Calculate the profit
    profit = yield_arbitrage(L, C, r_N, F_A, T)
    
    # Output the result
    print(f"\nNet arbitrage profit after {T} years is approximately: {profit:.2f} USDC")

if __name__ == "__main__":
    main()

