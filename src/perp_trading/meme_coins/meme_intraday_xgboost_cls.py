import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import warnings
import os

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
plt.style.use('fivethirtyeight')

class CryptoStrategyAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol.lower()
        self.df = None
        self.model = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.features = [
            'rank_volume', 'rank_quote_volume', 'rank_trades', 
            'rank_taker_base_volume', 'rank_taker_quote_volume', 
            'rank_returns', 'close'
        ]
        
    def load_data(self):
        """Load and preprocess the cryptocurrency data"""
        file_path = os.getenv("USER_HOME", "") + f'/tmp/perp_{self.symbol}usdt_5m.csv'
        self.df = pd.read_csv(file_path)
        
        # Convert timestamp and calculate features
        self.df['starttime'] = pd.to_datetime(self.df['starttime'], unit='ms')
        self._calculate_features()
        
    def _calculate_features(self):
        """Calculate all required features and targets"""
        rolling_window = 150
        
        # Calculate rolling ranks
        for col in ['volume', 'quote_volume', 'trades', 'taker_base_volume', 'taker_quote_volume']:
            self.df[f'rank_{col}'] = self.df[col].rolling(window=rolling_window, min_periods=1).rank()
        
        # Calculate returns and their ranks
        self.df['returns'] = self.df['close'] / self.df['close'].shift(1) - 1
        self.df['rank_returns'] = self.df['returns'].rolling(window=rolling_window, min_periods=1).rank()
        
        # Calculate target: max return > 10% in next 30 periods
        future_returns = [self.df['close'].shift(-i) / self.df['close'] - 1 for i in range(1, 31)]
        max_future_return = pd.concat(future_returns, axis=1).max(axis=1)
        self.df['target'] = (max_future_return > 0.1).astype(int)
        
        # Fill missing values
        self.df.fillna(0, inplace=True)
    
    def prepare_train_test(self):
        """Split data into training and testing sets"""
        train_size = int(len(self.df) * 0.7)
        
        X = self.df[self.features].copy()
        y = self.df['target'].copy()
        
        self.X_train, self.X_test = X.iloc[:train_size], X.iloc[train_size:]
        self.y_train, self.y_test = y.iloc[:train_size], y.iloc[train_size:]
        
    def train_model(self):
        """Train the XGBoost classifier"""
        self.model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        """Evaluate model performance on test set"""
        y_pred = self.model.predict(self.X_test)
        
        metrics = {
            'F1 Score': f1_score(self.y_test, y_pred),
            'Win Rate (Precision)': precision_score(self.y_test, y_pred),
            'Hit Rate (Recall)': recall_score(self.y_test, y_pred),
            'Accuracy': accuracy_score(self.y_test, y_pred)
        }
        
        for name, value in metrics.items():
            print(f"{name}: {value:.2%}")
        
        return metrics
    
    def analyze_strategies(self):
        """Analyze and compare trading strategies"""
        # Add predictions to test set
        X_test = self.X_test.copy()
        X_test['pred'] = self.model.predict(X_test)
        X_test['true'] = self.y_test.values
        X_test['future_close_30'] = self.df['close'].shift(-30).loc[X_test.index]
        
        # Strategy 1: Sell at close 30 steps later
        strategy1 = X_test[X_test['pred'] == 1].copy()
        strategy1 = strategy1.dropna(subset=['future_close_30'])
        strategy1['ret_fixed'] = (strategy1['future_close_30'] - strategy1['close']) / strategy1['close']
        
        # Strategy 2: Sell at max close within next 30 steps (capped at 20%)
        future_closes = self._sliding_window(X_test['close'].values, 30)
        max_future_returns = (np.max(future_closes, axis=1) - X_test['close'].values[:-29]) / X_test['close'].values[:-29]
        
        X_test['ret_max'] = np.nan
        X_test.iloc[:-29, X_test.columns.get_loc('ret_max')] = max_future_returns
        
        strategy2 = X_test[X_test['pred'] == 1].dropna(subset=['ret_max']).copy()
        strategy2['ret_max'] = np.minimum(strategy2['ret_max'], 0.2)
        
        return strategy1, strategy2
    
    def _sliding_window(self, arr, window):
        """Helper function to create sliding windows"""
        shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
        strides = arr.strides + (arr.strides[-1],)
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    
    def generate_reports(self, strategy1, strategy2):
        """Generate performance reports for strategies"""
        def _report(name, returns):
            total_trades = len(returns)
            profitable = (returns > 0).sum()
            win_rate = profitable / total_trades if total_trades else 0
            avg_return = returns.mean()
            total_return = (returns + 1).prod() - 1
            
            print(f"📌 {name}")
            print(f" - Total Trades: {total_trades}")
            print(f" - Profitable Trades: {profitable}")
            print(f" - Win Rate: {win_rate:.2%}")
            print(f" - Avg Return per Trade: {avg_return:.2%}")
            print(f" - Total Return: {total_return:.2%}")
            print("")
        
        _report("Strategy 1: Sell at Close After 30 Steps", strategy1['ret_fixed'])
        _report("Strategy 2: Sell at Max Close in Next 30 Steps", strategy2['ret_max'])
    
    def plot_results(self, strategy1, strategy2):
        """Visualize the results with volume subplot"""
        # Prepare data for plotting
        strategy1_sorted = strategy1.sort_index().copy()
        strategy2_sorted = strategy2.sort_index().copy()
        
        strategy1_sorted['cum_return'] = (1 + strategy1_sorted['ret_fixed']).cumprod()
        strategy2_sorted['cum_return'] = (1 + strategy2_sorted['ret_max']).cumprod()
        
        # Add datetime information
        self.X_test['datetime'] = self.df.loc[self.X_test.index, 'starttime']
        self.X_train['datetime'] = self.df.loc[self.X_train.index, 'starttime']
        strategy1_sorted['datetime'] = self.df.loc[strategy1_sorted.index, 'starttime']
        strategy2_sorted['datetime'] = self.df.loc[strategy2_sorted.index, 'starttime']
        
        # Create figure with 3 subplots
        fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Plot 1: Price and classifications
        self._plot_price_and_classifications(ax[0])
        
        # Plot 2: Volume (new subplot)
        self._plot_volume(ax[1])
        
        # Plot 3: Strategy performance
        self._plot_strategy_performance(ax[2], strategy1_sorted, strategy2_sorted)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_price_and_classifications(self, ax):
        """Plot price data with buy/sell signals"""
        ax.plot(self.X_train['datetime'], self.X_train['close'], label='Train Close', alpha=0.5)
        ax.plot(self.X_test['datetime'], self.X_test['close'], label='Test Close', alpha=0.5)
        
        # Plot classifications
        ax.scatter(self.X_train['datetime'][self.y_train == 1], 
                  self.X_train['close'][self.y_train == 1], 
                  color='red', s=10, alpha=0.4, label='Train Class 1')
        ax.scatter(self.X_test['datetime'][self.y_test == 1], 
                  self.X_test['close'][self.y_test == 1], 
                  color='red', s=15, alpha=0.7, label='Test Class 1')
        
        # Add split line and formatting
        split_time = self.X_test['datetime'].iloc[0]
        ax.axvline(x=split_time, color='k', linestyle='--', label='Train/Test Split')
        ax.set_title('Close Price with Classifications')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid()
    
    def _plot_volume(self, ax):
        """Plot trading volume as bar chart"""
        # Get volume data for train and test periods
        train_volume = self.df.loc[self.X_train.index, 'volume']
        test_volume = self.df.loc[self.X_test.index, 'volume']
        
        # Create green/red bars based on price direction
        colors_train = np.where(self.df.loc[self.X_train.index, 'close'].diff() > 0, 'g', 'r')
        colors_test = np.where(self.df.loc[self.X_test.index, 'close'].diff() > 0, 'g', 'r')
        
        # Plot volume bars
        ax.bar(self.X_train['datetime'], train_volume, color=colors_train, alpha=0.5, width=0.002)
        ax.bar(self.X_test['datetime'], test_volume, color=colors_test, alpha=0.7, width=0.002)
        
        ax.set_title('Trading Volume')
        ax.set_ylabel('Volume')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add split line
        split_time = self.X_test['datetime'].iloc[0]
        ax.axvline(x=split_time, color='k', linestyle='--')
    
    def _plot_strategy_performance(self, ax, strategy1, strategy2):
        """Plot strategy cumulative returns"""
        ax1 = ax
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(strategy1['datetime'], strategy1['cum_return'], 
                         color='blue', label='Sell @ Close After 30 Steps')
        line2 = ax2.plot(strategy2['datetime'], strategy2['cum_return'], 
                         color='green', label='Sell @ Max 20% Profit')
        
        ax1.set_ylabel('Cumulative Return - Strategy 1 (Left Axis)', color='blue')
        ax2.set_ylabel('Cumulative Return - Strategy 2 (Right Axis)', color='green')
        ax1.set_title('Cumulative Returns of Two Strategies')
        ax1.set_xlabel('Time')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid()


def main():
    # Initialize analyzer
    analyzer = CryptoStrategyAnalyzer(symbol='flm')  # Can change to other symbols
    
    # Execute analysis pipeline
    analyzer.load_data()
    analyzer.prepare_train_test()
    analyzer.train_model()
    metrics = analyzer.evaluate_model()
    
    # Analyze strategies
    strategy1, strategy2 = analyzer.analyze_strategies()
    analyzer.generate_reports(strategy1, strategy2)
    
    # Visualize results
    analyzer.plot_results(strategy1, strategy2)


if __name__ == "__main__":
    main()
