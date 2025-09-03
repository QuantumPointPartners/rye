# RYE — Risk & Yield Engine

<img width="500" height="500" alt="Quant Point (2)" src="https://github.com/user-attachments/assets/163175d4-2baf-4cca-ac28-afbda6c4b04b" />

> **Advanced Quantitative Trading Backtesting & Analysis Tool**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)]()

**RYE** is a zero-dependencies, single-file CLI tool for quantitative trading strategy development, backtesting, and risk analysis. Built for traders, researchers, and developers who need a fast, reliable, and feature-rich backtesting engine.

## **Key Features**

- **Advanced Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR
- **Multiple Trading Strategies**: Momentum, Mean Reversion, MACD, Bollinger Bands, Stochastic, Trend Following
- **Comprehensive Risk Metrics**: Sharpe, Sortino, Calmar ratios, VaR, Max Drawdown, Profit Factor
- **High Performance**: Optimized algorithms for large datasets
- **Configurable**: Easy strategy parameter customization
- **Beautiful Reports**: Interactive HTML reports with charts
- **Robust Error Handling**: Comprehensive validation and error reporting
- **Zero Dependencies**: Single Python file, no external packages required

## **Architecture**

RYE is built with a modular architecture that separates concerns:

```
rye.py
├── Data I/O (CSV reading/writing with validation)
├── Technical Indicators (SMA, EMA, RSI, MACD, etc.)
├── Trading Strategies (Signal generation)
├── Backtesting Engine (Portfolio simulation)
├── Risk Analytics (Performance metrics)
├── Reporting (HTML generation)
└── CLI Interface (Command management)
```

## **Installation**

### Prerequisites
- Python 3.8 or higher
- No external dependencies required

### Quick Start
```bash
wget https://raw.githubusercontent.com/QuantumPointPartners/rye/main/rye.py
chmod +x rye.py
python rye.py init
```

## **Usage**

### 1. **Initialize Project**
```bash
python rye.py init
```
Creates:
- `data/` - For CSV data files
- `reports/` - For HTML backtest reports
- `runs/` - For strategy outputs
- `rye_config.ini` - Configuration file
- `README_RYE.txt` - Project documentation

### 2. **Generate Synthetic Data**
```bash
python rye.py synth --symbol SPY --days 1250 --regime volatile
python rye.py synth --symbol AAPL --days 500 --start 2022-01-01 --price 150.0 --regime bull
```

If RYE is working correctly, this is what you should see:

<img width="1499" height="568" alt="Screenshot 2025-09-03 at 5 25 22 AM" src="https://github.com/user-attachments/assets/31d19cda-6293-46a7-9f53-51203c157663" />

Next, let's talk about the Regime Options that you have with RYE. 

**Regime Options:**
- `balanced` - Normal market conditions
- `bull` - Bull market (higher returns, lower volatility)
- `bear` - Bear market (negative returns, higher volatility)
- `volatile` - High volatility regime

### 3. **Run Backtests**
```bash
python rye.py backtest --input data/SPY.csv --strategy momentum
python rye.py backtest --input data/SPY.csv --strategy macd --fees-bps 10 --report
```

**Available Strategies:**
- `momentum` - SMA crossover (20 vs 50)
- `meanrev` - RSI-based mean reversion
- `macd` - MACD line crossover signals
- `bollinger` - Bollinger Bands mean reversion
- `stochastic` - Stochastic oscillator signals
- `trend` - EMA-based trend following with ATR stops

### 4. **Compare All Strategies**
```bash
python rye.py compare --input data/SPY.csv --report
```

### 5. **View Reports**
```bash
python rye.py serve --port 8080
```
Then open `http://localhost:8080` in your browser.

## **Technical Indicators**

### **Moving Averages**
- **SMA (Simple Moving Average)**: Traditional moving average
- **EMA (Exponential Moving Average)**: Weighted moving average

### **Momentum Indicators**
- **RSI (Relative Strength Index)**: Overbought/oversold levels
- **MACD**: Trend and momentum indicator
- **Stochastic**: Price momentum oscillator

### **Volatility Indicators**
- **Bollinger Bands**: Price channel with standard deviation
- **ATR (Average True Range)**: Volatility measurement

## **Trading Strategies**

### **Momentum Strategy**
- **Logic**: Go long when SMA(20) > SMA(50)
- **Exit**: Flat when SMA(20) < SMA(50)
- **Best for**: Trending markets

### **Mean Reversion Strategy**
- **Logic**: Buy when RSI < 30 (oversold)
- **Exit**: Sell when RSI > 70 (overbought)
- **Best for**: Range-bound markets

### **MACD Strategy**
- **Logic**: Buy on MACD line crossing above signal line
- **Exit**: Sell on MACD line crossing below signal line
- **Best for**: Trend changes and momentum

### **Bollinger Bands Strategy**
- **Logic**: Buy when price touches lower band
- **Exit**: Sell when price touches upper band
- **Best for**: Mean reversion in volatile markets

### **Stochastic Strategy**
- **Logic**: Buy when %K and %D < 20 (oversold)
- **Exit**: Sell when %K and %D > 80 (overbought)
- **Best for**: Range-bound markets with momentum

### **Trend Following Strategy**
- **Logic**: Buy when EMA(20) > EMA(50) and price > EMA(20)
- **Exit**: Sell on EMA crossover or ATR-based stop-loss
- **Best for**: Strong trending markets

## 📈 **Risk Metrics**

### **Return Metrics**
- **CAGR%**: Compound Annual Growth Rate
- **PnL%**: Total Profit/Loss percentage

### **Risk-Adjusted Returns**
- **Sharpe Ratio**: Return per unit of total risk
- **Sortino Ratio**: Return per unit of downside risk
- **Calmar Ratio**: CAGR to maximum drawdown ratio

### **Risk Metrics**
- **MaxDD%**: Maximum drawdown percentage
- **VaR 95%**: Value at Risk at 95% confidence
- **Exposure%**: Percentage of time in the market

### **Trade Analysis**
- **Win%**: Percentage of profitable trades
- **Profit Factor**: Gross profit to gross loss ratio
- **Trades**: Total number of trades executed

## ⚙️ **Configuration**

The `rye_config.ini` file allows customization of strategy parameters:

```ini
[momentum]
sma_short = 20
sma_long = 50

[meanrev]
rsi_period = 14
rsi_oversold = 30
rsi_overbought = 70

[macd]
fast_period = 12
slow_period = 26
signal_period = 9

[bollinger]
period = 20
std_dev = 2.0

[stochastic]
k_period = 14
d_period = 3
oversold = 20
overbought = 80

[trend]
ema_short = 20
ema_long = 50
atr_period = 14
atr_multiplier = 2.0

[backtest]
default_fees_bps = 5.0
```

## 📁 **File Formats**

### **Input CSV Format**
```csv
date,open,high,low,close,volume
2020-01-02,100.00,101.50,99.80,101.20,1000000
2020-01-03,101.20,102.30,100.90,102.10,1200000
```

**Required Columns:**
- `date`: YYYY-MM-DD format
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume (optional, defaults to 0)

### **Data Validation**
RYE automatically validates:
- File existence and permissions
- Required column presence
- OHLC data consistency (high ≥ low)
- Price relationship validity
- Volume data integrity

## **Error Handling**

RYE provides comprehensive error handling:

- **File Errors**: Missing files, permission issues
- **Data Errors**: Invalid formats, corrupted data
- **Validation Errors**: OHLC inconsistencies, missing columns
- **Strategy Errors**: Invalid parameters, calculation failures

All errors include:
- Clear error messages
- Colored output for visibility
- Graceful exit with error codes
- Detailed debugging information

## **Performance**

### **Optimizations**
- Efficient data structures for large datasets
- Optimized indicator calculations
- Memory-efficient backtesting engine
- Fast CSV parsing and validation

### **Scalability**
- Handles datasets with 100,000+ bars
- Efficient memory usage
- Fast strategy execution
- Quick report generation

## **Development**

### **Adding New Indicators**
```python
def my_indicator(prices: List[float], period: int) -> List[float]:
    return indicator_values
```

### **Adding New Strategies**
```python
def signal_mystrategy(closes: List[float]) -> List[int]:
    return signals
```

## Author 

Michael Mendy (c) 2025 _Quantum Point, LLC_ 
