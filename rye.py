# Author Michael Mendy (c) 2025 for Quantum Point, LLC.

from __future__ import annotations
import argparse, csv, json, math, os, random, statistics as stats, sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional


ANSI = {
    'g': '\x1b[92m', 'r': '\x1b[91m', 'y': '\x1b[93m', 'c': '\x1b[96m', 'b': '\x1b[94m',
    'dim': '\x1b[2m', 'reset': '\x1b[0m'
}

def ensure_dirs():
    try:
        for p in [Path('data'), Path('reports'), Path('runs')]:
            p.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"{ANSI['r']}✗{ANSI['reset']} Permission denied creating directories")
        sys.exit(1)
    except Exception as e:
        print(f"{ANSI['r']}✗{ANSI['reset']} Error creating directories: {e}")
        sys.exit(1)


def parse_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {s}. Expected YYYY-MM-DD")


def daterange(start: datetime, days: int) -> List[datetime]:
    out, d = [], start
    while len(out) < days:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


@dataclass
class Bar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


def read_csv(path: Path) -> List[Bar]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    bars: List[Bar] = []
    required_columns = {'date', 'open', 'high', 'low', 'close'}
    
    try:
        with open(path, 'r', newline='') as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                raise ValueError("CSV file is empty")
            
            missing_cols = required_columns - set(r.fieldnames)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            for i, row in enumerate(r, 1):
                try:
                    bar = Bar(
                        date=parse_date(row['date']),
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(float(row.get('volume', 0)))
                    )
                    
                    if bar.high < bar.low:
                        raise ValueError(f"High ({bar.high}) < Low ({bar.low}) at row {i}")
                    if bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close):
                        raise ValueError(f"OHLC values inconsistent at row {i}")
                    if bar.volume < 0:
                        raise ValueError(f"Negative volume at row {i}")
                    
                    bars.append(bar)
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Error parsing row {i}: {e}")
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    if not bars:
        raise ValueError("No valid data found in CSV file")
    
    return bars


def write_csv(path: Path, bars: List[Bar]):
    if not bars:
        raise ValueError("No bars to write")
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['date','open','high','low','close','volume'])
            for b in bars:
                w.writerow([b.date.strftime('%Y-%m-%d'), f'{b.open:.4f}', f'{b.high:.4f}', f'{b.low:.4f}', f'{b.close:.4f}', b.volume])
    except PermissionError:
        raise PermissionError(f"Permission denied writing to {path}")
    except Exception as e:
        raise ValueError(f"Error writing CSV file: {e}")



def synth_series(days: int, start_price: float=100.0, regime: str='balanced') -> List[float]:
    if regime == 'bull':
        mu, sigma = 0.12, 0.18
    elif regime == 'bear':
        mu, sigma = -0.05, 0.28
    elif regime == 'volatile':
        mu, sigma = 0.06, 0.35
    else:
        mu, sigma = 0.08, 0.22

    dt = 1.0/252.0
    price = start_price
    series = [price]
    for i in range(1, days):

        if random.random() < 0.02:
            mu += random.uniform(-0.03, 0.03)
            sigma = max(0.08, min(0.5, sigma + random.uniform(-0.05, 0.05)))
        z = random.gauss(0, 1)
        price = price * math.exp((mu - 0.5 * sigma*sigma)*dt + sigma * math.sqrt(dt) * z)
        series.append(price)
    return series


def synth_ohlcv(days: int, start_date: datetime, start_price: float=100.0, regime: str='balanced') -> List[Bar]:
    dates = daterange(start_date, days)
    closes = synth_series(len(dates), start_price, regime)
    bars: List[Bar] = []
    for i, d in enumerate(dates):
        c = closes[i]
        o = c * (1 + random.uniform(-0.003, 0.003))
        h = max(o, c) * (1 + random.uniform(0.000, 0.008))
        l = min(o, c) * (1 - random.uniform(0.000, 0.008))
        v = int(1_000_000 * (1 + random.uniform(-0.25, 0.25)))
        bars.append(Bar(d, o, h, l, c, v))
    return bars



def sma(vals: List[float], n: int) -> List[float]:
    out, buf = [], []
    s = 0.0
    for v in vals:
        buf.append(v); s += v
        if len(buf) > n:
            s -= buf.pop(0)
        out.append(s / len(buf))
    return out


def rsi(closes: List[float], n: int=14) -> List[float]:
    gains, losses = [0.0], [0.0]
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        gains.append(max(0.0, ch))
        losses.append(max(0.0, -ch))
    rsis = []
    avg_gain = sum(gains[1:n+1]) / n
    avg_loss = sum(losses[1:n+1]) / n
    for i in range(len(closes)):
        if i < n:
            rsis.append(50.0)
            continue
        if i > n:
            avg_gain = (avg_gain*(n-1) + gains[i]) / n
            avg_loss = (avg_loss*(n-1) + losses[i]) / n
        rs = avg_gain / avg_loss if avg_loss > 1e-12 else float('inf')
        r = 100 - 100/(1+rs) if rs != float('inf') else 100.0
        rsis.append(r)
    return rsis


def ema(vals: List[float], n: int) -> List[float]:
    if not vals:
        return []
    alpha = 2.0 / (n + 1)
    emas = [vals[0]]
    for i in range(1, len(vals)):
        ema_val = alpha * vals[i] + (1 - alpha) * emas[-1]
        emas.append(ema_val)
    return emas


def macd(closes: List[float], fast: int=12, slow: int=26, signal: int=9) -> Tuple[List[float], List[float], List[float]]:
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, histogram


def bollinger_bands(closes: List[float], n: int=20, std_dev: float=2.0) -> Tuple[List[float], List[float], List[float]]:
    sma_vals = sma(closes, n)
    upper_bands = []
    lower_bands = []
    
    for i in range(len(closes)):
        if i < n - 1:
            upper_bands.append(closes[i])
            lower_bands.append(closes[i])
        else:
            period_closes = closes[i-n+1:i+1]
            mean = sma_vals[i]
            variance = sum((x - mean) ** 2 for x in period_closes) / len(period_closes)
            std = math.sqrt(variance)
            upper_bands.append(mean + std_dev * std)
            lower_bands.append(mean - std_dev * std)
    
    return upper_bands, sma_vals, lower_bands


def stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int=14, d_period: int=3) -> Tuple[List[float], List[float]]:
    k_percent = []
    for i in range(len(closes)):
        if i < k_period - 1:
            k_percent.append(50.0)
        else:
            period_highs = highs[i-k_period+1:i+1]
            period_lows = lows[i-k_period+1:i+1]
            highest_high = max(period_highs)
            lowest_low = min(period_lows)
            if highest_high != lowest_low:
                k = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low)
            else:
                k = 50.0
            k_percent.append(k)
    
    d_percent = sma(k_percent, d_period)
    return k_percent, d_percent


def atr(highs: List[float], lows: List[float], closes: List[float], n: int=14) -> List[float]:
    tr_vals = []
    for i in range(len(closes)):
        if i == 0:
            tr_vals.append(highs[i] - lows[i])
        else:
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr_vals.append(max(tr1, tr2, tr3))
    
    return sma(tr_vals, n)



def signal_momentum(closes: List[float]) -> List[int]:
    s20, s50 = sma(closes, 20), sma(closes, 50)
    sig = []
    for i in range(len(closes)):
        sig.append(1 if s20[i] > s50[i] else 0)
    return sig


def signal_meanrev(closes: List[float]) -> List[int]:
    r = rsi(closes, 14)
    sig = []
    pos = 0
    for i in range(len(closes)):
        if r[i] < 30: pos = 1
        elif r[i] > 70: pos = 0
        sig.append(pos)
    return sig


def signal_macd(closes: List[float]) -> List[int]:
    macd_line, signal_line, _ = macd(closes)
    sig = []
    pos = 0
    for i in range(len(closes)):
        if i == 0:
            sig.append(0)
            continue
        if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
            pos = 1
        elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
            pos = 0
        sig.append(pos)
    return sig


def signal_bollinger(closes: List[float]) -> List[int]:
    upper, middle, lower = bollinger_bands(closes)
    sig = []
    pos = 0
    for i in range(len(closes)):
        if closes[i] <= lower[i] and pos == 0:
            pos = 1
        elif closes[i] >= upper[i] and pos == 1:
            pos = 0
        sig.append(pos)
    return sig


def signal_stochastic(highs: List[float], lows: List[float], closes: List[float]) -> List[int]:
    k_percent, d_percent = stochastic(highs, lows, closes)
    sig = []
    pos = 0
    for i in range(len(closes)):
        if k_percent[i] < 20 and d_percent[i] < 20 and pos == 0:
            pos = 1
        elif k_percent[i] > 80 and d_percent[i] > 80 and pos == 1:
            pos = 0
        sig.append(pos)
    return sig


def signal_trend_following(closes: List[float], highs: List[float], lows: List[float]) -> List[int]:
    ema_20 = ema(closes, 20)
    ema_50 = ema(closes, 50)
    atr_vals = atr(highs, lows, closes)
    sig = []
    pos = 0
    for i in range(len(closes)):
        if i < 50:
            sig.append(0)
            continue
        if ema_20[i] > ema_50[i] and closes[i] > ema_20[i] and pos == 0:
            pos = 1
        elif (ema_20[i] < ema_50[i] or closes[i] < ema_20[i] - 2*atr_vals[i]) and pos == 1:
            pos = 0
        sig.append(pos)
    return sig


@dataclass
class Result:
    equity_curve: List[float]
    dates: List[str]
    daily_returns: List[float]
    trades: int
    exposure: float
    metrics: Dict[str, float]


def backtest(bars: List[Bar], strategy: str='momentum', fees_bps: float=0.0) -> Result:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    dates = [b.date.strftime('%Y-%m-%d') for b in bars]
    
    if strategy == 'meanrev':
        sig = signal_meanrev(closes)
    elif strategy == 'macd':
        sig = signal_macd(closes)
    elif strategy == 'bollinger':
        sig = signal_bollinger(closes)
    elif strategy == 'stochastic':
        sig = signal_stochastic(highs, lows, closes)
    elif strategy == 'trend':
        sig = signal_trend_following(closes, highs, lows)
    else:
        sig = signal_momentum(closes)

    pos = 0
    trades = 0
    fees = fees_bps / 10000.0
    equity = 1.0
    eq_curve = [equity]
    dly = [0.0]
    time_in_mkt = 0

    for i in range(1, len(closes)):
        target = sig[i]
        if target != pos:

            equity *= (1 - fees)
            trades += 1
            pos = target
        ret = (closes[i] / closes[i-1]) - 1.0
        day_ret = pos * ret
        equity *= (1 + day_ret)
        dly.append(day_ret)
        eq_curve.append(equity)
        if pos: time_in_mkt += 1


    N = len(dly)
    mean_d = sum(dly)/N
    std_d = stats.pstdev(dly) if N > 1 else 0.0
    sharpe = (mean_d * math.sqrt(252) / std_d) if std_d > 1e-12 else 0.0
    cagr = (eq_curve[-1]) ** (252.0/N) - 1.0 if N>0 else 0.0


    peak = eq_curve[0]
    dd = 0.0
    max_dd = 0.0
    for v in eq_curve:
        peak = max(peak, v)
        dd = (v/peak) - 1.0
        max_dd = min(max_dd, dd)

    win_rate = sum(1 for x in dly if x>0) / N if N>0 else 0.0
    exposure = time_in_mkt / N if N>0 else 0.0
    pnl_pct = (eq_curve[-1]-1.0)*100.0
    
    negative_returns = [r for r in dly if r < 0]
    downside_std = stats.pstdev(negative_returns) if len(negative_returns) > 1 else 0.0
    sortino = (mean_d * math.sqrt(252) / downside_std) if downside_std > 1e-12 else 0.0
    
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0.0
    
    var_95 = sorted(dly)[int(0.05 * len(dly))] if len(dly) > 20 else 0.0
    
    positive_returns = [r for r in dly if r > 0]
    avg_win = stats.mean(positive_returns) if positive_returns else 0.0
    avg_loss = stats.mean(negative_returns) if negative_returns else 0.0
    profit_factor = (avg_win * len(positive_returns)) / (abs(avg_loss) * len(negative_returns)) if negative_returns and avg_loss != 0 else 0.0

    metrics = {
        'CAGR%': round(cagr*100, 2),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'Calmar': round(calmar, 2),
        'MaxDD%': round(max_dd*100, 2),
        'VaR95%': round(var_95*100, 2),
        'Win%': round(win_rate*100, 2),
        'ProfitFactor': round(profit_factor, 2),
        'Exposure%': round(exposure*100, 2),
        'PnL%': round(pnl_pct, 2),
        'Trades': trades
    }

    return Result(eq_curve, dates, dly, trades, exposure, metrics)


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RYE Report — {title}</title>
<style>
  :root{ --bg:#0b0d12; --panel:#121722; --text:#e8f1ff; --muted:#9fb2d9; --accent1:#00f2fe; --accent2:#4facfe }
  body{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial }
  .wrap{ max-width:1000px; margin:28px auto; padding:0 16px }
  h1{ font-size:22px; margin:0 0 8px }
  .meta{ color:var(--muted); margin-bottom:16px }
  .grid{ display:grid; grid-template-columns: 2fr 1fr; gap:16px }
  .card{ background:linear-gradient(180deg,#121722,#0f1420); border:1px solid #1a2338; border-radius:12px; padding:14px }
  .kv{ display:grid; grid-template-columns:auto 1fr; gap:4px 10px; font-family:ui-monospace,Menlo,monospace }
  .kv div:nth-child(odd){ color:#bcd0f7 }
  canvas{ width:100%; height:320px; background:#070a12; border:1px solid #17203a; border-radius:10px }
  .pill{ display:inline-block; margin:4px 6px 0 0; padding:6px 10px; border:1px solid #1a2338; border-radius:999px; background:#0e1422 }
</style>
</head>
<body>
  <div class="wrap">
    <h1>RYE — Risk & Yield Engine <span class="pill">{strategy}</span></h1>
    <div class="meta">Report time: {now} • Bars: {bars} • Fees(bps): {fees_bps}</div>
    <div class="grid">
      <div class="card">
        <canvas id="equity"></canvas>
      </div>
      <div class="card">
        <div class="kv">
          <div>CAGR</div><div>{CAGR}%</div>
          <div>Sharpe</div><div>{Sharpe}</div>
          <div>Sortino</div><div>{Sortino}</div>
          <div>Calmar</div><div>{Calmar}</div>
          <div>Max Drawdown</div><div>{MaxDD}%</div>
          <div>VaR 95%</div><div>{VaR95}%</div>
          <div>Win Rate</div><div>{Win}%</div>
          <div>Profit Factor</div><div>{ProfitFactor}</div>
          <div>Exposure</div><div>{Exposure}%</div>
          <div>PnL</div><div>{PnL}%</div>
          <div>Trades</div><div>{Trades}</div>
        </div>
      </div>
    </div>
  </div>
<script>
const DATES = {dates};
const EQ = {equity};
const ctx = document.getElementById('equity').getContext('2d');

(function draw(){
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const pad = 24;
  ctx.clearRect(0,0,W,H);

  const min = Math.min(...EQ), max = Math.max(...EQ);
  const x = i => pad + (W-2*pad) * (i/(EQ.length-1));
  const y = v => H-pad - (H-2*pad) * ((v-min)/(max-min||1));

  ctx.strokeStyle = '#1a2338'; ctx.lineWidth=1;
  for(let i=0;i<5;i++){ let yy = pad + i*(H-2*pad)/4; ctx.beginPath(); ctx.moveTo(pad,yy); ctx.lineTo(W-pad,yy); ctx.stroke(); }

  ctx.lineWidth = 2; ctx.strokeStyle = '#4facfe';
  ctx.beginPath(); ctx.moveTo(x(0), y(EQ[0]));
  for(let i=1;i<EQ.length;i++){ ctx.lineTo(x(i), y(EQ[i])); }
  ctx.stroke();
})();
</script>
</body>
</html>
"""


def write_report(path: Path, res: Result, title: str, strategy: str, fees_bps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.format(
        title=title,
        strategy=strategy,
        now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        bars=len(res.equity_curve),
        fees_bps=fees_bps,
        dates=json.dumps(res.dates),
        equity=json.dumps([round(x,6) for x in res.equity_curve]),
        CAGR=res.metrics['CAGR%'],
        Sharpe=res.metrics['Sharpe'],
        Sortino=res.metrics['Sortino'],
        Calmar=res.metrics['Calmar'],
        MaxDD=res.metrics['MaxDD%'],
        VaR95=res.metrics['VaR95%'],
        Win=res.metrics['Win%'],
        ProfitFactor=res.metrics['ProfitFactor'],
        Exposure=res.metrics['Exposure%'],
        PnL=res.metrics['PnL%'],
        Trades=res.metrics['Trades']
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)



def cmd_init(args):
    ensure_dirs()

    Path('README_RYE.txt').write_text('This repo was initialized by RYE (Quantum Point).\nFolders: data/, reports/, runs/\n', encoding='utf-8')
    
    config_content = """# RYE Configuration File
# Strategy parameters can be customized here

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
"""
    
    Path('rye_config.ini').write_text(config_content, encoding='utf-8')
    print(f"{ANSI['g']}✔{ANSI['reset']} Initialized folders: data/, reports/, runs/")
    print(f"{ANSI['g']}✔{ANSI['reset']} Created configuration file: rye_config.ini")


def cmd_synth(args):
    try:
        ensure_dirs()
        start = parse_date(args.start) if args.start else datetime.today() - timedelta(days=args.days*1.5)
        bars = synth_ohlcv(args.days, start, args.price, args.regime)
        out = Path('data') / f"{args.symbol}.csv"
        write_csv(out, bars)
        print(f"{ANSI['g']}✔{ANSI['reset']} Wrote synthetic data → {out}  ({len(bars)} bars, regime={args.regime})")
    except Exception as e:
        print(f"{ANSI['r']}✗{ANSI['reset']} Error generating synthetic data: {e}")
        sys.exit(1)


def cmd_backtest(args):
    try:
        ensure_dirs()
        bars = read_csv(Path(args.input))
        res = backtest(bars, args.strategy, args.fees_bps)

        print("\n=== RYE BACKTEST ===")
        print(f"Strategy  : {args.strategy}")
        print(f"Bars      : {len(bars)}")
        for k,v in res.metrics.items():
            label = f"{k:>9}"
            if isinstance(v, float):
                vv = f"{v:>8.2f}"
            else:
                vv = f"{v:>8}"
            color = 'g' if (k in ('CAGR%','Sharpe','Win%','PnL%') and float(v)>=0) or (k=='MaxDD%' and float(v)>=-10.0) else 'r'
            print(f"{label}: {ANSI[color]}{vv}{ANSI['reset']}")
        if args.report:
            title = Path(args.input).stem
            out = Path('reports') / f"rye_report_{title}_{args.strategy}.html"
            write_report(out, res, title, args.strategy, args.fees_bps)
            print(f"\n{ANSI['c']}Report:{ANSI['reset']} {out}")
    except Exception as e:
        print(f"{ANSI['r']}✗{ANSI['reset']} Error running backtest: {e}")
        sys.exit(1)


def cmd_compare(args):
    try:
        ensure_dirs()
        bars = read_csv(Path(args.input))
        strategies = ['momentum', 'meanrev', 'macd', 'bollinger', 'stochastic', 'trend']
        
        print("\n=== RYE STRATEGY COMPARISON ===")
        print(f"Data: {args.input} ({len(bars)} bars)")
        print(f"Fees: {args.fees_bps} bps")
        print()
        
        results = {}
        for strategy in strategies:
            res = backtest(bars, strategy, args.fees_bps)
            results[strategy] = res
        
        print(f"{'Strategy':<12} {'CAGR%':<8} {'Sharpe':<8} {'Sortino':<8} {'MaxDD%':<8} {'Win%':<8} {'Trades':<8}")
        print("-" * 70)
        
        for strategy, res in results.items():
            print(f"{strategy:<12} {res.metrics['CAGR%']:<8.2f} {res.metrics['Sharpe']:<8.2f} "
                  f"{res.metrics['Sortino']:<8.2f} {res.metrics['MaxDD%']:<8.2f} "
                  f"{res.metrics['Win%']:<8.2f} {res.metrics['Trades']:<8}")
        
        if args.report:
            for strategy, res in results.items():
                title = f"{Path(args.input).stem}_{strategy}"
                out = Path('reports') / f"rye_report_{title}.html"
                write_report(out, res, title, strategy, args.fees_bps)
            print(f"\n{ANSI['c']}Reports:{ANSI['reset']} Generated {len(strategies)} HTML reports in reports/")
            
    except Exception as e:
        print(f"{ANSI['r']}✗{ANSI['reset']} Error running comparison: {e}")
        sys.exit(1)


def cmd_serve(args):
    import http.server, socketserver
    ensure_dirs()
    port = args.port
    os.chdir('reports')
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving reports/ at http://localhost:{port}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def build_parser():
    p = argparse.ArgumentParser(prog='rye', description='RYE — Risk & Yield Engine (Quantum Point)')
    sub = p.add_subparsers(dest='cmd')

    sp = sub.add_parser('init', help='create data/, reports/, runs/')
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser('synth', help='generate synthetic OHLCV CSV')
    sp.add_argument('--symbol', default='SPY')
    sp.add_argument('--days', type=int, default=1250)
    sp.add_argument('--start', type=str, help='YYYY-MM-DD (business days only)')
    sp.add_argument('--price', type=float, default=100.0)
    sp.add_argument('--regime', choices=['balanced','bull','bear','volatile'], default='balanced')
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser('backtest', help='run a backtest on CSV data')
    sp.add_argument('--input', required=True, help='path to CSV with date,open,high,low,close,volume')
    sp.add_argument('--strategy', choices=['momentum','meanrev','macd','bollinger','stochastic','trend'], default='momentum')
    sp.add_argument('--fees-bps', type=float, default=5.0, dest='fees_bps', help='per-trade fee in basis points')
    sp.add_argument('--report', action='store_true', help='write HTML report to reports/')
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser('compare', help='compare all strategies on CSV data')
    sp.add_argument('--input', required=True, help='path to CSV with date,open,high,low,close,volume')
    sp.add_argument('--fees-bps', type=float, default=5.0, dest='fees_bps', help='per-trade fee in basis points')
    sp.add_argument('--report', action='store_true', help='write HTML reports for all strategies')
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser('serve', help='serve reports/ over HTTP')
    sp.add_argument('--port', type=int, default=8080)
    sp.set_defaults(func=cmd_serve)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, 'func'):
        parser.print_help()
        return 0
    return args.func(args)

if __name__ == '__main__':
    raise SystemExit(main())
