import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# ===============================
# 1️⃣ Download Data
# ===============================
symbol = "BTC-USD"

# Hourly data
# Daily data
train = yf.download(symbol, start="2022-01-01", end="2024-01-01", interval="1d")
test = yf.download(symbol, start="2024-01-01", end="2026-01-01", interval="1d")
train.dropna(inplace=True)
test.dropna(inplace=True)

# ===============================
# 2️⃣ Indicators
# ===============================
def add_indicators(df, short, long, rsi_period):
    df = df.copy()
    df["MA_short"] = df["Close"].rolling(short).mean()
    df["MA_long"] = df["Close"].rolling(long).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(24).var()
    return df

# ===============================
# 3️⃣ Backtest function
# ===============================
def backtest(df, params, capital=1000):
    short, long, rsi_period, rsi_overbought, rsi_oversold, vol_threshold = params

    # Protection pour rolling windows
    short = max(1, int(short))
    long = max(1, int(long))
    rsi_period = max(1, int(rsi_period))

    df = add_indicators(df, short, long, rsi_period)

    df["Signal"] = 0
    condition_buy = (
        (df["MA_short"] > df["MA_long"]) &
        (df["RSI"] < rsi_oversold) &
        (df["Volatility"] < vol_threshold)
    )
    condition_sell = (
        (df["MA_short"] < df["MA_long"]) |
        (df["RSI"] > rsi_overbought)
    )
    df.loc[condition_buy, "Signal"] = 1
    df.loc[condition_sell, "Signal"] = 0

    df["Position"] = df["Signal"].shift(1)
    df["Strategy"] = df["Return"] * df["Position"]
    df["Equity"] = capital * (1 + df["Strategy"]).cumprod()

    # Metrics with protection
    strategy_std = df["Strategy"].std()
    if strategy_std == 0 or np.isnan(strategy_std):
        sharpe = 0
    else:
        sharpe = np.sqrt(365) * df["Strategy"].mean() / strategy_std

    drawdown = (df["Equity"] / df["Equity"].cummax() - 1).min()

    fitness = df["Equity"].iloc[-1] + sharpe * 50 + drawdown * 500
    return fitness, df

# ===============================
# 4️⃣ Genetic Algorithm Setup
# ===============================
POP_SIZE = 50
GENERATIONS = 30
MUTATION_RATE = 0.3

# Population: [short, long, rsi_period, rsi_overbought, rsi_oversold, vol_threshold]
population = [
    [
        random.randint(5, 50),
        random.randint(60, 200),
        random.randint(10, 30),
        random.randint(60, 80),
        random.randint(20, 40),
        random.uniform(0.0001, 0.01)
    ]
    for _ in range(POP_SIZE)
]

# ===============================
# 5️⃣ Live visualization setup
# ===============================
plt.ion()
fig, axes = plt.subplots(2,2, figsize=(14,8))
fitness_history = []

# ===============================
# 6️⃣ Genetic Algorithm Loop
# ===============================
for gen in range(GENERATIONS):

    scores = []
    individuals_data = []

    for individual in population:
        short, long, *_ = individual
        if short >= long:
            scores.append(-999999)
            individuals_data.append(None)
        else:
            fitness, df = backtest(train, individual)
            scores.append(fitness)
            individuals_data.append(df)

    # Trier population par fitness
    sorted_data = sorted(
        zip(scores, population, individuals_data),
        key=lambda x: x[0],
        reverse=True
    )

    best_score, best_individual, best_df = sorted_data[0]
    fitness_history.append(best_score)

    # Sélection
    population = [x[1] for x in sorted_data[:POP_SIZE//2]]

    # Reproduction
    children = []
    while len(children) < POP_SIZE//2:
        p1 = random.choice(population)
        p2 = random.choice(population)
        child = [
            p1[0], p2[1],
            p1[2], p2[3],
            p1[4], p2[5]
        ]
        if random.random() < MUTATION_RATE:
            idx = random.randint(0,5)
            child[idx] *= random.uniform(0.8,1.2)
        children.append(child)
    population += children

    # ===============================
    # Live update plots
    # ===============================
    axes[0,0].cla()
    axes[0,0].plot(best_df["Equity"])
    axes[0,0].set_title(f"Best Equity - Gen {gen+1}")

    axes[0,1].cla()
    axes[0,1].plot(fitness_history)
    axes[0,1].set_title("Best Fitness Evolution")

    axes[1,0].cla()
    axes[1,0].hist(scores, bins=20)
    axes[1,0].set_title("Population Fitness Distribution")

    axes[1,1].cla()
    axes[1,1].bar(
        ["Short","Long","RSI_p","RSI_OB","RSI_OS","Vol"],
        best_individual
    )
    axes[1,1].set_title("Best Individual Parameters")

    plt.tight_layout()
    plt.pause(0.1)

    print(f"Generation {gen+1} | Best Fitness: {round(best_score,2)}")

plt.ioff()

# ===============================
# 7️⃣ Best parameters & live test
# ===============================
best_params = best_individual
print("\nBest Params:", best_params)

fitness, live_df = backtest(test, best_params, 1000)
print("Final Portfolio Value (Live Test):", round(live_df["Equity"].iloc[-1],2), "€")

# ===============================
# 8️⃣ Final Visualization (Live Test)
# ===============================
plt.figure(figsize=(14,10))

plt.subplot(3,1,1)
plt.plot(live_df["Close"])
plt.title("Live Price 2024-2026")

plt.subplot(3,1,2)
plt.plot(live_df["Equity"])
plt.title("Portfolio Equity Curve (Initial 1000€)")

plt.subplot(3,1,3)
plt.plot(live_df["Equity"] / live_df["Equity"].cummax() - 1)
plt.title("Drawdown")

plt.tight_layout()
plt.show()
