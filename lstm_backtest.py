import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler

# --- Parametreler ---
symbol = "BTCUSDT"
interval = "15m"
limit = 1000
window = 30
tp_ratio = 0.01   # %1 TP
sl_ratio = 0.01   # %1 SL
conf_long = 0.7
conf_short = 0.3

# --- Binance'ten veri çek ---
client = Client()
klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
])
df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
df = df[['open', 'high', 'low', 'close', 'volume']]

# --- Normalize ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
model = load_model("lstm_model.h5")

preds = []
for i in range(window, len(df)-2):  # -2: TP/SL için bir bar ileride veri gerekiyor
    window_data = scaled[i-window:i].reshape(1, window, 5)
    conf = float(model.predict(window_data, verbose=0)[0,0])
    preds.append({
        "index": i,
        "confidence": conf,
        "future_close": df['close'].iloc[i+1],
        "future_high": df['high'].iloc[i+1],
        "future_low": df['low'].iloc[i+1],
        "current_close": df['close'].iloc[i]
    })

# --- TP/SL ile backtest ---
results = []
for p in preds:
    action = None
    if p["confidence"] > conf_long:
        action = "long"
    elif p["confidence"] < conf_short:
        action = "short"

    if action:
        entry = p["current_close"]
        if action == "long":
            tp = entry * (1 + tp_ratio)
            sl = entry * (1 - sl_ratio)
            # Önce TP mi SL mi geliyor?
            if p["future_high"] >= tp:
                win = True
            elif p["future_low"] <= sl:
                win = False
            else:
                # Ne TP ne SL: close'a göre değerlendir
                win = p["future_close"] > entry
        elif action == "short":
            tp = entry * (1 - tp_ratio)
            sl = entry * (1 + sl_ratio)
            if p["future_low"] <= tp:
                win = True
            elif p["future_high"] >= sl:
                win = False
            else:
                win = p["future_close"] < entry
        results.append(win)

if results:
    winrate = sum(results) / len(results)
    print(f"LSTM confidence + TP/SL backtest sonucu: {len(results)} trade, Winrate: %{winrate*100:.1f}")
else:
    print("Hiç trade açılmadı (sınırları/conf sınırlarını düşürüp artırabilirsin).")