import numpy as np
import pandas as pd
from binance.client import Client
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Binance API anahtarsız da public veri çeker
client = Client()

symbol = "BTCUSDT"
interval = "15m"
limit = 2000  # Geniş tut!

# 1. Binance'ten veri çek
klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
])
df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
df = df[['open', 'high', 'low', 'close', 'volume']]

# 2. Feature scaling
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# 3. Window'lu veri seti oluştur (ör: 30 barlık pencere ile "gelecek close artacak mı?")
window = 30
X = []
y = []
for i in range(window, len(scaled)-1):
    X.append(scaled[i-window:i])
    # Sinyal: Sonraki kapanış > Şu anki kapanış ise 1, değilse 0
    y.append(int(df['close'].iloc[i+1] > df['close'].iloc[i]))

X = np.array(X)
y = np.array(y)

# 4. Modeli hazırla ve eğit
model = Sequential([
    LSTM(32, input_shape=(window, X.shape[2])),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=6, batch_size=32, verbose=1)
model.save("lstm_model.h5")
print("Gerçek OHLCV ile eğitilmiş model kaydedildi: lstm_model.h5")