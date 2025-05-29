from binance.client import Client
import pandas as pd
from price_action_engine import PriceActionPatterns

API_KEY = "senin_api_keyin"
API_SECRET = "senin_api_secretin"

client = Client(API_KEY, API_SECRET)
symbol = "BTCUSDT"
limit = 500  # Her zaman dilimi için 500 mum çekiyoruz

intervals = {
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR
}

for name, interval in intervals.items():
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    # Tip dönüşümleri
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  # Sadece gerekli kolonlar

    # Price Action analiz
    pa = PriceActionPatterns(df)
    print(f"\n{name} zaman dilimi için price action analizi:")
    print(pa.summary_report())