import ccxt
import pattern_stats  # Entegrasyon için ekledik
from datetime import datetime

class LiveTrader:
    def __init__(self, exchange_id, api_key, secret, password=None, testnet=False):
        exchange_class = getattr(ccxt, exchange_id)
        params = {'apiKey': api_key, 'secret': secret}
        if password:
            params['password'] = password
        if testnet and exchange_id == "binance":
            params['options'] = {'defaultType': 'future'}
            params['urls'] = {'api': {'private': 'https://testnet.binancefuture.com'}}
        self.exchange = exchange_class(params)

    def send_order(self, symbol, side, amount, price=None, order_type='market'):
        try:
            if order_type == 'market':
                return self.exchange.create_market_order(symbol, side, amount)
            else:
                return self.exchange.create_limit_order(symbol, side, amount, price)
        except Exception as e:
            print("Order Error:", str(e))
            return None

    def get_balance(self):
        return self.exchange.fetch_balance()

    def get_open_orders(self, symbol=None):
        return self.exchange.fetch_open_orders(symbol) if symbol else self.exchange.fetch_open_orders()

    def record_trade_result(self, pattern_key, is_win, close_time=None):
        """
        Pozisyon kapanınca çağır: pattern_key (örn: 'pinbar_long_15m'), is_win (bool)
        """
        if not close_time:
            close_time = datetime.utcnow().isoformat()
        pattern_stats.update_pattern_stats(pattern_key, is_win, close_time=close_time)
        print(f"[STATS] Pattern '{pattern_key}' için {'WIN' if is_win else 'LOSE'} kaydedildi.")

# Örnek kullanım:
# trader = LiveTrader("binance", "...", "...")
# trade kapanınca:
# trader.record_trade_result("pinbar_long_15m", is_win=True)