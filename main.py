import yaml
from binance.client import Client
import pandas as pd
from ta_engine import TechnicalAnalysisEngine
from price_action_engine import PriceActionPatterns
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Eklenen backtest ve canlı trade modülleri
from backtest import Backtester
from ccxt_live_trader import LiveTrader

# Sinyal bildirim entegrasyonu için
import datetime

# Config dosyasını yükle
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
BINANCE_API_KEY = config["binance"]["api_key"]
BINANCE_API_SECRET = config["binance"]["api_secret"]
TELEGRAM_TOKEN = config["telegram"]["bot_token"]

# Binance istemcisi
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Sinyal bildirim logu (tekrar sinyal spam'ini engellemek için)
signal_log = {}

def fetch_ohlcv(symbol="BTCUSDT", interval="15m", limit=200):
    interval_map = {
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR
    }
    klines = client.get_klines(
        symbol=symbol,
        interval=interval_map.get(interval, Client.KLINE_INTERVAL_15MINUTE),
        limit=limit
    )
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

# Otomatik sinyal bildirimi fonksiyonu (her 3 dakikada bir)
async def signal_notifier(application, chat_id, symbol="BTCUSDT", interval="15m"):
    df = fetch_ohlcv(symbol, interval)
    pa = PriceActionPatterns(df)
    summary = pa.summary_report()
    # Sadece yeni sinyali gönder (spam önlemi için)
    key = f"{symbol}_{interval}_{str(df['timestamp'].iloc[-1])}"
    if key not in signal_log or signal_log[key] != summary:
        signal_log[key] = summary
        if "🔥" in summary or "Trade Edilebilir" in summary:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            msg = f"[{now}] {symbol} ({interval})\n{s_summary_limiter(summary)}"
            await application.bot.send_message(chat_id=chat_id, text=msg)

def s_summary_limiter(summary, maxlen=3500):
    # Telegram mesaj limiti için
    return summary if len(summary) < maxlen else summary[:maxlen-10] + "..."

# Telegram komutu: /analyse BTCUSDT 15m
async def analyse(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) == 0:
        await update.message.reply_text("Sembol gir: /analyse BTCUSDT 15m")
        return
    symbol = args[0].upper()
    interval = args[1] if len(args) > 1 else "15m"
    df = fetch_ohlcv(symbol, interval)
    pa = PriceActionPatterns(df)
    result = pa.summary_report()
    await update.message.reply_text(f"{symbol} ({interval}) için Price Action Analizi:\n{s_summary_limiter(result)}")

# Telegram komutu: /ta BTCUSDT 15m
async def ta_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) == 0:
        await update.message.reply_text("Sembol gir: /ta BTCUSDT 15m")
        return
    symbol = args[0].upper()
    interval = args[1] if len(args) > 1 else "15m"
    df = fetch_ohlcv(symbol, interval)
    tae = TechnicalAnalysisEngine(df)
    result = tae.summary_report()
    await update.message.reply_text(f"{symbol} ({interval}) için Teknik Analiz Raporu:\n{s_summary_limiter(result)}")

# Telegram komutu: /backtest BTCUSDT 15m
async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) == 0:
        await update.message.reply_text("Sembol gir: /backtest BTCUSDT 15m")
        return
    symbol = args[0].upper()
    interval = args[1] if len(args) > 1 else "15m"
    df = fetch_ohlcv(symbol, interval)

    # Basit örnek strateji: Son bar trade edilebilir pinbar ise işleme gir
    def strategy_func(df, idx):
        pap = PriceActionPatterns(df)
        patterns = pap.detect_pinbar()
        if patterns and patterns[-1]['index'] == idx:
            p = patterns[-1]
            if p['type'] == 'bullish':
                return "buy"
            elif p['type'] == 'bearish':
                return "sell"
        return None

    def tp_sl_func(idx, entry_type):
        pap = PriceActionPatterns(df)
        return pap.dynamic_tp_sl(idx, entry_type)

    backtester = Backtester(df, strategy_func, tp_sl_func)
    backtester.run()
    summary = backtester.summary()
    msg = (
        f"Backtest Sonucu ({symbol} {interval}):\n"
        f"Son Bakiye: {summary['final_balance']:.2f}\n"
        f"Toplam İşlem: {summary['total_trades']}\n"
        f"Kazanç Oranı: {summary['win_rate']*100:.1f}%\n"
        f"Maksimum Drawdown: {summary['max_drawdown']*100:.1f}%"
    )
    await update.message.reply_text(msg)

# Telegram komutu: /trade BTCUSDT buy 0.001 [market/limit] [fiyat]
async def trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("Kullanım: /trade BTCUSDT buy 0.001 [market/limit] [fiyat]")
        return
    symbol = args[0].replace("USDT", "/USDT").upper()  # CCXT sembol formatı ör: BTC/USDT
    side = args[1].lower()
    amount = float(args[2])
    order_type = args[3] if len(args) > 3 else "market"
    price = float(args[4]) if len(args) > 4 and order_type == "limit" else None

    # CCXT ile canlı trade (Binance örneği)
    api_key = config["binance"]["api_key"]
    api_secret = config["binance"]["api_secret"]
    testnet = config["binance"].get("testnet", False)
    trader = LiveTrader("binance", api_key, api_secret, testnet=testnet)
    result = trader.send_order(symbol, side, amount, price, order_type)
    if result:
        await update.message.reply_text(f"Emir gönderildi: {result}")
    else:
        await update.message.reply_text("Emir gönderilemedi!")

# Telegram komutu: /startsignal BTCUSDT 15m
async def start_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 1:
        await update.message.reply_text("Kullanım: /startsignal BTCUSDT 15m")
        return
    symbol = args[0].upper()
    interval = args[1] if len(args) > 1 else "15m"
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"{symbol} ({interval}) için otomatik sinyal bildirimi başlatıldı.")

    # Her 3 dakikada bir price action sinyalini kullanıcıya bildir
    async def notifier_job(application):
        while True:
            try:
                await signal_notifier(application, chat_id, symbol, interval)
                await asyncio.sleep(180)  # 3 dakika bekle
            except Exception as e:
                print("Sinyal bildirici hata:", e)
                await asyncio.sleep(180)

    import asyncio
    application = update.application
    asyncio.create_task(notifier_job(application))

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("analyse", analyse))
    app.add_handler(CommandHandler("ta", ta_report))
    app.add_handler(CommandHandler("backtest", backtest_cmd))
    app.add_handler(CommandHandler("trade", trade))
    app.add_handler(CommandHandler("startsignal", start_signal))
    print("Bot başlatılıyor...")
    app.run_polling()