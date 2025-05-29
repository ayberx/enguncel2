import yaml
import asyncio
import pandas as pd
from binance.client import Client
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes
from telegram.error import BadRequest
from price_action_engine import PriceActionPatterns
import pattern_stats
from ta_engine import TechnicalAnalysisEngine  # <--- TA ENGINE ENTEGRASYONU

from datetime import datetime

# --- Config ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
BINANCE_API_KEY = config["binance"]["api_key"]
BINANCE_API_SECRET = config["binance"]["api_secret"]
TELEGRAM_TOKEN = config["telegram"]["bot_token"]
CHAT_ID = config["telegram"]["chat_id"]

# --- Binance ve Telegram client ---
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = Bot(token=TELEGRAM_TOKEN)

signal_log = {}

def fetch_ohlcv_futures(symbol="BTCUSDT", interval="15m", limit=200):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

def parse_pattern_line(line):
    import re
    pattern = {
        "pattern_emoji": "🔥",
        "pattern_name": "Pattern",
        "direction": "",
        "score": "",
        "confidence": "",
        "success": "",
        "index": "",
    }
    # Gelişmiş başarı oranlı pattern satırı
    m = re.search(r"🔥 LSTM Doğrulamalı ([\w\s]+) \(score=([\d]+)/10, conf=([\d\.]+), success=([\d\.]+)%\): index (\d+) \| (LONG|SHORT)", line)
    if m:
        pattern["pattern_name"] = m.group(1)
        pattern["score"] = m.group(2)
        pattern["confidence"] = m.group(3)
        pattern["success"] = m.group(4)
        pattern["index"] = m.group(5)
        pattern["direction"] = "long" if m.group(6) == "LONG" else "short"
    else:
        # Eski stil fallback
        m2 = re.search(r"🔥 LSTM Doğrulamalı ([\w\s]+) \(score=([\d]+)/10, conf=([\d\.]+)\): index (\d+) \| (LONG|SHORT)", line)
        if m2:
            pattern["pattern_name"] = m2.group(1)
            pattern["score"] = m2.group(2)
            pattern["confidence"] = m2.group(3)
            pattern["index"] = m2.group(4)
            pattern["direction"] = "long" if m2.group(5) == "LONG" else "short"
            pattern["success"] = ""
        else:
            m3 = re.search(r"🔥 Trade Edilebilir ([\w\s]+) \(score=([\d]+)/10\): index (\d+) \| (LONG|SHORT)", line)
            if m3:
                pattern["pattern_name"] = m3.group(1)
                pattern["score"] = m3.group(2)
                pattern["index"] = m3.group(3)
                pattern["direction"] = "long" if m3.group(4) == "LONG" else "short"
                pattern["confidence"] = "0.5"
                pattern["success"] = ""
    return pattern

def build_signal_buttons():
    buttons = [
        [InlineKeyboardButton("📊 Pattern Başarıları", callback_data="show_stats")]
    ]
    return InlineKeyboardMarkup(buttons)

async def send_telegram_signal(symbol, interval, pattern_info, df, ind_snapshot, ta_report, tp=None, sl=None, comment=None):
    pattern_emoji = pattern_info.get("pattern_emoji", "🔥")
    pattern_name = pattern_info.get("pattern_name", "Pattern")
    direction = pattern_info.get("direction", "")
    score = pattern_info.get("score", "")
    confidence = pattern_info.get("confidence", "")
    success = pattern_info.get("success", "")
    idx = int(pattern_info.get("index", -1))
    entry_price = df['close'].iloc[idx] if (0 <= idx < len(df)) else df['close'].iloc[-1]
    ts = df['timestamp'].iloc[idx] if (0 <= idx < len(df)) else df['timestamp'].iloc[-1]
    signal_time = ts.strftime("%Y-%m-%d %H:%M")
    if not tp or not sl:
        tp, sl = PriceActionPatterns(df).dynamic_tp_sl(idx, entry_type=direction)
    if not comment:
        comment = f"{pattern_name}, hacim ve skor filtreli otomatik sinyal."

    msg = (
        f"🔔 *Yeni Trade Sinyali!* 🔔\n\n"
        f"*Sembol:* `{symbol}`\n"
        f"*Timeframe:* `{interval}`\n"
        f"*Pattern:* {pattern_emoji} *{pattern_name}*\n"
        f"*Yön:* {'🟩 LONG' if direction=='long' else '🟥 SHORT'}\n"
        f"*Skor:* `{score}/10`\n"
        f"*LSTM Güven:* `{confidence}`\n"
        f"{f'*Başarı Oranı:* `{success}%`\n' if success != '' else ''}"
        f"*Tarih/Saat:* `{signal_time}`\n\n"
        f"*Entry:* `{entry_price}`\n"
        f"*TP:* `{tp}`\n"
        f"*SL:* `{sl}`\n"
        f"*Açıklama:* _{comment}_\n"
        f"\n"
        f"RSI: `{ind_snapshot.get('RSI_14', 'N/A')}` | EMA50: `{ind_snapshot.get('EMA_50', 'N/A')}` | EMA200: `{ind_snapshot.get('EMA_200', 'N/A')}`\n"
        f"MACD: `{ind_snapshot.get('MACD', 'N/A')}` | BB_UP: `{ind_snapshot.get('BB_upper', 'N/A')}` | BB_LOW: `{ind_snapshot.get('BB_lower', 'N/A')}`\n"
        f"\n"
        f"*Teknik Analiz Özeti:*\n{ta_report}\n"
    )
    print(f"[TELEGRAM] Kart gönderiliyor: {symbol} ({interval}) {pattern_name} {direction.upper()}")
    await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown", reply_markup=build_signal_buttons())

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query.data == "show_stats":
        stats = pattern_stats.get_top_patterns(10)
        msg = "*En Başarılı Patternler:*\n"
        for k, v in stats:
            total = v["win"] + v["lose"]
            winrate = v.get("winrate", 0.0) * 100  # zaman ağırlıklı winrate
            msg += f"`{k}`: {v['win']}/{total} (%{winrate:.1f})\n"
        await query.answer()
        # HATA YAKALAMA: Telegram "Message is not modified" hatasını yut
        try:
            await query.edit_message_text(msg, parse_mode="Markdown")
        except BadRequest as e:
            if "Message is not modified" in str(e):
                pass
            else:
                raise e

async def scan_and_notify():
    print("Binance Futures coinleri çekiliyor...")
    exchange_info = client.futures_exchange_info()
    symbols = [
        s['symbol'] for s in exchange_info['symbols']
        if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT'
    ]
    intervals = ["15m", "1h"]
    print(f"{len(symbols)} coin taranacak: {symbols[:5]}...")

    while True:
        for symbol in symbols:
            for interval in intervals:
                try:
                    df = fetch_ohlcv_futures(symbol, interval)
                    pa = PriceActionPatterns(df, timeframe=interval)
                    summary = pa.summary_report()
                    key = f"{symbol}_{interval}_{str(df['timestamp'].iloc[-1])}"

                    # Teknik analiz çalıştır
                    ta_engine = TechnicalAnalysisEngine(df, timeframe=interval)
                    ta_report = ta_engine.summary_report()

                    if "🔥" in summary and (key not in signal_log or signal_log[key] != summary):
                        signal_log[key] = summary
                        first_line = next((l for l in summary.splitlines() if "🔥" in l), None)
                        if first_line:
                            pattern_info = parse_pattern_line(first_line)
                            idx = int(pattern_info.get("index", -1))
                            ind_snapshot = pa.indicator_snapshot(idx)
                            tp, sl = pa.dynamic_tp_sl(idx, entry_type=pattern_info.get("direction", "long"))
                            await send_telegram_signal(symbol, interval, pattern_info, df, ind_snapshot, ta_report, tp, sl)
                        else:
                            msg = f"🔥 {symbol} ({interval})\n{summary[:3400]}\n\n*Teknik Analiz Özeti:*\n{ta_report}"
                            print(f"[SİNYAL] {symbol} ({interval}): {summary.splitlines()[0]}")
                            await bot.send_message(chat_id=CHAT_ID, text=msg)
                except Exception as e:
                    print(f"[{symbol} {interval}] hata: {e}")
                await asyncio.sleep(1.3)

        print("Tüm coinler tarandı, 3 dk uyku.")
        await asyncio.sleep(180)

async def periodic_scan(app):
    # Uygulama ile bağlantılı scan fonksiyonunu başlatır (kapanınca task durur)
    while True:
        await scan_and_notify()
        await asyncio.sleep(5)

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CallbackQueryHandler(handle_callback))

    async def start_scanner(app):  # post_init için application argümanı gerekli!
        asyncio.create_task(periodic_scan(app))

    application.post_init = start_scanner

    print("Otomatik Binance Futures tarama ve sinyal botu başlatılıyor (Telegram handler entegre)!")
    application.run_polling()

if __name__ == "__main__":
    main()