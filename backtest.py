import pandas as pd
import numpy as np

class Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        strategy_func,
        tp_sl_func,
        initial_balance=1000,
        fee_rate=0.0005,           # Komisyon oranı (örn: %0.05)
        slippage=0.0,              # Slipaj (örn: 0.5 USDT)
        risk_pct=1.0,              # İşlem başı risk yüzdesi
        plot_equity_curve=False,   # Equity curve çizilsin mi
        verbose=False              # Detaylı loglama
    ):
        """
        df: OHLCV DataFrame
        strategy_func: (df, idx) -> "buy" or "sell" or None
        tp_sl_func: (idx, entry_type) -> (tp, sl)
        fee_rate: Alım/satım başına toplam komisyon oranı (ör: 0.001 = %0.1)
        slippage: Emir gerçekleşme fiyatı ile beklenen fiyat arasındaki sapma (mutlak, ör: 0.5)
        risk_pct: İşlem başına toplam bakiyenin yüzde kaçı risk edilecek (%1 için 1.0)
        """
        self.df = df.copy()
        self.strategy_func = strategy_func
        self.tp_sl_func = tp_sl_func
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.risk_pct = risk_pct
        self.plot_equity_curve = plot_equity_curve
        self.verbose = verbose

        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.active_trade = None
        self.trades = []
        self.max_drawdown = 0
        self.peak_balance = initial_balance

    def run(self):
        for idx in range(len(self.df)):
            price = self.df['close'].iloc[idx]

            # Aktif trade yoksa ve sinyal varsa işleme gir
            if self.active_trade is None:
                signal = self.strategy_func(self.df, idx)
                if signal in ["buy", "sell"]:
                    entry_type = "long" if signal == "buy" else "short"
                    tp, sl = self.tp_sl_func(idx, entry_type)
                    # Dinamik pozisyon büyüklüğü (risk-parasal olarak)
                    if entry_type == "long":
                        risk_per_unit = abs(self.df['close'].iloc[idx] - sl)
                    else:
                        risk_per_unit = abs(sl - self.df['close'].iloc[idx])
                    if risk_per_unit == 0:
                        continue
                    max_risk_amt = self.balance * self.risk_pct / 100
                    qty = max_risk_amt / risk_per_unit
                    qty = max(qty, 0)
                    if qty == 0:
                        continue

                    entry_price = price + self.slippage if entry_type == "long" else price - self.slippage
                    fee = entry_price * qty * self.fee_rate
                    if self.verbose:
                        print(f"Giriş: {signal} idx={idx} Fiyat={entry_price:.2f} Miktar={qty:.4f} Fee={fee:.2f}")
                    self.active_trade = dict(
                        entry_idx=idx,
                        entry_price=entry_price,
                        type=entry_type,
                        tp=tp,
                        sl=sl,
                        qty=qty,
                        fee=fee
                    )
            # Aktif trade varsa çıkış kontrolü
            elif self.active_trade is not None:
                t = self.active_trade
                closed = False
                result = ""
                # Çıkış fiyatı slipajı uygula
                close_price = price - self.slippage if t["type"] == "long" else price + self.slippage
                # TP/SL kontrol
                if t["type"] == "long":
                    if close_price >= t["tp"]:
                        closed = True
                        result = "tp"
                        exit_price = t["tp"]
                    elif close_price <= t["sl"]:
                        closed = True
                        result = "sl"
                        exit_price = t["sl"]
                else:
                    if close_price <= t["tp"]:
                        closed = True
                        result = "tp"
                        exit_price = t["tp"]
                    elif close_price >= t["sl"]:
                        closed = True
                        result = "sl"
                        exit_price = t["sl"]
                # Pozisyon kapanışı
                if closed:
                    fee_close = exit_price * t["qty"] * self.fee_rate
                    if t["type"] == "long":
                        pnl = (exit_price - t["entry_price"]) * t["qty"] - t["fee"] - fee_close
                    else:
                        pnl = (t["entry_price"] - exit_price) * t["qty"] - t["fee"] - fee_close
                    self.trades.append(dict(
                        **t,
                        close_idx=idx,
                        close_price=exit_price,
                        result=result,
                        pnl=pnl,
                        fee_close=fee_close
                    ))
                    self.balance += pnl
                    self.active_trade = None
                    if self.verbose:
                        print(f"Çıkış: {result} idx={idx} Fiyat={exit_price:.2f} PnL={pnl:.2f} Yeni Bakiye={self.balance:.2f}")
            # Equity curve güncelle
            self.equity_curve.append(self.balance)
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

    def summary(self):
        results = pd.DataFrame(self.trades)
        returns = results['pnl'].sum() if not results.empty else 0
        win_trades = results[results['result'] == 'tp'] if not results.empty else []
        loss_trades = results[results['result'] == 'sl'] if not results.empty else []
        win_rate = len(win_trades) / len(results) if len(results) > 0 else 0
        avg_risk_reward = win_trades['pnl'].mean() / abs(loss_trades['pnl'].mean()) if len(win_trades) > 0 and len(loss_trades) > 0 else None

        summary = {
            "final_balance": self.balance,
            "total_trades": len(results),
            "win_rate": win_rate,
            "total_returns": returns,
            "max_drawdown": self.max_drawdown,
            "avg_risk_reward": avg_risk_reward,
            "results": results,
            "equity_curve": self.equity_curve
        }
        return summary

    def plot_equity(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.plot(self.equity_curve)
        plt.title("Equity Curve")
        plt.xlabel("Bar")
        plt.ylabel("Balance")
        plt.show()