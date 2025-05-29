import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import talib
from tensorflow.keras.models import load_model
import pattern_stats

class PriceActionPatterns:
    _lstm_model = None

    def __init__(self, df: pd.DataFrame, timeframe='1h', lstm_model=None, higher_df=None):
        self.df = df.copy()
        self.timeframe = timeframe
        self.df['ATR_14'] = self._atr()
        self.add_indicators()
        self.higher_df = higher_df

        if lstm_model is not None:
            self.lstm_model = lstm_model
        elif PriceActionPatterns._lstm_model is not None:
            self.lstm_model = PriceActionPatterns._lstm_model
        else:
            try:
                self.lstm_model = load_model("lstm_model.h5")
            except Exception as e:
                print("LSTM model yüklenemedi:", e)
                self.lstm_model = None
            PriceActionPatterns._lstm_model = self.lstm_model

    def add_indicators(self):
        try:
            self.df['RSI_14'] = talib.RSI(self.df['close'], timeperiod=14)
            self.df['EMA_50'] = talib.EMA(self.df['close'], timeperiod=50)
            self.df['EMA_200'] = talib.EMA(self.df['close'], timeperiod=200)
            macd, macdsig, macdhist = talib.MACD(self.df['close'])
            self.df['MACD'] = macd
            self.df['MACD_signal'] = macdsig
            upper, middle, lower = talib.BBANDS(self.df['close'])
            self.df['BB_upper'], self.df['BB_middle'], self.df['BB_lower'] = upper, middle, lower
            self.df['VWAP'] = self._vwap()
            self.df['Supertrend'] = self._supertrend()
            self.df['Pivot'], self.df['Pivot_S1'], self.df['Pivot_R1'] = self._pivot_points()
            self.df['HA_Close'] = self._heikin_ashi('close')
            self.df['HA_Open'] = self._heikin_ashi('open')
        except Exception as e:
            print("TA-Lib indikatörleri hesaplanamadı:", e)

    def _atr(self, period=14):
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _vwap(self):
        pv = (self.df['close'] * self.df['volume']).cumsum()
        vol = self.df['volume'].cumsum()
        return pv/vol

    def _supertrend(self, period=10, multiplier=3):
        hl2 = (self.df['high'] + self.df['low']) / 2
        atr = self.df['ATR_14']
        final_upperband = hl2 + (multiplier * atr)
        final_lowerband = hl2 - (multiplier * atr)
        supertrend = np.zeros(len(self.df))
        in_uptrend = True
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > final_upperband.iloc[i-1]:
                in_uptrend = True
            elif self.df['close'].iloc[i] < final_lowerband.iloc[i-1]:
                in_uptrend = False
            supertrend[i] = 1 if in_uptrend else -1
        return supertrend

    def _pivot_points(self):
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        pivot = (high + low + close) / 3
        s1 = (2 * pivot) - high
        r1 = (2 * pivot) - low
        return pivot, s1, r1

    def _heikin_ashi(self, typ='close'):
        o = self.df['open']
        h = self.df['high']
        l = self.df['low']
        c = self.df['close']
        ha_close = (o + h + l + c) / 4
        ha_open = np.zeros(len(self.df))
        ha_open[0] = (o.iloc[0] + c.iloc[0]) / 2
        for i in range(1, len(self.df)):
            ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
        return ha_close if typ == 'close' else ha_open

    ############## Tüm Pattern Fonksiyonları (Klasik, Modern, Harmonik) ##############

    def find_peaks(self, arr, distance=5):
        return list(argrelextrema(arr, np.greater, order=distance)[0])

    def find_troughs(self, arr, distance=5):
        return list(argrelextrema(arr, np.less, order=distance)[0])

    def detect_double_top(self, lookback=50, threshold=0.02):
        closes = self.df['close'].values
        peaks = self.find_peaks(closes, distance=lookback//4)
        double_tops = []
        for i in range(len(peaks)-1):
            if abs(closes[peaks[i]] - closes[peaks[i+1]])/closes[peaks[i]] < threshold:
                valley = np.min(closes[peaks[i]:peaks[i+1]])
                if valley < closes[peaks[i]] * (1 - threshold):
                    if self.df['RSI_14'].iloc[peaks[i+1]] > 60 and self.df['volume'].iloc[peaks[i+1]] > self.df['volume'].mean():
                        double_tops.append({'type':'double_top', 'indexes': (peaks[i], peaks[i+1]), 'rsi': self.df['RSI_14'].iloc[peaks[i+1]]})
        return double_tops

    def detect_double_bottom(self, lookback=50, threshold=0.02):
        closes = self.df['close'].values
        troughs = self.find_troughs(closes, distance=lookback//4)
        double_bottoms = []
        for i in range(len(troughs)-1):
            if abs(closes[troughs[i]] - closes[troughs[i+1]])/closes[troughs[i]] < threshold:
                peak = np.max(closes[troughs[i]:troughs[i+1]])
                if peak > closes[troughs[i]] * (1 + threshold):
                    if self.df['RSI_14'].iloc[troughs[i+1]] < 40 and self.df['volume'].iloc[troughs[i+1]] > self.df['volume'].mean():
                        double_bottoms.append({'type':'double_bottom', 'indexes': (troughs[i], troughs[i+1]), 'rsi': self.df['RSI_14'].iloc[troughs[i+1]]})
        return double_bottoms

    def detect_head_and_shoulders(self, lookback=60, threshold=0.02):
        closes = self.df['close'].values
        peaks = self.find_peaks(closes, distance=lookback//6)
        patterns = []
        for i in range(len(peaks)-2):
            l, h, r = peaks[i], peaks[i+1], peaks[i+2]
            if closes[l] < closes[h] and closes[r] < closes[h] and abs(closes[l]-closes[r])/closes[h] < threshold:
                macd = self.df['MACD'].iloc[h]
                if macd < 0 and self.df['volume'].iloc[h] > self.df['volume'].mean():
                    patterns.append({'type':'head_and_shoulders', 'indexes': (l, h, r), 'macd': macd})
        return patterns

    def detect_inverse_head_and_shoulders(self, lookback=60, threshold=0.02):
        closes = self.df['close'].values
        troughs = self.find_troughs(closes, distance=lookback//6)
        patterns = []
        for i in range(len(troughs)-2):
            l, h, r = troughs[i], troughs[i+1], troughs[i+2]
            if closes[l] > closes[h] and closes[r] > closes[h] and abs(closes[l]-closes[r])/closes[h] < threshold:
                macd = self.df['MACD'].iloc[h]
                if macd > 0 and self.df['volume'].iloc[h] > self.df['volume'].mean():
                    patterns.append({'type':'inverse_head_and_shoulders', 'indexes': (l, h, r), 'macd': macd})
        return patterns

    def detect_triple_top(self, lookback=60, threshold=0.02):
        closes = self.df['close'].values
        peaks = self.find_peaks(closes, distance=lookback//5)
        ttops = []
        for i in range(len(peaks)-2):
            a, b, c = peaks[i], peaks[i+1], peaks[i+2]
            if abs(closes[a]-closes[b])/closes[a] < threshold and abs(closes[b]-closes[c])/closes[b] < threshold:
                ttops.append({'type': 'triple_top', 'indexes': (a, b, c)})
        return ttops

    def detect_triple_bottom(self, lookback=60, threshold=0.02):
        closes = self.df['close'].values
        troughs = self.find_troughs(closes, distance=lookback//5)
        tbots = []
        for i in range(len(troughs)-2):
            a, b, c = troughs[i], troughs[i+1], troughs[i+2]
            if abs(closes[a]-closes[b])/closes[a] < threshold and abs(closes[b]-closes[c])/closes[b] < threshold:
                tbots.append({'type': 'triple_bottom', 'indexes': (a, b, c)})
        return tbots

    def detect_rising_wedge(self, lookback=40):
        highs = self.df['high'].values[-lookback:]
        lows = self.df['low'].values[-lookback:]
        x = np.arange(len(highs))
        high_coef = np.polyfit(x, highs, 1)
        low_coef = np.polyfit(x, lows, 1)
        if high_coef[0] > 0 and low_coef[0] > 0 and high_coef[0] < low_coef[0]:
            return {'type': 'rising_wedge', 'high_coef': high_coef, 'low_coef': low_coef}
        return None

    def detect_falling_wedge(self, lookback=40):
        highs = self.df['high'].values[-lookback:]
        lows = self.df['low'].values[-lookback:]
        x = np.arange(len(highs))
        high_coef = np.polyfit(x, highs, 1)
        low_coef = np.polyfit(x, lows, 1)
        if high_coef[0] < 0 and low_coef[0] < 0 and high_coef[0] > low_coef[0]:
            return {'type': 'falling_wedge', 'high_coef': high_coef, 'low_coef': low_coef}
        return None

    def detect_rectangle(self, lookback=40, threshold=0.01):
        highs = self.df['high'].values[-lookback:]
        lows = self.df['low'].values[-lookback:]
        if np.std(highs) < threshold * np.mean(highs) and np.std(lows) < threshold * np.mean(lows):
            return {'type': 'rectangle', 'top': np.mean(highs), 'bottom': np.mean(lows)}
        return None

    def detect_broadening(self, lookback=60):
        highs = self.df['high'].values[-lookback:]
        lows = self.df['low'].values[-lookback:]
        x = np.arange(len(highs))
        high_coef = np.polyfit(x, highs, 1)
        low_coef = np.polyfit(x, lows, 1)
        if high_coef[0] > 0 and low_coef[0] < 0:
            return {'type': 'broadening', 'high_coef': high_coef, 'low_coef': low_coef}
        return None

    def detect_cup_handle(self, lookback=80):
        closes = self.df['close'].values[-lookback:]
        min_idx = np.argmin(closes)
        left = closes[:min_idx]
        right = closes[min_idx+1:]
        if len(left) < 5 or len(right) < 5:
            return None
        if np.std(left) < 0.03 * np.mean(left) and np.std(right) < 0.03 * np.mean(right):
            if closes[-1] > closes[0]:
                return {'type': 'cup_and_handle', 'bottom': closes[min_idx]}
        return None

    def detect_inverse_cup_handle(self, lookback=80):
        closes = self.df['close'].values[-lookback:]
        max_idx = np.argmax(closes)
        left = closes[:max_idx]
        right = closes[max_idx+1:]
        if len(left) < 5 or len(right) < 5:
            return None
        if np.std(left) < 0.03 * np.mean(left) and np.std(right) < 0.03 * np.mean(right):
            if closes[-1] < closes[0]:
                return {'type': 'inverse_cup_and_handle', 'top': closes[max_idx]}
        return None

    def detect_flag(self, lookback=30):
        closes = self.df['close'].values[-lookback:]
        sharp_move = closes[-1] > closes[0] * 1.05 or closes[-1] < closes[0] * 0.95
        consolidation = np.std(closes[-10:]) < np.std(closes[-20:-10])
        if sharp_move and consolidation:
            return {'type': 'flag', 'move': closes[-1] - closes[0]}
        return None

    def detect_triangle(self, lookback=40, min_points=5):
        closes = self.df['close'].values[-lookback:]
        highs = self.df['high'].values[-lookback:]
        lows = self.df['low'].values[-lookback:]
        x = np.arange(len(closes))
        high_coef = np.polyfit(x, highs, 1)
        low_coef = np.polyfit(x, lows, 1)
        if (high_coef[0] < 0 and low_coef[0] > 0) or (abs(high_coef[0]) < 0.05 and low_coef[0] > 0) or (high_coef[0] < 0 and abs(low_coef[0]) < 0.05):
            return {'type':'triangle', 'high_coef': high_coef, 'low_coef': low_coef}
        return None

    # --- Harmonik Patternler ---
    def detect_abcd(self, order=3):
        closes = self.df['close'].values
        idx = argrelextrema(closes, np.greater, order=order)[0]
        result = []
        if len(idx) > 3:
            ab = closes[idx[-3]] - closes[idx[-4]]
            cd = closes[idx[-1]] - closes[idx[-2]]
            if 0.95 < abs(cd/ab) < 1.05:
                result.append({'type': 'abcd', 'points': idx[-4:]})
        return result

    def detect_shark(self, order=3):
        closes = self.df['close'].values
        idx = argrelextrema(closes, np.greater, order=order)[0]
        result = []
        if len(idx) > 4:
            xa = closes[idx[-4]] - closes[idx[-5]]
            ab = closes[idx[-3]] - closes[idx[-4]]
            bc = closes[idx[-2]] - closes[idx[-3]]
            cd = closes[idx[-1]] - closes[idx[-2]]
            if 1.33 < abs(bc/ab) < 2.24 and 1.618 < abs(cd/bc) < 2.24:
                result.append({'type': 'shark', 'points': idx[-5:]})
        return result

    def detect_deep_crab(self, order=3):
        closes = self.df['close'].values
        idx = argrelextrema(closes, np.greater, order=order)[0]
        result = []
        if len(idx) > 4:
            xa = closes[idx[-4]] - closes[idx[-5]]
            ab = closes[idx[-3]] - closes[idx[-4]]
            bc = closes[idx[-2]] - closes[idx[-3]]
            cd = closes[idx[-1]] - closes[idx[-2]]
            if 0.886 < abs(ab/xa) < 1.0 and 2.618 < abs(cd/xa) < 3.618:
                result.append({'type': 'deep_crab', 'points': idx[-5:]})
        return result

    def detect_alt_bat(self, order=3):
        closes = self.df['close'].values
        idx = argrelextrema(closes, np.greater, order=order)[0]
        result = []
        if len(idx) > 4:
            xa = closes[idx[-4]] - closes[idx[-5]]
            ab = closes[idx[-3]] - closes[idx[-4]]
            bc = closes[idx[-2]] - closes[idx[-3]]
            cd = closes[idx[-1]] - closes[idx[-2]]
            if 0.382 < abs(ab/xa) < 0.5 and 1.618 < abs(cd/bc) < 2.618:
                result.append({'type': 'alt_bat', 'points': idx[-5:]})
        return result

    def detect_50(self, order=3):
        closes = self.df['close'].values
        idx = argrelextrema(closes, np.greater, order=order)[0]
        result = []
        if len(idx) > 4:
            xa = closes[idx[-4]] - closes[idx[-5]]
            ab = closes[idx[-3]] - closes[idx[-4]]
            bc = closes[idx[-2]] - closes[idx[-3]]
            cd = closes[idx[-1]] - closes[idx[-2]]
            if 1.618 < abs(cd/bc) < 2.0 and 1.13 < abs(bc/ab) < 1.618:
                result.append({'type': 'five_zero', 'points': idx[-5:]})
        return result

    def detect_wolfe_wave(self, order=3):
        closes = self.df['close'].values
        idx_max = argrelextrema(closes, np.greater, order=order)[0]
        idx_min = argrelextrema(closes, np.less, order=order)[0]
        result = []
        if len(idx_max) >= 3 and len(idx_min) >= 3:
            result.append({'type': 'wolfe_wave', 'peaks': idx_max[-3:], 'troughs': idx_min[-3:]})
        return result

    def detect_gartley(self, order=3):
        closes = self.df['close'].values
        idx = argrelextrema(closes, np.greater, order=order)[0]
        result = []
        if len(idx) > 4:
            xa = closes[idx[-4]] - closes[idx[-5]]
            ab = closes[idx[-3]] - closes[idx[-4]]
            bc = closes[idx[-2]] - closes[idx[-3]]
            cd = closes[idx[-1]] - closes[idx[-2]]
            if 0.618 < abs(ab/xa) < 0.786 and 0.382 < abs(bc/ab) < 0.886:
                result.append({'type': 'gartley', 'points': idx[-5:]})
        return result

    def detect_butterfly(self, order=3):
        closes = self.df['close'].values
        idx = argrelextrema(closes, np.less, order=order)[0]
        result = []
        if len(idx) > 4:
            xa = closes[idx[-4]] - closes[idx[-5]]
            ab = closes[idx[-3]] - closes[idx[-4]]
            bc = closes[idx[-2]] - closes[idx[-3]]
            cd = closes[idx[-1]] - closes[idx[-2]]
            if 0.786 < abs(ab/xa) < 1.27 and 1.618 < abs(cd/bc) < 2.618:
                result.append({'type': 'butterfly', 'points': idx[-5:]})
        return result

    def detect_crab(self, order=3):
        closes = self.df['close'].values
        idx = argrelextrema(closes, np.greater, order=order)[0]
        result = []
        if len(idx) > 4:
            xa = closes[idx[-4]] - closes[idx[-5]]
            ab = closes[idx[-3]] - closes[idx[-4]]
            bc = closes[idx[-2]] - closes[idx[-3]]
            cd = closes[idx[-1]] - closes[idx[-2]]
            if 0.382 < abs(ab/xa) < 0.618 and 0.382 < abs(bc/ab) < 0.886 and 2.618 < abs(cd/xa) < 3.618:
                result.append({'type': 'crab', 'points': idx[-5:]})
        return result

    def detect_bat(self, order=3):
        closes = self.df['close'].values
        idx = argrelextrema(closes, np.greater, order=order)[0]
        result = []
        if len(idx) > 4:
            xa = closes[idx[-4]] - closes[idx[-5]]
            ab = closes[idx[-3]] - closes[idx[-4]]
            bc = closes[idx[-2]] - closes[idx[-3]]
            cd = closes[idx[-1]] - closes[idx[-2]]
            if 0.382 < abs(ab/xa) < 0.5 and 0.382 < abs(bc/ab) < 0.886 and 1.618 < abs(cd/xa) < 2.618:
                result.append({'type': 'bat', 'points': idx[-5:]})
        return result

    def detect_cypher(self, order=3):
        closes = self.df['close'].values
        idx = argrelextrema(closes, np.greater, order=order)[0]
        result = []
        if len(idx) > 4:
            xa = closes[idx[-4]] - closes[idx[-5]]
            ab = closes[idx[-3]] - closes[idx[-4]]
            bc = closes[idx[-2]] - closes[idx[-3]]
            cd = closes[idx[-1]] - closes[idx[-2]]
            if 0.382 < abs(ab/xa) < 0.618 and 1.13 < abs(bc/ab) < 1.414 and 0.618 < abs(cd/xa) < 0.786:
                result.append({'type': 'cypher', 'points': idx[-5:]})
        return result

    ####################### YARDIMCI & RAPORLAMA #######################

    def get_all_patterns(self):
        patterns = []
        patterns.extend(self.detect_double_top())
        patterns.extend(self.detect_double_bottom())
        patterns.extend(self.detect_head_and_shoulders())
        patterns.extend(self.detect_inverse_head_and_shoulders())
        patterns.extend(self.detect_triple_top())
        patterns.extend(self.detect_triple_bottom())
        wedge_r = self.detect_rising_wedge()
        wedge_f = self.detect_falling_wedge()
        if wedge_r: patterns.append(wedge_r)
        if wedge_f: patterns.append(wedge_f)
        rect = self.detect_rectangle()
        if rect: patterns.append(rect)
        broad = self.detect_broadening()
        if broad: patterns.append(broad)
        cup = self.detect_cup_handle()
        inv_cup = self.detect_inverse_cup_handle()
        if cup: patterns.append(cup)
        if inv_cup: patterns.append(inv_cup)
        flag = self.detect_flag()
        if flag: patterns.append(flag)
        triangle = self.detect_triangle()
        if triangle: patterns.append(triangle)
        patterns.extend(self.detect_wolfe_wave())
        patterns.extend(self.detect_gartley())
        patterns.extend(self.detect_butterfly())
        patterns.extend(self.detect_crab())
        patterns.extend(self.detect_bat())
        patterns.extend(self.detect_cypher())
        patterns.extend(self.detect_abcd())
        patterns.extend(self.detect_shark())
        patterns.extend(self.detect_deep_crab())
        patterns.extend(self.detect_alt_bat())
        patterns.extend(self.detect_50())
        return patterns

    def summary_report(self):
        report = []
        for pattern in self.get_all_patterns():
            report.append(f"{pattern['type'].upper()} detected: {pattern}")
        if not report:
            return "Şu anda trade edilebilir bir formasyon yok."
        return "\n".join(report)