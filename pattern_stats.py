import json
import threading
from datetime import datetime, timedelta

PATTERN_STATS_FILE = "pattern_stats.json"
_stats_lock = threading.Lock()

def load_stats():
    try:
        with _stats_lock, open(PATTERN_STATS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_stats(stats):
    with _stats_lock, open(PATTERN_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

def update_pattern_stats(pattern_key, is_win, close_time=None):
    """
    pattern_key: örn 'pinbar_long_15m'
    is_win: bool (True=win, False=lose)
    close_time: isoformat string ya da None (default: now)
    """
    stats = load_stats()
    if pattern_key not in stats:
        stats[pattern_key] = {"results": []}
    record = {
        "result": "win" if is_win else "lose",
        "timestamp": close_time if close_time else datetime.utcnow().isoformat(),
    }
    stats[pattern_key]["results"].append(record)
    # Opsiyonel: sadece son 100 işlemi tut
    stats[pattern_key]["results"] = stats[pattern_key]["results"][-100:]
    save_stats(stats)

def get_pattern_success_rate(pattern_key, decay_days=14):
    """
    Zaman ağırlıklı winrate (son 14 gün daha ağırlıklı).
    """
    stats = load_stats()
    if pattern_key not in stats or not stats[pattern_key]["results"]:
        return 0.0
    now = datetime.utcnow()
    weighted_sum = 0.0
    total_weight = 0.0
    for r in stats[pattern_key]["results"]:
        try:
            t = datetime.fromisoformat(r["timestamp"])
        except Exception:
            continue
        days_ago = (now - t).days
        # Üssel ağırlık: exp(-days_ago / decay_days) ≈ 0.5^(days_ago/decay_days)
        weight = pow(0.5, days_ago / decay_days)
        total_weight += weight
        weighted_sum += weight if r["result"] == "win" else 0
    return weighted_sum / total_weight if total_weight > 0 else 0.0

def get_top_patterns(n=10):
    stats = load_stats()
    ranking = []
    for k in stats:
        winrate = get_pattern_success_rate(k)
        wins = sum(1 for r in stats[k]["results"] if r["result"] == "win")
        loses = sum(1 for r in stats[k]["results"] if r["result"] == "lose")
        ranking.append((k, {"win": wins, "lose": loses, "winrate": winrate}))
    ranking.sort(key=lambda x: (x[1]["win"] + x[1]["lose"], x[1]["winrate"]), reverse=True)
    return ranking[:n]