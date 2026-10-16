"""Live market data for the dashboard: public exchange endpoints (no account, no key), polled by the server.

Each (venue, asset) has one background poller shared by every open browser tab. It keeps the last 12 hours of
1-minute bars and the latest best bid/ask, and turns them into chart events (the same event format as oql.live):
price, the efficiency ratio of the last hour (rolling), realized volatility, and the bid-ask spread.

Efficiency ratio over a window = |net move| / total distance travelled. Near 1 = price went one way; near 0 = back
and forth. It is the quantity the efficiency research tries to forecast; here it is simply measured as it happens.
Nothing here places orders or needs credentials.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np

from .live import _clean

VENUES = {
    "kraken": {"label": "Kraken", "BTC": "XBTUSD", "ETH": "ETHUSD"},
    "coinbase": {"label": "Coinbase", "BTC": "BTC-USD", "ETH": "ETH-USD"},
    "binanceus": {"label": "Binance.US", "BTC": "BTCUSDT", "ETH": "ETHUSDT"},
}
WINDOW = 60                                                  # minutes in the rolling efficiency / volatility window


# ---------------------------------------------------------------------------------------------- parsers (unit-tested)
def parse_kraken_ohlc(payload: dict) -> np.ndarray:
    """Kraken /0/public/OHLC -> rows [t_ms, open, high, low, close, volume] (oldest first)."""
    if payload.get("error"):
        raise ValueError(f"kraken: {payload['error']}")
    rows = next(v for k, v in payload["result"].items() if k != "last")
    return np.array([[r[0] * 1000, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6])] for r in rows], float)


def parse_kraken_ticker(payload: dict) -> tuple[float, float]:
    if payload.get("error"):
        raise ValueError(f"kraken: {payload['error']}")
    t = next(iter(payload["result"].values()))
    return float(t["b"][0]), float(t["a"][0])


def parse_coinbase_candles(payload: list) -> np.ndarray:
    """Coinbase /products/<id>/candles -> rows [t_ms, open, high, low, close, volume]; Coinbase sends newest first
    as [time, low, high, open, close, volume]."""
    rows = sorted(payload, key=lambda r: r[0])
    return np.array([[r[0] * 1000, float(r[3]), float(r[2]), float(r[1]), float(r[4]), float(r[5])] for r in rows], float)


def parse_coinbase_ticker(payload: dict) -> tuple[float, float]:
    return float(payload["bid"]), float(payload["ask"])


def parse_binance_klines(payload: list) -> np.ndarray:
    """Binance.US /api/v3/klines -> rows [t_ms, open, high, low, close, volume]."""
    return np.array([[float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in payload], float)


def parse_binance_book(payload: dict) -> tuple[float, float]:
    return float(payload["bidPrice"]), float(payload["askPrice"])


def urls(venue: str, asset: str) -> tuple[str, str]:
    s = VENUES[venue][asset]
    if venue == "kraken":
        return (f"https://api.kraken.com/0/public/OHLC?pair={s}&interval=1",
                f"https://api.kraken.com/0/public/Ticker?pair={s}")
    if venue == "coinbase":
        return (f"https://api.exchange.coinbase.com/products/{s}/candles?granularity=60",
                f"https://api.exchange.coinbase.com/products/{s}/ticker")
    return (f"https://api.binance.us/api/v3/klines?symbol={s}&interval=1m&limit=720",
            f"https://api.binance.us/api/v3/ticker/bookTicker?symbol={s}")


PARSERS = {"kraken": (parse_kraken_ohlc, parse_kraken_ticker), "coinbase": (parse_coinbase_candles, parse_coinbase_ticker),
           "binanceus": (parse_binance_klines, parse_binance_book)}


# ---------------------------------------------------------------------------------------------- measures (unit-tested)
def rolling_efficiency(close: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """|c_t - c_{t-w}| / sum of |one-bar moves| over the same w bars; NaN until w bars exist or if price never moved."""
    c = np.asarray(close, float)
    out = np.full(len(c), np.nan)
    if len(c) <= window:
        return out
    step = np.abs(np.diff(c))
    path = np.convolve(step, np.ones(window), "valid")              # path[i] = sum of steps i .. i+w-1
    net = np.abs(c[window:] - c[:-window])
    with np.errstate(invalid="ignore", divide="ignore"):
        out[window:] = np.where(path > 0, net / path, np.nan)
    return out


def rolling_vol(close: np.ndarray, window: int = WINDOW, per_year: float = 525_600) -> np.ndarray:
    """Annualized realized volatility of 1-minute log returns over the last `window` minutes."""
    r = np.diff(np.log(np.asarray(close, float)))
    out = np.full(len(close), np.nan)
    for i in range(window, len(close)):
        out[i] = r[i - window:i].std(ddof=1) * np.sqrt(per_year)
    return out


def events_from(bars: np.ndarray, bid: float | None, ask: float | None, venue: str, asset: str, spreads: list) -> list[dict]:
    """Turn the latest bars and quote into dashboard events."""
    t = [int(x) for x in bars[:, 0]]
    c = bars[:, 4]
    er, vol = rolling_efficiency(c), rolling_vol(c)
    name = f"{asset} on {VENUES[venue]['label']}"
    ev = [
        {"stat": "price", "label": f"{asset} last", "value": float(c[-1]), "fmt": ",.2f"},
        {"stat": "er", "label": f"Efficiency, last {WINDOW} min", "value": None if np.isnan(er[-1]) else float(er[-1]), "fmt": ".2f"},
        {"stat": "vol", "label": "Realized vol (annualized)", "value": None if np.isnan(vol[-1]) else float(vol[-1]), "fmt": ".0%"},
        {"chart": "price", "kind": "line", "x": "time", "title": f"{name}: 1-minute closes, last 12 hours",
         "series": [{"name": asset, "x": t, "y": c.tolist()}]},
        {"chart": "er", "kind": "line", "x": "time", "y_range": [0, 1],
         "title": f"Efficiency ratio over the last {WINDOW} minutes (1 = straight line, 0 = back and forth)",
         "series": [{"name": "efficiency", "x": t, "y": er.tolist()}],
         "hlines": [{"y": float(np.nanmedian(er)), "label": "12-hour median"}] if np.isfinite(er).any() else []},
        {"chart": "vol", "kind": "line", "x": "time", "title": f"Realized volatility, rolling {WINDOW} minutes (annualized)",
         "y_format": ".0%", "series": [{"name": "volatility", "x": t, "y": vol.tolist()}]},
    ]
    if bid and ask:
        mid = (bid + ask) / 2
        ev.append({"stat": "spread", "label": "Bid-ask spread", "value": (ask - bid) / mid * 1e4, "fmt": ".2f", "unit": " bp"})
        if spreads:
            ev.append({"chart": "spread", "kind": "line", "x": "time", "title": "Bid-ask spread since you opened this view (bp)",
                       "series": [{"name": "spread", "x": [s[0] for s in spreads], "y": [s[1] for s in spreads]}]})
    return ev


# ---------------------------------------------------------------------------------------------- shared pollers
class Feed:
    def __init__(self, venue: str, asset: str, getter=None, every: float = 5.0):
        if venue not in VENUES or asset not in ("BTC", "ETH"):
            raise ValueError(f"unknown market {venue}/{asset}")
        from .sources import fetch
        self.venue, self.asset, self.every = venue, asset, every
        self.get = getter or (lambda u: json.loads(fetch(u, timeout=10)))
        self.events: list[dict] = []
        self.version = 0
        self.last_seen = time.time()
        self.spreads: list = []
        self._cond = threading.Condition()
        self._bars, self._bars_at = None, 0.0
        threading.Thread(target=self._loop, daemon=True, name=f"feed-{venue}-{asset}").start()

    def poll_once(self):
        ohlc_url, quote_url = urls(self.venue, self.asset)
        p_ohlc, p_quote = PARSERS[self.venue]
        if self._bars is None or time.time() - self._bars_at > 20:      # candles every 20 s, quotes every poll
            self._bars, self._bars_at = p_ohlc(self.get(ohlc_url))[-720:], time.time()
        bid, ask = p_quote(self.get(quote_url))
        self.spreads.append((int(time.time() * 1000), (ask - bid) / ((ask + bid) / 2) * 1e4))
        self.spreads = self.spreads[-720:]
        return events_from(self._bars, bid, ask, self.venue, self.asset, self.spreads)

    def _loop(self):
        backoff = self.every
        while time.time() - self.last_seen < 90:                        # stop polling 90 s after the last viewer leaves
            try:
                ev = self.poll_once() + [{"log": f"updated {time.strftime('%H:%M:%S')}"}]
                backoff = self.every
            except Exception as e:                                       # network trouble: say so, retry more slowly
                ev = [{"log": f"{VENUES[self.venue]['label']} unreachable ({type(e).__name__}: {e}); retrying"}]
                backoff = min(backoff * 2, 60)
            with self._cond:
                self.events, self.version = ev, self.version + 1
                self._cond.notify_all()
            time.sleep(backoff)
        with _lock:
            _feeds.pop((self.venue, self.asset), None)

    def wait(self, seen: int, timeout: float = 15.0):
        self.last_seen = time.time()
        with self._cond:
            if self.version <= seen:
                self._cond.wait(timeout)
            return self.version, list(self.events)


_feeds: dict = {}
_lock = threading.Lock()


def feed(venue: str, asset: str) -> Feed:
    with _lock:
        key = (venue, asset)
        if key not in _feeds:
            _feeds[key] = Feed(venue, asset)
        _feeds[key].last_seen = time.time()
        return _feeds[key]


def sse(write, venue: str, asset: str) -> None:
    f = feed(venue, asset)
    seen = 0
    while True:
        version, events = f.wait(seen)
        if version == seen:
            write(": keep-alive\n\n")
            continue
        seen = version
        for ev in events:
            write(f"data: {json.dumps(_clean(ev))}\n\n")
