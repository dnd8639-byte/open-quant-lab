"""Check that the live demo page computes the same numbers as this Python lab.

    python demo/check_parity.py                     # the published page
    python demo/check_parity.py path/to/index.html  # a local copy

The page (https://dnd8639-byte.github.io/open-quant-lab/) runs a JavaScript port of the engine, rules and search.
This script takes the port and the market data out of the page, runs every strategy in Node.js, runs the same
strategies here in Python, and compares all the Sharpe ratios: development data and holdout, at two cost levels.
Needs Node.js (https://nodejs.org). Exits with status 1 if any number differs by more than 1e-9.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from oql.engine import prepare            # noqa: E402
from oql.experiment import Experiment     # noqa: E402
from oql.search import Scorer, sharpe     # noqa: E402
from oql.sources import _ssl_context      # noqa: E402
from oql.strategies import RULES, valid   # noqa: E402

URL = "https://dnd8639-byte.github.io/open-quant-lab/"
MTPY = Experiment(datasets=[]).min_trades_per_year
DRIVER = r"""
const OQL = require(process.argv[1]), M = JSON.parse(require("fs").readFileSync(process.argv[2]));
const genomes = JSON.parse(process.argv[3]), out = {};  // node -e: arguments start at argv[1]
for (const k of Object.keys(M)) for (const cost of [1, 10]) {
  const G = OQL.gate(M[k], {costBps: cost});
  out[k + "@" + cost] = {ppy: G.ppy, best: G.real,
    dev: genomes.map(g => G.scorer.score(g)),
    holdout: genomes.map(g => OQL.holdout(M[k], g, {costBps: cost}).real)};
}
console.log(JSON.stringify(out));
"""


def page_parts(html: str):
    markets = re.search(r'<script id="markets" type="application/json">(.*?)</script>', html, re.S).group(1)
    lab = re.search(r"<script>\s*(/\* Open Quant Lab in the browser.*?)</script>", html, re.S).group(1)
    return json.loads(markets), lab


def main(src: str = URL) -> int:
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=30, context=_ssl_context()) as r:
            html = r.read().decode()
    else:
        with open(src) as f:
            html = f.read()
    markets, lab = page_parts(html)
    genomes = [dict(rule=r, params=dict(zip(grid, vals)), side="both")
               for r, (_, grid) in RULES.items()
               for vals in __import__("itertools").product(*grid.values()) if valid(r, dict(zip(grid, vals)))]
    with tempfile.TemporaryDirectory() as tmp:
        lab_js, data = os.path.join(tmp, "lab.js"), os.path.join(tmp, "markets.json")
        with open(lab_js, "w") as f:
            f.write(lab)
        with open(data, "w") as f:
            json.dump(markets, f)
        run = subprocess.run(["node", "-e", DRIVER, lab_js, data, json.dumps(genomes)], capture_output=True, text=True)
        if run.returncode:
            sys.exit(f"node failed:\n{run.stderr}")
        js = json.loads(run.stdout)
    worst, count = 0.0, 0
    for k, m in markets.items():
        idx = pd.to_datetime(np.array(m["day"]) * 86400, unit="s", utc=True)
        df = pd.DataFrame({"open": m["o"], "high": m["h"], "low": m["l"], "close": m["c"], "volume": 0.0}, index=idx)
        split = int(len(df) * 0.7)
        for cost in (1, 10):
            j = js[f"{k}@{cost}"]
            dev = Scorer(prepare({k: df.iloc[:split]}, {"default": float(cost)}, warmup=250), RULES, MTPY)
            hold = Scorer(prepare({k: df}, {"default": float(cost)}, warmup=250, score_from=df.index[split]), RULES, MTPY)
            for g, a, b in zip(genomes, j["dev"], j["holdout"]):
                py_dev = dev.score(g, delay=1 / 23)
                py_hold = sharpe(hold.returns(g, delay=1 / 23)[0].to_numpy(), hold.prep.ppy)
                worst = max(worst, abs(py_dev - a), abs(py_hold - b))
                count += 2
            print(f"{m['name']:22s} cost {cost:2d} bp: best development Sharpe  Python {max(dev.score(g, delay=1 / 23) for g in genomes):.6f}"
                  f"   page {j['best']:.6f}")
    print(f"{count} Sharpe ratios compared; largest difference {worst:.1e}")
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
