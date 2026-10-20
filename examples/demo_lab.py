"""A demo lab for recording a walkthrough: synthetic markets only, kept apart from your real lab.

    python examples/demo_lab.py                 # creates ~/quant/demolab (once)
    OQL_LAB=~/quant/demolab OQL_CATALOG=~/quant/demolab/catalog oql serve --port 8766

It holds two cards already frozen and waiting for their gate (one on markets with a planted trend edge, one on pure
noise), plus two synthetic hourly markets for the Efficiency tab (planted efficiency regimes, and a random walk).
Run the gates from the Research board on camera: the first should PASS and the second FAIL. Nothing here touches
real data, so nothing needs counting in your real lab's ledger.
"""
import os
import sys

lab = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/quant/demolab")
if os.path.exists(os.path.join(lab, "board.json")):
    sys.exit(f"{lab} already has a board; delete the folder to start over.")
os.makedirs(lab, exist_ok=True)
os.environ["OQL_LAB"], os.environ["OQL_CATALOG"] = lab, os.path.join(lab, "catalog")

from oql import board, catalog, efficiency, synthetic  # noqa: E402


def card(title, mechanism, datasets):
    c = board.new_card(title, mechanism)
    d = board.card_dir(c)
    for name, pairs in (("PREREGISTRATION.md", [("TODO", "Demo card on synthetic data.")]),
                        ("experiment.py", [("datasets=[],", f"datasets={datasets!r},"), ("n_perm=99", "n_perm=99")])):
        path = os.path.join(d, name)
        with open(path) as f:
            text = f.read()
        for old, new in pairs:
            text = text.replace(old, new)
        with open(path, "w") as f:
            f.write(text)
    board.start(c["id"])
    board.freeze(c["id"])
    return c


if __name__ == "__main__":
    trend = [catalog.write(f"synth/trend{i}/1d", synthetic.trend_market(seed=i), license="own", origin="synthetic").id for i in range(4)]
    noise = [catalog.write(f"synth/noise{i}/1d", synthetic.noise_market(seed=50 + i), license="own", origin="synthetic").id for i in range(4)]
    catalog.write("synth/planted-efficiency/1h", efficiency.synthetic_bars(True, n_days=1500, seed=3), license="own", origin="synthetic")
    catalog.write("synth/random-walk/1h", efficiency.synthetic_bars(False, n_days=1500, seed=4), license="own", origin="synthetic")
    card("Trend following on markets with a planted edge", "Regimes persist by construction, so trends continue.", trend)
    card("Trend following on pure noise", "None: these markets are random walks, so the gate must fail.", noise)
    print(f"demo lab ready: {lab}\n"
          f"OQL_LAB={lab} OQL_CATALOG={lab}/catalog oql serve --port 8766")
