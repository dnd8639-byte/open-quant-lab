"""Quickstart: the whole pipeline on fake data, in a throwaway lab folder. About a minute.

    python examples/quickstart.py

Two cards go through the same gate: one on markets with a planted, real edge (persistent trends) and one on
pure noise. A trustworthy lab must pass the first and archive the second. Then it opens board.html.
"""
import os
import tempfile
import webbrowser

lab = tempfile.mkdtemp(prefix="oql-quickstart-")
os.environ["OQL_LAB"], os.environ["OQL_CATALOG"] = lab, os.path.join(lab, "catalog")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from oql import board, catalog, synthetic  # noqa: E402


def make_card(title, mechanism, datasets):
    c = board.new_card(title, mechanism)
    d = board.card_dir(c)
    for name, pairs in (("PREREGISTRATION.md", [("TODO", "Filled in for the quickstart.")]),
                        ("experiment.py", [("datasets=[],", f"datasets={datasets!r},"), ("n_perm=99", "n_perm=39")])):
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
    a = make_card("Trend following, planted edge", "Regimes persist by construction, so trends continue.", trend)
    b = make_card("Trend following, pure noise", "None: these markets are random walks.", noise)
    for c in (a, b):
        r = board.run_gate(c["id"])
        print(f"{c['title']}: gate {r['verdict']} (p = {r['p']:.3f}, dev Sharpe {r['dev_sharpe']:.2f})\n")
    r = board.run_holdout(a["id"])
    print(f"{a['title']}: holdout {r['verdict']} (Sharpe {r['holdout_sharpe']:.2f}, p = {r['p']:.3f})\n")
    print(board.render_text(board.load()))
    out = os.path.join(lab, "board.html")
    with open(out, "w") as f:
        f.write(board.render_html(board.load()))
    print(f"\nlab folder: {lab}\nboard:      {out}")
    webbrowser.open("file://" + out)
