# Running the lab on the Raspberry Pi

The Pi collects data (daily) and paper-trades deployed cards (hourly). Research and the dashboard run on the Mac.

```
Mac  ── rsync code ──>  Pi:~/open-quant-lab   (code)
                        Pi:~/lab              (collect.json, catalog/, paper/)
Mac  <── rsync catalog/ and paper/ ──  Pi
```

**Install (on the Pi):**
```bash
cd ~/open-quant-lab && python3 -m venv .venv && .venv/bin/pip install -e .
mkdir -p ~/lab && cd ~/lab && ~/open-quant-lab/.venv/bin/oql collect init
~/open-quant-lab/.venv/bin/oql collect run          # first run downloads years of history: 10-30 minutes
sudo cp ~/open-quant-lab/deploy/oql-collect.* /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now oql-collect.timer
```

**Bring the data to the Mac (on the Mac):**
```bash
rsync -a --exclude _raw david@ahl-node.local:~/lab/catalog/ ~/Desktop/ahl-node/open-quant-lab/catalog/
```
