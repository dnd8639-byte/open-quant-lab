// Live charts for Open Quant Lab: the Projects tab and the Live market tab.
// Every computation streams events (see oql/live.py); each chart event carries the whole chart, so the page simply
// keeps the latest version of each chart and redraws it. Colors are the validated categorical palette in index.html.
(() => {
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const SERIES = n => css(`--s${(n % 8) + 1}`);
const fin = v => v !== null && v !== undefined && Number.isFinite(+v);

// ------------------------------------------------------------------ number formats (a small d3-like subset)
function fmt(v, f) {
  if (!fin(v)) return "–";
  v = +v; f = f || "";
  const m = /^([+]?)(,?)\.(\d+)([f%])$/.exec(f);
  if (!m) return Math.abs(v) >= 1e4 ? v.toLocaleString(undefined, {maximumFractionDigits: 0}) : +v.toPrecision(4) + "";
  const [, plus, comma, d, kind] = m;
  let x = kind === "%" ? v * 100 : v;
  let s = comma ? x.toLocaleString(undefined, {minimumFractionDigits: +d, maximumFractionDigits: +d}) : x.toFixed(+d);
  if (plus && x > 0) s = "+" + s;
  return s + (kind === "%" ? "%" : "");
}
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const step0 = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(step0))), e = step0 / p;
  const step = (e >= 5 ? 10 : e >= 2 ? 5 : e >= 1 ? 2 : 1) * p;
  const out = []; for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(+t.toPrecision(12));
  return out;
}
function logTicks(lo, hi) {
  const few = Math.log10(hi / lo) < 2.5, out = [];                 // short ranges also get 2x and 5x ticks
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    (few ? [1, 2, 5] : [1]).forEach(m => out.push(m * Math.pow(10, e)));
  return out.filter(t => t >= lo * 0.999 && t <= hi * 1.001);
}
const timeLabel = (ms, span) => { const d = new Date(ms);
  if (span < 36e5) return d.toISOString().slice(11, 19);
  if (span < 2 * 864e5) return d.toISOString().slice(11, 16);
  if (span < 400 * 864e5) return d.toISOString().slice(5, 10);
  return d.toISOString().slice(0, 7); };

// ------------------------------------------------------------------ one chart
function drawChart(box, spec) {
  const W = Math.max(300, Math.round(box.clientWidth || 600)), H = spec.height || (spec.kind === "heatmap" ? 230 : 250);
  const series = spec.series || [];
  const legend = series.length > 1 ? `<div class="legend">${series.map((s, i) =>
      `<span><i style="background:${SERIES(i)}${s.dash ? ";height:2px" : ""}"></i>${esc(s.name)}</span>`).join("")}</div>` : "";
  box.innerHTML = `<div class="ctitle">${esc(spec.title || "")}</div>${legend}<div class="cplot" style="position:relative"></div>` +
                  (spec.notes ? `<div class="note">${spec.notes.map(esc).join(" · ")}</div>` : "");
  const plot = box.querySelector(".cplot");
  const pad = {l: 54, r: 14, t: 10, b: spec.kind === "bar" && (spec.categories || []).length > 8 ? 46 : 32};
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  let g = "", tips = [];
  const K = spec.kind;
  if (K === "heatmap") return heatmap(plot, spec, W, H);
  if (K === "bar" && spec.horizontal) return hbar(plot, spec, W);
  // ---- x and y domains
  let xs = [], ys = [];
  if (K === "line" || K === "scatter") series.forEach(s => { (s.x || []).forEach((x, i) => { if (fin(x) && fin(s.y[i])) { xs.push(+x); ys.push(+s.y[i]); } }); });
  if (K === "bar") series.forEach(s => s.y.forEach((y, i) => { if (fin(y)) { ys.push(+y + (s.err ? +s.err[i] : 0)); ys.push(+y - (s.err ? +s.err[i] : 0)); } }));
  if (K === "hist") series.forEach(s => s.counts.forEach(c => ys.push(c)));
  (spec.hlines || []).forEach(h => ys.push(+h.y));
  if (!ys.length) { plot.innerHTML = '<p class="empty">waiting for data…</p>'; return; }
  let [y0, y1] = spec.y_range || [Math.min(...ys), Math.max(...ys)];
  if (K === "bar" || K === "hist") { y0 = Math.min(0, y0); y1 = Math.max(0, y1); }
  if (y1 === y0) { y1 += 1; y0 -= (K === "hist" ? 0 : 1); }
  if (!spec.y_range && !spec.y_log) { const m = (y1 - y0) * 0.05; if (K !== "hist" && K !== "bar") y0 -= m; y1 += m; }
  const ylog = !!spec.y_log && y0 > 0, xlog = !!spec.x_log;
  const Y = v => pad.t + ih - (ylog ? (Math.log(v) - Math.log(y0)) / (Math.log(y1) - Math.log(y0)) : (v - y0) / (y1 - y0)) * ih;
  let X, x0, x1, xticks = [], xlab = t => fmt(t);
  if (K === "line" || K === "scatter") {
    [x0, x1] = spec.x_range || [Math.min(...xs), Math.max(...xs)]; if (x1 === x0) x1 = x0 + 1;
    X = v => pad.l + (xlog ? (Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) : (v - x0) / (x1 - x0)) * iw;
    if (spec.x === "time") { xticks = niceTicks(x0, x1, 5); xlab = t => timeLabel(t, x1 - x0); }
    else xticks = xlog ? logTicks(x0, x1) : niceTicks(x0, x1, 6);
  }
  // ---- grid and y axis
  const yt = ylog ? logTicks(y0, y1) : niceTicks(y0, y1, 5);
  yt.forEach(t => g += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="${css("--grid")}"/>` +
                    `<text x="${pad.l - 6}" y="${Y(t) + 4}" text-anchor="end">${esc(fmt(t, spec.y_format))}</text>`);
  if (spec.ylabel) g += `<text x="12" y="${pad.t + ih / 2}" transform="rotate(-90 12 ${pad.t + ih / 2})" text-anchor="middle">${esc(spec.ylabel)}</text>`;
  if (spec.xlabel) g += `<text x="${pad.l + iw / 2}" y="${H - 2}" text-anchor="middle">${esc(spec.xlabel)}</text>`;
  // ---- marks
  if (K === "line" || K === "scatter") {
    (spec.bands || []).forEach(b => g += `<rect x="${X(b.x0)}" y="${pad.t}" width="${Math.max(0, X(b.x1) - X(b.x0))}" height="${ih}" fill="${css("--acc-soft")}"/>` +
                                         `<text x="${X(b.x0) + 4}" y="${pad.t + 12}">${esc(b.label || "")}</text>`);
    xticks.forEach(t => { if (t >= x0 && t <= x1) g += `<text x="${X(t)}" y="${pad.t + ih + 16}" text-anchor="middle">${esc(xlab(t))}</text>`; });
    series.forEach((s, si) => {
      const col = SERIES(si), asLine = K === "line" ? s.line !== false : !!s.line;
      if (asLine) {
        let d = "", pen = false;
        s.x.forEach((x, i) => { const y = s.y[i]; if (!fin(x) || !fin(y) || (ylog && y <= 0) || (xlog && x <= 0)) { pen = false; return; }
          d += (pen ? "L" : "M") + X(+x).toFixed(1) + " " + Y(Math.min(Math.max(+y, y0), y1)).toFixed(1); pen = true; });
        g += `<path d="${d}" fill="none" stroke="${col}" stroke-width="2" ${s.dash ? 'stroke-dasharray="5 4"' : ""} stroke-linejoin="round"/>`;
      }
      if (s.points || (K === "scatter" && !s.line)) s.x.forEach((x, i) => { if (fin(x) && fin(s.y[i]) && (!ylog || s.y[i] > 0))
        g += `<circle cx="${X(+x).toFixed(1)}" cy="${Y(+s.y[i]).toFixed(1)}" r="${K === "scatter" ? 3 : 4}" fill="${col}" ${K === "scatter" ? 'fill-opacity=".65"' : `stroke="${css("--panel")}" stroke-width="2"`}/>`; });
    });
    (spec.vlines || []).forEach(v => { const x = X(+v.x); if (x >= pad.l - 1 && x <= W - pad.r + 1)
      g += `<line x1="${x}" x2="${x}" y1="${pad.t}" y2="${pad.t + ih}" stroke="${css("--ink")}" stroke-dasharray="3 3"/><text x="${x + 4}" y="${pad.t + 12}" class="ink">${esc(v.label || "")}</text>`; });
  }
  if (K === "bar") {
    const cats = spec.categories, n = cats.length, gw = iw / n, bw = Math.max(1, (gw - (n > 40 ? 0 : 6)) / series.length - (series.length > 1 ? 2 : 0));
    const every = Math.ceil(n / Math.max(1, Math.floor(iw / 46)));
    const show = i => spec.label_multiple ? +cats[i] % spec.label_multiple === 0 : (i % every === 0 || (!spec.dense && n <= 12));
    cats.forEach((c, i) => { if (show(i)) g += `<text x="${pad.l + gw * (i + 0.5)}" y="${pad.t + ih + 15}" text-anchor="middle">${esc(c)}</text>`; });
    series.forEach((s, si) => s.y.forEach((y, i) => { if (!fin(y)) return;
      const x = pad.l + gw * i + (n > 40 ? 0 : 3) + si * (bw + 2), top = Y(Math.max(+y, 0)), bot = Y(Math.min(+y, 0));
      g += `<rect x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(1, bot - top).toFixed(1)}" rx="${n > 40 ? 0 : 3}" fill="${SERIES(si)}" data-tip="${esc(`${cats[i]}: ${fmt(y, spec.y_format)}${s.err ? " ± " + fmt(s.err[i], spec.y_format) : ""}`)}"/>`;
      if (s.err) g += `<line x1="${x + bw / 2}" x2="${x + bw / 2}" y1="${Y(+y + +s.err[i])}" y2="${Y(+y - +s.err[i])}" stroke="${css("--ink")}"/>`; }));
    g += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="${css("--mut")}"/>`;
    (spec.vlines || []).forEach(v => { const i = cats.indexOf(v.x); if (i >= 0) { const x = pad.l + gw * (i + 0.5);
      g += `<line x1="${x}" x2="${x}" y1="${pad.t}" y2="${pad.t + ih}" stroke="${css("--ink")}" stroke-dasharray="3 3"/><text x="${x + 4}" y="${pad.t + 12}" class="ink">${esc(v.label)}</text>`; } });
  }
  if (K === "hist") {
    const e = spec.edges, ex0 = e[0], ex1 = e[e.length - 1], XH = v => pad.l + (v - ex0) / (ex1 - ex0) * iw;
    niceTicks(ex0, ex1, 6).forEach(t => g += `<text x="${XH(t)}" y="${pad.t + ih + 16}" text-anchor="middle">${esc(fmt(t))}</text>`);
    series.forEach((s, si) => s.counts.forEach((c, i) => { if (!c) return;
      const x = XH(e[i]) + 1, w = Math.max(1, XH(e[i + 1]) - XH(e[i]) - 2);
      g += `<rect x="${x.toFixed(1)}" y="${Y(c).toFixed(1)}" width="${w.toFixed(1)}" height="${(Y(0) - Y(c)).toFixed(1)}" rx="2" fill="${SERIES(si)}" fill-opacity="${series.length > 1 ? 0.55 : 0.85}" data-tip="${esc(`${s.name}: ${c} between ${fmt(e[i])} and ${fmt(e[i + 1])}`)}"/>`; }));
    g += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="${css("--mut")}"/>`;
    (spec.vlines || []).forEach((v, k) => { const x = XH(Math.min(Math.max(+v.x, ex0), ex1));
      g += `<line x1="${x}" x2="${x}" y1="${pad.t}" y2="${pad.t + ih}" stroke="${css("--ink")}" stroke-width="2" ${k ? 'stroke-dasharray="4 3"' : ""}/>` +
           `<text x="${x + (x > W * 0.75 ? -5 : 5)}" y="${pad.t + 12 + 14 * k}" text-anchor="${x > W * 0.75 ? "end" : "start"}" class="ink">${esc(v.label || "")}</text>`; });
  }
  (spec.hlines || []).forEach(h => g += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(+h.y)}" y2="${Y(+h.y)}" stroke="${css("--ink")}" stroke-dasharray="4 4"/>` +
                                        `<text x="${W - pad.r - 4}" y="${Y(+h.y) - 5}" text-anchor="end" class="ink">${esc(h.label || "")}</text>`);
  plot.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${esc(spec.title || "")}">${g}</svg><div class="tip hide"></div>`;
  hover(plot, spec, {X, Y, x0, x1, pad, iw, W});
}

// ranked categories with long names: one row per category, bars from zero, an optional reference line (spec.ref)
function hbar(plot, spec, W) {
  const cats = spec.categories, n = cats.length, s = spec.series[0], row = 22;
  const labW = Math.min(Math.round(W * 0.45), 8 + 6.6 * Math.max(...cats.map(c => String(c).length)));
  const pad = {l: labW, r: 16, t: 6, b: 24}, H = pad.t + pad.b + n * row, iw = W - pad.l - pad.r;
  const vals = s.y.filter(fin).concat(spec.ref ? [+spec.ref.value] : []);
  let x0 = Math.min(0, ...vals), x1 = Math.max(0, ...vals); if (x1 === x0) x1 += 1;
  const m = (x1 - x0) * 0.04; x0 = x0 < 0 ? x0 - m : x0; x1 += m;
  const X = v => pad.l + (v - x0) / (x1 - x0) * iw;
  let g = "";
  niceTicks(x0, x1, 5).forEach(t => g += `<line x1="${X(t)}" x2="${X(t)}" y1="${pad.t}" y2="${H - pad.b}" stroke="${css("--grid")}"/>` +
    `<text x="${X(t)}" y="${H - 6}" text-anchor="middle">${esc(fmt(t, spec.y_format))}</text>`);
  cats.forEach((c, i) => { const y = pad.t + i * row, v = s.y[i];
    g += `<text x="${pad.l - 8}" y="${y + row / 2 + 4}" text-anchor="end">${esc(c)}</text>`;
    if (!fin(v)) return;
    const a = X(Math.min(0, v)), b = X(Math.max(0, v));
    g += `<rect x="${a.toFixed(1)}" y="${(y + 4).toFixed(1)}" width="${Math.max(1, b - a).toFixed(1)}" height="${row - 8}" rx="3" fill="${SERIES(0)}" data-tip="${esc(`${c}: ${fmt(v, spec.y_format)}`)}"/>`; });
  g += `<line x1="${X(0)}" x2="${X(0)}" y1="${pad.t}" y2="${H - pad.b}" stroke="${css("--mut")}"/>`;
  if (spec.ref) { const x = X(+spec.ref.value);
    g += `<line x1="${x}" x2="${x}" y1="${pad.t}" y2="${H - pad.b}" stroke="${css("--ink")}" stroke-dasharray="4 3"/>` +
         `<text x="${x + 4}" y="${H - pad.b - 4}" class="ink">${esc(spec.ref.label || "")}</text>`; }
  plot.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${esc(spec.title || "")}">${g}</svg><div class="tip hide"></div>`;
  hover(plot, spec, {pad, W});
}

function heatmap(plot, spec, W, H) {
  const z = spec.z, ny = z.length, nx = z[0].length, pad = {l: 40, r: 70, t: 6, b: 28};
  const cw = (W - pad.l - pad.r) / nx, ch = (H - pad.t - pad.b) / ny;
  const vals = z.flat().filter(fin), mx = Math.max(...vals.map(Math.abs)), lo = Math.min(...vals), hi = Math.max(...vals);
  const color = v => { if (!fin(v)) return css("--grid");
    if (spec.diverging) { const t = Math.max(-1, Math.min(1, v / (mx || 1)));
      return t >= 0 ? mix(css("--div-mid"), css("--div-pos"), t) : mix(css("--div-mid"), css("--div-neg"), -t); }
    return mix(css("--seq-lo"), css("--seq-hi"), (v - lo) / ((hi - lo) || 1)); };
  let g = "";
  z.forEach((row, j) => row.forEach((v, i) => g += `<rect x="${(pad.l + i * cw).toFixed(1)}" y="${(pad.t + j * ch).toFixed(1)}" width="${(cw - 1).toFixed(1)}" height="${(ch - 1).toFixed(1)}" fill="${color(v)}" data-tip="${esc(`${spec.ycats[j]} ${spec.xcats[i]}: ${fmt(v)} ${spec.unit || ""}`)}"/>`));
  spec.ycats.forEach((c, j) => g += `<text x="${pad.l - 6}" y="${pad.t + (j + 0.5) * ch + 4}" text-anchor="end">${esc(c)}</text>`);
  spec.xcats.forEach((c, i) => { if (i % Math.ceil(nx / 12) === 0) g += `<text x="${pad.l + (i + 0.5) * cw}" y="${H - 12}" text-anchor="middle">${esc(c)}</text>`; });
  if (spec.xlabel) g += `<text x="${pad.l + (W - pad.l - pad.r) / 2}" y="${H}" text-anchor="middle">${esc(spec.xlabel)}</text>`;
  const L = spec.diverging ? [-mx, 0, mx] : [lo, (lo + hi) / 2, hi], lx = W - pad.r + 14;
  L.forEach((v, k) => g += `<rect x="${lx}" y="${pad.t + k * 22}" width="14" height="14" rx="2" fill="${color(v)}"/><text x="${lx + 18}" y="${pad.t + k * 22 + 11}">${esc(fmt(v))}</text>`);
  g += `<text x="${lx}" y="${pad.t + 3 * 22 + 10}">${esc(spec.unit || "")}</text>`;
  plot.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="${esc(spec.title || "")}">${g}</svg><div class="tip hide"></div>`;
  hover(plot, spec, null);
}

function mix(a, b, t) {
  const p = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)), A = p(a), B = p(b);
  return "#" + A.map((v, i) => Math.round(v + (B[i] - v) * Math.max(0, Math.min(1, t))).toString(16).padStart(2, "0")).join("");
}

function hover(plot, spec, geo) {
  const tip = plot.querySelector(".tip"), svg = plot.querySelector("svg");
  const show = (html, ev) => { tip.innerHTML = html; tip.classList.remove("hide");
    const r = plot.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    tip.style.left = Math.min(x + 12, r.width - 230) + "px"; tip.style.top = Math.max(0, y - 10) + "px"; };
  svg.addEventListener("mouseleave", () => { tip.classList.add("hide"); const c = svg.querySelector(".cross"); if (c) c.remove(); });
  svg.addEventListener("mousemove", ev => {
    const t = ev.target.getAttribute && ev.target.getAttribute("data-tip");
    if (t) return show(esc(t), ev);
    if (!geo || !(spec.kind === "line" || spec.kind === "scatter")) return;
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (geo.W / r.width);
    if (px < geo.pad.l || px > geo.pad.l + geo.iw) return;
    let best = null;
    (spec.series || []).forEach((s, si) => s.x.forEach((x, i) => { if (!fin(x) || !fin(s.y[i])) return;
      const d = Math.abs(geo.X(+x) - px); if (!best || d < best.d) best = {d, x: +x}; }));
    if (!best) return;
    const rows = (spec.series || []).map((s, si) => { let k = -1, dd = Infinity;
      s.x.forEach((x, i) => { const d = Math.abs(+x - best.x); if (fin(s.y[i]) && d < dd) { dd = d; k = i; } });
      return k < 0 ? "" : `<div><i style="background:${SERIES(si)}"></i>${esc(s.name)}: <b>${esc(fmt(s.y[k], spec.y_format))}</b></div>`; }).join("");
    const xl = spec.x === "time" ? new Date(best.x).toISOString().replace("T", " ").slice(0, 16) + " UTC" : fmt(best.x);
    let c = svg.querySelector(".cross"); if (!c) { c = document.createElementNS("http://www.w3.org/2000/svg", "line"); c.setAttribute("class", "cross"); svg.appendChild(c); }
    const cx = geo.X(best.x); c.setAttribute("x1", cx); c.setAttribute("x2", cx); c.setAttribute("y1", geo.pad.t); c.setAttribute("y2", geo.pad.t + (svg.viewBox.baseVal.height - geo.pad.t - 32));
    c.setAttribute("stroke", css("--mut")); c.setAttribute("stroke-dasharray", "2 3");
    show(`<div class="mut">${esc(xl)}</div>${rows}`, ev);
  });
}

// ------------------------------------------------------------------ a live output area (stats, checks, charts, tables)
function makeOutput(root) {
  root.innerHTML = `<div class="stats live-stats"></div><div class="checks"></div><div class="bar"><div></div></div>
    <p class="note live-log"></p><div class="charts"></div><div class="tables"></div>`;
  const state = {stats: {}, checks: {}, charts: {}, tables: {}, order: []};
  let pending = false;
  const paint = () => { pending = false;
    root.querySelector(".live-stats").innerHTML = Object.values(state.stats).map(s =>
      `<div class="stat"><div class="k">${esc(s.label)}</div><div class="v">${esc(fmt(s.value, s.fmt))}${esc(s.unit || "")}</div></div>`).join("");
    root.querySelector(".checks").innerHTML = Object.values(state.checks).map(c =>
      `<div class="checkrow"><span class="badge ${c.passed ? "PASS" : "FAIL"}">${c.passed ? "✓ check passed" : "✗ check failed"}</span> <b>${esc(c.check)}</b> <span class="note">${esc(c.detail)}</span></div>`).join("");
    const box = root.querySelector(".charts");
    Object.entries(state.charts).forEach(([id, spec]) => {
      let el = box.querySelector(`[data-chart="${CSS.escape(id)}"]`);
      if (!el) { el = document.createElement("div"); el.className = "panel chart" + (spec.kind === "heatmap" || spec.wide ? " wide" : ""); el.dataset.chart = id; box.appendChild(el); }
      if (el._spec !== spec) { el._spec = spec; drawChart(el, spec); }
    });
    root.querySelector(".tables").innerHTML = Object.values(state.tables).map(t => `<div class="panel"><h2>${esc(t.title)}</h2><table><tr>${t.columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr>` +
      t.rows.map(r => `<tr>${r.map(v => `<td class="${typeof v === "number" ? "num" : ""}">${esc(typeof v === "number" ? fmt(v) : v)}</td>`).join("")}</tr>`).join("") + "</table></div>").join("");
  };
  return {
    add(ev) {
      if ("stat" in ev) state.stats[ev.stat] = {...ev, label: ev.label};
      else if ("check" in ev) state.checks[ev.check] = ev;
      else if ("chart" in ev) state.charts[ev.chart] = ev;
      else if ("table" in ev) state.tables[ev.table] = ev;
      else if ("log" in ev) root.querySelector(".live-log").textContent = ev.log;
      else if ("progress" in ev) root.querySelector(".bar > div").style.width = (100 * ev.progress) + "%";
      if (!pending) { pending = true; requestAnimationFrame(paint); }
    },
    redraw() { root.querySelectorAll(".chart").forEach(el => el._spec && drawChart(el, el._spec)); },
  };
}
window.addEventListener("resize", () => document.querySelectorAll(".chart").forEach(el => el._spec && drawChart(el, el._spec)));

// ------------------------------------------------------------------ Projects tab
let PROJECTS = [], current = null, source = null;
async function loadProjects() {
  if (PROJECTS.length) return;
  PROJECTS = await (await fetch("/api/projects")).json();
  $("#plist").innerHTML = PROJECTS.map((p, i) => `<div class="card pcard" data-i="${i}"><div class="id">project ${i + 1}</div><div class="t">${esc(p.title)}</div>
    <div class="m">${esc(p.question)}</div>${p.missing.length ? `<span class="badge FAIL">needs data</span>` : ""}</div>`).join("");
  document.querySelectorAll(".pcard").forEach(el => el.onclick = () => openProject(+el.dataset.i));
}
function openProject(i) {
  document.querySelectorAll(".pcard").forEach(el => el.classList.toggle("sel", +el.dataset.i === i));
  const p = PROJECTS[i]; current = p;
  $("#pdetail").classList.remove("hide");
  $("#pdetail").innerHTML = `<div class="panel"><h2>Project ${i + 1}</h2><div style="font-size:17px;font-weight:650">${esc(p.title)}</div><p class="note">${esc(p.question)}</p>
    ${p.missing.length ? `<p class="err">Missing data: ${p.missing.map(esc).join(", ")}. Get it with <code>${esc(p.how_to_get)}</code>. The built-in checks still run.</p>` : ""}
    <div class="controls">${p.params.map(q => `<label>${esc(q.label)}<select data-p="${esc(q.name)}">${(q.options || []).map(o =>
      `<option ${String(o) === String(q.default) ? "selected" : ""}>${esc(o)}</option>`).join("")}</select></label>`).join("")}
      <button class="go" id="prun">Run</button><button class="ghost" id="pstop" disabled>Stop</button><span class="note" id="pstatus"></span></div>
    <div id="pout"></div></div>`;
  $("#prun").onclick = runProject;
  $("#pstop").onclick = () => current && current.job && fetch(`/api/jobs/${current.job}/stop`, {method: "POST"});
}
async function runProject() {
  const params = {}; document.querySelectorAll("#pdetail [data-p]").forEach(s => params[s.dataset.p] = isNaN(+s.value) ? s.value : +s.value);
  const out = makeOutput($("#pout"));
  $("#prun").disabled = true; $("#pstop").disabled = false; $("#pstatus").textContent = "running…";
  const r = await fetch("/api/jobs", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({project: current.id, params})});
  const j = await r.json(); if (!r.ok) { $("#pstatus").textContent = j.error; $("#prun").disabled = false; return; }
  current.job = j.id; if (source) source.close();
  source = new EventSource(`/api/jobs/${j.id}/stream`);
  source.onmessage = m => { const ev = JSON.parse(m.data);
    if (ev.done || ev.error) { $("#pstatus").textContent = ev.error ? "error: " + ev.error : ev.stopped ? "stopped" : `finished in ${ev.seconds} s`;
      $("#prun").disabled = false; $("#pstop").disabled = true; source.close(); }
    else out.add(ev); };
  source.addEventListener("end", () => source.close());
}

// ------------------------------------------------------------------ Live market tab
let mstream = null, mout = null;
async function initMarket() {
  if ($("#mvenue").options.length) return;
  const v = await (await fetch("/api/market/venues")).json();
  $("#mvenue").innerHTML = Object.entries(v).map(([k, l]) => `<option value="${k}">${esc(l)}</option>`).join("");
  $("#mstart").onclick = startMarket;
  $("#mstop").onclick = () => { if (mstream) mstream.close(); mstream = null; $("#mstate").textContent = "paused"; $("#mstop").disabled = true; };
}
function startMarket() {
  if (mstream) mstream.close();
  mout = makeOutput($("#mout"));
  $("#mstate").textContent = "connecting…"; $("#mstop").disabled = false;
  mstream = new EventSource(`/api/market/stream?venue=${$("#mvenue").value}&asset=${$("#masset").value}`);
  mstream.onmessage = m => { const ev = JSON.parse(m.data); if ("log" in ev) $("#mstate").textContent = ev.log; else mout.add(ev); };
  mstream.onerror = () => { $("#mstate").textContent = "connection lost; retrying…"; };
}

// ------------------------------------------------------------------ Efficiency tab
let EFF = null;
const jobs = {};
function runJob(kind, params, outEl, btn, stopBtn, stateEl, onDone) {
  const out = makeOutput(outEl);
  btn.disabled = true; stopBtn.disabled = false; stateEl.textContent = "running…";
  fetch("/api/jobs", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({project: kind, params})})
    .then(r => r.json().then(j => ({ok: r.ok, j}))).then(({ok, j}) => {
      if (!ok) { stateEl.textContent = j.error; btn.disabled = false; stopBtn.disabled = true; return; }
      if (jobs[kind]) jobs[kind].src.close();
      const src = new EventSource(`/api/jobs/${j.id}/stream`);
      jobs[kind] = {id: j.id, src};
      src.onmessage = m => { const ev = JSON.parse(m.data);
        if (ev.done || ev.error) { stateEl.textContent = ev.error ? "error: " + ev.error : ev.stopped ? "stopped" : `finished in ${ev.seconds} s`;
          btn.disabled = false; stopBtn.disabled = true; src.close(); onDone && onDone(); }
        else out.add(ev); };
      src.addEventListener("end", () => src.close());
    });
  stopBtn.onclick = () => jobs[kind] && fetch(`/api/jobs/${jobs[kind].id}/stop`, {method: "POST"});
}
function effHistory() {
  const rows = EFF.history || [];
  $("#ehist").innerHTML = rows.length ? `<tr><th>UTC</th><th>data</th><th>what</th><th>tried</th><th>result</th></tr>` + rows.map(r =>
    `<tr><td>${esc(r.utc)}</td><td>${esc(r.data)}</td><td>${esc(r.finalist)}</td><td class="num">${esc(r.n_trials)}</td><td>${esc(r.note)}</td></tr>`).join("")
    : `<tr><td class="note">None yet. Each run here is added to the ledger as an exploration.</td></tr>`;
}
async function loadEfficiency(ds) {
  EFF = await (await fetch("/api/efficiency" + (ds ? `?dataset=${encodeURIComponent(ds)}` : ""))).json();
  const opt = (v, sel) => `<option ${v === sel ? "selected" : ""}>${esc(v)}</option>`;
  $("#eds").innerHTML = EFF.datasets.map(d => opt(d, EFF.dataset)).join("");
  $("#egroups").innerHTML = EFF.groups.map(g => `<label class="${g.ok ? "" : "off"}" title="${esc(g.note || g.label)}">
    <input type="checkbox" data-g="${g.id}" ${g.id === "price" ? "checked disabled" : g.ok ? "checked" : "disabled"}>${esc(g.id)}<span class="note">: ${esc(g.ok ? g.label : g.note)}</span></label>`).join("");
  if (!$("#gds").options.length) {
    $("#gds").innerHTML = EFF.gate_datasets.map(d => opt(d, EFF.dataset)).join("");
    $("#gstrat").innerHTML = EFF.strategies.map(s => opt(s, "momentum lookback=20")).join("");
  }
  $("#emodel").innerHTML = `<option value="ridge">ridge (linear)</option>` + (EFF.trees ? `<option value="trees">boosted trees</option>` : "");
  effHistory();
}
async function initEfficiency() {
  if (EFF) return;
  await loadEfficiency();
  $("#eds").onchange = () => loadEfficiency($("#eds").value);
  $("#erun").onclick = () => {
    const params = {dataset: $("#eds").value, model: $("#emodel").value};
    document.querySelectorAll("#egroups [data-g]").forEach(c => params[c.dataset.g] = c.checked);
    runJob("efficiency", params, $("#eout"), $("#erun"), $("#estop"), $("#estate"), () => loadEfficiency($("#eds").value));
  };
  $("#grun").onclick = () => runJob("gatecheck", {dataset: $("#gds").value, strategy: $("#gstrat").value, cost_bps: +$("#gcost").value, band: +$("#gband").value},
    $("#gout"), $("#grun"), $("#gstop"), $("#gstate"), () => loadEfficiency($("#eds").value));
}

window.OQLLive = {loadProjects, initMarket, initEfficiency, drawChart, fmt};
})();
