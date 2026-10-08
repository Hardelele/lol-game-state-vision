"use strict";
// Смотрелка предсказаний координат камеры. Данные берутся по HTTP у
// tools/serve_inspect.py, картинки грузятся по мере надобности, поэтому
// доступны все кадры ролика, а не выборка.

const $ = (id) => document.getElementById(id);
const BIG = 0.133;   // промах больше половины вьюпорта считаем крупным
const GOOD = 0.05;
const UNSURE = 0.15; // разброс ответа, выше которого модель сомневается
const CELLS = ["8x5", "12x7", "16x9", "24x13"];

// Полосы таймлайна и цифры в тексте окрашены по-разному: тёмные заливки
// хорошо читаются полосой, а на тёмном фоне текста нужны светлее.
const BAR = { ok: "#3fa863", mid: "#c9922f", bad: "#d6483c" };
const errClass = (e) => (e < GOOD ? "ok" : e < BIG ? "mid" : "bad");

const TL_H = 58;     // высота холста таймлайна
const FUNNEL = 16;   // верхняя полоса холста: воронка от ленты к её отрезку

const S = {
  runs: [], run: null, video: null, frames: [], pos: 0,
  proj: null, mapUnits: 14800, cells: [], cellSpec: "8x5",
  grid: true, heat: true, trail: false, fErr: false, fUns: false, fLab: false,
  playing: null, trailLen: 40, filmN: 15,
  mode: "end",        // "end" — сквозной режим, "net" — разбор по слоям
  stages: [], stage: 0, chan: 0, why: false,
  // Счётчики запросов: ответ, пришедший после следующего переключения,
  // отбрасывается. Иначе при быстром листании поздний ответ перерисовывает
  // панель данными уже не того кадра или этапа.
  netSeq: 0, chanSeq: 0,
};

// Пояснение к этапу: что именно здесь происходит и на что смотреть.
const WHY = {
  input: "Кадр после маски и обрезки, сжатый до входного размера. Это всё, " +
    "что сеть видит: HUD и миникарта вырезаны, поэтому подсмотреть ответ " +
    "на миникарте она не может.",
  conv: "Свёртка прикладывает набор небольших ядер к каждой точке предыдущего " +
    "этапа. Каждое ядро даёт свою карту признаков: одно отзывается на край, " +
    "другое на пятно нужного цвета, третье на сочетание.",
  act: "Нормировка выравнивает масштаб откликов, а ReLU обнуляет отрицательные. " +
    "Отсюда доля «живых» точек: если канал почти весь погас, он на этом кадре " +
    "ничего не нашёл.",
  pool: "Усреднение до мелкой сетки. Сетка, а не одно число на канал: " +
    "положение найденного в кадре — это и есть подсказка о месте на карте.",
  heat: "Голова превращает признаки в распределение вероятности по карте. " +
    "Два пятна означают, что сеть видит два подходящих места.",
};

const KEYS = [
  [["←", "→"], "кадр"], [["⇧", "←", "→"], "10 кадров"], [["N", "P"], "крупный промах"],
  [["Space"], "играть / пауза"], [["G"], "сетка"], [["H"], "тепло"], [["T"], "след"],
  [["Home", "End"], "начало / конец"], [["Tab"], "режим по кругу"],
  [["[", "]"], "этап сети"],
  [["?"], "эта подсказка"],
];

const pad = (i) => String(i).padStart(6, "0");
const sceneSrc = (f) => `/img/scene/${S.video}/${pad(f.i)}.jpg`;

// ---------- загрузка ----------

async function boot() {
  const idx = await (await fetch("/api/index")).json();
  S.runs = idx.runs;
  S.proj = idx.projection && idx.projection.outline ? idx.projection : null;
  S.meta = idx.videos || {};
  S.mapUnits = idx.map_units;
  $("run").innerHTML = S.runs.map((r) => `<option>${r.id}</option>`).join("");
  $("run").onchange = () => selectRun($("run").value);
  $("video").onchange = () => loadFrames($("video").value);
  $("keys").innerHTML = KEYS.map(([caps, d]) =>
    `<span>${caps.map((c) => `<kbd>${c}</kbd>`).join("")}</span><span>${d}</span>`).join("");
  wire();
  wireNet();
  wireChannels();
  await selectRun(S.runs[0].id);
}

async function selectRun(id) {
  S.run = S.runs.find((r) => r.id === id);
  const train = S.run.trained_on || [];
  // В списке — человекочитаемая подпись, иначе виден только идентификатор.
  $("video").innerHTML = S.run.videos.map((v) => {
    const m = S.meta[v] || {};
    const tag = train.includes(v) ? "обучающий" : "ОТЛОЖЕННЫЙ";
    const who = [m.role, m.side].filter(Boolean).join(" ");
    return `<option value="${v}">${tag} · ${who || v}</option>`;
  }).join("");
  // По умолчанию открываем отложенный ролик: на обучающем смотреть нечего,
  // модель эти кадры видела.
  const first = S.run.videos.find((v) => !train.includes(v)) || S.run.videos[0];
  $("video").value = first;
  await loadFrames(first);
}

async function loadFrames(vid) {
  S.video = vid;
  const d = await (await fetch(`/api/frames/${S.run.id}/${vid}`)).json();
  S.frames = d.frames;
  const train = (S.run.trained_on || []).includes(vid);
  $("holdout").title = train ? "обучающий ролик — модель его видела"
    : "отложенный ролик — модель его не видела";
  $("holdout").className = "dot" + (train ? " train" : "");
  showMeta(vid, train);
  await loadCells(S.cellSpec);
  summary();
  go(0);
}

async function loadCells(spec) {
  S.cellSpec = spec;
  $("cells").textContent = spec.replace("x", "×");
  S.cells = S.proj ? await (await fetch(`/api/cells/${spec}`)).json() : [];
  buildGrids();
  render();
}

function summary() {
  const e = S.frames.map((f) => f.err).sort((a, b) => a - b);
  const med = e[Math.floor(e.length / 2)] || 0;
  $("sMed").title = `медианный промах ≈ ${Math.round(med * S.mapUnits)} ед. · ` +
    `кадров ${S.frames.length}`;
  $("sMed").innerHTML = `<b>${med.toFixed(3)}</b> <i>медиана</i>`;
  $("sGood").textContent = S.frames.filter((f) => f.err < GOOD).length;
  $("sBad").textContent = S.frames.filter((f) => f.err >= BIG).length;
  $("posTotal").textContent = S.frames.length
    ? `/ ${fmtTime(S.frames[S.frames.length - 1].t)}` : "";
}

// ---------- фильтры и перемещение ----------

function passes(f) {
  if (S.fErr && f.err < BIG) return false;
  if (S.fUns && f.spread < UNSURE) return false;
  if (S.fLab && !f.label) return false;
  return true;
}

function step(delta) {
  const n = S.frames.length;
  let i = S.pos;
  for (let k = 0; k < n; k++) {
    i = (i + delta + n) % n;
    if (passes(S.frames[i])) return go(i);
  }
  go(S.pos);
}

function jumpWorst(dir) {
  const n = S.frames.length;
  let i = S.pos;
  for (let k = 0; k < n; k++) {
    i = (i + dir + n) % n;
    if (S.frames[i].err >= BIG) return go(i);
  }
}

function go(i) {
  S.pos = Math.max(0, Math.min(S.frames.length - 1, i));
  render();
  strip();
  if (S.mode === "net") loadNet();
}

// ---------- отрисовка кадра ----------

function fmtTime(t) {
  const s = Math.round(t);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function render() {
  const f = S.frames[S.pos];
  if (!f) return;
  $("scene").src = sceneSrc(f);
  $("mini").src = `/img/mini/${S.video}/${pad(f.i)}.jpg`;
  $("heatimg").src = `/img/heat/${S.run.id}/${S.video}/${pad(f.i)}.png`;
  $("heatimg").style.display = S.heat ? "" : "none";

  $("vErr").textContent = f.err.toFixed(3);
  $("vErr").className = "big " + errClass(f.err);
  $("vUnits").textContent = `${Math.round(f.err * S.mapUnits)} ед.`;
  $("vTrue").textContent = `${f.cx.toFixed(3)}, ${f.cy.toFixed(3)}`;
  $("vPred").textContent = `${f.px.toFixed(3)}, ${f.py.toFixed(3)}`;
  $("vSpread").textContent = f.spread.toFixed(2);
  $("vSpread").className = f.spread >= UNSURE ? "mid" : "";
  $("vQ").textContent = f.q.toFixed(2);
  $("vLabel").textContent = f.label || "";
  $("vLabel").className = "chip" + (f.label ? "" : " hidden") +
    (f.label === "top" ? " top" : "");
  $("yt").href = `https://www.youtube.com/watch?v=${S.video}&t=${Math.round(f.t)}s`;
  $("why").href = `/explain/${S.run.id}/${S.video}/${f.i}.html`;
  if (document.activeElement !== $("jump")) $("jump").value = fmtTime(f.t);
  $("posNo").textContent = `#${f.i}`;
  drawMap(f);
}

function poly(pts, ox, oy) {
  return pts.map((p) => `${(ox + p[0]).toFixed(5)},${(oy + p[1]).toFixed(5)}`).join(" ");
}

function drawMap(f) {
  const svg = $("mapSvg");
  const out = S.proj ? S.proj.outline : null;
  let s = "";
  if (S.trail) {
    const a = Math.max(0, S.pos - S.trailLen);
    const d = S.frames.slice(a, S.pos + 1)
      .map((g, k) => `${k ? "L" : "M"}${g.cx.toFixed(4)},${g.cy.toFixed(4)}`).join("");
    s += `<path class="trail" d="${d}"/>`;
  }
  if (out) {
    s += `<polygon class="out" points="${poly(out, f.cx, f.cy)}"/>`;
    s += `<polygon class="pred" points="${poly(out, f.px, f.py)}"/>`;
    s += S.cells.map((c) =>
      `<polygon class="cellpoly" data-cell="${c.id}" points="${poly(c.pts, f.cx, f.cy)}"/>`
    ).join("");
  }
  s += `<circle class="t" cx="${f.cx}" cy="${f.cy}" r="0.012"/>`;
  s += `<line class="cross" x1="${f.px - 0.02}" y1="${f.py}" x2="${f.px + 0.02}" y2="${f.py}"/>`;
  s += `<line class="cross" x1="${f.px}" y1="${f.py - 0.02}" x2="${f.px}" y2="${f.py + 0.02}"/>`;
  s += `<line class="link" x1="${f.cx}" y1="${f.cy}" x2="${f.px}" y2="${f.py}"/>`;
  svg.innerHTML = s;
  hookCells(svg);
}

function buildGrids() {
  const [cols, rows] = S.cellSpec.split("x").map(Number);
  const size = S.proj ? S.proj.scene_size : [720, 372];
  const g = $("sceneGrid");
  g.setAttribute("viewBox", `0 0 ${size[0]} ${size[1]}`);
  let s = "";
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const id = `${r + 1}.${c + 1}`;
      s += `<rect class="cellrect" data-cell="${id}" x="${c * size[0] / cols}" ` +
           `y="${r * size[1] / rows}" width="${size[0] / cols}" height="${size[1] / rows}"/>`;
    }
  }
  g.innerHTML = s;
  hookCells(g);
  applyGrid();
}

// Ячейка картинки и ячейка карты носят один номер — по нему и связываются.
function hookCells(scope) {
  scope.querySelectorAll("[data-cell]").forEach((el) => {
    el.onmouseenter = () => {
      const id = el.dataset.cell;
      document.querySelectorAll(`[data-cell="${id}"]`)
        .forEach((x) => x.classList.add("lit"));
      const f = S.frames[S.pos];
      const q = S.cells.find((c) => c.id === id);
      let txt = id;
      if (q && f) {
        const cx = f.cx + q.pts.reduce((a, p) => a + p[0], 0) / 4;
        const cy = f.cy + q.pts.reduce((a, p) => a + p[1], 0) / 4;
        txt = `${id}  →  ${cx.toFixed(3)}, ${cy.toFixed(3)}`;
      }
      $("cellTip").textContent = txt;
      $("cellTip").classList.remove("hidden");
    };
    el.onmouseleave = () => {
      document.querySelectorAll(".lit").forEach((x) => x.classList.remove("lit"));
      $("cellTip").classList.add("hidden");
    };
  });
}

function applyGrid() {
  document.body.classList.toggle("hidden-grid", !S.grid);
  $("grid").classList.toggle("on", S.grid);
}

// ---------- лента и таймлайн ----------

// Лента — окно вокруг текущего кадра, таймлайн — весь ролик. Воронка в
// верхней полосе холста показывает, какой отрезок ролика сейчас в ленте.
function strip() {
  film();
  drawTimeline();
}

function film() {
  const box = $("film"), K = (S.filmN - 1) / 2;
  let s = "";
  for (let k = -K; k <= K; k++) {
    const j = S.pos + k, f = S.frames[j];
    if (!f) { s += '<div class="th void"><div class="pic"></div><div class="eb"></div></div>'; continue; }
    const op = k === 0 ? 1 : passes(f) ? 0.62 : 0.2;
    s += `<div class="th${k === 0 ? " cur" : ""}" data-i="${j}" title="${fmtTime(f.t)}" ` +
      `style="opacity:${op}"><div class="pic"><img loading="lazy" src="${sceneSrc(f)}"></div>` +
      `<div class="eb" style="background:${BAR[errClass(f.err)]}"></div></div>`;
  }
  box.innerHTML = s;
  box.querySelectorAll(".th[data-i]").forEach((el) => {
    el.onclick = () => go(+el.dataset.i);
  });
}

function fitFilm() {
  let n = Math.max(5, Math.floor(($("film").clientWidth + 4) / 100));
  if (n % 2 === 0) n--;
  if (n !== S.filmN) { S.filmN = n; film(); }
  drawTimeline();
}

function drawTimeline() {
  const c = $("tl"), dpr = devicePixelRatio || 1, w = c.clientWidth;
  c.width = w * dpr; c.height = TL_H * dpr;
  const x = c.getContext("2d");
  x.scale(dpr, dpr);
  x.clearRect(0, 0, w, TL_H);
  const n = S.frames.length || 1, bw = Math.max(1, w / n);
  const base = 53, span = 26, minH = 3;
  S.frames.forEach((f, i) => {
    const on = passes(f);
    x.globalAlpha = on ? 1 : 0.14;
    x.fillStyle = BAR[errClass(f.err)];
    const bh = minH + Math.min(span, f.err / 0.3 * span);
    x.fillRect(i * w / n, base - bh, bw, bh);
    if (f.label) {
      x.globalAlpha = on ? 0.9 : 0.2;
      x.fillStyle = f.label === "top" ? "#8ab4ff" : "#5b6472";
      x.fillRect(i * w / n, base + 1, bw, 3);
    }
  });
  x.globalAlpha = 1;
  const p = S.pos, px = p * w / n;
  x.fillStyle = "#fff";
  x.fillRect(px - 1, FUNNEL, Math.max(2, bw), TL_H - FUNNEL);

  const K = (S.filmN - 1) / 2, L = FUNNEL;
  const x0 = Math.max(0, (p - K) * w / n), x1 = Math.min(w, (p + K + 1) * w / n);
  x.beginPath(); x.moveTo(0, 0); x.lineTo(w, 0); x.lineTo(x1, L); x.lineTo(x0, L); x.closePath();
  x.fillStyle = "rgba(138,180,255,.07)"; x.fill();
  x.strokeStyle = "rgba(138,180,255,.3)"; x.lineWidth = 1;
  x.beginPath(); x.moveTo(0.5, 0); x.lineTo(x0, L); x.moveTo(w - 0.5, 0); x.lineTo(x1, L); x.stroke();
  x.fillStyle = "rgba(138,180,255,.75)"; x.fillRect(x0, L - 1, Math.max(2, x1 - x0), 2);
  x.strokeStyle = "#fff"; x.lineWidth = 1.5;
  x.beginPath(); x.moveTo(w / 2, 0); x.lineTo(px, L); x.stroke();
}

function tlIndex(ev) {
  const r = $("tl").getBoundingClientRect();
  return Math.max(0, Math.min(S.frames.length - 1,
    Math.floor((ev.clientX - r.left) / r.width * S.frames.length)));
}

function hover(ev) {
  const r = $("tl").getBoundingClientRect(), f = S.frames[tlIndex(ev)];
  if (!f) return;
  const box = $("hov");
  box.style.left = `${Math.max(72, Math.min(r.width - 72, ev.clientX - r.left))}px`;
  $("hovImg").src = sceneSrc(f);
  $("hovTime").textContent = `${fmtTime(f.t)} · #${f.i}`;
  $("hovErr").textContent = f.err.toFixed(3);
  $("hovErr").className = errClass(f.err);
  box.classList.remove("hidden");
}

// ---------- управление ----------

function play() {
  if (S.playing) {
    clearInterval(S.playing); S.playing = null;
    $("play").textContent = "▶";
  } else {
    S.playing = setInterval(() => step(1), 220);
    $("play").textContent = "❚❚";
  }
}

function toggle(key) {
  S[key] = !S[key];
  $(key).classList.toggle("on", S[key]);
  if (key === "grid") applyGrid();
  else if (key === "heat" || key === "trail") render();
  else strip();
}

function jump(v) {
  let target = -1;
  if (v.includes(":")) {
    const [m, s] = v.split(":").map(Number);
    const sec = m * 60 + s;
    target = S.frames.reduce((best, f, i) =>
      Math.abs(f.t - sec) < Math.abs(S.frames[best].t - sec) ? i : best, 0);
  } else if (v) {
    target = S.frames.findIndex((f) => f.i === Number(v));
  }
  if (target >= 0) go(target);
}

function help(show) {
  $("help").classList.toggle("hidden", show === undefined
    ? !$("help").classList.contains("hidden") : !show);
}

function wire() {
  $("prev").onclick = (e) => step(e.shiftKey ? -10 : -1);
  $("next").onclick = (e) => step(e.shiftKey ? 10 : 1);
  $("play").onclick = play;
  ["grid", "heat", "trail", "fErr", "fUns", "fLab"].forEach((k) => {
    $(k).onclick = () => toggle(k);
  });
  $("cells").onclick = () =>
    loadCells(CELLS[(CELLS.indexOf(S.cellSpec) + 1) % CELLS.length]);

  const j = $("jump");
  j.onfocus = () => { j.value = ""; };
  j.onblur = () => { const f = S.frames[S.pos]; if (f) j.value = fmtTime(f.t); };
  j.onkeydown = (e) => {
    if (e.key === "Enter") { const v = j.value.trim(); j.blur(); jump(v); }
    else if (e.key === "Escape") j.blur();
  };

  let drag = false;
  $("tl").onmousedown = (e) => { drag = true; go(tlIndex(e)); };
  addEventListener("mouseup", () => { drag = false; });
  $("tl").onmousemove = (e) => { hover(e); if (drag) go(tlIndex(e)); };
  $("tl").onmouseleave = () => $("hov").classList.add("hidden");

  document.querySelectorAll(".fact[data-lit]").forEach((el) => {
    const flag = "lit-" + el.dataset.lit;
    el.onmouseenter = () => document.body.classList.add(flag);
    el.onmouseleave = () => document.body.classList.remove(flag);
  });

  $("helpBtn").onclick = () => help();
  $("help").onclick = () => help(false);

  // Буквы проверяются и в русской раскладке: переключать её ради
  // горячих клавиш неудобно.
  addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    const k = e.key, l = k.toLowerCase();
    if (k === "ArrowRight") { step(e.shiftKey ? 10 : 1); e.preventDefault(); }
    else if (k === "ArrowLeft") { step(e.shiftKey ? -10 : -1); e.preventDefault(); }
    else if (k === " ") { play(); e.preventDefault(); }
    else if (k === "Home") go(0);
    else if (k === "End") go(S.frames.length - 1);
    else if (l === "n" || l === "т") jumpWorst(1);
    else if (l === "p" || l === "з") jumpWorst(-1);
    else if (l === "g" || l === "п") toggle("grid");
    else if (l === "h" || l === "р") toggle("heat");
    else if (l === "t" || l === "е") toggle("trail");
    else if (k === "Tab") {
      setMode(MODES[(MODES.indexOf(S.mode) + 1) % MODES.length]);
      e.preventDefault();
      return;
    }
    // В витрине каналов листать кадры нечем, а пробел и буквы только мешают.
    else if (S.mode === "chan") { if (k === "?" || k === "Escape") help(); }
    else if (k === "?" || k === ",") help();
    else if (k === "Escape") help(false);
    else if (S.mode === "net" && (k === "[" || l === "х")) gotoStage(-1);
    else if (S.mode === "net" && (k === "]" || l === "ъ")) gotoStage(1);
  });

  new ResizeObserver(fitFilm).observe($("film"));
}

boot();


// ---------- режим «по слоям» ----------

const MODES = ["end", "net", "chan"];

function setMode(m) {
  S.mode = m;
  $("mEnd").classList.toggle("on", m === "end");
  $("mNet").classList.toggle("on", m === "net");
  $("mCh").classList.toggle("on", m === "chan");
  $("endview").classList.toggle("hidden", m !== "end");
  $("netview").classList.toggle("hidden", m !== "net");
  $("chview").classList.toggle("hidden", m !== "chan");
  // В витрине каналов лента кадров и строка про матч не о чём: там ещё не
  // выбран ролик датасета.
  $("stripview").classList.toggle("hidden", m === "chan");
  $("vmeta").classList.toggle("hidden", m === "chan");
  if (S.playing && m !== "end") play();
  if (m === "net") loadNet();
  if (m === "chan") openChannels();
}

async function loadNet() {
  const f = S.frames[S.pos];
  if (!f) return;
  const seq = ++S.netSeq;
  const r = await fetch(`/api/net/${S.run.id}/${S.video}/${f.i}`);
  if (seq !== S.netSeq) return;
  if (!r.ok) { $("stNote").textContent = "не удалось получить этапы сети"; return; }
  const body = await r.json();
  if (seq !== S.netSeq) return;
  S.stages = body.stages;
  if (S.stage >= S.stages.length) S.stage = 0;
  drawPipe();
  showStage();
}

const cap = (s) => s[0].toUpperCase() + s.slice(1);

// Этапы группируются по блоку сети: «блок 2 · свёртка 1» попадает в «блок 2»,
// одиночные этапы вроде входа и пулинга идут каждый своей группой.
function drawPipe() {
  const groups = [];
  S.stages.forEach((st, i) => {
    const label = st.name.split(" · ")[0];
    const last = groups[groups.length - 1];
    if (last && last.label === label) last.ids.push(i);
    else groups.push({ label, ids: [i] });
  });
  $("pipe").innerHTML = groups.map((g) =>
    `<div class="grp${g.ids.includes(S.stage) ? " cur" : ""}"><div>` +
    g.ids.map((i) => `<button data-i="${i}" title="${S.stages[i].name}" class="` +
      `${i === S.stage ? "cur" : i < S.stage ? "done" : ""}"></button>`).join("") +
    `</div><span>${g.label}</span></div>`).join("");
  $("pipe").querySelectorAll("button").forEach((el) => {
    el.onclick = () => { S.stage = +el.dataset.i; S.chan = 0; drawPipe(); showStage(); };
  });
}

function gotoStage(d) {
  if (!S.stages.length) return;
  S.stage = Math.max(0, Math.min(S.stages.length - 1, S.stage + d));
  S.chan = 0;
  drawPipe();
  showStage();
}

function stageUrl(st, what) {
  const f = S.frames[S.pos];
  return `/img/net/${S.run.id}/${S.video}/${f.i}/${st.id}/${what}`;
}

function stageImg(st) {
  const f = S.frames[S.pos];
  return st.id === "input"
    ? `/img/scenefull/${S.run.id}/${S.video}/${f.i}.png`
    : stageUrl(st, "grid.png");
}

function showStage() {
  const st = S.stages[S.stage];
  if (!st) return;
  const prev = S.stage > 0 ? S.stages[S.stage - 1] : null;
  const inSt = prev || st;

  $("stName").textContent = cap(st.name);
  $("stShape").textContent = st.shape.join(" × ");
  $("inName").textContent = prev ? prev.name : "кадр";
  $("inShape").textContent = inSt.shape.join(" × ");
  $("inGrid").src = stageImg(inSt);
  $("outName").textContent = st.name;
  $("outShape").textContent = st.shape.join(" × ");
  $("outGrid").src = stageImg(st);

  const kind = st.id === "pool" ? "pool" : st.kind;
  let extra = "";
  if (st.kind === "conv") {
    extra = st.stride > 1
      ? ` Шаг ${st.stride} уменьшает картинку вдвое, поэтому дальше каждая ` +
        "точка охватывает больший кусок кадра."
      : " Шаг 1: размер не меняется, свёртка только пересобирает признаки.";
    if (prev && prev.shape[0] !== st.shape[0]) {
      extra += ` Каналов стало ${st.shape[0]} вместо ${prev.shape[0]}.`;
    }
  }
  $("stNote").textContent = `${cap(st.note)}.`;
  $("explain").innerHTML =
    `${WHY[kind] || ""}${extra} Форма: <b>${st.shape[0]}</b> карт признаков по ` +
    `<b>${st.shape[1]}×${st.shape[2]}</b>; ` +
    `${st.kind === "act" ? "ненулевых" : "положительных"} значений ` +
    `<b>${Math.round(st.alive * 100)}%</b>.`;
  showChan();
}

async function showChan() {
  const st = S.stages[S.stage];
  if (!st) return;
  S.chan = Math.max(0, Math.min(st.channels - 1, S.chan));
  const f = S.frames[S.pos];
  $("chNo").textContent = S.chan + 1;
  $("chTotal").textContent = `/ ${st.channels}`;
  $("chBig").src = stageUrl(st, `ch${S.chan}.png`);

  const isConv = st.kind === "conv";
  $("kernBox").classList.toggle("hidden", !isConv);
  if (isConv) $("kernImg").src = `/img/kernel/${S.run.id}/${st.id}/${S.chan}.png`;

  const seq = ++S.chanSeq;
  const r = await fetch(`/api/chan/${S.run.id}/${S.video}/${f.i}/${st.id}`);
  if (seq !== S.chanSeq || !r.ok) return;
  const all = await r.json();
  if (seq !== S.chanSeq) return;
  const stats = all[S.chan];
  if (!stats) return;
  // До ReLU отрицательные значения осмысленны, после — обнулены, поэтому
  // одна и та же доля называется по-разному.
  const lbl = (st.kind === "act" || st.id === "pool")
    ? "доля ненулевых" : "доля положительных";
  const pc = stats.live * 100;
  $("chStats").innerHTML =
    `<span><i>ср.</i> ${stats.mean}</span>` +
    `<span><i>макс.</i> ${stats.max}</span>` +
    `<span title="${lbl}"><i>живых</i> ${pc < 1 ? pc.toFixed(2) : pc.toFixed(0)}%</span>`;
}

function pickChannel(ev) {
  const st = S.stages[S.stage];
  if (!st || !st.grid) return;
  const img = $("outGrid"), r = img.getBoundingClientRect();
  const k = img.naturalWidth / r.width;              // лист может быть сжат по ширине
  const x = (ev.clientX - r.left) * k, y = (ev.clientY - r.top) * k;
  const g = st.grid;
  const c = Math.floor((x - g.pad) / (g.cw + g.pad));
  const rr = Math.floor((y - g.pad) / (g.ch + g.pad));
  const i = rr * g.cols + c;
  if (i >= 0 && i < st.channels) { S.chan = i; showChan(); }
}

function wireNet() {
  $("mEnd").onclick = () => setMode("end");
  $("mNet").onclick = () => setMode("net");
  $("mCh").onclick = () => setMode("chan");
  $("outGrid").onclick = pickChannel;
  $("stPrev").onclick = () => gotoStage(-1);
  $("stNext").onclick = () => gotoStage(1);
  $("chPrev").onclick = () => { S.chan--; showChan(); };
  $("chNext").onclick = () => { S.chan++; showChan(); };
  $("whyBtn").onclick = () => {
    S.why = !S.why;
    $("explain").classList.toggle("hidden", !S.why);
    $("whyBtn").textContent = S.why ? "скрыть" : "подробнее";
  };
}


// Какой это матч и можно ли верить числам. Без этой строки в интерфейсе виден
// только идентификатор вида 4AIX8QtRid4, и ошибка в 22 единицы выглядит
// достижением, хотя означает лишь, что ролик был в обучении.
function showMeta(vid, train) {
  const m = (S.meta || {})[vid] || {};
  const bits = [];
  bits.push(train
    ? '<span class="warn">ОБУЧАЮЩИЙ · модель эти кадры видела</span>'
    : '<span class="held">ОТЛОЖЕННЫЙ · модель этих кадров не видела</span>');
  if (m.title) bits.push(`<b>${m.title}</b>`);
  if (m.role) bits.push(`роль ${m.role}`);
  if (m.side && m.side !== "?") {
    bits.push(`сторона ${m.side}${m.side_sure ? "" : " (вероятно)"}`);
  }
  bits.push(`<code>${vid}</code>`);
  if (train) bits.push("числа ниже — это память, а не предсказание");
  $("vmeta").innerHTML = bits.join(" · ");
}


// ---------- режим «Каналы»: витрина, поиск, плеер ----------
// Выбирать следующий ролик по идентификатору было неудобно: не видно ни
// канала, ни матча, ни того, брали мы его. Здесь каталог каналов (по одному
// на чемпиона), поиск, список роликов канала и плеер YouTube — ролик при
// этом никуда не скачивается.

const CH = {
  all: [], have: {}, handle: null, url: null, vids: [], sel: null,
  q: "", vq: "", role: null, onlyMine: false, limit: 60, timer: null,
  vseq: 0, src: "",
};
const VROLES = ["Top", "Jungle", "Mid", "ADC", "Support"];

const dur = (s) => (s == null ? "" :
  `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`);
const esc = (s) => String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

async function openChannels() {
  if (!CH.all.length) {
    const d = await (await fetch("/api/channels")).json();
    CH.all = d.channels;
    CH.have = d.have || {};
    drawChannels();
    drawJobs(d.jobs || []);
  }
  pollJobs();
}

function chanRows() {
  const q = CH.q.trim().toLowerCase();
  return CH.all.filter((c) => {
    if (CH.onlyMine && !c.mine) return false;
    if (!q) return true;
    return (c.name || "").toLowerCase().includes(q) ||
      (c.handle || "").toLowerCase().includes(q) ||
      (c.title || "").toLowerCase().includes(q);
  });
}

function drawChannels() {
  const rows = chanRows();
  $("chCount").textContent = rows.length === CH.all.length
    ? `${CH.all.length}` : `${rows.length} из ${CH.all.length}`;
  $("chList").innerHTML = rows.map((c) =>
    `<button class="crow${c.handle === CH.handle ? " cur" : ""}" ` +
    `data-h="${esc(c.handle)}" data-u="${esc(c.url)}" title="${esc(c.handle)}">` +
    `<span class="cname">${esc(c.name)}</span>` +
    (c.mine ? `<span class="badge mine" title="роликов этого канала в датасете">${c.mine}</span>` : "") +
    (c.custom ? `<span class="badge own" title="добавлен вами">свой</span>` : "") +
    `<span class="grow"></span><i>${esc(c.videos || c.subs || "")}</i>` +
    (c.custom ? `<span class="x" data-drop="${esc(c.handle)}" title="убрать">×</span>` : "") +
    `</button>`).join("");
  $("chList").querySelectorAll(".crow").forEach((el) => {
    el.onclick = (ev) => {
      const drop = ev.target.closest("[data-drop]");
      if (drop) { ev.stopPropagation(); return dropChannel(drop.dataset.drop); }
      openChannel(el.dataset.h, el.dataset.u);
    };
  });
}

async function openChannel(handle, url, refresh) {
  CH.handle = handle; CH.url = url;
  drawChannels();
  const c = CH.all.find((x) => x.handle === handle) || {};
  $("vidsOf").textContent = c.name || handle;
  $("vidsCount").textContent = "читаю…";
  $("vidList").innerHTML = `<div class="msg">запрашиваю список у YouTube…</div>`;
  const seq = ++CH.vseq;
  const qs = new URLSearchParams({ handle, limit: String(CH.limit) });
  if (url) qs.set("url", url);
  if (refresh) qs.set("refresh", "1");
  const r = await fetch(`/api/channel?${qs}`);
  if (seq !== CH.vseq) return;
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    $("vidsCount").textContent = "";
    $("vidList").innerHTML =
      `<div class="msg bad">не вышло: ${esc(e.error || r.status)}</div>`;
    return;
  }
  const d = await r.json();
  if (seq !== CH.vseq) return;
  CH.vids = d.items || [];
  CH.src = d.source;
  drawVids();
}

function vidRows() {
  const q = CH.vq.trim().toLowerCase();
  return CH.vids.filter((v) => {
    if (CH.role && v.role !== CH.role) return false;
    if (!q) return true;
    return (v.title || "").toLowerCase().includes(q) ||
      (v.vs || "").toLowerCase().includes(q) ||
      (v.patch || "").includes(q) || (v.id || "").toLowerCase() === q;
  });
}

function drawVids() {
  const rows = vidRows();
  $("vidsCount").textContent =
    `${rows.length}${rows.length !== CH.vids.length ? ` из ${CH.vids.length}` : ""}` +
    (CH.src ? ` · ${CH.src}` : "");
  $("roleRow").innerHTML =
    `<button class="pill${CH.role ? "" : " on"}" data-r="">все</button>` +
    VROLES.map((r) => {
      const n = CH.vids.filter((v) => v.role === r).length;
      return `<button class="pill${CH.role === r ? " on" : ""}" data-r="${r}"` +
        `${n ? "" : " disabled"}>${r}<i> ${n}</i></button>`;
    }).join("");
  $("roleRow").querySelectorAll(".pill").forEach((el) => {
    el.onclick = () => { CH.role = el.dataset.r || null; drawVids(); };
  });

  $("vidList").innerHTML = rows.map((v) => {
    const h = CH.have[v.id];
    const tags = [
      v.role ? `<span class="tag">${esc(v.role)}</span>` : "",
      v.vs ? `<span class="tag dim">vs ${esc(v.vs)}</span>` : "",
      v.patch ? `<span class="tag dim">${esc(v.patch)}</span>` : "",
      v.region ? `<span class="tag dim">${esc(v.region)}</span>` : "",
      h && h.frames ? `<span class="tag have">в датасете · ${h.frames} кадров</span>`
        : h ? `<span class="tag dim">описание есть</span>` : "",
    ].join("");
    return `<button class="vrow${v.id === CH.sel ? " cur" : ""}" data-id="${v.id}">` +
      `<img loading="lazy" src="${esc(v.thumb || "")}" alt="">` +
      `<span class="vtxt"><span class="vt">${esc(v.title)}</span>` +
      `<span class="vtags">${tags}</span></span>` +
      `<span class="vdur">${dur(v.duration)}</span></button>`;
  }).join("") || `<div class="msg">ничего не нашлось</div>`;
  $("vidList").querySelectorAll(".vrow").forEach((el) => {
    el.onclick = () => selectVideo(el.dataset.id);
  });
}

// ---------- плеер и карточка ролика ----------

function selectVideo(id) {
  CH.sel = id;
  drawVids();
  const v = CH.vids.find((x) => x.id === id) || {};
  $("pId").textContent = id;
  $("pYt").href = `https://www.youtube.com/watch?v=${id}`;
  // Плеер в iframe: ролик идёт с серверов YouTube, у нас он не оседает.
  $("ytbox").innerHTML =
    `<iframe src="https://www.youtube-nocookie.com/embed/${id}?rel=0" ` +
    `title="ролик" allow="accelerometer; autoplay; clipboard-write; ` +
    `encrypted-media; picture-in-picture" allowfullscreen loading="lazy"></iframe>`;
  const run = S.runs.find((r) => r.videos.includes(id));
  $("pOpen").disabled = !run;
  $("pOpen").title = run ? `прогон ${run.id}`
    : "предсказаний для этого ролика пока нет: сначала возьмите его в датасет и обучите модель";
  const h = CH.have[id];
  // Ролик уже в датасете — кнопка не исчезает: кадры можно пересобрать с
  // другой частотой, но подпись должна предупреждать, что это перезапись.
  const taken = !!(h && h.frames);
  $("pTake").textContent = taken ? "пересобрать кадры" : "взять в датасет потоком";
  $("pTake").classList.toggle("go", !taken);
  $("pmeta").innerHTML =
    `<div class="pline"><b>${esc(v.title || id)}</b></div>` +
    `<div class="pline"><i>длительность</i> ${dur(v.duration)}` +
    (h && h.frames
      ? ` · <span class="tag have">уже в датасете, ${h.frames} кадров</span>` : "") +
    `</div><div class="pline" id="pSide"><i>сторона</i> узнаю по описанию…</div>`;
  loadSide(id);
}

async function loadSide(id) {
  const r = await fetch(`/api/video/${id}`);
  if (CH.sel !== id) return;
  const box = $("pSide");
  if (!box) return;
  if (!r.ok) { box.innerHTML = `<i>сторону определить не удалось</i>`; return; }
  const a = await r.json();
  const bits = [];
  if (a.role) bits.push(`<i>роль</i> ${esc(a.role)}`);
  bits.push(`<i>итог</i> ${esc(a.result)}`);
  bits.push(`<i>сторона</i> <b>${esc(a.side_ru)}</b>` +
    (a.confident ? "" : ` <span class="tag dim">вероятно</span>`));
  if (a.size) bits.push(`<i>кадр</i> ${esc(a.size)}`);
  box.innerHTML = bits.join(" · ") + `<div class="msg">${esc(a.why)}</div>`;
}

function openInViewer(id) {
  const run = S.runs.find((r) => r.videos.includes(id));
  if (!run) return;
  setMode("end");
  $("run").value = run.id;
  selectRun(run.id).then(() => { $("video").value = id; loadFrames(id); });
}

// ---------- забор ролика и ход работ ----------

async function take() {
  if (!CH.sel) return;
  const fps = Number($("pFps").value);
  const r = await fetch("/api/ingest", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: CH.sel, fps }),
  });
  const d = await r.json();
  if (!r.ok) {
    $("jobs").innerHTML = `<div class="msg bad">${esc(d.error)}</div>`;
    return;
  }
  pollJobs(true);
}

function drawJobs(jobs) {
  if (!jobs.length) { $("jobs").innerHTML = ""; return; }
  $("jobs").innerHTML = `<div class="jhead">сборка кадров</div>` + jobs.map((j) =>
    `<div class="job ${j.state === "идёт" ? "run" : j.state === "готово" ? "ok" : "bad"}">` +
    `<div class="jtop"><code>${esc(j.id)}</code><i>${esc(j.state)}</i>` +
    `<span class="grow"></span><i>${j.fps} кадр/с</i></div>` +
    `<pre>${esc(j.log)}</pre></div>`).join("");
}

async function pollJobs(force) {
  const d = await (await fetch("/api/jobs")).json();
  CH.have = d.have || CH.have;
  drawJobs(d.jobs || []);
  if (CH.vids.length) drawVids();
  const running = (d.jobs || []).some((j) => j.state === "идёт");
  clearTimeout(CH.timer);
  // Опрашиваем, только пока что-то идёт и витрина открыта: греть сервер
  // пустыми запросами незачем.
  if (running && S.mode === "chan") CH.timer = setTimeout(pollJobs, 4000);
  else if (force && S.mode === "chan") CH.timer = setTimeout(pollJobs, 2000);
}

// ---------- добавление канала ----------

async function addChannel() {
  const url = $("chAdd").value.trim();
  if (!url) return;
  $("chMsg").textContent = "спрашиваю YouTube…";
  const r = await fetch("/api/channels/add", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  const d = await r.json();
  if (!r.ok) {
    $("chMsg").innerHTML = `<span class="bad">${esc(d.error)}</span>`;
    return;
  }
  $("chAdd").value = "";
  $("chMsg").textContent = `добавлен ${d.name || d.handle}`;
  CH.all = [];
  await openChannels();
  openChannel(d.handle, d.url);
}

async function dropChannel(handle) {
  await fetch("/api/channels/drop", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ handle }),
  });
  CH.all = [];
  if (CH.handle === handle) { CH.handle = null; CH.vids = []; drawVids(); }
  await openChannels();
}

function wireChannels() {
  $("chSearch").oninput = (e) => { CH.q = e.target.value; drawChannels(); };
  $("vSearch").oninput = (e) => { CH.vq = e.target.value; drawVids(); };
  $("chOnlyMine").onclick = () => {
    CH.onlyMine = !CH.onlyMine;
    $("chOnlyMine").classList.toggle("on", CH.onlyMine);
    drawChannels();
  };
  $("chAddBtn").onclick = addChannel;
  $("chAdd").onkeydown = (e) => { if (e.key === "Enter") addChannel(); };
  $("vLimit").onchange = (e) => {
    CH.limit = Number(e.target.value);
    if (CH.handle) openChannel(CH.handle, CH.url);
  };
  $("vRefresh").onclick = () => CH.handle && openChannel(CH.handle, CH.url, true);
  $("pTake").onclick = take;
  $("pOpen").onclick = () => CH.sel && openInViewer(CH.sel);
}
