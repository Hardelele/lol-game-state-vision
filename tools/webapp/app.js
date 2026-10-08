"use strict";
// Смотрелка предсказаний координат камеры. Данные берутся по HTTP у
// tools/serve_inspect.py, картинки грузятся по мере надобности, поэтому
// доступны все кадры ролика, а не выборка.

const $ = (id) => document.getElementById(id);
const BIG = 0.133;   // промах больше половины вьюпорта считаем крупным
const GOOD = 0.05;
const UNSURE = 0.15; // разброс ответа, выше которого модель сомневается

const S = {
  runs: [], run: null, video: null, frames: [], pos: 0,
  proj: null, mapUnits: 14800, cells: [], cellSpec: "8x5",
  playing: null, trail: 40,
  mode: "end",        // "end" — сквозной режим, "net" — разбор по слоям
  stages: [], stage: 0, chan: 0,
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

// ---------- загрузка ----------

async function boot() {
  const idx = await (await fetch("/api/index")).json();
  S.runs = idx.runs;
  S.proj = idx.projection && idx.projection.outline ? idx.projection : null;
  S.mapUnits = idx.map_units;
  $("run").innerHTML = S.runs.map((r) => `<option>${r.id}</option>`).join("");
  $("run").onchange = () => selectRun($("run").value);
  $("video").onchange = () => loadFrames($("video").value);
  await selectRun(S.runs[0].id);
  wire();
}

async function selectRun(id) {
  S.run = S.runs.find((r) => r.id === id);
  $("video").innerHTML = S.run.videos.map((v) => `<option>${v}</option>`).join("");
  // По умолчанию открываем отложенный ролик: на обучающем смотреть нечего,
  // модель эти кадры видела.
  const train = S.run.trained_on || [];
  const first = S.run.videos.find((v) => !train.includes(v)) || S.run.videos[0];
  $("video").value = first;
  await loadFrames(first);
}

async function loadFrames(vid) {
  S.video = vid;
  const d = await (await fetch(`/api/frames/${S.run.id}/${vid}`)).json();
  S.frames = d.frames;
  const train = (S.run.trained_on || []).includes(vid);
  $("holdout").textContent = train ? "обучающий — не отложенный" : "отложенный";
  $("holdout").className = "badge" + (train ? " train" : "");
  await loadCells(S.cellSpec);
  summary();
  go(0);
  drawTimeline();
}

async function loadCells(spec) {
  S.cellSpec = spec;
  S.cells = S.proj ? await (await fetch(`/api/cells/${spec}`)).json() : [];
  buildGrids();
}

function summary() {
  const e = S.frames.map((f) => f.err).sort((a, b) => a - b);
  const med = e[Math.floor(e.length / 2)] || 0;
  const bad = S.frames.filter((f) => f.err >= BIG).length;
  const good = S.frames.filter((f) => f.err < GOOD).length;
  $("stats").innerHTML =
    `кадров <b>${S.frames.length}</b> · медианный промах <b>${med.toFixed(3)}</b> ` +
    `(≈${Math.round(med * S.mapUnits)} ед.) · точных <b class="ok">${good}</b> ` +
    `· крупных промахов <b class="bad">${bad}</b>`;
}

// ---------- фильтры и перемещение ----------

function passes(f) {
  if ($("fErr").checked && f.err < BIG) return false;
  if ($("fUns").checked && f.spread < UNSURE) return false;
  if ($("fLab").checked && !f.label) return false;
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
  drawTimeline();
  film();
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
  $("scene").src = `/img/scene/${S.video}/${String(f.i).padStart(6, "0")}.jpg`;
  $("mini").src = `/img/mini/${S.video}/${String(f.i).padStart(6, "0")}.jpg`;
  $("heatimg").src = `/img/heat/${S.run.id}/${S.video}/${String(f.i).padStart(6, "0")}.png`;
  $("heatimg").style.display = $("heat").checked ? "" : "none";

  $("vTrue").textContent = `${f.cx.toFixed(3)}, ${f.cy.toFixed(3)}`;
  $("vPred").textContent = `${f.px.toFixed(3)}, ${f.py.toFixed(3)}`;
  $("vErr").textContent = f.err.toFixed(3);
  $("vUnits").textContent = Math.round(f.err * S.mapUnits);
  $("vSpread").textContent = f.spread.toFixed(3);
  $("vQ").textContent = f.q.toFixed(2);
  $("vLabel").textContent = f.label || "—";
  $("yt").href = `https://www.youtube.com/watch?v=${S.video}&t=${Math.round(f.t)}s`;
  $("yt").textContent = `▶ ${fmtTime(f.t)} на YouTube`;
  drawMap(f);
}

function poly(pts, ox, oy) {
  return pts.map((p) => `${(ox + p[0]).toFixed(5)},${(oy + p[1]).toFixed(5)}`).join(" ");
}

function drawMap(f) {
  const svg = $("mapSvg");
  const out = S.proj ? S.proj.outline : null;
  let s = "";
  if ($("trail").checked) {
    const a = Math.max(0, S.pos - S.trail);
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
  g.removeAttribute("preserveAspectRatio");
  let s = "";
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const id = `${r + 1}.${c + 1}`;
      const x = c * size[0] / cols, y = r * size[1] / rows;
      s += `<rect class="cellrect" data-cell="${id}" x="${x}" y="${y}" ` +
           `width="${size[0] / cols}" height="${size[1] / rows}"/>`;
      s += `<text class="celllab" x="${x + 3}" y="${y + 10}">${id}</text>`;
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
      if (q && f) {
        const cx = f.cx + q.pts.reduce((a, p) => a + p[0], 0) / 4;
        const cy = f.cy + q.pts.reduce((a, p) => a + p[1], 0) / 4;
        $("vCell").textContent = `${id} → ${cx.toFixed(3)}, ${cy.toFixed(3)}`;
      } else {
        $("vCell").textContent = id;
      }
    };
    el.onmouseleave = () => {
      document.querySelectorAll(".lit").forEach((x) => x.classList.remove("lit"));
      $("vCell").textContent = "—";
    };
  });
}

function applyGrid() {
  document.body.classList.toggle("hidden-grid", !$("grid").checked);
}

// ---------- таймлайн ----------

function drawTimeline() {
  const c = $("tl"), dpr = devicePixelRatio || 1;
  const w = c.clientWidth, h = c.height / dpr || 46;
  c.width = w * dpr; c.height = 46 * dpr;
  const x = c.getContext("2d");
  x.scale(dpr, dpr);
  x.clearRect(0, 0, w, 46);
  const n = S.frames.length || 1;
  const bw = Math.max(1, w / n);
  S.frames.forEach((f, i) => {
    const on = passes(f);
    x.globalAlpha = on ? 1 : 0.18;
    x.fillStyle = f.err < GOOD ? "#3fa863" : (f.err < BIG ? "#c9922f" : "#d6483c");
    const bh = 10 + Math.min(26, f.err / 0.3 * 26);
    x.fillRect(i * w / n, 40 - bh, bw, bh);
    if (f.label) {
      x.globalAlpha = on ? 0.9 : 0.2;
      x.fillStyle = f.label === "top" ? "#8ab4ff" : "#5b6472";
      x.fillRect(i * w / n, 41, bw, 4);
    }
  });
  x.globalAlpha = 1;
  const px = S.pos * w / n;
  x.fillStyle = "#fff";
  x.fillRect(px - 1, 0, Math.max(2, bw), 46);
}

function tlIndex(ev) {
  const c = $("tl"), r = c.getBoundingClientRect();
  return Math.max(0, Math.min(S.frames.length - 1,
    Math.round((ev.clientX - r.left) / r.width * S.frames.length)));
}

// ---------- лента кадров ----------

function film() {
  const box = $("film");
  const a = Math.max(0, S.pos - 10), b = Math.min(S.frames.length, S.pos + 11);
  box.innerHTML = S.frames.slice(a, b).map((f, k) => {
    const i = a + k;
    const cls = f.err < GOOD ? "e-ok" : (f.err < BIG ? "e-mid" : "e-bad");
    return `<figure class="${cls}${i === S.pos ? " cur" : ""}" data-i="${i}">` +
      `<img loading="lazy" src="/img/scene/${S.video}/${String(f.i).padStart(6, "0")}.jpg">` +
      `<figcaption>${fmtTime(f.t)}</figcaption></figure>`;
  }).join("");
  box.querySelectorAll("figure").forEach((el) => {
    el.onclick = () => go(+el.dataset.i);
  });
}

// ---------- управление ----------

function play() {
  if (S.playing) {
    clearInterval(S.playing); S.playing = null;
    $("play").textContent = "▶ играть"; $("play").classList.remove("on");
  } else {
    S.playing = setInterval(() => step(1), 220);
    $("play").textContent = "⏸ пауза"; $("play").classList.add("on");
  }
}

function wire() {
  $("prev").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  $("play").onclick = play;
  $("grid").onchange = applyGrid;
  $("heat").onchange = render;
  $("trail").onchange = render;
  $("cells").onchange = () => loadCells($("cells").value);
  ["fErr", "fUns", "fLab"].forEach((id) => { $(id).onchange = () => { drawTimeline(); }; });

  $("jump").onchange = () => {
    const v = $("jump").value.trim();
    let target;
    if (v.includes(":")) {
      const [m, s] = v.split(":").map(Number);
      const sec = m * 60 + s;
      target = S.frames.reduce((best, f, i) =>
        Math.abs(f.t - sec) < Math.abs(S.frames[best].t - sec) ? i : best, 0);
    } else {
      target = S.frames.findIndex((f) => f.i === Number(v));
    }
    if (target >= 0) go(target);
  };

  let drag = false;
  $("tl").onmousedown = (e) => { drag = true; go(tlIndex(e)); };
  addEventListener("mouseup", () => { drag = false; });
  $("tl").onmousemove = (e) => {
    const i = tlIndex(e), f = S.frames[i];
    if (f) {
      $("tlhover").textContent =
        `кадр ${f.i} · ${fmtTime(f.t)} · промах ${f.err.toFixed(3)} ` +
        `(${Math.round(f.err * S.mapUnits)} ед.) · разброс ${f.spread.toFixed(2)}` +
        (f.label ? ` · метка ${f.label}` : "");
    }
    if (drag) go(i);
  };

  [["hl-true", "lit-true"], ["hl-pred", "lit-pred"], ["hl-link", "lit-link"]]
    .forEach(([cls, flag]) => {
      document.querySelectorAll("tr." + cls).forEach((tr) => {
        tr.onmouseenter = () => document.body.classList.add(flag);
        tr.onmouseleave = () => document.body.classList.remove(flag);
      });
    });

  $("helpBtn").onclick = () => $("help").classList.toggle("hidden");
  $("helpClose").onclick = () => $("help").classList.add("hidden");

  addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    const k = e.key;
    if (k === "ArrowRight") { step(e.shiftKey ? 10 : 1); e.preventDefault(); }
    else if (k === "ArrowLeft") { step(e.shiftKey ? -10 : -1); e.preventDefault(); }
    else if (k === " ") { play(); e.preventDefault(); }
    else if (k === "Home") go(0);
    else if (k === "End") go(S.frames.length - 1);
    else if (k.toLowerCase() === "n") jumpWorst(1);
    else if (k.toLowerCase() === "p") jumpWorst(-1);
    else if (k.toLowerCase() === "g") { $("grid").checked = !$("grid").checked; applyGrid(); }
    else if (k.toLowerCase() === "h") { $("heat").checked = !$("heat").checked; render(); }
    else if (k.toLowerCase() === "t") { $("trail").checked = !$("trail").checked; render(); }
    else if (k === "Escape") $("help").classList.add("hidden");
  });

  addEventListener("resize", drawTimeline);
}

boot();


// ---------- режим «по слоям» ----------

function setMode(m) {
  S.mode = m;
  $("mEnd").classList.toggle("on", m === "end");
  $("mNet").classList.toggle("on", m === "net");
  document.querySelector("main").classList.toggle("hidden", m !== "end");
  $("stripbox").classList.toggle("hidden", m !== "end");
  $("netview").classList.toggle("hidden", m !== "net");
  if (m === "net") loadNet();
}

async function loadNet() {
  const f = S.frames[S.pos];
  if (!f) return;
  const seq = ++S.netSeq;
  const r = await fetch(`/api/net/${S.run.id}/${S.video}/${f.i}`);
  if (seq !== S.netSeq) return;
  if (!r.ok) { $("explain").textContent = "не удалось получить этапы сети"; return; }
  const body = await r.json();
  if (seq !== S.netSeq) return;
  S.stages = body.stages;
  if (S.stage >= S.stages.length) S.stage = 0;
  drawPipe();
  showStage();
}

function drawPipe() {
  $("stNum").textContent = `${S.stage + 1} / ${S.stages.length}`;
  $("pipe").innerHTML = S.stages.map((st, i) => {
    const sh = st.shape;
    return (i ? '<span class="arrow">→</span>' : "") +
      `<div class="st${i === S.stage ? " cur" : ""}" data-i="${i}">` +
      `<b>${st.name}</b><span>${sh[0]}×${sh[1]}×${sh[2]}</span></div>`;
  }).join("");
  $("pipe").querySelectorAll(".st").forEach((el) => {
    el.onclick = () => { S.stage = +el.dataset.i; S.chan = 0; drawPipe(); showStage(); };
  });
  const cur = $("pipe").querySelector(".st.cur");
  if (cur) cur.scrollIntoView({ block: "nearest", inline: "center" });
}

function gotoStage(d) {
  S.stage = Math.max(0, Math.min(S.stages.length - 1, S.stage + d));
  S.chan = 0;
  drawPipe();
  showStage();
}

function stageUrl(st, what) {
  const f = S.frames[S.pos];
  return `/img/net/${S.run.id}/${S.video}/${f.i}/${st.id}/${what}`;
}

function showStage() {
  const st = S.stages[S.stage];
  if (!st) return;
  const prev = S.stage > 0 ? S.stages[S.stage - 1] : null;
  const f = S.frames[S.pos];

  if (prev) {
    $("inTitle").textContent = `что пришло — ${prev.name}`;
    $("inShape").textContent = prev.shape.join(" × ");
    $("inGrid").src = prev.id === "input"
      ? `/img/scenefull/${S.run.id}/${S.video}/${f.i}.png`
      : stageUrl(prev, "grid.png");
  } else {
    $("inTitle").textContent = "что пришло — кадр";
    $("inShape").textContent = st.shape.join(" × ");
    $("inGrid").src = `/img/scenefull/${S.run.id}/${S.video}/${f.i}.png`;
  }

  $("outTitle").textContent = `что получилось — ${st.name}`;
  $("outShape").textContent = st.shape.join(" × ");
  $("outGrid").src = st.id === "input"
    ? `/img/scenefull/${S.run.id}/${S.video}/${f.i}.png`
    : stageUrl(st, "grid.png");

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
  $("explain").innerHTML =
    `<b>${st.name}.</b> ${st.note}.${extra} ${WHY[kind] || ""} ` +
    `Форма: <b>${st.shape[0]}</b> карт признаков по ` +
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
  $("chNo").textContent = `${S.chan + 1} из ${st.channels}`;
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
    `<tr><td>среднее</td><td>${stats.mean}</td></tr>` +
    `<tr><td>максимум</td><td>${stats.max}</td></tr>` +
    `<tr><td>${lbl}</td><td>${pc < 1 ? pc.toFixed(2) : pc.toFixed(0)}%</td></tr>`;
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
  $("outGrid").onclick = pickChannel;
  $("stPrev").onclick = () => gotoStage(-1);
  $("stNext").onclick = () => gotoStage(1);
  $("chPrev").onclick = () => { S.chan--; showChan(); };
  $("chNext").onclick = () => { S.chan++; showChan(); };
  addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "Tab") { setMode(S.mode === "end" ? "net" : "end"); e.preventDefault(); }
    if (S.mode !== "net") return;
    if (e.key === "[") gotoStage(-1);
    if (e.key === "]") gotoStage(1);
  });
}
wireNet();
