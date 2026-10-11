"use strict";
// Разметка эталона юнитов (tools/label_units_gold.py). Координаты меток —
// пиксели кадра 1920×1080; svg поверх картинки с тем же viewBox, поэтому
// зум и сдвиг — один CSS transform у обёртки.

const $ = (id) => document.getElementById(id);
const KIND = {
  champion: { key: "1", name: "чемпион", r: 26, shape: "circle" },
  minion: { key: "2", name: "миньон", r: 13, shape: "circle" },
  ward: { key: "3", name: "вард", r: 12, shape: "diamond" },
  monster: { key: "4", name: "монстр", r: 20, shape: "square" },
  other: { key: "5", name: "прочее", r: 13, shape: "triangle" },
};
const TEAM = { ORDER: "синие", CHAOS: "красные", NEUTRAL: "нейтр." };
const S = {
  frames: [], labels: {}, cur: 0, units: [], sel: -1, brush: "minion",
  undo: [], dirty: false, t0: 0, zoom: 1, px: 0, py: 0, hide: false, drag: null,
};

// ---------- данные ----------

async function load() {
  const d = await (await fetch("/api/queue")).json();
  S.frames = d.frames;
  S.labels = d.labels;
  const first = S.frames.findIndex((f) => !done(f.frame));
  buildBrushes();
  buildLegend();
  open(first < 0 ? S.frames.length - 1 : first);
}

const done = (frame) => ["done", "skip"].includes((S.labels[frame] || {}).status);

function open(i) {
  if (i < 0 || i >= S.frames.length) return;
  S.cur = i;
  const f = S.frames[i];
  const lab = S.labels[f.frame];
  // Своя разметка (даже черновик) важнее предразметки сети.
  S.units = lab ? lab.units.map((u) => ({ ...u, pred: false }))
    : f.pred.map((u) => ({ kind: u.kind, team: u.team, x: u.x, y: u.y, pred: true }));
  S.sel = -1;
  S.undo = [];
  S.dirty = false;
  S.t0 = performance.now();
  $("img").src = `/frame/${i}.jpg`;
  resetZoom();
  render();
  // Следующий кадр — в кэш браузера заранее: переход без ожидания.
  if (i + 1 < S.frames.length) new Image().src = `/frame/${i + 1}.jpg`;
}

async function save(status) {
  const f = S.frames[S.cur];
  const prev = S.labels[f.frame];
  const sec = (prev ? prev.sec || 0 : 0) + (performance.now() - S.t0) / 1000;
  const body = { frame: f.frame, status, sec,
    units: S.units.map(({ kind, team, x, y }) => ({ kind, team, x, y })) };
  try {
    const r = await fetch("/api/save", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error);
    S.labels[f.frame] = { ...body, saved: d.saved };
    S.dirty = false;
    $("saved").textContent = `сохранено ${d.saved.slice(11)}`;
    return true;
  } catch (e) {
    toast(`не сохранилось: ${e.message}`, true);
    return false;
  }
}

async function leave(to, status) {
  // Уходя с кадра, ничего не теряем: правки — хотя бы черновиком.
  if (status) {
    if (!(await save(status))) return;
  } else if (S.dirty) {
    const st = done(S.frames[S.cur].frame) ? S.labels[S.frames[S.cur].frame].status : "draft";
    if (!(await save(st))) return;
  }
  if (to >= S.frames.length) {
    toast("Это был последний кадр. Спасибо!");
    render();
    return;
  }
  open(to);
}

// ---------- правка ----------

function push() {
  S.undo.push(JSON.stringify(S.units));
  if (S.undo.length > 200) S.undo.shift();
  S.dirty = true;
}

function add(x, y, team) {
  push();
  const kind = S.brush;
  S.units.push({ kind, team: kind === "monster" ? "NEUTRAL" : team, x, y, pred: false });
  S.sel = S.units.length - 1;
  render();
}

function remove(k) {
  if (k < 0) return;
  push();
  S.units.splice(k, 1);
  S.sel = -1;
  render();
}

function setBrush(kind) {
  S.brush = kind;
  if (S.sel >= 0) {
    push();
    const u = S.units[S.sel];
    u.kind = kind;
    if (kind === "monster") u.team = "NEUTRAL";
    else if (u.team === "NEUTRAL") u.team = "ORDER";
  }
  render();
}

function flipTeam() {
  if (S.sel < 0) return;
  push();
  const u = S.units[S.sel];
  const order = ["ORDER", "CHAOS", "NEUTRAL"];
  u.team = order[(order.indexOf(u.team) + 1) % 3];
  render();
}

// ---------- отрисовка ----------

function shape(u, k) {
  const { r, shape: sh } = KIND[u.kind];
  const cls = `m ${u.team} ${u.pred ? "pred" : ""} ${k === S.sel ? "sel" : ""}`;
  const at = `data-k="${k}" class="${cls}"`;
  const { x, y } = u;
  if (sh === "circle") return `<circle ${at} cx="${x}" cy="${y}" r="${r}"/>`;
  if (sh === "square") return `<rect ${at} x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" rx="3"/>`;
  if (sh === "diamond") return `<path ${at} d="M${x} ${y - r}L${x + r} ${y}L${x} ${y + r}L${x - r} ${y}Z"/>`;
  return `<path ${at} d="M${x} ${y - r}L${x + r} ${y + r * .8}L${x - r} ${y + r * .8}Z"/>`;
}

function render() {
  const f = S.frames[S.cur];
  // Под каждой меткой — тёмная подложка: цвет команды не теряется на
  // красной / синей обводке юнита в самой игре.
  $("marks").innerHTML = S.units.map((u, k) =>
    shape(u, -2).replace('class="m', 'class="halo m') + shape(u, k) +
    `<circle class="dot" cx="${u.x}" cy="${u.y}" r="3.5"/>`).join("");
  $("marks").classList.toggle("hidden", S.hide);
  const nDone = S.frames.filter((g) => (S.labels[g.frame] || {}).status === "done").length;
  const nSkip = S.frames.filter((g) => (S.labels[g.frame] || {}).status === "skip").length;
  $("pos").textContent = `${S.cur + 1}`;
  $("posOf").textContent = ` / ${S.frames.length} · готово ${nDone}` + (nSkip ? ` · пропущено ${nSkip}` : "");
  $("vid").textContent = `${f.video} · ${f.frame.split("_t").pop().replace(".jpg", "")} с`;
  const st = (S.labels[f.frame] || {}).status;
  $("status").textContent = st === "done" ? "✓ готов" : st === "skip" ? "пропущен"
    : st === "draft" ? "черновик" : "предразметка сети — проверь";
  $("status").className = `badge ${st || "pred"}`;
  const n = (kind, team) => S.units.filter((u) => u.kind === kind && u.team === team).length;
  $("counts").innerHTML = ["champion", "minion", "ward"].map((k) =>
    `<span title="${KIND[k].name}: синие / красные">${KIND[k].name} ` +
    `<b class="ORDER">${n(k, "ORDER")}</b>/<b class="CHAOS">${n(k, "CHAOS")}</b></span>`).join("");
  for (const b of document.querySelectorAll(".brush")) b.classList.toggle("on", b.dataset.kind === S.brush);
  renderStrip();
}

function renderStrip() {
  const strip = $("strip");
  if (strip.childElementCount !== S.frames.length) {
    strip.innerHTML = S.frames.map((g, i) => `<i data-i="${i}" title="${i + 1}: ${g.video}"></i>`).join("");
  }
  [...strip.children].forEach((el, i) => {
    const st = (S.labels[S.frames[i].frame] || {}).status || "";
    el.className = `${st}${i === S.cur ? " cur" : ""}`;
  });
}

function buildBrushes() {
  $("brushes").innerHTML = Object.entries(KIND).map(([k, v]) =>
    `<button class="brush" data-kind="${k}" title="кисть: ${v.name} (${v.key})"><kbd>${v.key}</kbd>${v.name}</button>`).join("");
  for (const b of document.querySelectorAll(".brush")) b.onclick = () => setBrush(b.dataset.kind);
}

function buildLegend() {
  $("legend").innerHTML = Object.entries(KIND).map(([k, v]) => {
    const u = { kind: k, team: k === "monster" ? "NEUTRAL" : "ORDER", x: 30, y: 30, pred: false };
    return `<div><svg viewBox="0 0 60 60" width="34" height="34">${shape(u, -2)}</svg>` +
      `<kbd>${v.key}</kbd> ${v.name}${k === "other" ? " — питомцы, клоны, ловушки, растения" : ""}</div>`;
  }).join("");
}

// ---------- зум и сдвиг ----------

function applyView() {
  $("view").style.transform = `translate(${S.px}px, ${S.py}px) scale(${S.zoom})`;
}

function resetZoom() {
  S.zoom = 1;
  S.px = S.py = 0;
  applyView();
}

function framePoint(ev) {
  const r = $("marks").getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width * 1920, (ev.clientY - r.top) / r.height * 1080];
}

$("stage").addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const r = $("stage").getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const z = Math.min(8, Math.max(1, S.zoom * (ev.deltaY < 0 ? 1.25 : 0.8)));
  // Точка под курсором остаётся на месте.
  S.px = mx - (mx - S.px) * (z / S.zoom);
  S.py = my - (my - S.py) * (z / S.zoom);
  S.zoom = z;
  if (z === 1) S.px = S.py = 0;
  applyView();
}, { passive: false });

// ---------- мышь ----------

$("stage").addEventListener("contextmenu", (ev) => ev.preventDefault());

$("stage").addEventListener("mousedown", (ev) => {
  if (ev.button === 1 || (ev.button === 0 && ev.shiftKey)) {
    ev.preventDefault();
    S.drag = { pan: true, x: ev.clientX, y: ev.clientY, px: S.px, py: S.py };
    return;
  }
  const k = ev.target.dataset && ev.target.dataset.k !== undefined ? +ev.target.dataset.k : -1;
  if (ev.button === 2) {
    if (k >= 0) remove(k);
    else add(...framePoint(ev), "CHAOS");
    return;
  }
  if (ev.button !== 0) return;
  if (k >= 0) {
    S.sel = k;
    S.drag = { k, moved: false };
    render();
  } else {
    add(...framePoint(ev), "ORDER");
  }
});

window.addEventListener("mousemove", (ev) => {
  const d = S.drag;
  if (!d) return;
  if (d.pan) {
    S.px = d.px + ev.clientX - d.x;
    S.py = d.py + ev.clientY - d.y;
    applyView();
    return;
  }
  if (!d.moved) { push(); d.moved = true; }
  const [x, y] = framePoint(ev);
  Object.assign(S.units[d.k], { x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10, pred: false });
  render();
});

window.addEventListener("mouseup", () => { S.drag = null; });

// ---------- клавиши (по e.code — работает и в русской раскладке) ----------

window.addEventListener("keydown", (ev) => {
  if (!$("help").classList.contains("hidden")) {
    if (["Escape", "Enter", "Space", "Slash"].includes(ev.code)) { ev.preventDefault(); toggleHelp(false); }
    return;
  }
  const c = ev.code;
  if ((c === "KeyZ" && !ev.shiftKey) || (c === "KeyZ" && ev.ctrlKey)) {
    ev.preventDefault();
    if (S.undo.length) { S.units = JSON.parse(S.undo.pop()); S.sel = -1; S.dirty = true; render(); }
    return;
  }
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const digit = { Digit1: "champion", Digit2: "minion", Digit3: "ward", Digit4: "monster", Digit5: "other",
    Numpad1: "champion", Numpad2: "minion", Numpad3: "ward", Numpad4: "monster", Numpad5: "other" }[c];
  if (digit) setBrush(digit);
  else if (c === "KeyT") flipTeam();
  else if (c === "KeyX" || c === "Delete") remove(S.sel);
  else if (c === "Enter" || c === "Space" || c === "NumpadEnter") { ev.preventDefault(); leave(S.cur + 1, "done"); }
  else if (c === "KeyS") leave(S.cur + 1, "skip");
  else if (c === "ArrowRight") leave(S.cur + 1);
  else if (c === "ArrowLeft" || c === "Backspace") { ev.preventDefault(); leave(S.cur - 1); }
  else if (c === "KeyH") { if (!S.hide) { S.hide = true; render(); } }
  else if (c === "Digit0" || c === "Numpad0") resetZoom();
  else if (c === "Slash" || c === "F1") { ev.preventDefault(); toggleHelp(true); }
  else if (c === "Escape") { S.sel = -1; render(); }
});

window.addEventListener("keyup", (ev) => {
  if (ev.code === "KeyH") { S.hide = false; render(); }
});

window.addEventListener("beforeunload", (ev) => {
  if (S.dirty) { save("draft"); ev.preventDefault(); }
});

// ---------- прочее ----------

function toggleHelp(on) {
  $("help").classList.toggle("hidden", !on);
}

let toastTimer = 0;
function toast(text, bad) {
  const t = $("toast");
  t.textContent = text;
  t.className = `show${bad ? " bad" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = ""; }, 2600);
}

$("bPrev").onclick = () => leave(S.cur - 1);
$("bNext").onclick = () => leave(S.cur + 1);
$("bSkip").onclick = () => leave(S.cur + 1, "skip");
$("bDone").onclick = () => leave(S.cur + 1, "done");
$("bHelp").onclick = () => toggleHelp(true);
$("bHelpClose").onclick = () => toggleHelp(false);
$("strip").addEventListener("click", (ev) => {
  if (ev.target.dataset.i !== undefined) leave(+ev.target.dataset.i);
});

load().then(() => {
  // Первый заход — сразу с правилами.
  try {
    if (!localStorage.getItem("goldHelpSeen")) { toggleHelp(true); localStorage.setItem("goldHelpSeen", "1"); }
  } catch (e) { /* без хранилища — просто без подсказки */ }
});
