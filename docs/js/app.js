"use strict";
/* Interfaz: bucle de simulación, controles, telemetría, curva de aprendizaje y carga de modelos de Python. */

// ═══ App state ═════════════════════════════════════════════════════════
const $ = (id) => document.getElementById(id);
const app = {
  envKey: "cartpole", algo: "dqn", speed: "10", mode: "idle",
  hp: null, env: null, agent: null, obs: null, epReturn: 0, returns: [],
  keys: new Set(), banner: null, bannerUntil: 0,
};

function colors() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n) => cs.getPropertyValue(n).trim();
  return { skyTop: g("--sky-top"), skyBottom: g("--sky-bottom"), ground: g("--ground"), groundEdge: g("--ground-edge"),
    craft: g("--craft"), accent: g("--accent"), warm: g("--warm"), good: g("--good"), bad: g("--bad"),
    muted: g("--muted"), ink: g("--ink"), line: g("--line"), panel: g("--panel") };
}
let COL = colors();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { COL = colors(); drawChart(); });
new MutationObserver(() => { COL = colors(); drawChart(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

function buildAgent() {
  const Env = ENVS[app.envKey];
  app.env = new Env(makeRng(app.hp.seed * 31 + 7));
  app.agent = app.algo === "dqn" ? new DQNAgent(Env, app.hp) : new QLearningAgent(Env, app.hp, app.envKey);
  app.obs = app.env.reset(); app.epReturn = 0; app.returns = [];
  buildQBars(); buildTelemetry(); renderStats(); drawChart();
  const goal = Env.solved;
  $("solved").textContent = `Meta: media de ${goal} en los últimos 100 episodios.`;
  $("solved").dataset.state = "no";
  $("keys-hint").innerHTML = app.envKey === "cartpole"
    ? "Pilotear yo: <kbd>←</kbd> <kbd>→</kbd> empujan el carro."
    : "Pilotear yo: <kbd>↑</kbd> motor principal, <kbd>←</kbd> <kbd>→</kbd> motores laterales.";
  $("btn-pretrained").disabled = !(app.envKey === "cartpole" && app.algo === "dqn");
  $("model-msg").textContent = app.envKey === "cartpole" && app.algo === "dqn" ? "" : "Disponible con CartPole + DQN.";
  $("model-msg").dataset.kind = "";
}

function resetHp() {
  app.hp = { ...PRESETS[app.algo][app.envKey] };
  buildHpForm();
  buildAgent();
}

function buildHpForm() {
  const grid = $("hpgrid"); grid.textContent = "";
  for (const [key, label, type, step] of HP_FIELDS[app.algo]) {
    const lab = document.createElement("label");
    lab.htmlFor = `hp-${key}`; lab.textContent = label;
    let input;
    if (type === "bool") {
      input = document.createElement("select");
      input.innerHTML = '<option value="true">Sí</option><option value="false">No</option>';
      input.value = String(app.hp[key]);
    } else {
      input = document.createElement("input");
      input.type = type; input.value = app.hp[key];
      if (step) input.step = step;
      if (type === "number") input.min = 0;
    }
    input.id = `hp-${key}`;
    input.addEventListener("change", () => {
      let v = input.value;
      if (type === "number") { v = Number(v); if (!Number.isFinite(v) || v < 0) { input.value = app.hp[key]; return; } }
      if (type === "bool") v = v === "true";
      app.hp[key] = v;
      setMode("idle"); buildAgent();
    });
    lab.appendChild(input); grid.appendChild(lab);
  }
}

function buildQBars() {
  const box = $("qbars"); box.textContent = "";
  ENVS[app.envKey].actionNames.forEach((name, i) => {
    const row = document.createElement("div"); row.className = "qbar"; row.id = `q-${i}`;
    row.innerHTML = `<span class="name" title="${name}">${name}</span><span class="track"><span class="fill"></span></span><span class="val">0.00</span>`;
    box.appendChild(row);
  });
}
function renderQ(q, chosen) {
  const n = q.length; let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { lo = Math.min(lo, q[i]); hi = Math.max(hi, q[i]); }
  const span = hi - lo || 1;
  for (let i = 0; i < n; i++) {
    const row = $(`q-${i}`); if (!row) continue;
    row.classList.toggle("best", i === chosen);
    row.querySelector(".fill").style.width = `${8 + 92 * (q[i] - lo) / span}%`;
    row.querySelector(".val").textContent = q[i].toFixed(2);
  }
}

function buildTelemetry() {
  const dl = $("telemetry"); dl.textContent = "";
  ENVS[app.envKey].obsLabels.forEach((lab, i) => {
    const d = document.createElement("div");
    d.innerHTML = `<dt>s[${i}] ${lab}</dt><dd id="obs-${i}">0.000</dd>`;
    dl.appendChild(d);
  });
}
function renderTelemetry() {
  app.obs.forEach((v, i) => { const el = $(`obs-${i}`); if (el) el.textContent = (v >= 0 ? " " : "") + v.toFixed(3); });
}

function mean(a) { return a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN; }
function renderStats() {
  const ag = app.agent, last = app.returns[app.returns.length - 1];
  $("st-ep").textContent = ag.episodes.toLocaleString("es-CO");
  $("st-steps").textContent = ag.steps.toLocaleString("es-CO");
  $("st-eps").textContent = ag.epsilon.toFixed(3);
  $("st-last").textContent = last === undefined ? "–" : last.toFixed(1);
  const avg = mean(app.returns.slice(-100));
  $("st-avg").textContent = Number.isFinite(avg) ? avg.toFixed(1) : "–";
  $("st-extra-l").textContent = ag.extraLabel();
  $("st-extra").textContent = ag.extraValue();
  const goal = ENVS[app.envKey].solved;
  if (app.returns.length >= 100 && avg >= goal) {
    $("solved").textContent = `Resuelto: media de ${avg.toFixed(1)} en los últimos 100 episodios (meta ${goal}).`;
    $("solved").dataset.state = "yes";
  }
}

// ── simulation step ───────────────────────────────────────────────────
function envStep(learn, greedy) {
  const env = app.env, ag = app.agent, s = app.obs;
  const action = app.mode === "manual" ? env.manualAction(app.keys) : ag.act(s, greedy);
  const { obs, reward, terminated, truncated } = env.step(action);
  if (learn) ag.observe(s, action, reward, obs, terminated); // truncated is NOT terminal
  app.obs = obs; app.epReturn += reward; app.lastAction = action;
  if (terminated || truncated) {
    if (learn) { ag.episodeEnd(); app.returns.push(app.epReturn); }
    app.banner = `${env.outcome || "Fin"} · retorno ${app.epReturn.toFixed(1)}`;
    app.bannerUntil = performance.now() + (app.mode === "train" && app.speed !== "1" ? 400 : 1400);
    app.obs = env.reset(); app.epReturn = 0;
    return true;
  }
  return false;
}

let chartDirty = false;
function frame(now) {
  const m = app.mode;
  if (m === "train") {
    const budgetEnd = now + 12;
    if (app.speed === "max") {
      let n = 0; while (performance.now() < budgetEnd && n < 20000) { if (envStep(true, false)) chartDirty = true; n++; }
    } else {
      const n = Number(app.speed);
      for (let i = 0; i < n; i++) if (envStep(true, false)) chartDirty = true;
    }
  } else if (m === "watch" || m === "manual") {
    envStep(false, true);
  }
  render(now);
  requestAnimationFrame(frame);
}

let lastStats = 0;
function render(now) {
  const cv = $("sim"), ctx = cv.getContext("2d");
  // Match the backing store to the displayed size so text stays crisp at any width.
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 900, h = w * 2 / 3;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  app.env.draw(ctx, w, h, COL);
  // HUD
  const fs = Math.max(11, Math.round(w / 48));
  ctx.font = `500 ${fs}px "IBM Plex Mono", ui-monospace, monospace`;
  ctx.fillStyle = COL.ink; ctx.textBaseline = "top";
  const pad = Math.max(10, w / 40);
  ctx.fillText(`${ENVS[app.envKey].id} · ${app.algo === "dqn" ? "DQN" : "Q-Learning"}`, pad, pad);
  ctx.fillStyle = COL.muted;
  ctx.fillText(`paso ${app.env.t}   retorno ${app.epReturn.toFixed(1)}`, pad, pad + fs * 1.4);
  if (app.banner && now < app.bannerUntil) {
    ctx.font = `600 ${Math.max(18, Math.round(w / 24))}px "Saira Condensed", sans-serif`;
    // Result pill in the middle of the scene, clear of the craft's spawn point.
    const tw = ctx.measureText(app.banner).width, fsz = Math.max(18, Math.round(w / 24));
    const bx = w / 2 - tw / 2 - fsz * 0.6, by = h * 0.4, bw = tw + fsz * 1.2, bh = fsz * 1.6;
    ctx.globalAlpha = 0.9; ctx.fillStyle = COL.panel;
    roundRect(ctx, bx, by, bw, bh, bh / 2); ctx.fill();
    ctx.globalAlpha = 1; ctx.strokeStyle = COL.line; ctx.lineWidth = 1; ctx.stroke();
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = COL.ink;
    ctx.fillText(app.banner, w / 2, by + bh / 2 + 1);
    ctx.textAlign = "left"; ctx.textBaseline = "top";
  }
  if (now - lastStats > 90) {
    lastStats = now;
    renderTelemetry(); renderStats();
    const q = app.agent.qValues(app.obs);
    renderQ(q, app.mode === "manual" ? app.lastAction : argmax(q));
    if (chartDirty) { drawChart(); chartDirty = false; }
  }
}

// ── learning curve ────────────────────────────────────────────────────
function drawChart() {
  const cv = $("chart"); if (!cv) return;
  const dpr = window.devicePixelRatio || 1, cw = cv.clientWidth, ch = cv.clientHeight;
  if (cv.width !== Math.round(cw * dpr)) { cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr); }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const data = app.returns, goal = ENVS[app.envKey].solved;
  const padL = 46, padR = 12, padT = 10, padB = 26;
  const W = cw - padL - padR, H = ch - padT - padB;
  let lo = app.envKey === "cartpole" ? 0 : Math.max(-500, Math.min(-200, ...data)), hi = app.envKey === "cartpole" ? 500 : Math.max(goal + 50, ...data);
  const step = niceStep((hi - lo) / 5); lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const n = Math.max(data.length, 10);
  const X = (i) => padL + (i / (n - 1 || 1)) * W, Y = (v) => padT + H - ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * H;
  ctx.font = `500 11px "IBM Plex Mono", monospace`; ctx.textBaseline = "middle"; ctx.textAlign = "right";
  for (let v = lo; v <= hi + 1e-9; v += step) {
    ctx.strokeStyle = COL.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(padL + W, Y(v)); ctx.stroke();
    ctx.fillStyle = COL.muted; ctx.fillText(String(Math.round(v)), padL - 6, Y(v));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const xs = niceStep(n / 6);
  for (let i = 0; i <= n; i += xs) { if (i === 0) continue; ctx.fillText(String(i), X(i - 1), padT + H + 8); }
  ctx.setLineDash([5, 4]); ctx.strokeStyle = COL.good; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(padL, Y(goal)); ctx.lineTo(padL + W, Y(goal)); ctx.stroke(); ctx.setLineDash([]);
  if (!data.length) {
    ctx.fillStyle = COL.muted; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.font = `500 13px "IBM Plex Sans", sans-serif`;
    ctx.fillText(W < 420 ? "Pulsa Entrenar para ver la curva." : "Pulsa Entrenar: cada episodio terminado aparece aquí.", padL + W / 2, padT + H / 2);
    return;
  }
  ctx.fillStyle = COL.muted; ctx.globalAlpha = 0.45;
  const r = data.length > 800 ? 1 : 1.8;
  data.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(i), Y(v), r, 0, Math.PI * 2); ctx.fill(); });
  ctx.globalAlpha = 1;
  ctx.strokeStyle = COL.accent; ctx.lineWidth = 2.2; ctx.beginPath();
  let s = 0; const win = 50;
  data.forEach((v, i) => { s += v; if (i >= win) s -= data[i - win]; const m = s / Math.min(i + 1, win); i ? ctx.lineTo(X(i), Y(m)) : ctx.moveTo(X(i), Y(m)); });
  ctx.stroke();
  const lastI = data.length - 1; let tail = 0; for (let i = Math.max(0, lastI - win + 1); i <= lastI; i++) tail += data[i];
  ctx.fillStyle = COL.accent; ctx.beginPath(); ctx.arc(X(lastI), Y(tail / Math.min(data.length, win)), 4, 0, Math.PI * 2); ctx.fill();
}
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9)))), f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

// ── controls ──────────────────────────────────────────────────────────
const MODE_TEXT = { idle: "En espera", train: "Entrenando", watch: "Política codiciosa", manual: "Piloto manual" };
function setMode(m) {
  app.mode = m;
  $("status").dataset.mode = m; $("status-text").textContent = MODE_TEXT[m];
  $("btn-train").textContent = m === "train" ? "Pausar" : "Entrenar";
  $("btn-watch").setAttribute("aria-pressed", String(m === "watch"));
  $("btn-manual").setAttribute("aria-pressed", String(m === "manual"));
  if (m !== "train") { app.obs = app.env.reset(); app.epReturn = 0; }
}
function segSelect(_unused, attr, value) {
  document.querySelectorAll(`[${attr}]`).forEach((b) => b.setAttribute("aria-pressed", String(b.getAttribute(attr) === value)));
}
document.querySelectorAll("[data-env]").forEach((b) => b.addEventListener("click", () => {
  app.envKey = b.dataset.env; segSelect(null, "data-env", app.envKey); setModeQuiet(); resetHp();
}));
document.querySelectorAll("[data-algo]").forEach((b) => b.addEventListener("click", () => {
  app.algo = b.dataset.algo; segSelect(null, "data-algo", app.algo); setModeQuiet(); resetHp();
}));
document.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => {
  app.speed = b.dataset.speed; segSelect(null, "data-speed", app.speed);
}));
function setModeQuiet() { app.mode = "idle"; $("status").dataset.mode = "idle"; $("status-text").textContent = MODE_TEXT.idle; $("btn-train").textContent = "Entrenar"; $("btn-watch").setAttribute("aria-pressed", "false"); $("btn-manual").setAttribute("aria-pressed", "false"); }
$("btn-train").addEventListener("click", () => {
  if (app.mode === "train") { setMode("idle"); return; }
  if (app.mode !== "idle") { app.obs = app.env.reset(); app.epReturn = 0; }
  app.mode = "train"; setModeLabelsOnly("train");
});
function setModeLabelsOnly(m) {
  $("status").dataset.mode = m; $("status-text").textContent = MODE_TEXT[m];
  $("btn-train").textContent = m === "train" ? "Pausar" : "Entrenar";
  $("btn-watch").setAttribute("aria-pressed", String(m === "watch"));
  $("btn-manual").setAttribute("aria-pressed", String(m === "manual"));
}
$("btn-watch").addEventListener("click", () => setMode(app.mode === "watch" ? "idle" : "watch"));
$("btn-manual").addEventListener("click", () => { setMode(app.mode === "manual" ? "idle" : "manual"); $("sim").focus?.(); });
$("btn-reset").addEventListener("click", () => { setModeQuiet(); buildAgent(); });
$("btn-step").addEventListener("click", () => {
  if (app.mode !== "idle") setMode("idle");
  const prev = app.mode; app.mode = "train"; envStep(true, false); app.mode = prev; chartDirty = true;
});
const GAME_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", " "]);
window.addEventListener("keydown", (e) => {
  if (app.mode !== "manual" || !GAME_KEYS.has(e.key)) return;
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === "INPUT" || tag === "SELECT") return;
  e.preventDefault(); app.keys.add(e.key);
});
window.addEventListener("keyup", (e) => app.keys.delete(e.key));
window.addEventListener("blur", () => app.keys.clear());
window.addEventListener("resize", () => drawChart());

// ── Python-trained model ──────────────────────────────────────────────
function applyModel(spec, source) {
  const msg = $("model-msg");
  try {
    if (!spec || spec.format !== "rl_games-dqn-mlp" || !Array.isArray(spec.layers)) throw new Error("El archivo no es un modelo exportado con 'rlgames export'.");
    if (spec.env_id && spec.env_id !== "CartPole-v1") throw new Error(`Modelo de ${spec.env_id}; este simulador solo reproduce exactamente CartPole-v1.`);
    if (app.envKey !== "cartpole" || app.algo !== "dqn") { app.envKey = "cartpole"; app.algo = "dqn"; segSelect(null, "data-env", "cartpole"); segSelect(null, "data-algo", "dqn"); resetHp(); }
    app.agent.loadLayers(spec.layers);
    setMode("watch");
    msg.dataset.kind = "ok";
    msg.textContent = `${source}: red ${app.agent.online.sizes.join("→")}, ${spec.training_episodes ?? "?"} episodios en Python. Mostrando su política.`;
  } catch (err) {
    msg.dataset.kind = "err"; msg.textContent = err.message;
  }
}
$("btn-pretrained").addEventListener("click", async () => {
  const msg = $("model-msg"); msg.dataset.kind = ""; msg.textContent = "Cargando…";
  try {
    const res = await fetch("models/dqn_CartPole-v1.json");
    if (!res.ok) throw new Error(`No se encontró models/dqn_CartPole-v1.json (HTTP ${res.status}).`);
    applyModel(await res.json(), "Modelo incluido");
  } catch (err) { msg.dataset.kind = "err"; msg.textContent = err.message; }
});
$("file-model").addEventListener("change", async (e) => {
  const f = e.target.files && e.target.files[0]; if (!f) return;
  try { applyModel(JSON.parse(await f.text()), f.name); }
  catch (err) { $("model-msg").dataset.kind = "err"; $("model-msg").textContent = "JSON inválido: " + err.message; }
  e.target.value = "";
});

// ── boot ──────────────────────────────────────────────────────────────
resetHp();
requestAnimationFrame(frame);
