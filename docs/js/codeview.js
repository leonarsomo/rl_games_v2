"use strict";
/* Visor de código: muestra, junto al simulador, el código JavaScript que se está
   ejecutando y su equivalente en el paquete Python, y resalta en vivo las
   funciones activas según el modo (entrenar, ver política, pilotear). */

const REPO_BLOB = "https://github.com/leonarsomo/rl_games_v2/blob/main/";

const FILES = [
  { group: "JavaScript · se ejecuta aquí" },
  { id: "js-agents", label: "agents.js", path: "js/agents.js", repo: "docs/js/agents.js", lang: "javascript", desc: "Agentes DQN y Q-Learning del simulador" },
  { id: "js-nn", label: "nn.js", path: "js/nn.js", repo: "docs/js/nn.js", lang: "javascript", desc: "Red neuronal y optimizador Adam desde cero" },
  { id: "js-envs", label: "envs.js", path: "js/envs.js", repo: "docs/js/envs.js", lang: "javascript", desc: "Física de CartPole y Lunar Lander" },
  { id: "js-app", label: "app.js", path: "js/app.js", repo: "docs/js/app.js", lang: "javascript", desc: "Bucle de simulación e interfaz" },
  { group: "Python · paquete rl_games" },
  { id: "py-dqn", label: "dqn.py", path: "src/rl_games/agents/dqn.py", lang: "python", desc: "DQN en PyTorch (equivale a agents.js + nn.js)" },
  { id: "py-q", label: "qlearning.py", path: "src/rl_games/agents/qlearning.py", lang: "python", desc: "Q-Learning tabular" },
  { id: "py-base", label: "base.py", path: "src/rl_games/agents/base.py", lang: "python", desc: "Bucle de entrenamiento común y política ε-greedy" },
  { id: "py-replay", label: "replay.py", path: "src/rl_games/agents/replay.py", lang: "python", desc: "Buffer de repetición de experiencias" },
  { id: "py-config", label: "config.py", path: "src/rl_games/config.py", lang: "python", desc: "Hiperparámetros validados y presets por entorno" },
];
FILES.filter((f) => f.path && f.path.startsWith("src/")).forEach((f) => { f.repo = f.path; });

// Functions that run in each situation: [fileId, className | null, functionName]
function liveTargets(mode, algo, env) {
  const envClass = env === "cartpole" ? "CartPole" : "LunarLite";
  const t = [];
  if (mode === "train") {
    t.push(["js-app", null, "envStep"], ["js-envs", envClass, "step"], ["py-base", "BaseAgent", "train"]);
    if (algo === "dqn") {
      t.push(["js-agents", "DQNAgent", "act"], ["js-agents", "DQNAgent", "observe"], ["js-agents", "DQNAgent", "learn"],
        ["js-nn", "MLP", "forward"], ["js-nn", "Adam", "backward"], ["js-nn", "Adam", "step"],
        ["py-dqn", "DQNAgent", "_observe"], ["py-dqn", "DQNAgent", "_learn"], ["py-dqn", "DQNAgent", "_on_step_end"],
        ["py-replay", "ReplayBuffer", "push"], ["py-replay", "ReplayBuffer", "sample"]);
    } else {
      t.push(["js-agents", "QLearningAgent", "act"], ["js-agents", "QLearningAgent", "key"], ["js-agents", "QLearningAgent", "observe"],
        ["js-agents", "QLearningAgent", "episodeEnd"],
        ["py-q", "QLearningAgent", "discretize"], ["py-q", "QLearningAgent", "_observe"], ["py-q", "QLearningAgent", "_on_episode_end"],
        ["py-base", "BaseAgent", "select_action"]);
    }
  } else if (mode === "watch") {
    t.push(["js-app", null, "envStep"], ["js-envs", envClass, "step"], ["py-base", "BaseAgent", "select_action"], ["py-base", "BaseAgent", "predict"]);
    if (algo === "dqn") t.push(["js-agents", "DQNAgent", "act"], ["js-agents", "DQNAgent", "qValues"], ["js-nn", "MLP", "forward"], ["py-dqn", "DQNAgent", "_greedy_action"], ["py-dqn", "DQNAgent", "q_values"]);
    else t.push(["js-agents", "QLearningAgent", "act"], ["js-agents", "QLearningAgent", "key"], ["py-q", "QLearningAgent", "_greedy_action"], ["py-q", "QLearningAgent", "discretize"]);
  } else if (mode === "manual") {
    t.push(["js-app", null, "envStep"], ["js-envs", envClass, "manualAction"], ["js-envs", envClass, "step"]);
  }
  return t;
}
// File to show first when "follow" is on.
function preferredFile(mode, algo) {
  if (mode === "manual") return "js-envs";
  if (mode === "idle") return null;
  return "js-agents";
}

// ── source loading ──────────────────────────────────────────────────────
const cache = new Map();
async function loadSource(file) {
  if (cache.has(file.id)) return cache.get(file.id);
  const candidates = file.path.startsWith("src/") ? [file.path, "../" + file.path] : [file.path];
  let lastErr = null;
  for (const url of candidates) {
    try {
      const res = await fetch(url, { cache: "no-cache" });
      if (res.ok) { const text = await res.text(); cache.set(file.id, text); return text; }
      lastErr = new Error(`HTTP ${res.status}`);
    } catch (err) { lastErr = err; }
  }
  throw lastErr || new Error("no disponible");
}

// ── function ranges (1-based, inclusive) ────────────────────────────────
function findRange(lines, lang, cls, fn) {
  let from = 0;
  if (cls) {
    const re = lang === "python" ? new RegExp(`^class ${cls}\\b`) : new RegExp(`^class ${cls}\\b`);
    from = lines.findIndex((l) => re.test(l));
    if (from < 0) return null;
  }
  if (lang === "python") {
    const re = new RegExp(`^(\\s*)def ${fn}\\(`);
    for (let i = from; i < lines.length; i++) {
      const m = re.exec(lines[i]);
      if (!m) continue;
      if (cls && i > from && /^class /.test(lines[i])) return null;
      const indent = m[1].length;
      // A signature can span several lines: skip to the line that closes it.
      let sigEnd = i, depth = 0;
      for (let j = i; j < lines.length; j++) {
        for (const ch of lines[j]) { if (ch === "(" || ch === "[") depth++; else if (ch === ")" || ch === "]") depth--; }
        sigEnd = j;
        if (depth <= 0 && /:\s*(#.*)?$/.test(lines[j])) break;
      }
      let end = sigEnd;
      for (let j = sigEnd + 1; j < lines.length; j++) {
        const l = lines[j];
        if (!l.trim()) continue;
        if (l.length - l.trimStart().length <= indent) break;
        end = j;
      }
      let start = i;
      while (start > 0 && /^\s*@/.test(lines[start - 1])) start--;
      return [start + 1, end + 1];
    }
    return null;
  }
  const re = cls
    ? new RegExp(`^\\s{2}(static\\s+)?(async\\s+)?${fn}\\s*\\([^)]*\\)\\s*\\{`)
    : new RegExp(`^(async\\s+)?function\\s+${fn}\\s*\\(`);
  for (let i = from; i < lines.length; i++) {
    if (cls && i > from && /^class /.test(lines[i])) return null;
    if (!re.test(lines[i])) continue;
    let depth = 0, seen = false;
    for (let j = i; j < lines.length; j++) {
      for (const ch of lines[j]) {
        if (ch === "{") { depth++; seen = true; } else if (ch === "}") depth--;
      }
      if (seen && depth <= 0) return [i + 1, j + 1];
    }
    return [i + 1, i + 1];
  }
  return null;
}

// ── highlighted HTML split per line (keeps multi-line spans balanced) ───
function escapeHtml(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function highlightLines(src, lang) {
  let html;
  try { html = window.hljs ? window.hljs.highlight(src, { language: lang, ignoreIllegals: true }).value : escapeHtml(src); }
  catch { html = escapeHtml(src); }
  const out = [], open = [];
  let cur = "";
  const tagRe = /<\/?span[^>]*>|\n|[^<\n]+/g;
  let m;
  while ((m = tagRe.exec(html))) {
    const tok = m[0];
    if (tok === "\n") {
      out.push(cur + "</span>".repeat(open.length));
      cur = open.join("");
    } else if (tok.startsWith("</span")) { open.pop(); cur += tok; }
    else if (tok.startsWith("<span")) { open.push(tok); cur += tok; }
    else cur += tok;
  }
  out.push(cur + "</span>".repeat(open.length));
  return out;
}

// ── view state ──────────────────────────────────────────────────────────
const view = { fileId: "js-agents", key: "", live: [], rendered: null };

function buildTabs() {
  const box = document.getElementById("filetabs");
  box.textContent = "";
  for (const f of FILES) {
    if (f.group) { const g = document.createElement("span"); g.className = "grp"; g.textContent = f.group; box.appendChild(g); continue; }
    const b = document.createElement("button");
    b.type = "button"; b.id = `tab-${f.id}`; b.setAttribute("role", "tab"); b.textContent = f.label;
    b.setAttribute("aria-selected", String(f.id === view.fileId));
    b.title = f.desc;
    b.addEventListener("click", () => { document.getElementById("code-follow").checked = false; showFile(f.id, true); });
    box.appendChild(b);
  }
}

async function showFile(id, scrollToLive) {
  const file = FILES.find((f) => f.id === id);
  view.fileId = id;
  document.querySelectorAll("#filetabs button").forEach((b) => b.setAttribute("aria-selected", String(b.id === `tab-${id}`)));
  document.getElementById("code-desc").textContent = file.desc;
  const link = document.getElementById("code-link");
  link.href = REPO_BLOB + file.repo;
  const code = document.getElementById("code");
  let src;
  try { src = await loadSource(file); }
  catch {
    code.innerHTML = `<span class="ln" data-n="">No se pudo leer ${escapeHtml(file.path)}. Ábrelo en GitHub con el enlace de arriba.</span>`;
    view.rendered = null; return;
  }
  if (view.fileId !== id) return; // user switched meanwhile
  const lines = highlightLines(src, file.lang);
  code.innerHTML = lines.map((l, i) => `<span class="ln" data-n="${i + 1}">${l || " "}</span>`).join("");
  view.rendered = { id, lines: src.split("\n"), lang: file.lang };
  applyLive(scrollToLive);
}

function applyLive(scroll) {
  const r = view.rendered; if (!r) return;
  const rows = document.querySelectorAll("#code .ln");
  rows.forEach((row) => row.classList.remove("live"));
  let first = null, ranges = [];
  for (const [fid, cls, fn] of view.live) {
    if (fid !== r.id) continue;
    const range = findRange(r.lines, r.lang, cls, fn);
    if (!range) continue;
    ranges.push(range);
    for (let n = range[0]; n <= range[1]; n++) rows[n - 1]?.classList.add("live");
    if (!first || range[0] < first) first = range[0];
  }
  const file = FILES.find((f) => f.id === r.id);
  if (ranges.length) {
    const [a, b] = ranges.sort((x, y) => x[0] - y[0])[0];
    document.getElementById("code-link").href = `${REPO_BLOB}${file.repo}#L${a}-L${b}`;
  }
  if (scroll && first) {
    const body = document.getElementById("codebody"), row = rows[first - 1];
    if (row) body.scrollTo({ top: Math.max(0, row.offsetTop - 48), behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }
  const names = view.live.filter(([fid]) => fid === r.id).map(([, c, f]) => (c ? `${c}.${f}` : f));
  document.getElementById("code-live").textContent = names.length
    ? `En ejecución en este archivo: ${names.join(", ")}`
    : (view.live.length ? "Este archivo no se está ejecutando ahora; los archivos con punto naranja sí." : "Pulsa Entrenar, Ver política o Pilotear yo para ver qué funciones se ejecutan.");
}

function syncWithApp() {
  if (typeof app === "undefined") return;
  const key = `${app.mode}|${app.algo}|${app.envKey}`;
  if (key === view.key) return;
  view.key = key;
  view.live = liveTargets(app.mode, app.algo, app.envKey);
  const liveFiles = new Set(view.live.map(([fid]) => fid));
  document.querySelectorAll("#filetabs button").forEach((b) => b.dataset.live = String(liveFiles.has(b.id.slice(4))));
  const follow = document.getElementById("code-follow").checked;
  const want = follow ? preferredFile(app.mode, app.algo) : null;
  if (want && want !== view.fileId) showFile(want, true);
  else applyLive(follow);
}

buildTabs();
showFile(view.fileId, false);
setInterval(syncWithApp, 250);
