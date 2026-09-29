"use strict";
/* Agentes: DQN (Double DQN, Huber, red objetivo, buffer de repetición) y Q-Learning tabular. */

// ═══ Agents ═════════════════════════════════════════════════════════════
class DQNAgent {
  constructor(Env, hp) {
    this.Env = Env; this.hp = hp; this.rng = makeRng(hp.seed * 7919 + 17);
    const hidden = String(hp.hidden).split(",").map((v) => parseInt(v, 10)).filter((v) => v > 0);
    this.hidden = hidden.length ? hidden : [64, 64];
    this.online = new MLP([Env.obsDim, ...this.hidden, Env.nActions], this.rng);
    this.target = new MLP([Env.obsDim, ...this.hidden, Env.nActions], null);
    this.target.copyFrom(this.online);
    this.opt = new Adam(this.online, hp.lr);
    const cap = hp.buffer, d = Env.obsDim;
    this.buf = { s: new Float32Array(cap * d), s2: new Float32Array(cap * d), a: new Int8Array(cap), r: new Float32Array(cap), term: new Uint8Array(cap), pos: 0, size: 0, cap };
    this.epsilon = 1; this.steps = 0; this.episodes = 0; this.lastLoss = NaN;
    this.q = new Float32Array(Env.nActions);
  }
  qValues(obs) { this.q.set(this.online.forward(obs)); return this.q; }
  act(obs, greedy) {
    if (!greedy && this.rng.random() < this.epsilon) return this.rng.int(this.Env.nActions);
    return argmax(this.qValues(obs));
  }
  observe(s, a, r, s2, terminated) {
    const B = this.buf, d = this.Env.obsDim, i = B.pos;
    B.s.set(s, i * d); B.s2.set(s2, i * d); B.a[i] = a; B.r[i] = r; B.term[i] = terminated ? 1 : 0;
    B.pos = (B.pos + 1) % B.cap; B.size = Math.min(B.size + 1, B.cap);
    this.steps += 1;
    const hp = this.hp;
    let loss = null;
    if (B.size >= Math.max(hp.learningStarts, hp.batch) && this.steps % hp.trainFreq === 0) loss = this.learn();
    const frac = Math.min(1, this.steps / hp.epsDecaySteps);
    this.epsilon = 1 + frac * (hp.epsEnd - 1);
    if (this.steps % hp.targetUpdate === 0) this.target.copyFrom(this.online);
    return loss;
  }
  learn() {
    const B = this.buf, d = this.Env.obsDim, n = this.hp.batch, gamma = this.hp.gamma;
    const nA = this.Env.nActions, net = this.online, L = net.L;
    this.opt.zero();
    let lossSum = 0;
    for (let k = 0; k < n; k++) {
      const j = this.rng.int(B.size);
      const s2 = B.s2.subarray(j * d, j * d + d);
      let y = B.r[j];
      if (!B.term[j]) {
        let aStar;
        if (this.hp.doubleDqn) aStar = argmax(net.forward(s2)); // online selects
        const qt = this.target.forward(s2);
        if (!this.hp.doubleDqn) aStar = argmax(qt);
        y += gamma * qt[aStar]; // target evaluates
      }
      const q = net.forward(B.s.subarray(j * d, j * d + d));
      const a = B.a[j], diff = q[a] - y, ad = Math.abs(diff);
      lossSum += ad < 1 ? 0.5 * diff * diff : ad - 0.5; // Huber (SmoothL1, beta = 1)
      const dOut = net.deltas[L];
      for (let o = 0; o < nA; o++) dOut[o] = 0;
      dOut[a] = (ad < 1 ? diff : Math.sign(diff)) / n;
      this.opt.backward();
    }
    this.opt.step(10);
    this.lastLoss = lossSum / n;
    return this.lastLoss;
  }
  episodeEnd() { this.episodes += 1; }
  extraLabel() { return "Pérdida"; }
  extraValue() { return Number.isFinite(this.lastLoss) ? this.lastLoss.toFixed(4) : "calentando"; }
  loadLayers(layers) {
    const net = MLP.fromLayers(layers);
    if (net.sizes[0] !== this.Env.obsDim || net.sizes[net.L] !== this.Env.nActions) {
      throw new Error(`El modelo espera ${net.sizes[0]} entradas y ${net.sizes[net.L]} acciones; este entorno usa ${this.Env.obsDim} y ${this.Env.nActions}.`);
    }
    this.online = net; this.target = MLP.fromLayers(layers); this.opt = new Adam(this.online, this.hp.lr);
    this.hidden = net.sizes.slice(1, -1); this.epsilon = this.hp.epsEnd;
    this.steps = Math.max(this.steps, this.hp.epsDecaySteps);
  }
}

class QLearningAgent {
  constructor(Env, hp, envKey) {
    this.Env = Env; this.hp = hp; this.rng = makeRng(hp.seed * 104729 + 3);
    const spec = OBS_BOUNDS[envKey];
    this.bounds = spec.bounds; this.nBinary = spec.binary;
    this.edges = this.bounds.map(([lo, hi]) => {
      const e = []; for (let i = 1; i < hp.bins; i++) e.push(lo + (hi - lo) * i / hp.bins); return e;
    });
    this.table = new Map();
    this.epsilon = 1; this.steps = 0; this.episodes = 0; this.lastTd = NaN;
    this.q = new Float32Array(Env.nActions);
  }
  key(obs) {
    const parts = [];
    for (let i = 0; i < this.bounds.length; i++) {
      const [lo, hi] = this.bounds[i]; const v = Math.min(hi, Math.max(lo, obs[i]));
      let b = 0; for (const e of this.edges[i]) if (v >= e) b++; parts.push(b);
    }
    for (let i = this.bounds.length; i < obs.length; i++) parts.push(Math.round(obs[i]));
    return parts.join(",");
  }
  row(k) { let r = this.table.get(k); if (!r) { r = new Float64Array(this.Env.nActions); this.table.set(k, r); } return r; }
  qValues(obs) { const r = this.table.get(this.key(obs)); this.q.set(r || new Float64Array(this.Env.nActions)); return this.q; }
  act(obs, greedy) {
    if (!greedy && this.rng.random() < this.epsilon) return this.rng.int(this.Env.nActions);
    const r = this.row(this.key(obs)); const m = Math.max(...r);
    const best = []; r.forEach((v, i) => { if (v === m) best.push(i); });
    return best.length === 1 ? best[0] : best[this.rng.int(best.length)]; // random tie-break
  }
  observe(s, a, r, s2, terminated) {
    this.steps += 1;
    const q = this.row(this.key(s)), q2 = this.row(this.key(s2));
    const target = r + (terminated ? 0 : this.hp.gamma * Math.max(...q2));
    const td = target - q[a];
    q[a] += this.hp.lr * td;
    this.lastTd = td * td;
    return this.lastTd;
  }
  episodeEnd() { this.episodes += 1; this.epsilon = Math.max(this.hp.epsEnd, this.epsilon * this.hp.epsDecay); }
  extraLabel() { return "Estados visitados"; }
  extraValue() { return this.table.size.toLocaleString("es-CO"); }
}

function argmax(arr) { let b = 0; for (let i = 1; i < arr.length; i++) if (arr[i] > arr[b]) b = i; return b; }
