"use strict";
/* Entornos del simulador: RNG con semilla, CartPole-v1 (puerto exacto de Gymnasium) y Lunar Lander ligero. */

// ── seeded RNG (mulberry32) ─────────────────────────────────────────────
function makeRng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    random: next,
    uniform: (lo, hi) => lo + (hi - lo) * next(),
    int: (n) => Math.floor(next() * n),
  };
}

// ═══ Environment: CartPole-v1 (exact port of gymnasium/envs/classic_control/cartpole.py)
class CartPole {
  static id = "CartPole-v1";
  static obsDim = 4;
  static nActions = 2;
  static actionNames = ["Empujar izquierda", "Empujar derecha"];
  static obsLabels = ["x carro", "v carro", "θ poste", "ω poste"];
  static maxSteps = 500;
  static solved = 475;
  constructor(rng) {
    this.rng = rng;
    this.g = 9.8; this.mc = 1.0; this.mp = 0.1; this.total = 1.1;
    this.len = 0.5; this.pml = 0.05; this.force = 10.0; this.tau = 0.02;
    this.thetaMax = 12 * 2 * Math.PI / 360; this.xMax = 2.4;
    this.reset();
  }
  reset() {
    this.s = [0, 0, 0, 0].map(() => this.rng.uniform(-0.05, 0.05));
    this.t = 0; this.lastAction = null;
    return this.s.slice();
  }
  step(action) {
    let [x, xd, th, thd] = this.s;
    const f = action === 1 ? this.force : -this.force;
    const c = Math.cos(th), sn = Math.sin(th);
    const temp = (f + this.pml * thd * thd * sn) / this.total;
    const thacc = (this.g * sn - c * temp) / (this.len * (4 / 3 - this.mp * c * c / this.total));
    const xacc = temp - this.pml * thacc * c / this.total;
    x += this.tau * xd; xd += this.tau * xacc; th += this.tau * thd; thd += this.tau * thacc;
    this.s = [x, xd, th, thd];
    this.t += 1; this.lastAction = action;
    const terminated = x < -this.xMax || x > this.xMax || th < -this.thetaMax || th > this.thetaMax;
    const truncated = !terminated && this.t >= CartPole.maxSteps;
    this.outcome = terminated ? "El poste cayó" : truncated ? "¡500 pasos en equilibrio!" : null;
    return { obs: this.s.slice(), reward: 1.0, terminated, truncated };
  }
  manualAction(keys) {
    if (keys.has("ArrowLeft")) return 0;
    if (keys.has("ArrowRight")) return 1;
    return this.t % 2; // no key: alternate so the net force is zero
  }
  draw(ctx, w, h, col) {
    const sky = ctx.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, col.skyTop); sky.addColorStop(1, col.skyBottom);
    ctx.fillStyle = sky; ctx.fillRect(0, 0, w, h);
    const scale = w / 5.6, cx = w / 2, trackY = h * 0.72;
    const X = (x) => cx + x * scale;
    ctx.strokeStyle = col.groundEdge; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(-2.8), trackY); ctx.lineTo(X(2.8), trackY); ctx.stroke();
    ctx.fillStyle = col.bad;
    for (const lim of [-2.4, 2.4]) ctx.fillRect(X(lim) - 1.5, trackY - 14, 3, 28);
    const [x, , th] = this.s;
    const cw = 0.5 * scale, ch = 0.3 * scale, cartX = X(x), cartY = trackY - ch / 2 - 4;
    ctx.fillStyle = col.craft;
    roundRect(ctx, cartX - cw / 2, cartY - ch / 2, cw, ch, 6); ctx.fill();
    const L = 1.0 * scale;
    ctx.strokeStyle = col.warm; ctx.lineWidth = Math.max(6, scale * 0.09); ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(cartX, cartY - ch * 0.2);
    ctx.lineTo(cartX + L * Math.sin(th), cartY - ch * 0.2 - L * Math.cos(th)); ctx.stroke();
    ctx.fillStyle = col.accent;
    ctx.beginPath(); ctx.arc(cartX, cartY - ch * 0.2, scale * 0.05, 0, Math.PI * 2); ctx.fill();
    if (this.lastAction !== null) {
      const dir = this.lastAction === 1 ? 1 : -1;
      ctx.strokeStyle = col.accent; ctx.lineWidth = 3;
      const ax = cartX - dir * (cw / 2 + 8), ay = cartY;
      ctx.beginPath(); ctx.moveTo(ax - dir * 26, ay); ctx.lineTo(ax, ay);
      ctx.lineTo(ax - dir * 9, ay - 7); ctx.moveTo(ax, ay); ctx.lineTo(ax - dir * 9, ay + 7); ctx.stroke();
    }
  }
}

// ═══ Environment: Lunar Lander (lite). Same observation layout and reward
//     shaping as gymnasium LunarLander-v3; simplified rigid-body physics.
class LunarLite {
  static id = "LunarLander-lite";
  static obsDim = 8;
  static nActions = 4;
  static actionNames = ["No hacer nada", "Motor izq. (gira ↺)", "Motor principal", "Motor der. (gira ↻)"];
  static obsLabels = ["x", "y", "vx", "vy", "ángulo", "ω", "pata izq.", "pata der."];
  static maxSteps = 1000;
  static solved = 200;
  constructor(rng) {
    this.rng = rng;
    this.W = 20; this.H = 400 / 30; this.FPS = 50; this.dt = 1 / 50; this.g = -10;
    this.padY = this.H / 4; this.legRest = 0.95;
    this.hull = [[-14, 17], [-17, 0], [-17, -10], [17, -10], [17, 0], [14, 17]].map(([a, b]) => [a / 30, b / 30]);
    this.legs = [[-0.75, -0.95], [0.75, -0.95]];
    this.particles = [];
    this.reset();
  }
  reset() {
    const n = 11, W = this.W, H = this.H;
    const raw = Array.from({ length: n }, () => this.rng.uniform(0, H / 2));
    for (let i = n / 2 - 2 | 0; i <= (n / 2 | 0) + 2; i++) raw[i] = this.padY;
    this.cx = Array.from({ length: n }, (_, i) => (W / (n - 1)) * i);
    this.cy = raw.map((v, i) => (i >= (n / 2 | 0) - 2 && i <= (n / 2 | 0) + 2) ? this.padY
      : (raw[Math.max(0, i - 1)] + v + raw[Math.min(n - 1, i + 1)]) / 3);
    this.padX1 = this.cx[(n / 2 | 0) - 1]; this.padX2 = this.cx[(n / 2 | 0) + 1];
    this.x = W / 2; this.y = H - 1.2; this.vx = this.rng.uniform(-2.5, 2.5); this.vy = this.rng.uniform(-2, 0);
    this.a = 0; this.w = 0; this.contact = [0, 0];
    this.t = 0; this.prevShaping = null; this.lastAction = 0; this.particles = []; this.outcome = null;
    return this.obs();
  }
  ground(x) {
    const cx = this.cx, cy = this.cy;
    if (x <= cx[0]) return cy[0];
    for (let i = 1; i < cx.length; i++) {
      if (x <= cx[i]) { const f = (x - cx[i - 1]) / (cx[i] - cx[i - 1]); return cy[i - 1] + f * (cy[i] - cy[i - 1]); }
    }
    return cy[cy.length - 1];
  }
  toWorld([px, py]) {
    const c = Math.cos(this.a), s = Math.sin(this.a);
    return [this.x + px * c - py * s, this.y + px * s + py * c];
  }
  obs() {
    const { W, H, FPS } = this;
    return [
      (this.x - W / 2) / (W / 2),
      (this.y - (this.padY + this.legRest)) / (H / 2),
      this.vx * (W / 2) / FPS,
      this.vy * (H / 2) / FPS,
      this.a,
      20 * this.w / FPS,
      this.contact[0], this.contact[1],
    ];
  }
  step(action) {
    const dt = this.dt, c = Math.cos(this.a), s = Math.sin(this.a);
    let ax = 0, ay = this.g, alpha = 0, mPower = 0, sPower = 0;
    if (action === 2) {
      mPower = 1;
      const thrust = 20 + this.rng.uniform(-0.8, 0.8);
      ax += -s * thrust; ay += c * thrust;
      this.emit(0, -0.4, -s, c, 1);
    } else if (action === 1 || action === 3) {
      sPower = 1;
      // Side engines sit slightly below the centre of mass: the left one (action 1)
      // pushes the craft right and turns it counter-clockwise (angle grows),
      // the same sign convention as gymnasium's heuristic controller.
      const dir = action === 1 ? 1 : -1;
      ax += dir * c * 2.2; ay += dir * s * 2.2;
      alpha = dir * (3.0 + this.rng.uniform(-0.2, 0.2));
      this.emit(-dir * 0.55, -0.1, -dir * c, -dir * s, 0.4);
    }
    this.vx += ax * dt; this.vy += ay * dt; this.w += alpha * dt;
    this.x += this.vx * dt; this.y += this.vy * dt; this.a += this.w * dt;

    // ground contact on the leg tips
    let crashed = false;
    const tips = this.legs.map((p) => this.toWorld(p));
    const pen = tips.map(([lx, ly]) => this.ground(lx) - ly);
    this.contact = pen.map((p) => (p >= -0.02 ? 1 : 0));
    const maxPen = Math.max(pen[0], pen[1]);
    if (maxPen > 0) {
      this.y += maxPen;
      if (this.vy < -5) crashed = true; // hard landing
      if (this.vy < 0) this.vy = 0;
      this.vx *= 0.85;
      this.w *= 0.85;
      if (this.contact[0] && !this.contact[1]) this.w -= 5 * dt;
      else if (this.contact[1] && !this.contact[0]) this.w += 5 * dt;
      else { this.a *= 0.9; this.w *= 0.5; }
    }
    // hull touching the ground = crash
    for (const p of this.hull) { const [hx, hy] = this.toWorld(p); if (hy < this.ground(hx) - 0.02) crashed = true; }

    const o = this.obs();
    const shaping = -100 * Math.hypot(o[0], o[1]) - 100 * Math.hypot(o[2], o[3]) - 100 * Math.abs(o[4])
      + 10 * o[6] + 10 * o[7];
    let reward = this.prevShaping === null ? 0 : shaping - this.prevShaping;
    this.prevShaping = shaping;
    reward -= mPower * 0.3 + sPower * 0.03;

    let terminated = false;
    this.outcome = null;
    if (crashed || Math.abs(o[0]) >= 1 || this.y > this.H * 1.6) {
      terminated = true; reward = -100; this.outcome = "Choque";
    } else if (this.contact[0] && this.contact[1] && Math.abs(this.vx) < 0.05 && Math.abs(this.w) < 0.05 && action !== 2) {
      terminated = true; reward = 100; this.outcome = "¡Aterrizaje!";
    }
    this.t += 1; this.lastAction = action;
    const truncated = !terminated && this.t >= LunarLite.maxSteps;
    if (truncated) this.outcome = "Tiempo agotado";
    return { obs: o, reward, terminated, truncated };
  }
  emit(px, py, dx, dy, power) {
    if (this.particles.length > 120) this.particles.splice(0, 20);
    const [wx, wy] = this.toWorld([px, py]);
    for (let i = 0; i < 3; i++) {
      this.particles.push({ x: wx, y: wy, vx: -dx * (3 + this.rng.random() * 3) * power + this.vx, vy: -dy * (3 + this.rng.random() * 3) * power + this.vy, life: 1 });
    }
  }
  manualAction(keys) {
    if (keys.has("ArrowUp") || keys.has(" ")) return 2;
    if (keys.has("ArrowLeft")) return 1;
    if (keys.has("ArrowRight")) return 3;
    return 0;
  }
  draw(ctx, w, h, col) {
    const k = w / this.W;
    const X = (x) => x * k, Y = (y) => h - y * k;
    const sky = ctx.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, col.skyTop); sky.addColorStop(1, col.skyBottom);
    ctx.fillStyle = sky; ctx.fillRect(0, 0, w, h);
    // stars (fixed pattern)
    ctx.fillStyle = col.muted; ctx.globalAlpha = 0.35;
    for (let i = 0; i < 40; i++) { const sx = (i * 97.13) % w, sy = (i * 53.7) % (h * 0.6); ctx.fillRect(sx, sy, 1.6, 1.6); }
    ctx.globalAlpha = 1;
    // terrain
    ctx.fillStyle = col.ground; ctx.strokeStyle = col.groundEdge; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, h);
    this.cx.forEach((x, i) => ctx.lineTo(X(x), Y(this.cy[i])));
    ctx.lineTo(w, h); ctx.closePath(); ctx.fill(); ctx.stroke();
    // flags
    for (const fx of [this.padX1, this.padX2]) {
      ctx.strokeStyle = col.muted; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(X(fx), Y(this.padY)); ctx.lineTo(X(fx), Y(this.padY + 1.3)); ctx.stroke();
      ctx.fillStyle = col.warm;
      ctx.beginPath(); ctx.moveTo(X(fx), Y(this.padY + 1.3)); ctx.lineTo(X(fx) + 16, Y(this.padY + 1.15)); ctx.lineTo(X(fx), Y(this.padY + 1.0)); ctx.fill();
    }
    // particles
    for (const p of this.particles) {
      p.x += p.vx * this.dt; p.y += p.vy * this.dt; p.life -= 0.06;
    }
    this.particles = this.particles.filter((p) => p.life > 0);
    for (const p of this.particles) {
      ctx.globalAlpha = Math.max(0, p.life);
      ctx.fillStyle = col.warm;
      ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 2 + 3 * p.life, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    // legs
    ctx.strokeStyle = col.craft; ctx.lineWidth = 3;
    this.legs.forEach((lp, i) => {
      const [ax, ay] = this.toWorld([lp[0] * 0.55, -0.2]);
      const [bx, by] = this.toWorld(lp);
      ctx.strokeStyle = this.contact[i] ? col.good : col.craft;
      ctx.beginPath(); ctx.moveTo(X(ax), Y(ay)); ctx.lineTo(X(bx), Y(by)); ctx.stroke();
    });
    // hull
    ctx.fillStyle = col.craft;
    ctx.beginPath();
    this.hull.forEach((p, i) => { const [hx, hy] = this.toWorld(p); i ? ctx.lineTo(X(hx), Y(hy)) : ctx.moveTo(X(hx), Y(hy)); });
    ctx.closePath(); ctx.fill();
    const [wx, wy] = this.toWorld([0, 0.2]);
    ctx.fillStyle = col.accent; ctx.beginPath(); ctx.arc(X(wx), Y(wy), k * 0.14, 0, Math.PI * 2); ctx.fill();
  }
}

const ENVS = { cartpole: CartPole, lunar: LunarLite };

// Observation bounds for tabular Q-learning (same as src/rl_games/envs.py).
const OBS_BOUNDS = {
  cartpole: { bounds: [[-2.4, 2.4], [-3, 3], [-0.21, 0.21], [-3, 3]], binary: 0 },
  lunar: { bounds: [[-1.5, 1.5], [-0.5, 1.5], [-5, 5], [-5, 5], [-3.14, 3.14], [-5, 5]], binary: 2 },
};


function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

