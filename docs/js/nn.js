"use strict";
/* Red neuronal desde cero: perceptrón multicapa con ReLU y optimizador Adam con recorte de gradiente. */

// ═══ Neural network: MLP with ReLU, Adam, Huber loss, gradient clipping
class MLP {
  constructor(sizes, rng) {
    this.sizes = sizes;
    this.L = sizes.length - 1;
    this.W = []; this.b = [];
    for (let l = 0; l < this.L; l++) {
      const nin = sizes[l], nout = sizes[l + 1], bound = 1 / Math.sqrt(nin); // PyTorch nn.Linear default
      const W = new Float32Array(nin * nout), b = new Float32Array(nout);
      for (let i = 0; i < W.length; i++) W[i] = rng ? rng.uniform(-bound, bound) : 0;
      for (let i = 0; i < b.length; i++) b[i] = rng ? rng.uniform(-bound, bound) : 0;
      this.W.push(W); this.b.push(b);
    }
    this.acts = sizes.map((n) => new Float32Array(n));
    this.deltas = sizes.map((n) => new Float32Array(n));
  }
  forward(x) { // returns internal buffer: copy if you need to keep it
    const a0 = this.acts[0];
    for (let i = 0; i < a0.length; i++) a0[i] = x[i];
    for (let l = 0; l < this.L; l++) {
      const inp = this.acts[l], out = this.acts[l + 1], W = this.W[l], b = this.b[l];
      const nin = inp.length, nout = out.length, last = l === this.L - 1;
      for (let o = 0; o < nout; o++) {
        let z = b[o]; const row = o * nin;
        for (let i = 0; i < nin; i++) z += W[row + i] * inp[i];
        out[o] = last ? z : (z > 0 ? z : 0);
      }
    }
    return this.acts[this.L];
  }
  copyFrom(other) {
    for (let l = 0; l < this.L; l++) { this.W[l].set(other.W[l]); this.b[l].set(other.b[l]); }
  }
  toJSON() {
    return this.W.map((W, l) => {
      const nin = this.sizes[l], nout = this.sizes[l + 1];
      return { W: Array.from({ length: nout }, (_, o) => Array.from(W.subarray(o * nin, (o + 1) * nin))), b: Array.from(this.b[l]) };
    });
  }
  static fromLayers(layers) {
    const sizes = [layers[0].W[0].length, ...layers.map((ly) => ly.b.length)];
    const net = new MLP(sizes, null);
    layers.forEach((ly, l) => { net.W[l].set(ly.W.flat()); net.b[l].set(ly.b); });
    return net;
  }
}

class Adam {
  constructor(net, lr) {
    this.net = net; this.lr = lr; this.t = 0; this.b1 = 0.9; this.b2 = 0.999; this.eps = 1e-8;
    const z = (arrs) => arrs.map((a) => new Float32Array(a.length));
    this.gW = z(net.W); this.gb = z(net.b);
    this.mW = z(net.W); this.vW = z(net.W); this.mb = z(net.b); this.vb = z(net.b);
  }
  zero() { this.gW.forEach((g) => g.fill(0)); this.gb.forEach((g) => g.fill(0)); }
  // Backprop dLoss/dOutput (already in net.deltas[L]) for the current forward pass.
  backward() {
    const net = this.net;
    for (let l = net.L - 1; l >= 0; l--) {
      const d = net.deltas[l + 1], inp = net.acts[l], W = net.W[l], gW = this.gW[l], gb = this.gb[l];
      const nin = inp.length, nout = d.length;
      for (let o = 0; o < nout; o++) {
        const dv = d[o]; if (dv === 0) continue;
        gb[o] += dv; const row = o * nin;
        for (let i = 0; i < nin; i++) gW[row + i] += dv * inp[i];
      }
      if (l > 0) {
        const dprev = net.deltas[l];
        for (let i = 0; i < nin; i++) {
          if (inp[i] <= 0) { dprev[i] = 0; continue; } // ReLU derivative
          let s = 0; for (let o = 0; o < nout; o++) s += W[o * nin + i] * d[o];
          dprev[i] = s;
        }
      }
    }
  }
  step(maxNorm) {
    let sq = 0;
    for (const g of [...this.gW, ...this.gb]) for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
    const norm = Math.sqrt(sq), scale = norm > maxNorm ? maxNorm / (norm + 1e-6) : 1;
    this.t += 1;
    const bc1 = 1 - Math.pow(this.b1, this.t), bc2 = 1 - Math.pow(this.b2, this.t);
    const upd = (p, g, m, v) => {
      for (let i = 0; i < p.length; i++) {
        const gi = g[i] * scale;
        m[i] = this.b1 * m[i] + (1 - this.b1) * gi;
        v[i] = this.b2 * v[i] + (1 - this.b2) * gi * gi;
        p[i] -= this.lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + this.eps);
      }
    };
    for (let l = 0; l < this.net.L; l++) {
      upd(this.net.W[l], this.gW[l], this.mW[l], this.vW[l]);
      upd(this.net.b[l], this.gb[l], this.mb[l], this.vb[l]);
    }
  }
}

