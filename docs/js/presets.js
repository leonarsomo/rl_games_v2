"use strict";
/* Hiperparámetros iniciales: los mismos presets de src/rl_games/config.py, escalados al navegador. */

// Presets = the Python presets (src/rl_games/config.py), scaled for the browser.
const PRESETS = {
  dqn: {
    cartpole: { lr: 1e-3, gamma: 0.99, hidden: "64,64", batch: 64, buffer: 50000, learningStarts: 1000, trainFreq: 1, targetUpdate: 500, epsDecaySteps: 10000, epsEnd: 0.05, doubleDqn: true, seed: 1 },
    lunar: { lr: 5e-4, gamma: 0.99, hidden: "128,128", batch: 64, buffer: 100000, learningStarts: 5000, trainFreq: 4, targetUpdate: 1000, epsDecaySteps: 60000, epsEnd: 0.05, doubleDqn: true, seed: 1 },
  },
  qlearning: {
    cartpole: { lr: 0.1, gamma: 0.99, bins: 10, epsDecay: 0.998, epsEnd: 0.01, seed: 1 },
    lunar: { lr: 0.1, gamma: 0.99, bins: 8, epsDecay: 0.9995, epsEnd: 0.01, seed: 1 },
  },
};

const HP_FIELDS = {
  dqn: [
    ["lr", "Tasa de aprendizaje", "number", 0.0001],
    ["gamma", "γ descuento", "number", 0.001],
    ["hidden", "Capas ocultas", "text"],
    ["batch", "Tamaño de lote", "number", 1],
    ["learningStarts", "Pasos de calentamiento", "number", 100],
    ["trainFreq", "Actualizar cada N pasos", "number", 1],
    ["targetUpdate", "Sincronizar red objetivo", "number", 50],
    ["epsDecaySteps", "Pasos de decaimiento de ε", "number", 1000],
    ["epsEnd", "ε final", "number", 0.01],
    ["seed", "Semilla", "number", 1],
    ["doubleDqn", "Double DQN", "bool"],
  ],
  qlearning: [
    ["lr", "Tasa de aprendizaje α", "number", 0.01],
    ["gamma", "γ descuento", "number", 0.001],
    ["bins", "Intervalos por dimensión", "number", 1],
    ["epsDecay", "Decaimiento de ε por episodio", "number", 0.0001],
    ["epsEnd", "ε final", "number", 0.01],
    ["seed", "Semilla", "number", 1],
  ],
};

