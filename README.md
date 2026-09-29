# rl_games · edición mejorada

[![CI](https://github.com/leonarsomo/rl_games_mejorado/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/leonarsomo/rl_games_mejorado/actions/workflows/ci.yml)
[![Simulador](https://img.shields.io/badge/simulador-HTML5-2a57d6)](https://leonarsomo.github.io/rl_games_mejorado/)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Licencia Apache 2.0](https://img.shields.io/badge/licencia-Apache%202.0-lightgrey)

Laboratorio de aprendizaje por refuerzo con **Q-Learning tabular** y **Deep Q-Network (DQN)** sobre [Gymnasium](https://gymnasium.farama.org/), más un **simulador HTML5** donde los mismos algoritmos entrenan en vivo en el navegador.

Es un espejo mejorado de [emiliomunozai/rl_games](https://github.com/emiliomunozai/rl_games). El original deja los algoritmos como ejercicios; esta edición trae una implementación de referencia probada, corrige varios errores de fondo y añade herramientas para experimentar de forma reproducible. El detalle de cada cambio está en [REVISION.md](REVISION.md).

> Si estás resolviendo los ejercicios del curso, trabaja en el repositorio original. Usa este para comparar tu solución cuando termines.

## Simulador en el navegador

**[leonarsomo.github.io/rl_games_mejorado](https://leonarsomo.github.io/rl_games_mejorado/)** (o abre `docs/index.html` localmente).

- **CartPole-v1** es un puerto exacto de la física de Gymnasium: la diferencia máxima entre ambas trayectorias es de 1e-7, la precisión de float32. Por eso una DQN entrenada en Python funciona en la página sin cambios; el modelo incluido obtiene 500/500 en ambos lados.
- **Lunar Lander** es una versión ligera con la misma observación de 8 valores y la misma recompensa que `LunarLander-v3`, pero con física de cuerpo rígido simplificada en lugar de Box2D. El controlador heurístico oficial de Gymnasium, trasladado sin cambios, aterriza en 20 de 20 intentos.
- DQN (Double DQN, Huber, recorte de gradiente, red objetivo) y Q-Learning escritos en JavaScript puro, sin dependencias.
- Modos: entrenar (tiempo real, ×10 o turbo), ver la política codiciosa y pilotear con el teclado.
- Muestra en vivo la curva de aprendizaje, los valores Q(s,a) del estado actual y el vector de observación.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.11.

```bash
git clone https://github.com/leonarsomo/rl_games_mejorado.git
cd rl_games_mejorado
uv sync --all-extras        # dependencias + matplotlib para las gráficas
source .venv/bin/activate   # macOS / Linux  (Windows: .venv\Scripts\activate)
```

En Linux y Windows se instala PyTorch para CPU (≈ 1 GB en lugar de ≈ 7 GB con CUDA).

## Uso rápido

```bash
rlgames inspect --env CartPole-v1                     # espacios y transiciones de ejemplo
rlgames train dqn --env CartPole-v1 --episodes 500 --seed 7 --eval-every 20
rlgames load  dqn --env CartPole-v1 --best --eval     # evalúa el mejor punto de control
rlgames plot  dqn --env CartPole-v1                   # saves/dqn_CartPole-v1.png
rlgames export dqn --env CartPole-v1 --best           # docs/models/dqn_CartPole-v1.json
rlgames render dqn --env CartPole-v1 --best           # ventana gráfica
```

| Comando | Qué hace |
|---|---|
| `inspect` | Muestra los espacios de observación y acción y transiciones aleatorias |
| `init` | Crea un agente sin entrenar |
| `train` | Entrena, o reanuda si ya existe un guardado, y añade el registro CSV |
| `load` | Muestra la configuración; con `--eval`, evalúa la política codiciosa |
| `sim` / `render` | Ejecuta episodios en texto o en ventana |
| `plot` | Curva de aprendizaje (PNG) a partir del CSV |
| `export` | Pesos de la DQN en JSON para el simulador |
| `list` / `delete` / `version` | Utilidades |

Opciones comunes: `--env`, `--seed`, `--hp clave=valor` (repetible) y `--best`.

```bash
# Comparar DQN simple contra Double DQN con la misma semilla
rlgames train dqn --env CartPole-v1 --seed 1 --hp double_dqn=false
rlgames delete dqn --env CartPole-v1
rlgames train dqn --env CartPole-v1 --seed 1 --hp double_dqn=true
```

Las claves válidas de `--hp` son los campos de `QLearningConfig` y `DQNConfig` en [`src/rl_games/config.py`](src/rl_games/config.py). Una clave desconocida detiene la ejecución con la lista de claves válidas.

## Como librería

```python
from rl_games import evaluate, registry

agent = registry.create("dqn", "CartPole-v1", {"lr": 1e-3}, seed=0)
log = agent.train(300, eval_every=50)
print(log.mean_reward(100))
print(evaluate.summarize(evaluate.run_episodes(agent, n_episodes=20, seed=1000)))
```

## Estructura

```
src/rl_games/
├── cli.py            # argumentos y salida; sin lógica de aprendizaje
├── config.py         # QLearningConfig, DQNConfig, presets por entorno, --hp
├── registry.py       # agentes registrados, rutas de guardado, carga/creación
├── envs.py           # creación de entornos (con semilla) y límites de observación
├── evaluate.py       # evaluación codiciosa reproducible
├── metrics.py        # registro por episodio, CSV y media móvil
├── utils.py          # semillas globales
└── agents/
    ├── base.py       # bucle de entrenamiento único, ε-greedy, persistencia
    ├── qlearning.py  # Q-Learning tabular (guardado .npz, sin pickle)
    ├── dqn.py        # DQN: Double DQN, Huber, recorte, red objetivo por pasos
    └── replay.py     # buffer de repetición con arreglos NumPy
docs/
├── index.html        # simulador HTML5 (GitHub Pages)
└── models/           # modelo DQN de CartPole exportado desde Python
tests/                # pytest: unidades, CLI y pruebas de aprendizaje
```

## Conceptos clave

**Q-Learning** (Watkins & Dayan, 1992) actualiza una tabla indexada por el estado discretizado:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\,\bigl[r + \gamma \max_{a'} Q(s',a')\,(1-\text{terminated}) - Q(s,a)\bigr]$$

**DQN** (Mnih et al., 2015) reemplaza la tabla por una red neuronal y la estabiliza con un buffer de repetición y una red objetivo. Esta edición usa **Double DQN** (van Hasselt et al., 2016): la red en línea elige la acción siguiente y la red objetivo la evalúa, lo que reduce la sobreestimación.

$$y = r + \gamma\, Q_{\text{obj}}\!\bigl(s',\, \arg\max_{a'} Q_{\theta}(s',a')\bigr)\,(1-\text{terminated})$$

**Terminado frente a truncado.** Gymnasium distingue `terminated` (el episodio terminó de verdad: choque, aterrizaje, caída del poste) de `truncated` (se acabó el tiempo). Solo el primero anula el valor futuro. Tratar el límite de tiempo como estado terminal sesga el aprendizaje (Pardo et al., 2018), y era el error principal del proyecto original.

## Resultados verificados

Resultados obtenidos al construir esta edición (CPU, Python 3.11):

| Agente | Entorno | Entrenamiento | Evaluación codiciosa |
|---|---|---|---|
| Q-Learning | CartPole-v1 | 3000 episodios, semilla 0 (15 s) | 139,9 ± 41,4 en 20 episodios |
| DQN | CartPole-v1 | 500 episodios, semilla 7, mejor punto de control | **500,0 ± 0,0** en 30 episodios |
| DQN (navegador) | Lunar Lander (ligero) | ≈ 45 000 pasos, 90 s en Chromium | media móvil de 50 episodios ≈ 220 |

Una política aleatoria obtiene unos 22 puntos en CartPole. La discretización limita al Q-Learning tabular; ese es el motivo para pasar a la DQN.

## Calidad

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest                 # todas (≈ 35 s)
uv run pytest -m "not slow"   # solo unidades (≈ 3 s)
```

La CI de GitHub Actions ejecuta lint, formato, tipos y pruebas en Ubuntu y macOS, y el flujo `pages.yml` publica `docs/` en GitHub Pages.

## Créditos y licencia

Proyecto original: [emiliomunozai/rl_games](https://github.com/emiliomunozai/rl_games), cuyo historial se conserva aquí. Edición mejorada: Leonar Socarrás Molina. Licencia Apache 2.0; ver [LICENSE](LICENSE) y [NOTICE](NOTICE).

## Referencias

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529-533. https://doi.org/10.1038/nature14236

Pardo, F., Tavakoli, A., Levdik, V., & Kormushev, P. (2018). Time limits in reinforcement learning. En *Proceedings of the 35th International Conference on Machine Learning* (Vol. 80, pp. 4045-4054). PMLR.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2.ª ed.). MIT Press.

Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., De Cola, G., Deleu, T., Goulão, M., Kallinteris, A., Krimmel, M., KG, A., Perez-Vicente, R., Pierré, A., Schulhoff, S., Tai, J. J., Tan, H., & Younis, O. G. (2024). *Gymnasium: A standard interface for reinforcement learning environments* (arXiv:2407.17032). arXiv. https://doi.org/10.48550/arXiv.2407.17032

van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *Proceedings of the AAAI Conference on Artificial Intelligence, 30*(1). https://doi.org/10.1609/aaai.v30i1.10295

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning, 8*(3-4), 279-292. https://doi.org/10.1007/BF00992698
