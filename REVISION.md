# Revisión técnica de `emiliomunozai/rl_games`

Revisión del commit `7a4e130` (rama `main`, 2026) realizada antes de construir esta edición. El proyecto original es un buen andamiaje docente: separa CLI, registro, entornos y agentes, y deja como ejercicio justamente las líneas donde ocurre el aprendizaje. Los hallazgos siguientes no le restan valor; señalan qué cambia cuando ese andamiaje se usa para entrenar agentes que deben aprender de forma fiable y reproducible.

Cada hallazgo indica severidad, ubicación en el original y cómo se resolvió aquí.

## 1. Correctitud del aprendizaje

| # | Severidad | Dónde (original) | Hallazgo | Resolución |
|---|---|---|---|---|
| 1.1 | Alta | `dqn.py` → `train()`; `qlearning.py` → `train()` | Se guarda `done = terminated or truncated` y ese valor anula el término futuro del objetivo de Bellman. Cuando un episodio se corta por límite de tiempo (500 pasos en CartPole, 1000 en LunarLander) el estado no es terminal, pero el agente aprende que su valor futuro es cero. Pardo et al. (2018) muestran que este error sesga las estimaciones y desestabiliza el entrenamiento. | El bucle único de `BaseAgent.train()` pasa solo `terminated` al aprendiz; `truncated` únicamente cierra el episodio. Hay pruebas que lo verifican (`test_terminal_transition_does_not_bootstrap`, `test_terminal_mask_removes_bootstrap`). |
| 1.2 | Media | `dqn.py` → `train()` | La red objetivo se sincroniza cada 10 **episodios**. En LunarLander un episodio dura hasta 1000 pasos, así que la frecuencia real varía entre decenas y miles de actualizaciones. | Sincronización cada `target_update_steps` **pasos** de entorno, como en Mnih et al. (2015). |
| 1.3 | Media | `base.py` → `_decay_epsilon()` | ε decae una vez por episodio de forma exponencial. Con la DQN (0,995) los 500 episodios por defecto dejan ε ≈ 0,08; con Q-Learning (0,9995) hacen falta ≈ 9200 episodios para llegar a 0,01. La exploración depende de cuánto duren los episodios, no de cuánta experiencia se ha acumulado. | La DQN usa un calendario lineal por pasos (`epsilon_decay_steps`). Q-Learning conserva el decaimiento por episodio, con presets ajustados por entorno. |
| 1.4 | Media | `dqn.py` → `_learn()` | Pérdida MSE sin recorte de gradiente y objetivo `max` de la red objetivo (DQN simple), que sobreestima los valores Q. | Pérdida Huber, recorte de norma de gradiente y Double DQN (van Hasselt et al., 2016), todos configurables para comparar. |
| 1.5 | Baja | `dqn.py` | El aprendizaje empieza con el primer lote de 64 transiciones, casi todas de una política aleatoria y muy correlacionadas. | Parámetro `learning_starts` (calentamiento del buffer) y `train_freq`. |
| 1.6 | Baja | `qlearning.py` → `select_action` (esperado) | Con una fila de ceros, `np.argmax` siempre devuelve la acción 0 y sesga el comportamiento inicial. | Desempate aleatorio entre las acciones con valor máximo. |

## 2. Persistencia y seguridad

| # | Severidad | Dónde | Hallazgo | Resolución |
|---|---|---|---|---|
| 2.1 | Alta | `dqn.py` → `load()` | `torch.load(path, weights_only=False)` deserializa con pickle: abrir un `.pt` descargado puede ejecutar código arbitrario. | `weights_only=True`; el archivo solo contiene tensores y tipos básicos. |
| 2.2 | Media | `qlearning.py` → `save()/load()` | La tabla Q se guarda con `pickle`, con el mismo riesgo. | Formato `.npz` con cabecera JSON y `allow_pickle=False`. |
| 2.3 | Media | `dqn.py` → `save()/load()` | `hidden` no se guarda. Un agente entrenado con otra arquitectura no se puede volver a cargar (error de forma de tensores). | La configuración completa se guarda y se restaura como una unidad (`DQNConfig`). Prueba: `test_save_load_roundtrip_with_custom_hidden`. |
| 2.4 | Baja | `load()` de ambos agentes | `epsilon_start` se reemplaza por el ε guardado, así que la configuración cargada ya no describe el experimento original. | Estado (ε, contadores) y configuración se guardan por separado, con `format_version`. |
| 2.5 | Baja | `cli.py` → `train` | La DQN guarda solo los últimos pesos. Su desempeño oscila, y los últimos rara vez son los mejores. | `--eval-every N` evalúa la política codiciosa y guarda el mejor punto de control en `*_best.pt`; `--best` lo usa en `load`, `sim`, `render` y `export`. |

## 3. Reproducibilidad y medición

| # | Severidad | Hallazgo | Resolución |
|---|---|---|---|
| 3.1 | Alta | No hay semillas: dos ejecuciones idénticas dan resultados distintos, lo que impide comparar hiperparámetros (Henderson et al., 2018). | `--seed` siembra Python, NumPy, PyTorch, el entorno y el espacio de acciones. Cada agente usa su propio `np.random.Generator`. Prueba: `test_seeded_training_is_reproducible`. |
| 3.2 | Media | Las recompensas de entrenamiento solo se imprimen; no quedan registros. | Registro CSV por episodio (retorno, pasos, ε, pérdida) junto al guardado y comando `rlgames plot`. |
| 3.3 | Media | La evaluación usa estados iniciales aleatorios distintos en cada llamada. | `run_episodes(..., seed=)` evalúa siempre sobre el mismo conjunto de estados iniciales. |
| 3.4 | Baja | Los hiperparámetros son argumentos sueltos del constructor, sin validación. | `dataclasses` inmutables con validación (`gamma` en [0,1], ε coherente…), presets por entorno y `--hp clave=valor` con tipos. Una clave mal escrita falla con un mensaje claro. |

## 4. Ingeniería de software

| # | Hallazgo | Resolución |
|---|---|---|
| 4.1 | Los dos agentes duplican el mismo bucle de entrenamiento. | Un único bucle en `BaseAgent.train()` con ganchos (`_observe`, `_on_step_end`, `_on_episode_end`). |
| 4.2 | `agents/__init__.py` importa la DQN de forma inmediata, así que usar Q-Learning importa PyTorch aunque el registro diga lo contrario. | Importación diferida (`__getattr__`). Verificado: `torch` no se carga para Q-Learning. |
| 4.3 | `ReplayBuffer` usa un `deque` de tuplas; cada muestreo arma 64 arreglos pequeños en Python. | Arreglos NumPy preasignados y muestreo con índices vectorizados. |
| 4.4 | No hay pruebas; la CI solo ejecuta `ruff` y `uv build`. | 32 pruebas con `pytest` (84 % de cobertura), incluidas dos que exigen que los agentes **aprendan** CartPole. CI con ruff (lint y formato), mypy y pruebas en Ubuntu y macOS. |
| 4.5 | `ruff` es dependencia de ejecución. | Movido al grupo `dev` junto con pytest y mypy. |
| 4.6 | En Linux, `uv sync` descarga PyTorch con CUDA (≈ 6,9 GB de `.venv`). | Índice de ruedas CPU para Linux y Windows: 1,1 GB. |
| 4.7 | El mensaje de error de `envs.py` remite a `rl_games/envs/__init__.py`, que no existe. | Ruta corregida; además se rechazan con un mensaje claro los entornos de acciones continuas. |
| 4.8 | `Ctrl+C` durante el entrenamiento termina con una traza. | Salida controlada con código 130; los errores de configuración salen con código 2 y un mensaje legible. |

## 5. Lo que se conservó

La interfaz de línea de comandos (`rlgames init/train/load/sim/render/list/inspect/delete`), la estructura de módulos, el uso de `uv`, la licencia Apache 2.0 y el historial de commits del autor original. La edición añade `plot` y `export`.

## Referencias

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep reinforcement learning that matters. *Proceedings of the AAAI Conference on Artificial Intelligence, 32*(1). https://doi.org/10.1609/aaai.v32i1.11694

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529-533. https://doi.org/10.1038/nature14236

Pardo, F., Tavakoli, A., Levdik, V., & Kormushev, P. (2018). Time limits in reinforcement learning. En *Proceedings of the 35th International Conference on Machine Learning* (Vol. 80, pp. 4045-4054). PMLR.

van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *Proceedings of the AAAI Conference on Artificial Intelligence, 30*(1). https://doi.org/10.1609/aaai.v30i1.10295
