# Changelog

## 0.2.1 (2026-09-28)

### Añadido
- Visor de código junto al simulador: JavaScript en ejecución y su equivalente en Python, con las funciones activas resaltadas en vivo y enlaces a las líneas en GitHub.
- El JavaScript del simulador se separa en módulos (`docs/js/`).
- GitHub Pages publica también `src/rl_games` para el visor.

## 0.2.0 · rl_games v2, edición mejorada (2026-09-28)

### Añadido
- Implementación de referencia de Q-Learning tabular y DQN.
- Double DQN, pérdida Huber, recorte de gradiente, `learning_starts` y `train_freq`.
- Configuración validada (`QLearningConfig`, `DQNConfig`), presets por entorno y `--hp clave=valor`.
- Semillas (`--seed`), registro CSV por episodio, `rlgames plot` y `rlgames export`.
- Evaluación periódica (`--eval-every`) con mejor punto de control (`--best`).
- Simulador HTML5 en `docs/` (CartPole exacto y Lunar Lander ligero) con entrenamiento en el navegador.
- 32 pruebas con pytest, mypy, reglas ampliadas de ruff, pre-commit, CI en Ubuntu y macOS, y despliegue en GitHub Pages.

### Corregido
- `truncated` ya no se trata como estado terminal en el objetivo de Bellman.
- La DQN guarda y restaura su arquitectura (`hidden`).
- Carga segura: `torch.load(weights_only=True)` y tabla Q en `.npz` sin pickle.
- Red objetivo sincronizada por pasos, ε lineal por pasos en la DQN.
- Q-Learning ya no importa PyTorch; ruta correcta en el mensaje de `envs.py`.

### Cambiado
- Un único bucle de entrenamiento en `BaseAgent`; buffer de repetición con arreglos NumPy.
- PyTorch para CPU en Linux y Windows; `ruff` pasa al grupo de desarrollo.
- Formato de guardado v2 (los guardados del original no son compatibles).
