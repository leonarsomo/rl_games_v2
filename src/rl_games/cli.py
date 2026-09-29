"""Command-line interface for the `rlgames` command.

Contents:
  - cmd_*()         : one function per subcommand (version, list, inspect, init,
                      train, delete, load, sim, render, plot, export)
  - build_parser()  : the argparse parser wiring commands to their options
  - main()          : entry point (returns an exit code)

Argument parsing and output formatting only; behaviour lives in
rl_games.registry, rl_games.evaluate, rl_games.envs and the agents.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rl_games import envs, evaluate, registry
from rl_games.config import parse_overrides
from rl_games.registry import AGENT_CHOICES

if TYPE_CHECKING:
    from rl_games.agents.base import BaseAgent

ENV_ID = envs.DEFAULT_ENV_ID


def _version() -> str:
    try:
        return version("rl_games")
    except Exception:  # pragma: no cover - running from a source tree
        return "dev"


def _path(args: argparse.Namespace) -> Path:
    if getattr(args, "best", False):
        return registry.best_path(args.agent, args.env)
    return registry.save_path(args.agent, args.env)


def _load(args: argparse.Namespace) -> BaseAgent:
    return registry.agent_class(args.agent).load(_path(args))


def _require_save(args: argparse.Namespace) -> bool:
    path = _path(args)
    if not path.exists():
        print(f"No save found at {path}.")
        print(f"Train one with: rlgames train {args.agent} --env {args.env}")
        return False
    return True


# ── commands ─────────────────────────────────────────────────────────────


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"rl_games {_version()}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    print(f"Available agents for {args.env}:\n")
    for agent in AGENT_CHOICES:
        path = registry.save_path(agent, args.env)
        status = "saved" if path.exists() else "no save"
        print(f"  {agent:<14} [{status}]  {path}")
    print(f"\nEnvs with hand-written bounds: {', '.join(envs.OBS_BOUNDS)}")
    print("Others work too if their observation space is bounded.")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    env = envs.make(args.env, seed=args.seed)
    print(f"Environment: {args.env}\n")
    print(f"Observation space : {env.observation_space}")
    print(f"  shape           : {env.observation_space.shape}")
    low = getattr(env.observation_space, "low", None)
    high = getattr(env.observation_space, "high", None)
    if low is not None and high is not None:
        print(f"  low             : {low}")
        print(f"  high            : {high}")
    print(f"\nAction space      : {env.action_space}")
    print(f"Max episode steps : {env.spec.max_episode_steps if env.spec else 'N/A'}")

    print(f"\n-- Sample transitions ({args.steps} steps, random policy) --\n")
    obs, _ = env.reset(seed=args.seed)
    print(f"  Initial state: {np.array2string(obs, precision=3)}\n")
    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(
            f"  step {step:>3} | action={action} | reward={float(reward):+.3f} | "
            f"terminated={terminated} | truncated={truncated}"
        )
        print(f"           state -> {np.array2string(obs, precision=3)}")
        if terminated or truncated:
            obs, _ = env.reset()
            print("           [episode ended, resetting]")
        print()
    env.close()
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    path = registry.save_path(args.agent, args.env)
    if path.exists():
        print(f"Save already exists at {path}. Run 'rlgames delete {args.agent}' first.")
        return 1
    overrides = parse_overrides(args.agent, args.hp)
    registry.create(args.agent, args.env, overrides, seed=args.seed).save(path)
    print(f"Initialized {args.agent} agent.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    overrides = parse_overrides(args.agent, args.hp)
    if args.seed is not None:
        from rl_games.utils import set_global_seed

        set_global_seed(args.seed)
    agent = registry.load_or_create(args.agent, args.env, overrides, seed=args.seed)
    print(
        f"Training {agent.label} on {args.env} for {args.episodes} episodes "
        f"(already trained: {agent.training_episodes})\n"
    )

    # With --eval-every, keep a copy of the best greedy policy seen so far:
    # DQN performance oscillates, and the last weights are often not the best.
    best_path = registry.best_path(args.agent, args.env)
    best = {"score": float("-inf")}

    def on_eval(score: float) -> None:
        if score > best["score"]:
            best["score"] = score
            agent.save(best_path)
            print(f"  [eval] new best {score:.2f} -> {best_path}")

    try:
        log = agent.train(
            total_episodes=args.episodes,
            log_interval=args.log_interval,
            eval_every=args.eval_every,
            eval_callback=on_eval if args.eval_every else None,
        )
    except KeyboardInterrupt:
        print("\nInterrupted -- the partial run is NOT saved. (Use fewer episodes per call.)")
        return 130

    agent.save(registry.save_path(args.agent, args.env))
    csv = registry.log_path(args.agent, args.env)
    log.to_csv(csv, append=True)
    print(f"Training log appended to {csv}")
    print(f"Last-100 mean reward: {log.mean_reward(100):.2f}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    removed = False
    paths = (
        registry.save_path(args.agent, args.env),
        registry.best_path(args.agent, args.env),
        registry.log_path(args.agent, args.env),
    )
    for path in paths:
        if path.exists():
            path.unlink()
            print(f"Deleted {path}")
            removed = True
    if not removed:
        print(f"No save found for {args.agent} on {args.env}")
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    if not _require_save(args):
        return 1
    agent = _load(args)
    print(agent.info())
    if args.eval:
        print(f"\nEvaluating greedily ({args.episodes} episodes) ...")
        stats = evaluate.summarize(
            evaluate.run_episodes(agent, n_episodes=args.episodes, seed=args.seed)
        )
        print(
            f"  Mean reward: {stats['mean']:.2f} ± {stats['std']:.2f} "
            f"(min {stats['min']:.1f}, max {stats['max']:.1f})"
        )
    return 0


def cmd_sim(args: argparse.Namespace) -> int:
    if not _require_save(args):
        return 1
    agent = _load(args)
    env = envs.make(args.env)
    all_rewards: list[float] = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=None if args.seed is None else args.seed + ep)
        done, total, step = False, 0.0, 0
        terminated = truncated = False
        print(f"== Episode {ep}/{args.episodes} ==\n")
        print(f"  initial state: {np.array2string(obs, precision=3)}\n")

        while not done:
            step += 1
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)
            if args.steps is None or step <= args.steps:
                print(
                    f"  step {step:>4} | action={action:>2} | "
                    f"reward={float(reward):+8.3f} | total={total:+9.2f}"
                )
                if args.verbose:
                    print(f"           state -> {np.array2string(obs, precision=3)}")

        if args.steps is not None and step > args.steps:
            print(f"  ... ({step - args.steps} more steps) ...")
        outcome = "TRUNCATED (time limit)" if truncated else "TERMINATED"
        print(f"\n  Result: {outcome} | Steps: {step} | Total reward: {total:+.2f}\n")
        all_rewards.append(total)

    env.close()
    if len(all_rewards) > 1:
        print(
            f"Summary over {len(all_rewards)} episodes: "
            f"mean={np.mean(all_rewards):+.2f} ± {np.std(all_rewards):.2f}"
        )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    if not _require_save(args):
        return 1
    agent = _load(args)
    env = envs.make(args.env, render_mode="human")
    try:
        returns = evaluate.run_episodes(agent, env, n_episodes=args.episodes, seed=args.seed)
        for ep, total in enumerate(returns, 1):
            print(f"Episode {ep}/{args.episodes} | Reward: {total:.2f}")
    finally:
        env.close()
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from rl_games.metrics import TrainingLog, moving_average

    csv = registry.log_path(args.agent, args.env)
    if not csv.exists():
        print(f"No training log at {csv}")
        return 1
    records = TrainingLog.read_csv(csv)
    rewards = [r.reward for r in records]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Install it with: uv sync --extra plot")
        ma = moving_average(rewards, args.window)
        print(f"{len(rewards)} episodes. Last moving average ({args.window}): {ma[-1]:.2f}")
        return 1

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    x = [r.episode for r in records]
    ax.plot(x, rewards, lw=0.8, alpha=0.35, label="Retorno por episodio")
    ax.plot(x, moving_average(rewards, args.window), lw=2, label=f"Media móvil ({args.window})")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Retorno")
    ax.set_title(f"{args.agent} en {args.env}")
    ax.grid(alpha=0.3)
    ax.legend()
    out = Path(args.out) if args.out else csv.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out)
    print(f"Saved plot to {out}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    if args.agent != "dqn":
        print("Only DQN agents can be exported to the web simulator.")
        return 1
    if not _require_save(args):
        return 1
    agent = _load(args)
    out = Path(args.out or f"docs/models/dqn_{args.env}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agent.export_json()))  # type: ignore[attr-defined]
    print(f"Exported {agent.label} weights to {out} ({out.stat().st_size / 1024:.1f} KB)")
    return 0


# ── argument parser ──────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlgames",
        description="Train and evaluate RL agents on Gymnasium environments",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def env_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--env", default=ENV_ID, help=f"Gymnasium env ID (default: {ENV_ID})")

    def seed_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    def agent_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("agent", choices=AGENT_CHOICES)

    def best_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--best",
            action="store_true",
            help="Use the best checkpoint saved by 'train --eval-every'",
        )

    def hp_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--hp",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Override a hyperparameter, e.g. --hp lr=3e-4 --hp hidden=64,64",
        )

    p = sub.add_parser("version", help="Show the package version")
    p.set_defaults(func=cmd_version)

    p = sub.add_parser("list", help="List agents and their save status")
    env_arg(p)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("inspect", help="Show spaces and sample random transitions")
    env_arg(p)
    seed_arg(p)
    p.add_argument("--steps", type=int, default=5, help="Random steps to sample (default: 5)")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("init", help="Initialize a new (untrained) agent and save it")
    agent_arg(p)
    env_arg(p)
    seed_arg(p)
    hp_arg(p)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("train", help="Train an agent (resumes a save if present)")
    agent_arg(p)
    env_arg(p)
    seed_arg(p)
    hp_arg(p)
    p.add_argument("--episodes", type=int, default=1_000, help="Training episodes (default: 1000)")
    p.add_argument("--log-interval", type=int, default=None, help="Print every N episodes")
    p.add_argument(
        "--eval-every", type=int, default=0, help="Greedy evaluation every N episodes (0 = off)"
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("delete", help="Delete a saved agent and its training log")
    agent_arg(p)
    env_arg(p)
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("load", help="Load a saved agent, show its info, optionally evaluate")
    agent_arg(p)
    env_arg(p)
    seed_arg(p)
    p.add_argument("--eval", action="store_true", help="Run a greedy evaluation")
    p.add_argument("--episodes", type=int, default=10, help="Evaluation episodes (default: 10)")
    best_arg(p)
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("sim", help="Simulate episodes with a trained agent (text output)")
    agent_arg(p)
    env_arg(p)
    seed_arg(p)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=None, help="Only print the first N steps")
    p.add_argument("--verbose", action="store_true", help="Print full state vectors")
    best_arg(p)
    p.set_defaults(func=cmd_sim)

    p = sub.add_parser("render", help="Render episodes in a graphical window")
    agent_arg(p)
    env_arg(p)
    seed_arg(p)
    p.add_argument("--episodes", type=int, default=1)
    best_arg(p)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("plot", help="Plot the learning curve from the training log (PNG)")
    agent_arg(p)
    env_arg(p)
    p.add_argument("--window", type=int, default=50, help="Moving-average window")
    p.add_argument("--out", default=None, help="Output PNG path")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("export", help="Export DQN weights as JSON for docs/index.html")
    agent_arg(p)
    env_arg(p)
    p.add_argument("--out", default=None, help="Output JSON path")
    best_arg(p)
    p.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:  # config / env validation errors → clean message
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
