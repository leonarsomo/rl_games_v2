"""rl_games: Q-learning and DQN laboratory on Gymnasium.

The CLI (`rlgames`) is a thin wrapper; everything is reachable from Python:

    from rl_games import registry, evaluate
    agent = registry.create("dqn", "CartPole-v1", seed=0)
    agent.train(300)
    print(evaluate.summarize(evaluate.run_episodes(agent, n_episodes=10)))
"""

__all__ = ["config", "envs", "evaluate", "metrics", "registry", "utils"]
